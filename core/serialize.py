# core/serialize.py — LangGraph 최종 state → 프론트엔드(우측 패널)용 페이로드 변환
"""그래프 실행 결과(state)에서 프론트가 바로 렌더링할 수 있는 두 가지 구조를 만든다.

1) graph    : 기업 관계도 (Neo4j graph_facts/graph_paths 기반) — react-force-graph 규격
2) documents: 원본 문서 (Vector 청크 + 그래프 근거 rcept_no) — MariaDB document_index 로 메타 보강

외부 DB(Neo4j/Qdrant)가 꺼져 있어 검색 결과가 비면 빈 구조를 반환한다(프론트가 탭을 숨김).
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from tool.rdb_client import execute_sql_query
# 관계 한글라벨·회사명 정규화는 SSOT(config)에서. _norm_co 는 답변↔노드명 매칭용 별칭.
from config.relations import REL_LABELS
from config.entities import normalize_corp_name as _norm_co
from config.graphrag import (
    PANEL_CURATION_MIN_EDGES,
    PANEL_CURATION_KEEP_MIN,
    PANEL_MENTION_MIN_LEN,
)

_RCEPT_RE = re.compile(r"[^0-9A-Za-z]")


def _humanize_rel(rel: str) -> str:
    return REL_LABELS.get(rel, rel.replace("_", " ").strip() or "관계")



# IFRS/DART account_id → 한글 라벨
_ACCOUNT_KR: dict[str, str] = {
    "ifrs-full_Revenue": "매출액",
    "dart_OperatingIncomeLoss": "영업이익",
    "ifrs-full_ProfitLoss": "당기순이익",
    "ifrs-full_GrossProfit": "매출총이익",
    "ifrs-full_Assets": "총자산",
    "ifrs-full_Liabilities": "총부채",
    "ifrs-full_Equity": "자본총계",
    "ifrs-full_CashAndCashEquivalents": "현금및현금성자산",
    "ifrs-full_CurrentAssets": "유동자산",
    "ifrs-full_CurrentLiabilities": "유동부채",
    "ifrs-full_CostOfSales": "매출원가",
    "ifrs-full_ProfitLossBeforeTax": "세전순이익",
}

_SUMMARY_ORDER = ["매출액", "영업이익", "당기순이익"]


def _fmt_krw(v: Any) -> str:
    """원화 금액을 조/억 단위로 사람이 읽기 쉽게 변환."""
    try:
        n = float(v)
        if abs(n) >= 1e12:
            return f"{n / 1e12:,.1f}조원"
        if abs(n) >= 1e8:
            return f"{n / 1e8:,.0f}억원"
        return f"{n:,.0f}원"
    except Exception:
        return str(v)


def _build_rdb_documents(rdb_results: list[dict]) -> list[dict]:
    """rdb_row 타입 UnifiedResult를 corp/year 단위로 묶어 재무카드 문서 리스트를 반환.

    rdb_results 는 SQL 한 행 = 재무 지표 하나이므로 같은 기업·연도 지표를
    하나의 카드로 집약한다. 중복 행도 이 단계에서 제거한다.
    """
    # (corp_name, year) → {account_id: amount}  (중복은 마지막 값 우선)
    groups: dict[tuple[str, Any], dict[str, Any]] = defaultdict(dict)

    for r in rdb_results:
        if r.get("type") != "rdb_row":
            continue
        val = r.get("value") or {}
        if not isinstance(val, dict):
            continue
        account_id = str(val.get("account_id") or "")
        amount = val.get("value")
        year = val.get("bsns_year")
        corp_name = r.get("name") or ""
        if not corp_name or not account_id:
            continue
        groups[(corp_name, year)][account_id] = amount

    documents: list[dict] = []
    for (corp_name, year), metrics in groups.items():
        label_map = {
            _ACCOUNT_KR.get(aid, aid): amt for aid, amt in metrics.items()
        }
        lines = [f"{lbl}: {_fmt_krw(amt)}" for lbl, amt in label_map.items()]
        summary = " · ".join(
            f"{p} {_fmt_krw(label_map[p])}"
            for p in _SUMMARY_ORDER
            if p in label_map
        )
        documents.append(
            {
                "rcept_no": "",
                "chunk_id": f"rdb_{corp_name}_{year}",
                "corp_name": corp_name,
                "title": f"주요 재무지표 ({year}년)",
                "doc_type": "재무수치",
                "date": str(year) if year else "",
                "summary": summary,
                "section_path": "재무제표",
                "year": year,
                "score": None,
                "text": "\n".join(lines),
                "source_kind": "rdb",
            }
        )

    # 연도 최신 순 정렬
    documents.sort(key=lambda d: d["year"] or 0, reverse=True)
    return documents


# embedding_text 첫 줄 헤더: [회사 · 문서 (YYYY.MM) · 섹션 …]
_RDB_HEADER_RE = re.compile(r"^\s*\[(?P<body>.+?)\]\s*(?:\n|$)", re.DOTALL)
# "사업보고서 (2025.12)" → doc_type + 연도 추출용
_DOC_DATE_RE = re.compile(r"\((?P<date>\d{4})[.\-/]?(?P<month>\d{1,2})?\)")

# 본문 정리용 — PDF 추출 과정에서 깨진 공백/줄바꿈을 읽기 좋게 다듬는다.
_LEAD_HEADER_RE = re.compile(r"^\s*\[[^\]]*\]\s*")          # 맨 앞 [회사 · 문서 · 섹션] 헤더 줄
# 한 줄 전체가 '연 결 재 무 제 표'처럼 한 글자씩 벌어진 경우(뒤 구두점 허용)
_SPACED_LINE_RE = re.compile(r"^(?:[가-힣] ){2,}[가-힣][.,)\]]?$")
_WS_RE = re.compile(r"\s+")                                   # 깨진 줄바꿈·연속 공백
# 표 인코딩: '헤더: 열1 | 열2 | 열3' 줄 + '맨앞값 | 키=값; 키=값' 데이터 줄
_TABLE_HEADER_RE = re.compile(r"헤더:\s*(?P<cols>.+)$")
# 넓은 단일행 표(예: 종속기업(1)…(26))는 항목/값 2열로 전치할 때의 열 수 임계
_TABLE_TRANSPOSE_MIN_COLS = 6


def _md_cell(v: Any) -> str:
    """마크다운 표 셀로 안전화 — 구분자 '|' 와 줄바꿈 제거."""
    return str(v or "").replace("|", "/").replace("\n", " ").strip()


def _parse_table_row(line: str, cols: list[str]) -> list[str]:
    """데이터 줄 → 헤더 열 순서에 맞춘 값 리스트.

    형식: '맨앞값 | 키=값; 키=값; …'  또는  '키=값; 키=값; …'(맨앞값 없음).
    맨앞값은 첫 열, 나머지는 '키=값'을 헤더명으로 매칭한다.
    """
    bare: str | None = None
    rest = line
    if " | " in line:
        bare, rest = line.split(" | ", 1)
    pairs: dict[str, str] = {}
    for kv in rest.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            pairs[k.strip()] = v.strip()
    values: list[str] = []
    for idx, h in enumerate(cols):
        if idx == 0 and bare is not None:
            values.append(bare.strip())
        else:
            values.append(pairs.get(h, ""))
    return values


def _rows_to_markdown(cols: list[str], rows: list[list[str]], caption: str = "") -> str:
    """헤더·데이터 → GFM 마크다운 표 문자열.

    넓은 단일행(열 많고 행 1개)은 '항목 | 값' 2열로 전치해 읽기 쉽게 한다.
    """
    if len(rows) == 1 and len(cols) > _TABLE_TRANSPOSE_MIN_COLS:
        hdr = ["항목", "값"]
        body = [
            [_md_cell(cols[k]), _md_cell(rows[0][k])]
            for k in range(min(len(cols), len(rows[0])))
            if _md_cell(rows[0][k])
        ]
    else:
        hdr = [_md_cell(c) for c in cols]
        body = [
            [_md_cell(v) for v in (row + [""] * len(hdr))[: len(hdr)]]
            for row in rows
        ]
    out: list[str] = []
    if caption:
        out.append(caption)
    out.append("| " + " | ".join(hdr) + " |")
    out.append("| " + " | ".join(["---"] * len(hdr)) + " |")
    for row in body:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _parse_table_block(lines: list[str], start: int) -> tuple[str | None, int]:
    """lines[start] 가 '헤더:' 표 머리줄이면 표 블록을 파싱해 (마크다운, 다음인덱스) 반환.

    표가 아니면 (None, start+1).
    """
    header_line = lines[start].strip()
    m = _TABLE_HEADER_RE.search(header_line)
    if not m:
        return None, start + 1
    caption = header_line[: m.start()].strip()  # '(단위 : 백만원)' 같은 머리말
    cols = [c.strip() for c in m.group("cols").split("|") if c.strip()]
    if len(cols) < 2:
        return None, start + 1
    rows: list[list[str]] = []
    j = start + 1
    while j < len(lines):
        l = lines[j].strip()
        # 데이터 줄은 키=값(=) 또는 셀 구분(|) 을 포함한다. 그 외/빈 줄이면 표 종료.
        if not l or ("=" not in l and "|" not in l) or "헤더:" in l:
            break
        rows.append(_parse_table_row(l, cols))
        j += 1
    if not rows:
        return None, start + 1
    return _rows_to_markdown(cols, rows, caption), j


def _clean_excerpt_text(text: Any, drop_header: bool = True) -> str:
    """공시 원문 발췌를 읽기 좋게 정리(마크다운 반환).

    1) 카드 제목과 중복되는 맨 앞 '[…]' 헤더 줄 제거
    2) '헤더: a | b' 표 블록은 GFM 마크다운 표로 변환(프론트가 <table> 로 렌더)
    3) PDF에서 한 줄 전체가 글자 단위로 벌어진 줄('연 결 재 무')은 공백 제거
    4) 표가 아닌 깨진 줄바꿈은 단일 공백으로 합쳐 흐르는 문장으로
    """
    s = str(text or "")
    if not s.strip():
        return ""
    if drop_header:
        s = _LEAD_HEADER_RE.sub("", s, count=1)
    lines = s.split("\n")
    blocks: list[str] = []
    prose: list[str] = []

    def flush_prose() -> None:
        if prose:
            joined = _WS_RE.sub(" ", " ".join(prose)).strip()
            if joined:
                blocks.append(joined)
            prose.clear()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if "헤더:" in line and "|" in line:
            md, nxt = _parse_table_block(lines, i)
            if md:
                flush_prose()
                blocks.append(md)
                i = nxt
                continue
        if _SPACED_LINE_RE.match(line):
            line = line.replace(" ", "")
        prose.append(line)
        i += 1
    flush_prose()
    # 표 블록은 앞뒤로 빈 줄이 있어야 마크다운 표로 인식된다.
    return "\n\n".join(blocks)


def _dedup_key(text: Any) -> str:
    """내용 기준 중복 판정용 키 — 공백 접고 앞 200자. rdb 원문↔vec 청크 교차 중복 제거."""
    return _WS_RE.sub(" ", str(text or "")).strip()[:200]


def _parse_rdb_text_header(text: str) -> dict[str, Any]:
    """embedding_text 첫 줄 '[회사 · 문서(YYYY.MM) · 섹션]' → 메타 dict.

    형식이 어긋나면 빈 메타를 돌려준다(방어). 본문 text 는 호출부에서 통째로 쓴다.
    """
    meta: dict[str, Any] = {
        "corp_name": "", "doc_type": "", "title": "",
        "section_path": "", "year": None, "date": "",
    }
    m = _RDB_HEADER_RE.match(text or "")
    if not m:
        return meta
    parts = [p.strip() for p in m.group("body").split("·") if p.strip()]
    if not parts:
        return meta
    meta["corp_name"] = parts[0]
    if len(parts) >= 2:
        doc = parts[1]
        dm = _DOC_DATE_RE.search(doc)
        if dm:
            meta["year"] = int(dm.group("date"))
            mon = dm.group("month")
            meta["date"] = f"{dm.group('date')}.{mon}" if mon else dm.group("date")
        meta["doc_type"] = _DOC_DATE_RE.sub("", doc).strip()
        meta["title"] = doc
    if len(parts) >= 3:
        meta["section_path"] = " · ".join(parts[2:])
    return meta


def _build_rdb_text_documents(rdb_results: list[dict]) -> list[dict]:
    """rdb_row 중 '공시 원문 발췌'(value.embedding_text) 행 → 원본 문서 카드 리스트.

    embedding_text 를 가진 행이면 모두 대상. 재무지표와 청크를 JOIN 한 SQL 은
    account_id 와 embedding_text 를 한 행에 함께 담으므로(예: 매출액 × 청크),
    account_id 유무로 거르지 않는다. 대신 같은 청크가 여러 지표 행에 중복으로
    딸려 오므로 embedding_text 기준으로 중복을 제거한다.
    재무수치(account_id/value)는 _build_rdb_documents 가 별도로 카드를 만든다.
    """
    documents: list[dict] = []
    seen_text: set[str] = set()
    for i, r in enumerate(rdb_results):
        if r.get("type") != "rdb_row":
            continue
        val = r.get("value")
        if not isinstance(val, dict):
            continue
        body = val.get("embedding_text")
        if not body:
            continue
        key = str(body)
        if key in seen_text:
            continue
        seen_text.add(key)
        meta = _parse_rdb_text_header(key)
        documents.append(
            {
                "rcept_no": str(r.get("source") or ""),
                "chunk_id": f"rdbtext_{i}",
                "corp_name": meta["corp_name"] or r.get("name") or "",
                "title": meta["title"] or "공시 원문",
                "doc_type": meta["doc_type"] or "공시원문",
                "date": meta["date"],
                "summary": "",
                "section_path": meta["section_path"],
                "year": meta["year"],
                "score": None,
                "text": _clean_excerpt_text(body),
                "source_kind": "rdb_text",
            }
        )
    return documents


_CO_STRIP_RE = re.compile(r"주식회사|\(주\)|㈜|\s+")


def _norm_co(s: str) -> str:
    """회사명 정규화 — 접미사·공백 제거 + SK↔에스케이/LG↔엘지 별칭. 답변 매칭용."""
    s = _CO_STRIP_RE.sub("", s or "")
    s = s.replace("에스케이", "SK").replace("엘지", "LG")
    return s.lower()


def _doc_meta_by_rcept(rcept_nos: list[str]) -> dict[str, dict[str, Any]]:
    """document_index 에서 rcept_no 묶음의 메타(회사·공시일·제목·요약)를 조회한다.

    execute_sql_query 는 파라미터 바인딩이 없으므로 rcept_no 를 영숫자만 남겨
    인젝션 여지를 없앤 뒤 IN 리스트로 만든다(rcept_no 는 14자리 숫자).
    """
    safe = sorted({_RCEPT_RE.sub("", r) for r in rcept_nos if r})
    if not safe:
        return {}
    in_list = ", ".join(f"'{r}'" for r in safe)
    sql = (
        "SELECT rcept_no, corp_name, doc_type, date, title, summary_short "
        f"FROM document_index WHERE rcept_no IN ({in_list})"
    )
    result = execute_sql_query(sql, max_rows=len(safe))
    if not result.get("ok"):
        return {}
    meta: dict[str, dict[str, Any]] = {}
    for row in result["rows"]:
        meta[str(row.get("rcept_no", ""))] = {
            "corp_name": row.get("corp_name") or "",
            "doc_type": row.get("doc_type") or "",
            "date": str(row.get("date") or ""),
            "title": row.get("title") or "",
            "summary": row.get("summary_short") or "",
        }
    return meta


# 작은따옴표만 이스케이프(execute_sql_query 가 파라미터 바인딩을 안 하므로)
def _sql_str(s: str) -> str:
    return str(s or "").replace("'", "''")


def _rcept_by_corp_title(pairs: set[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """(회사명, 보고서 라벨) → rcept_no 역조회.

    rdb 원문은 JOIN SQL 이 rcept_no 를 SELECT 하지 않아 DART 링크를 만들 수 없다.
    document_index.title 은 DART report_nm('사업보고서 (2025.12)' 등)이고, 원문 헤더
    가운데 라벨이 바로 그 값이라 (corp_name, title) 로 정확히 되짚을 수 있다.
    """
    conds = [
        f"(corp_name = '{_sql_str(c)}' AND title = '{_sql_str(t)}')"
        for c, t in pairs
        if c and t
    ]
    if not conds:
        return {}
    sql = (
        "SELECT corp_name, title, rcept_no FROM document_index "
        f"WHERE {' OR '.join(conds)} ORDER BY date"
    )
    result = execute_sql_query(sql, max_rows=len(conds) * 5)
    if not result.get("ok"):
        return {}
    out: dict[tuple[str, str], str] = {}
    for row in result["rows"]:
        # ORDER BY date 오름차순 → 같은 보고서명이 여러 건이면 최신(마지막)으로 덮인다
        out[(str(row.get("corp_name") or ""), str(row.get("title") or ""))] = str(
            row.get("rcept_no") or ""
        )
    return out


def build_graph(state: dict, answer: str = "") -> dict:
    """graph_facts/graph_paths → {nodes:[{id,label,category}], edges:[{...}]}.

    graph_paths[i] 는 [노드, 관계, 노드, 관계, …] 로 교차 구성되고(rag._row_to_unified),
    graph_facts[i].source 가 그 경로의 근거 rcept_no 다(둘은 행 단위로 정렬됨).

    answer 가 주어지면, 답변이 실제로 언급한 회사들의 부분그래프만 남긴다(멀티홉 뼈대).
    이웃 전체를 덤프하지 않고 '답의 근거 구조'만 보여주기 위함. 큐레이션 결과가 너무
    적으면(엣지<3) 전체를 유지한다(빈 패널 방지).
    """
    facts = state.get("graph_facts") or []
    paths = state.get("graph_paths") or []
    if not paths:
        return {"nodes": [], "edges": []}

    # 인물(임원·개인 주주)은 회사 관계 그래프에서 제외 — 속성이지 회사↔회사 망이 아님
    person_names = {
        h.get("name") for h in (state.get("graph_hits") or [])
        if h.get("label") == "person" and h.get("name")
    }

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for i, path in enumerate(paths):
        if not path:
            continue
        rcept_no = ""
        if i < len(facts):
            rcept_no = str(facts[i].get("source") or "")

        names = [p for idx, p in enumerate(path) if idx % 2 == 0 and p]
        rels = [p for idx, p in enumerate(path) if idx % 2 == 1]

        for name in names:
            if name in person_names:
                continue
            nodes.setdefault(name, {"id": name, "label": name, "category": "기업"})

        for j in range(len(names) - 1):
            rel = rels[j] if j < len(rels) else ""
            src, tgt = names[j], names[j + 1]
            if src in person_names or tgt in person_names:
                continue
            key = (src, tgt, rel)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append(
                {
                    "source": src,
                    "target": tgt,
                    "type": rel,
                    "label": _humanize_rel(rel),
                    "rcept_no": rcept_no,
                }
            )

    # 큐레이션 1순위 — 시드 + 시드의 '직접 이웃' 사이 엣지만. 2차 이웃(질의와 무관한
    # 다른 회사로 뻗는 가지)을 잘라 답의 핵심 구조만 남긴다.
    seeds = {s.get("name") for s in (state.get("graph_seeds") or []) if s.get("name")}
    seed_nodes = {n for n in nodes if n in seeds}
    if seed_nodes and len(edges) > PANEL_CURATION_MIN_EDGES:
        core = set(seed_nodes)
        for e in edges:
            if e["source"] in seed_nodes:
                core.add(e["target"])
            if e["target"] in seed_nodes:
                core.add(e["source"])
        cur = [e for e in edges if e["source"] in core and e["target"] in core]
        if len(cur) >= PANEL_CURATION_KEEP_MIN:
            keep = {e["source"] for e in cur} | {e["target"] for e in cur}
            return {"nodes": [nodes[n] for n in nodes if n in keep], "edges": cur}

    # 큐레이션 2순위(시드 매칭 실패 시) — 답변이 언급한 회사들 사이 엣지만
    na = _norm_co(answer)
    if na and len(edges) > PANEL_CURATION_MIN_EDGES:
        mentioned = {
            nid for nid in nodes
            if len(_norm_co(nid)) >= PANEL_MENTION_MIN_LEN and _norm_co(nid) in na
        }
        cur = [e for e in edges if e["source"] in mentioned and e["target"] in mentioned]
        if len(cur) >= PANEL_CURATION_KEEP_MIN:
            keep = {e["source"] for e in cur} | {e["target"] for e in cur}
            return {"nodes": [nodes[n] for n in nodes if n in keep], "edges": cur}

    return {"nodes": list(nodes.values()), "edges": edges}


def build_documents(state: dict) -> list[dict]:
    """Vector 청크 → 원본 문서 리스트(메타 보강).

    각 항목은 우측 '원본 문서' 탭 카드 한 장에 대응한다. Vector 청크는 본문(text)을 포함한다.
    로그에 기록된 vec_results만 표시한다.
    """
    vec_results = state.get("vec_results") or []

    rcept_nos: list[str] = []
    for v in vec_results:
        rn = str((v.get("extra") or {}).get("rcept_no") or "")
        if rn:
            rcept_nos.append(rn)

    meta = _doc_meta_by_rcept(rcept_nos)

    documents: list[dict] = []
    seen: set[str] = set()
    # 같은 본문이 rdb 원문·vec 청크로 두 번 나오지 않게 내용 기준으로도 거른다
    seen_text: set[str] = set()

    # 0) RDB 정형 데이터(재무수치) — 벡터 문서보다 앞에 배치
    rdb_results = state.get("rdb_results") or []
    rdb_docs = _build_rdb_documents(rdb_results)
    documents.extend(rdb_docs)
    seen.update(d["chunk_id"] for d in rdb_docs if d["chunk_id"])

    # 0-1) RDB 공시 원문 발췌(embedding_text) — 그동안 버려지던 원문을 문서로 살린다
    rdb_text_docs = _build_rdb_text_documents(rdb_results)
    # JOIN SQL 이 rcept_no 를 빠뜨려 링크가 없는 원문은 (회사명, 보고서명)으로 역조회해 채운다
    need = {
        (d["corp_name"], d["title"])
        for d in rdb_text_docs
        if not d["rcept_no"] and d["corp_name"] and d["title"]
    }
    if need:
        rcept_map = _rcept_by_corp_title(need)
        for d in rdb_text_docs:
            if not d["rcept_no"]:
                d["rcept_no"] = rcept_map.get((d["corp_name"], d["title"]), "")
    for d in rdb_text_docs:
        tkey = _dedup_key(d["text"])
        if tkey and tkey in seen_text:
            continue
        if tkey:
            seen_text.add(tkey)
        documents.append(d)
        if d["chunk_id"]:
            seen.add(d["chunk_id"])

    # 1) Vector 청크 = 본문이 있는 원본 발췌
    for v in vec_results:
        extra = v.get("extra") or {}
        rcept_no = str(extra.get("rcept_no") or "")
        chunk_id = str(v.get("source") or "")
        dedup_key = chunk_id or rcept_no
        if dedup_key and dedup_key in seen:
            continue
        text = _clean_excerpt_text(v.get("value"))
        tkey = _dedup_key(text)
        if tkey and tkey in seen_text:  # rdb 원문으로 이미 나온 본문이면 건너뛴다
            continue
        if dedup_key:
            seen.add(dedup_key)
        if tkey:
            seen_text.add(tkey)
        m = meta.get(rcept_no, {})
        documents.append(
            {
                "rcept_no": rcept_no,
                "chunk_id": chunk_id,
                "corp_name": extra.get("corp_name") or v.get("name") or m.get("corp_name") or "",
                "title": extra.get("title") or m.get("title") or "",
                "doc_type": extra.get("doc_type") or m.get("doc_type") or "",
                "date": m.get("date") or "",
                "summary": m.get("summary") or "",
                "section_path": extra.get("section_path") or "",
                "year": extra.get("year"),
                "score": extra.get("score"),
                "text": text,
                "source_kind": "vector",
            }
        )

    return documents


def build_financials(state: dict) -> list[dict]:
    """rdb_results → 프론트 차트용 재무지표 그룹 리스트.

    반환 형태:
      [
        {
          "corp_name": "삼성전자",
          "year": 2025,
          "unit": "조원",          # 해당 그룹의 통일 단위
          "metrics": [
            {"label": "매출액", "value": 333.6, "unit": "조원"},
            ...
          ]
        },
        ...
      ]
    연도 내림차순 정렬, 동일 기업·연도 중복 제거.
    """
    rdb_results = state.get("rdb_results") or []
    # (corp_name, year) → {account_id: float_amount}
    groups: dict[tuple[str, Any], dict[str, float]] = defaultdict(dict)

    for r in rdb_results:
        if r.get("type") != "rdb_row":
            continue
        val = r.get("value") or {}
        if not isinstance(val, dict):
            continue
        aid  = str(val.get("account_id") or "")
        amt  = val.get("value")
        year = val.get("bsns_year")
        corp = r.get("name") or ""
        if corp and aid and amt is not None:
            try:
                groups[(corp, year)][aid] = float(amt)
            except (TypeError, ValueError):
                pass

    if not groups:
        return []

    result: list[dict] = []
    account_order = list(_ACCOUNT_KR.keys())

    for (corp, year), raw in sorted(groups.items(), key=lambda x: -(x[0][1] or 0)):
        max_abs = max(abs(v) for v in raw.values()) if raw else 0
        if max_abs >= 1e12:
            unit, divisor = "조원", 1e12
        else:
            unit, divisor = "억원", 1e8

        # 지정 순서대로 정렬 후 나머지 추가
        ordered_ids = [aid for aid in account_order if aid in raw]
        ordered_ids += [aid for aid in raw if aid not in account_order]

        metrics = [
            {
                "label": _ACCOUNT_KR.get(aid, aid),
                "value": round(raw[aid] / divisor, 1),
                "unit": unit,
            }
            for aid in ordered_ids
        ]

        result.append({"corp_name": corp, "year": year, "unit": unit, "metrics": metrics})

    return result


def serialize_state(state: dict) -> dict:
    """최종 state → {graph, documents, financials, panel}.

    panel: 프론트가 자동으로 펼칠 탭 힌트.
      'graph'     → 관계도 데이터 있음
      'documents' → 원본 문서만 있음
      'none'      → 우측 패널 표시 안 함
    """
    msgs = state.get("messages") or []
    answer = ""
    if msgs:
        last = msgs[-1]
        answer = getattr(last, "content", "") or (
            last.get("content", "") if isinstance(last, dict) else ""
        )
    graph      = build_graph(state, answer)
    documents  = build_documents(state)
    financials = build_financials(state)

    if graph["edges"]:
        panel = "graph"
    elif documents:
        panel = "documents"
    else:
        panel = "none"

    return {"graph": graph, "documents": documents, "financials": financials, "panel": panel}
