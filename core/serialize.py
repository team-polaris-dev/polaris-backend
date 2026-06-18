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


def build_graph(state: dict, answer: str = "") -> dict:
    """graph_facts/graph_paths → {nodes:[{id,label,category}], edges:[{...}]}.

    graph_paths[i] 는 [노드, 관계, 노드, 관계, …] 로 교차 구성되고, 엣지별 근거는
    graph_path_sources[i](문서 rcept_no)·graph_path_chunks[i](추출 엣지의 chunk_id)에
    행 단위로 정렬돼 들어온다(adapt_to_legacy 가 paths 와 같은 루프에서 채움).
    예전엔 graph_facts[i].source 로 읽었는데 facts(전체 hit)와 paths(망 엣지만)의 길이가
    달라 엣지·출처가 어긋났다 — 그 버그를 정렬 배열로 교정.

    answer 가 주어지면, 답변이 실제로 언급한 회사들의 부분그래프만 남긴다(멀티홉 뼈대).
    이웃 전체를 덤프하지 않고 '답의 근거 구조'만 보여주기 위함. 큐레이션 결과가 너무
    적으면(엣지<3) 전체를 유지한다(빈 패널 방지).
    """
    paths = state.get("graph_paths") or []
    path_sources = state.get("graph_path_sources") or []
    path_chunks = state.get("graph_path_chunks") or []
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
        rcept_no = str(path_sources[i]) if i < len(path_sources) else ""
        chunk_id = str(path_chunks[i]) if i < len(path_chunks) else ""

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
                    "rcept_no": rcept_no,   # 문서 출처(모든 엣지)
                    "chunk_id": chunk_id,   # 청크 출처(추출 엣지만, 없으면 '')
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

    # 0) RDB 정형 데이터(재무수치) — 벡터 문서보다 앞에 배치
    rdb_docs = _build_rdb_documents(state.get("rdb_results") or [])
    documents.extend(rdb_docs)
    seen.update(d["chunk_id"] for d in rdb_docs if d["chunk_id"])

    # 1) Vector 청크 = 본문이 있는 원본 발췌
    for v in vec_results:
        extra = v.get("extra") or {}
        rcept_no = str(extra.get("rcept_no") or "")
        chunk_id = str(v.get("source") or "")
        dedup_key = chunk_id or rcept_no
        if dedup_key and dedup_key in seen:
            continue
        if dedup_key:
            seen.add(dedup_key)
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
                "text": v.get("value") or "",
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
