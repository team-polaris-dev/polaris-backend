# core/serialize.py — LangGraph 최종 state → 프론트엔드(우측 패널)용 페이로드 변환
"""그래프 실행 결과(state)에서 프론트가 바로 렌더링할 수 있는 두 가지 구조를 만든다.

1) graph    : 기업 관계도 (Neo4j graph_facts/graph_paths 기반) — react-force-graph 규격
2) documents: 원본 문서 (Vector 청크 + 그래프 근거 rcept_no) — MariaDB document_index 로 메타 보강

외부 DB(Neo4j/Qdrant)가 꺼져 있어 검색 결과가 비면 빈 구조를 반환한다(프론트가 탭을 숨김).
"""
from __future__ import annotations

import re
from typing import Any

from tool.rdb_client import execute_sql_query

# 그래프 관계 타입 → 한글 라벨
REL_LABELS: dict[str, str] = {
    "IS_SUBSIDIARY_OF": "자회사",
    "EXECUTIVE_OF": "임원",
    "IS_MAJOR_SHAREHOLDER_OF": "대주주",
    "SUPPLIES_TO": "공급",
    "ACQUIRES": "인수",
    "INVESTS": "투자",
}

_RCEPT_RE = re.compile(r"[^0-9A-Za-z]")


def _humanize_rel(rel: str) -> str:
    return REL_LABELS.get(rel, rel.replace("_", " ").strip() or "관계")


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


def build_graph(state: dict) -> dict:
    """graph_facts/graph_paths → {nodes:[{id,label,category}], edges:[{...}]}.

    graph_paths[i] 는 [노드, 관계, 노드, 관계, …] 로 교차 구성되고(rag._row_to_unified),
    graph_facts[i].source 가 그 경로의 근거 rcept_no 다(둘은 행 단위로 정렬됨).
    """
    facts = state.get("graph_facts") or []
    paths = state.get("graph_paths") or []
    if not paths:
        return {"nodes": [], "edges": []}

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
            nodes.setdefault(name, {"id": name, "label": name, "category": "기업"})

        for j in range(len(names) - 1):
            rel = rels[j] if j < len(rels) else ""
            src, tgt = names[j], names[j + 1]
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

    return {"nodes": list(nodes.values()), "edges": edges}


def build_documents(state: dict) -> list[dict]:
    """Vector 청크 + 그래프 근거 rcept_no → 원본 문서 리스트(메타 보강).

    각 항목은 우측 '원본 문서' 탭 카드 한 장에 대응한다. Vector 청크는 본문(text)을
    포함하고, 그래프 근거 문서는 메타만(본문 없음) 포함한다.
    """
    vec_results = state.get("vec_results") or []
    graph_prov = list(state.get("graph_provenance") or [])
    for fact in state.get("graph_facts") or []:
        if fact.get("source"):
            graph_prov.append(str(fact["source"]))

    rcept_nos: list[str] = []
    for v in vec_results:
        rn = str((v.get("extra") or {}).get("rcept_no") or "")
        if rn:
            rcept_nos.append(rn)
    rcept_nos.extend(graph_prov)

    meta = _doc_meta_by_rcept(rcept_nos)

    documents: list[dict] = []
    seen: set[str] = set()

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

    # 2) 그래프 근거 문서 = 메타만(본문 없음), 중복 제외
    for rn in graph_prov:
        rn = str(rn)
        if not rn or rn in seen:
            continue
        seen.add(rn)
        m = meta.get(rn, {})
        documents.append(
            {
                "rcept_no": rn,
                "chunk_id": "",
                "corp_name": m.get("corp_name") or "",
                "title": m.get("title") or "",
                "doc_type": m.get("doc_type") or "",
                "date": m.get("date") or "",
                "summary": m.get("summary") or "",
                "section_path": "",
                "year": None,
                "score": None,
                "text": "",
                "source_kind": "graph",
            }
        )

    return documents


def serialize_state(state: dict) -> dict:
    """최종 state → {graph, documents, panel}.

    panel: 프론트가 자동으로 펼칠 탭 힌트.
      'graph'     → 관계도 데이터 있음
      'documents' → 원본 문서만 있음
      'none'      → 우측 패널 표시 안 함
    """
    graph = build_graph(state)
    documents = build_documents(state)

    if graph["edges"]:
        panel = "graph"
    elif documents:
        panel = "documents"
    else:
        panel = "none"

    return {"graph": graph, "documents": documents, "panel": panel}
