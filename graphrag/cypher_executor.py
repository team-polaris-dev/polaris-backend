"""검증된 text-to-Cypher 를 실행해 (hits, meta) 로 매핑한다.

cypher_generator 가 통과시킨 GeneratedCypher 를 받아 $anchors 를 주입하고 실행,
행을 structured_executor 의 canonical 매퍼(_node_hit/_rel_hit)로 변환한다. 엣지는
structured 경로와 동일하게 _score_edge_evidence 로 근거를 붙여 fail-closed 존재
attestation 을 보존한다. meta 는 _execute_two_hop_list 의 모양을 미러해 downstream
(node/_assemble_local + adapt_to_legacy)이 그대로 동작하게 한다.

_run_cypher 는 monkeypatch 가능한 모듈 수준 seam(_two_hop_list_rows 패턴 미러).
"""
from __future__ import annotations

from typing import Any

from graphrag.cypher_generator import GeneratedCypher
from graphrag.schema import GraphHit
from graphrag.structured_executor import (
    _fetch_chunk_texts,
    _node_hit,
    _rel_hit,
    _score_edge_evidence,
)

# 근거 level → 노드 hit score (structured_executor 와 동일 의미). 강근거=1.0, 중=0.75.
_NODE_SCORE_ANCHOR = 1.0
_NODE_SCORE_OTHER = 0.75
_REL_SCORE = 0.75

# 행이 노드별로 싣는 role. 알 수 없으면 neighbor.
_VALID_ROLES = {"anchor", "bridge", "sibling", "neighbor"}


def _run_cypher(cypher: str, params: dict) -> list[dict[str, Any]]:
    """검증된 Cypher 를 실행해 행 dict 리스트를 돌려준다(structured 경로 패턴 미러).

    테스트는 이 함수를 monkeypatch 해 Neo4j 없이 행을 주입한다.
    """
    from tool.graph_client import neo4j_driver  # noqa: PLC0415

    try:
        with neo4j_driver.session() as session:
            return [r.data() for r in session.run(cypher, **(params or {}))]
    except Exception as exc:
        print(f"⚠️ [cypher_executor] cypher run failed: {exc!r}")
        return []


def _role_of(value: Any) -> str:
    role = str(value or "").strip()
    return role if role in _VALID_ROLES else "neighbor"


def _anchor_code(anchor: dict) -> str:
    """앵커 dict 에서 corp_code 추출. 정규화 앵커(corp_code 키)와 raw Seed
    (key_type=='corp_code' → key_value, schema.Seed) 양쪽을 모두 받는다."""
    code = anchor.get("corp_code")
    if code:
        return str(code)
    if anchor.get("key_type") == "corp_code" and anchor.get("key_value"):
        return str(anchor["key_value"])
    return ""


def _node_score(role: str) -> float:
    return _NODE_SCORE_ANCHOR if role == "anchor" else _NODE_SCORE_OTHER


def run(
    generated: GeneratedCypher,
    anchors: list[dict],
    query: str,
) -> tuple[list[GraphHit], dict] | None:
    """검증된 Cypher 실행 → (hits, meta) 또는 None(행 0 → 호출자 폴백/abstain).

    params 에 $anchors = [a['corp_code'] for a in anchors if a.get('corp_code')] 를
    generated.params 와 병합해 주입한다(프롬프트의 $anchors 와 일치).
    """
    anchor_codes = list(dict.fromkeys(
        c for c in (_anchor_code(a) for a in (anchors or [])) if c
    ))
    params: dict[str, Any] = {**(generated.params or {}), "anchors": anchor_codes}

    rows = _run_cypher(generated.cypher, params)
    if not rows:
        return None

    # 엣지 근거: structured 경로와 동일하게 chunk 본문을 한 번에 fetch 후 점수화.
    chunk_ids = [str(r.get("chunk_id") or "") for r in rows]
    chunk_texts = _fetch_chunk_texts(chunk_ids)

    hits: list[GraphHit] = []
    seen_nodes: set[str] = set()
    seen_rels: set[str] = set()
    # 노드는 (name, id) asc, 엣지는 (from_name, to_name, rel_type) asc 로 결정성 정렬.
    node_buf: list[tuple[tuple[str, str], GraphHit]] = []
    rel_buf: list[tuple[tuple[str, str, str], GraphHit]] = []

    def add_node(id_: str, name: str, role: str, stock_code: Any = None) -> None:
        if not id_ or id_ in seen_nodes:
            return
        seen_nodes.add(id_)
        attrs: dict[str, Any] = {"structured_role": role}
        if stock_code:
            attrs["stock_code"] = stock_code
        node_buf.append(((name or "", id_), _node_hit(id_, name, attrs, _node_score(role))))

    for row in rows:
        from_id = str(row.get("from_id") or "")
        from_name = str(row.get("from_name") or "")
        to_id = str(row.get("to_id") or "")
        to_name = str(row.get("to_name") or "")
        rel_type = str(row.get("rel_type") or "")
        if not (from_id and to_id and rel_type):
            continue

        add_node(from_id, from_name, _role_of(row.get("from_role")), row.get("from_stock_code"))
        add_node(to_id, to_name, _role_of(row.get("to_role")), row.get("to_stock_code"))

        rel_id = f"rel:{rel_type}:{from_id}:{to_id}"
        if rel_id in seen_rels:
            continue
        seen_rels.add(rel_id)
        edge: dict[str, Any] = {
            "rel_type": rel_type,
            "from_id": from_id, "from_name": from_name,
            "to_id": to_id, "to_name": to_name,
            "role": str(row.get("role") or ""),
            "source": str(row.get("source") or ""),
            "chunk_id": str(row.get("chunk_id") or ""),
        }
        edge["evidence"] = _score_edge_evidence(edge, chunk_texts)
        rel_buf.append(((from_name or "", to_name or "", rel_type), _rel_hit(edge, _REL_SCORE)))

    node_buf.sort(key=lambda x: x[0])
    rel_buf.sort(key=lambda x: x[0])
    hits.extend(h for _, h in node_buf)
    hits.extend(h for _, h in rel_buf)

    if not hits:
        return None

    n_edges = len(rel_buf)
    structured = {
        "mode": "structured",
        "kind": "text2cypher",
        "query": query,
        "cypher": generated.cypher,
        "reason": generated.reason,
        "anchors": anchor_codes,
        "answer_edges": [],
        "abstained": False,
        "quality_notes": [
            "스키마 가드 text-to-Cypher 로 생성한 read-only 관계 조회 결과",
            "재무 지표 랭킹은 이 경로에서 다루지 않는다(구조/관계/존재만)",
        ],
    }
    meta = {
        "mode": "structured",
        "structured": structured,
        "patterns_run": ["text2cypher"],
        "n_hits": len(hits),
        "n_edges": n_edges,
        "fallback_used": False,
        "errors": [],
    }
    return hits, meta
