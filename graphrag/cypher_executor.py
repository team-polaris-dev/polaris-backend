"""공식 Text2CypherRetriever 가 돌려준 행을 canonical (hits, meta) 로 매핑한다.

라이브러리(text2cypher.run_relationship_query)는 read-only Cypher 한 개를 생성·실행해
행 dict 리스트만 돌려준다. 그 위에 도메인 레이어로 행→hit 변환을 얹는다: 노드는
_node_hit, 엣지는 _rel_hit 로 바꾸고 _score_edge_evidence 로 fail-closed 존재 attestation
근거를 붙인다(라이브러리가 안 하는 부분). meta 모양은 structured 경로를 미러해 downstream
(node/_assemble_local + adapt_to_legacy)이 그대로 동작하게 한다.
"""
from __future__ import annotations

from typing import Any

from config.relations import has_rank_intent, metric_for_query
from tool.rdb_client import parse_year

from graphrag.schema import GraphHit
from graphrag.structured_executor import (
    _METRIC_LABEL,
    _NODE_SCORE_HIGH,
    _fetch_chunk_texts,
    _fetch_metric_values,
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


def _supply_role(
    rel_type: str,
    from_id: str,
    to_id: str,
    anchor_codes: set[str],
    raw: Any,
) -> str:
    """SUPPLIES_TO 엣지에 한해 앵커 기준 방향 라벨(supplier/buyer)을 결정적으로 채운다.

    traverse._run_supply_chain 의 계약 미러: 앵커가 to 면 from 이 공급사(supplier),
    앵커가 from 이면 to 가 매출처(buyer). LLM 이 row.role 을 안 내보내도 패널 렌더링이
    이 라벨로 방향성을 그리므로 결정적으로 보강한다. 다른 관계 타입은 원본 값 보존.
    """
    raw_role = str(raw or "")
    if rel_type != "SUPPLIES_TO":
        return raw_role
    if to_id in anchor_codes:
        return "supplier"
    if from_id in anchor_codes:
        return "buyer"
    return raw_role


def map_results(
    rows: list[dict[str, Any]],
    cypher: str,
    anchors: list[dict],
    query: str,
    reason: str = "",
) -> tuple[list[GraphHit], dict] | None:
    """공식 retriever 행 리스트 → (hits, meta) 또는 None(행 0 → 호출자 폴백/abstain).

    rows 는 text2cypher.run_relationship_query 가 돌려준 행(from_id/from_name/to_id/
    to_name/rel_type/source/chunk_id/from_role/to_role)이고, cypher 는 생성된 read-only
    Cypher 원문(meta 기록용)이다.
    """
    anchor_codes = list(dict.fromkeys(
        c for c in (_anchor_code(a) for a in (anchors or [])) if c
    ))
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

    anchor_set = set(anchor_codes)
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
            "role": _supply_role(rel_type, from_id, to_id, anchor_set, row.get("role")),
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
        "cypher": cypher,
        "reason": reason,
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


def rank_results(
    hits: list[GraphHit],
    meta: dict,
    query: str,
    anchor_codes: list[str],
) -> tuple[list[GraphHit], dict]:
    """랭킹 의도가 있으면 text2cypher 후보를 결정적 SQL 지표로 줄세워 hits/meta 에 주석한다.

    text2cypher 경로(map_results)는 관계/구조/존재만 만든다 — 재무 지표 랭킹은 그래프가 아니라
    MariaDB 에 있기 때문이다. 그 위에 value-add 로, 질문에 랭킹 의도(has_rank_intent)가 있고
    지표가 해소되면(metric_for_query) 비앵커 후보(corp_code)를 _fetch_metric_values 로 가져와
    내림차순 줄세운다. 1위 재무수치를 fin_metric hit 로 더해 답변이 수치를 노출하게 하고
    (community_member_rank 와 동일 계약: adapt_to_legacy→render._fmt_graph), 1위 org 노드의
    role/score 를 올리며 structured 에 members/selected/rank_metric 를 기록한다.

    랭킹 의도 없음·지표 미해소·비앵커 후보 0·지표행 0 이면 입력을 그대로 돌려준다(additive no-op).
    그래서 플래그 ON 이어도 관계/구조 질문의 동작은 바뀌지 않는다.
    """
    if not has_rank_intent(query):
        return hits, meta
    metric_id = metric_for_query(query)
    if not metric_id:
        return hits, meta

    anchor_set = {str(c) for c in (anchor_codes or []) if c}
    candidate_codes: list[str] = []
    name_by_code: dict[str, str] = {}
    for hit in hits:
        if hit.get("label") != "organization":
            continue
        cid = str(hit.get("id") or "")
        if not cid or cid in anchor_set or cid in name_by_code:
            continue
        candidate_codes.append(cid)
        name_by_code[cid] = str(hit.get("name") or cid)
    if not candidate_codes:
        return hits, meta

    year = parse_year(query)
    metric_rows = _fetch_metric_values(candidate_codes, metric_id, year)
    ranked: list[dict[str, Any]] = []
    for row in metric_rows:
        try:
            value = float(row.get("value"))
        except (TypeError, ValueError):
            continue
        code = str(row.get("corp_code") or "")
        ranked.append({
            "corp_code": code,
            "name": name_by_code.get(code, str(row.get("corp_name") or code)),
            "value": value,
            "year": row.get("bsns_year"),
            "unit": row.get("unit") or "KRW",
            "source": str(row.get("rcept_no") or ""),
        })
    if not ranked:
        return hits, meta
    ranked.sort(key=lambda m: (-m["value"], m["corp_code"]))
    winner = ranked[0]
    rank_by_code = {m["corp_code"]: i + 1 for i, m in enumerate(ranked)}

    new_hits: list[GraphHit] = []
    for hit in hits:
        if hit.get("label") != "organization":
            new_hits.append(hit)
            continue
        rank = rank_by_code.get(str(hit.get("id") or ""))
        if not rank:
            new_hits.append(hit)
            continue
        attrs = dict(hit.get("attrs") or {})
        attrs["rank"] = rank
        updated = dict(hit, attrs=attrs)
        if rank == 1:
            attrs["structured_role"] = "selected"
            updated["score"] = _NODE_SCORE_HIGH
        new_hits.append(updated)

    # 1위 재무수치 — render._fmt_graph 가 account_id/value/bsns_year 로 수치를 노출한다.
    new_hits.append({
        "id": f"fin:{metric_id}:{winner['corp_code']}",
        "label": "fin_metric",
        "name": winner["name"],
        "attrs": {
            "account_id": metric_id,
            "value": winner["value"],
            "bsns_year": winner["year"],
            "unit": winner["unit"],
            "corp_code": winner["corp_code"],
            "metric_label": _METRIC_LABEL.get(metric_id, metric_id),
            "rank": 1,
        },
        "score": _NODE_SCORE_HIGH,
        "source": winner["source"],
        "seed_origin": "structured",
    })

    structured = dict(meta.get("structured") or {})
    structured["rank_metric"] = metric_id
    structured["metric_label"] = _METRIC_LABEL.get(metric_id, metric_id)
    structured["year"] = year
    structured["members"] = ranked
    structured["selected"] = winner
    structured["rankable"] = True
    structured["quality_notes"] = [
        *(structured.get("quality_notes") or []),
        "재무 지표 랭킹은 MariaDB 결정적 SQL(_fetch_metric_values)로 후처리한다",
    ]
    new_meta = dict(meta)
    new_meta["structured"] = structured
    new_meta["n_hits"] = len(new_hits)
    new_meta["patterns_run"] = [*(meta.get("patterns_run") or []), "sql_rank"]
    return new_hits, new_meta
