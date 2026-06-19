"""Executor for structured GraphRAG plans.

Runs constrained graph traversals and metric ranking. This path is for
questions whose semantics are closer to "graph query + ORDER BY" than broad
neighborhood retrieval.
"""
from __future__ import annotations

from typing import Any

from tool.graph_client import neo4j_driver
from tool.rdb_client import mariadb_conn, parse_year

from config.graphrag import (
    STRUCTURED_MIN_EVIDENCE,
    STRUCTURED_MIN_EVIDENCE_OPERATING,
)
from config.relations import NETWORK_REL_TYPES
from graphrag.plan_schema import BranchRankStep, MetricRankStep, RelationStep, StructuredPlan
from graphrag.schema import GraphHit, Seed


_METRIC_LABEL = {
    "ifrs-full_Revenue": "매출액",
    "dart_OperatingIncomeLoss": "영업이익",
    "ifrs-full_ProfitLoss": "당기순이익",
    "ifrs-full_Assets": "총자산",
}

_GOVERNANCE_RELS = {
    "INTERLOCKING_DIRECTORATE",
    "IS_MAJOR_SHAREHOLDER_OF",
    "IS_SUBSIDIARY_OF",
    "INVESTS_IN",
}
_HUB_DEGREE_SOFT_CAP = 25
_HUB_DEGREE_HARD_CAP = 80
_FIN_FS_DIV = "CFS"
_FIN_REPRT_CODE = "11011"

_RELATION_EVIDENCE_TERMS = {
    "SUPPLIES_TO": ("공급", "납품", "매출처", "고객", "거래", "판매"),
    "RELATED_PARTY": ("특수관계", "관계기업", "관계회사", "관련회사", "계열"),
    "INVESTS_IN": ("투자", "지분", "관계기업", "공동기업", "주식"),
}
_TYPE_ATTESTED_RELS = {"RELATED_PARTY", "INVESTS_IN"}


def _placeholders(n: int) -> str:
    return ", ".join(["%s"] * n)


def _fetch_metric_values(
    corp_codes: list[str],
    account_id: str = "ifrs-full_Revenue",
    year: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch one annual CFS metric per candidate for structured ranking."""
    codes = [c for c in dict.fromkeys(corp_codes) if c]
    if not codes or not account_id:
        return []

    code_ph = _placeholders(len(codes))
    params: list[Any] = [*codes, account_id, _FIN_FS_DIV, _FIN_REPRT_CODE]

    if year is not None:
        year_clause = "AND f.bsns_year = %s"
        params.append(year)
    else:
        year_clause = (
            "AND f.bsns_year = ("
            "  SELECT MAX(f2.bsns_year) FROM fin_metric f2"
            "  WHERE f2.corp_code = f.corp_code"
            "    AND f2.account_id = f.account_id"
            "    AND f2.fs_div = %s AND f2.reprt_code = %s"
            ")"
        )
        params.extend([_FIN_FS_DIV, _FIN_REPRT_CODE])

    sql = (
        "SELECT f.corp_code, "
        "  (SELECT d.corp_name FROM document_index d "
        "   WHERE d.corp_code = f.corp_code AND d.corp_name IS NOT NULL LIMIT 1) AS corp_name, "
        "  f.bsns_year, f.account_id, f.value, f.unit, f.fs_div, f.rcept_no "
        "FROM fin_metric f "
        f"WHERE f.corp_code IN ({code_ph}) "
        "  AND f.account_id = %s "
        "  AND f.fs_div = %s AND f.reprt_code = %s "
        f"  {year_clause} "
        "ORDER BY f.value DESC"
    )
    try:
        with mariadb_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            return list(cur.fetchall())
    except Exception as exc:
        print(f"⚠️ [structured_executor] metric query failed: {exc!r}")
        return []


def _fetch_chunk_texts(chunk_ids: list[str]) -> dict[str, str]:
    ids = [c for c in dict.fromkeys(chunk_ids) if c]
    if not ids:
        return {}

    sql = (
        f"SELECT chunk_id, embedding_text FROM chunk_index "
        f"WHERE chunk_id IN ({_placeholders(len(ids))})"
    )
    try:
        with mariadb_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(ids))
            return {str(r.get("chunk_id") or ""): str(r.get("embedding_text") or "") for r in cur.fetchall()}
    except Exception as exc:
        print(f"⚠️ [structured_executor] chunk evidence query failed: {exc!r}")
        return {}


def _norm_evidence_text(value: str) -> str:
    out = (value or "").lower()
    for token in ("주식회사", "(주)", "㈜", " ", "\t", "\n", "-", "_", ".", ",", "·"):
        out = out.replace(token, "")
    return out


def _name_variants(name: str) -> set[str]:
    base = _norm_evidence_text(name)
    variants = {base} if base else set()
    if base.startswith("에스케이"):
        variants.add("sk" + base[len("에스케이"):])
    if base.startswith("sk"):
        variants.add("에스케이" + base[2:])
    if "하이닉스" in base:
        variants.add("하이닉스")
    return {v for v in variants if len(v) >= 2}


def _mentions_name(text_norm: str, name: str) -> bool:
    return any(v in text_norm for v in _name_variants(name))


def _score_edge_evidence(edge: dict[str, Any], chunk_texts: dict[str, str]) -> dict[str, Any]:
    rel_type = str(edge.get("rel_type") or "")
    source = str(edge.get("source") or "")
    chunk_id = str(edge.get("chunk_id") or "")
    text = chunk_texts.get(chunk_id, "")
    text_norm = _norm_evidence_text(text)
    warnings: list[str] = []

    has_source = bool(source)
    has_chunk_ref = bool(chunk_id)
    has_chunk_text = bool(text)
    from_mentioned = bool(text_norm and _mentions_name(text_norm, str(edge.get("from_name") or "")))
    to_mentioned = bool(text_norm and _mentions_name(text_norm, str(edge.get("to_name") or "")))
    relation_term = bool(text_norm and any(t in text for t in _RELATION_EVIDENCE_TERMS.get(rel_type, ())))

    if has_source and has_chunk_text and from_mentioned and to_mentioned and relation_term:
        confidence = 0.95
    elif has_source and has_chunk_text and from_mentioned and to_mentioned:
        confidence = 0.8
        warnings.append("chunk_names_relation_term_missing")
    elif has_source and has_chunk_text:
        confidence = 0.55
        warnings.append("chunk_does_not_name_both_endpoints")
    elif has_source and has_chunk_ref:
        confidence = 0.45
        warnings.append("chunk_reference_not_found")
    elif has_source:
        confidence = 0.35
        warnings.append("document_source_without_chunk")
    else:
        confidence = 0.1
        warnings.append("missing_source")

    if rel_type in {"RELATED_PARTY", "INVESTS_IN"} and confidence < 0.8:
        warnings.append("weak_evidence_for_accounting_or_investment_relation")

    level = "high" if confidence >= 0.8 else "medium" if confidence >= 0.55 else "low"
    return {
        "confidence": confidence,
        "level": level,
        "source": source,
        "chunk_id": chunk_id,
        "chunk_found": has_chunk_text,
        "from_mentioned": from_mentioned,
        "to_mentioned": to_mentioned,
        "relation_term_found": relation_term,
        "warnings": warnings,
    }


def _candidate_edges(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    edges = []
    if candidate.get("edge"):
        edges.append(candidate["edge"])
    edges.extend(candidate.get("anchor_edges") or [])
    return edges


def _attach_candidate_evidence(candidates: list[dict[str, Any]]) -> None:
    edges = [edge for cand in candidates for edge in _candidate_edges(cand)]
    chunk_texts = _fetch_chunk_texts([str(edge.get("chunk_id") or "") for edge in edges])
    for edge in edges:
        edge["evidence"] = _score_edge_evidence(edge, chunk_texts)
    for cand in candidates:
        edge = cand.get("edge") or {}
        cand["evidence"] = edge.get("evidence") or {}


def _org_id_expr(var: str) -> str:
    return f"coalesce({var}.corp_code, 'org:' + coalesce({var}.er_name, {var}.name))"


def _anchor_match(anchor: dict[str, Any], var: str = "anchor") -> tuple[str, dict[str, Any]]:
    key_type = anchor.get("key_type")
    key_value = anchor.get("key_value")
    if not key_type:
        if anchor.get("corp_code"):
            key_type, key_value = "corp_code", anchor.get("corp_code")
        elif anchor.get("er_name"):
            key_type, key_value = "er_name", anchor.get("er_name")
    if key_type == "corp_code":
        return f"MATCH ({var}:Organization {{corp_code: $key_value}})", {"key_value": key_value}
    if key_type == "er_name":
        return f"MATCH ({var}:Organization {{er_name: $key_value}})", {"key_value": key_value}
    raise ValueError(f"unsupported structured anchor: {anchor!r}")


def _node_hit(id_: str, name: str, attrs: dict[str, Any] | None = None, score: float = 1.0) -> GraphHit:
    return {
        "id": id_,
        "label": "organization",
        "name": name,
        "attrs": attrs or {},
        "score": score,
        "seed_origin": "structured",
    }


def _rel_hit(edge: dict[str, Any], score: float = 0.9) -> GraphHit:
    attrs = {
        "rel_type": edge["rel_type"],
        "from_id": edge["from_id"],
        "from_name": edge["from_name"],
        "to_id": edge["to_id"],
        "to_name": edge["to_name"],
        "role": edge.get("role") or "",
    }
    if edge.get("chunk_id"):
        attrs["chunk_id"] = edge["chunk_id"]
    if edge.get("branch_kind"):
        attrs["branch_kind"] = edge["branch_kind"]
    evidence = edge.get("evidence") or {}
    if evidence:
        attrs["evidence_confidence"] = evidence.get("confidence")
        attrs["evidence_level"] = evidence.get("level")
        attrs["evidence_warning"] = ";".join(evidence.get("warnings") or [])
        attrs["evidence_chunk_found"] = evidence.get("chunk_found")
        attrs["evidence_from_mentioned"] = evidence.get("from_mentioned")
        attrs["evidence_to_mentioned"] = evidence.get("to_mentioned")
        attrs["evidence_relation_term_found"] = evidence.get("relation_term_found")
    hit: GraphHit = {
        "id": f"rel:{edge['rel_type']}:{edge['from_id']}:{edge['to_id']}",
        "label": "relationship",
        "name": f"{edge['from_name']} → {edge['to_name']}",
        "attrs": attrs,
        "score": score,
        "seed_origin": "structured",
    }
    if edge.get("source"):
        hit["source"] = edge["source"]
    return hit


def _clean_candidate(row: dict[str, Any], step: RelationStep) -> dict[str, Any]:
    return {
        "id": str(row.get("cand_id") or ""),
        "corp_code": str(row.get("corp_code") or ""),
        "er_name": str(row.get("er_name") or ""),
        "name": str(row.get("cand_name") or ""),
        "source": str(row.get("source") or ""),
        "chunk_id": str(row.get("chunk_id") or ""),
        "anchor_rels": [str(v) for v in (row.get("anchor_rels") or []) if v],
        "graph_degree": int(row.get("graph_degree") or 0),
        "edge": {
            "rel_type": step.rel_type,
            "from_id": str(row.get("from_id") or ""),
            "from_name": str(row.get("from_name") or ""),
            "to_id": str(row.get("to_id") or ""),
            "to_name": str(row.get("to_name") or ""),
            "role": step.role,
            "source": str(row.get("source") or ""),
            "chunk_id": str(row.get("chunk_id") or ""),
        },
    }


def _relation_candidates(
    anchor: dict[str, Any],
    step: RelationStep,
    *,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    match, params = _anchor_match(anchor)
    rel = step.rel_type
    if step.direction == "incoming":
        pattern = f"(cand:Organization)-[r:{rel}]->(anchor)"
    elif step.direction == "outgoing":
        pattern = f"(anchor)-[r:{rel}]->(cand:Organization)"
    else:
        pattern = f"(anchor)-[r:{rel}]-(cand:Organization)"

    cypher = f"""
{match}
MATCH {pattern}
WHERE cand <> anchor
  AND coalesce(r.valid_to, '') = ''
  AND r.qc_disabled_at IS NULL
  AND NOT coalesce(cand.name, '') IN ['계','소계','합계','-','주','']
WITH DISTINCT anchor, cand, r,
     {_org_id_expr('anchor')} AS anchor_id,
     {_org_id_expr('cand')} AS cand_id
WHERE NOT (cand_id IN $exclude_ids)
WITH anchor, cand, r, anchor_id, cand_id, startNode(r) = anchor AS anchor_is_start
OPTIONAL MATCH (cand)-[other]-(anchor)
WITH anchor, cand, r, anchor_id, cand_id, anchor_is_start,
     [x IN collect(DISTINCT type(other)) WHERE x IS NOT NULL AND x <> $rel_type] AS anchor_rels
OPTIONAL MATCH (cand)-[deg_rel]-(deg_node:Organization)
WHERE type(deg_rel) IN $network_rel_types
WITH anchor, cand, r, anchor_id, cand_id, anchor_is_start, anchor_rels,
     count(DISTINCT deg_rel) AS graph_degree
RETURN
  cand_id,
  cand.corp_code AS corp_code,
  cand.er_name AS er_name,
  cand.name AS cand_name,
  CASE WHEN anchor_is_start THEN anchor_id ELSE cand_id END AS from_id,
  CASE WHEN anchor_is_start THEN anchor.name ELSE cand.name END AS from_name,
  CASE WHEN anchor_is_start THEN cand_id ELSE anchor_id END AS to_id,
  CASE WHEN anchor_is_start THEN cand.name ELSE anchor.name END AS to_name,
  r.rcept_no AS source,
  r.chunk_id AS chunk_id,
  anchor_rels,
  graph_degree
ORDER BY (r.chunk_id IS NOT NULL) DESC, (r.rcept_no IS NOT NULL) DESC, cand_id ASC
LIMIT 200
"""
    params["exclude_ids"] = list(exclude_ids or set())
    params["rel_type"] = rel
    params["network_rel_types"] = list(NETWORK_REL_TYPES)
    with neo4j_driver.session() as session:
        rows = [r.data() for r in session.run(cypher, **params)]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        cand = _clean_candidate(row, step)
        if not cand["id"] or cand["id"] in seen:
            continue
        if step.direction == "undirected" and step.rel_type == "RELATED_PARTY":
            cand["edge"]["from_id"] = str(anchor.get("id") or "")
            cand["edge"]["from_name"] = str(anchor.get("name") or "")
            cand["edge"]["to_id"] = cand["id"]
            cand["edge"]["to_name"] = cand["name"]
        seen.add(cand["id"])
        out.append(cand)
    return out


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_policy_bucket(candidate: dict[str, Any], policy: str) -> tuple[float, dict[str, Any]]:
    anchor_rels = set(candidate.get("anchor_rels") or [])
    evidence = candidate.get("evidence") or candidate.get("edge", {}).get("evidence") or {}
    evidence_confidence = _as_float(evidence.get("confidence"))
    graph_degree = int(_as_float(candidate.get("graph_degree"), 0.0))
    flags = {
        "governance_linked": bool(anchor_rels & _GOVERNANCE_RELS),
        "related_party_linked": "RELATED_PARTY" in anchor_rels,
        "graph_degree": graph_degree,
        "evidence_confidence": evidence_confidence,
        "evidence_level": evidence.get("level") or "",
        "evidence_relation_term_found": bool(evidence.get("relation_term_found")),
    }
    score = 0.0
    reasons: list[str] = []

    if evidence_confidence >= 0.8:
        score += 1.0
        reasons.append("strong_edge_evidence")
    elif evidence_confidence >= 0.55:
        score += 0.25
        reasons.append("medium_edge_evidence")
    elif evidence_confidence > 0:
        score -= 0.5
        reasons.append("weak_edge_evidence")

    if flags["evidence_relation_term_found"]:
        score += 0.25
        reasons.append("relation_term_found")

    if policy == "operating_counterparty":
        if flags["governance_linked"]:
            score -= 2.0
            reasons.append("governance_linked")
        if flags["related_party_linked"]:
            score -= 1.0
            reasons.append("related_party_linked")

        if graph_degree >= _HUB_DEGREE_HARD_CAP:
            score -= 1.0
            reasons.append("hard_hub_degree")
        elif graph_degree >= _HUB_DEGREE_SOFT_CAP:
            score -= 0.5
            reasons.append("soft_hub_degree")

        if not flags["governance_linked"] and not flags["related_party_linked"]:
            score += 0.5
            reasons.append("pure_operating_counterparty")

    flags["score"] = score
    flags["reasons"] = reasons
    return score, flags


def _rank_candidates(
    candidates: list[dict[str, Any]],
    metric_id: str,
    year: int | None,
    *,
    policy: str = "default",
) -> list[dict[str, Any]]:
    codes = [c["corp_code"] for c in candidates if c.get("corp_code")]
    metric_rows = _fetch_metric_values(codes, metric_id, year)
    by_code: dict[str, dict[str, Any]] = {str(r.get("corp_code")): r for r in metric_rows}

    ranked: list[dict[str, Any]] = []
    for cand in candidates:
        row = by_code.get(cand.get("corp_code") or "")
        item = dict(cand)
        policy_bucket, policy_flags = _candidate_policy_bucket(cand, policy)
        item["policy"] = {"name": policy, "bucket": policy_bucket, **policy_flags}
        if row:
            try:
                metric_value = float(row.get("value"))
            except (TypeError, ValueError):
                metric_value = None
            item["metric"] = {
                "account_id": str(row.get("account_id") or metric_id),
                "label": _METRIC_LABEL.get(metric_id, metric_id),
                "value": metric_value,
                "raw_value": str(row.get("value") or ""),
                "year": row.get("bsns_year"),
                "unit": row.get("unit") or "KRW",
                "source": str(row.get("rcept_no") or ""),
            }
        else:
            item["metric"] = {
                "account_id": metric_id,
                "label": _METRIC_LABEL.get(metric_id, metric_id),
                "value": None,
                "raw_value": "",
                "year": year,
                "unit": "KRW",
                "source": "",
            }
        ranked.append(item)

    ranked.sort(
        key=lambda c: (
            c.get("policy", {}).get("bucket", 0),
            c.get("metric", {}).get("value") is not None,
            c.get("metric", {}).get("value") or float("-inf"),
            c.get("id") or c.get("name") or "",
        ),
        reverse=True,
    )
    return ranked


def _evidence_confidence(candidate: dict[str, Any]) -> float:
    evidence = candidate.get("evidence") or candidate.get("edge", {}).get("evidence") or {}
    return _as_float(evidence.get("confidence"))


def _relation_type_attested(candidate: dict[str, Any]) -> bool:
    evidence = candidate.get("evidence") or candidate.get("edge", {}).get("evidence") or {}
    return bool(evidence.get("relation_term_found"))


def _candidate_supported(candidate: dict[str, Any], rel_type: str) -> bool:
    if rel_type in _TYPE_ATTESTED_RELS:
        return _relation_type_attested(candidate)
    return _evidence_confidence(candidate) >= _evidence_floor(rel_type)


def _evidence_floor(rel_type: str) -> float:
    if rel_type == "SUPPLIES_TO":
        return STRUCTURED_MIN_EVIDENCE_OPERATING
    return STRUCTURED_MIN_EVIDENCE


def _select_supported(
    ranked: list[dict[str, Any]],
    rel_type: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    supported = [c for c in ranked if _candidate_supported(c, rel_type)]
    metric_supported = [
        c for c in supported
        if c.get("metric", {}).get("value") is not None
    ]
    selected = metric_supported[0] if metric_supported else None
    unranked_confirmed = [
        c for c in supported
        if c.get("metric", {}).get("value") is None
    ]
    return selected, unranked_confirmed


def _supply_direction_from_counts(incoming_count: int, outgoing_count: int) -> str:
    if outgoing_count > incoming_count:
        return "outgoing"
    return "incoming"


def _resolve_supply_direction(anchor: dict[str, Any]) -> str:
    match, params = _anchor_match(anchor)
    cypher = f"""
{match}
OPTIONAL MATCH (src:Organization)-[rin:SUPPLIES_TO]->(anchor)
WHERE coalesce(rin.valid_to, '') = '' AND rin.qc_disabled_at IS NULL
WITH anchor, count(DISTINCT rin) AS incoming_count
OPTIONAL MATCH (anchor)-[rout:SUPPLIES_TO]->(dst:Organization)
WHERE coalesce(rout.valid_to, '') = '' AND rout.qc_disabled_at IS NULL
RETURN incoming_count, count(DISTINCT rout) AS outgoing_count
"""
    try:
        with neo4j_driver.session() as session:
            row = session.run(cypher, **params).single()
        if not row:
            return "incoming"
        return _supply_direction_from_counts(
            int(row.get("incoming_count") or 0),
            int(row.get("outgoing_count") or 0),
        )
    except Exception as exc:
        print(f"⚠️ [structured_executor] supply direction probe failed: {exc!r}")
        return "incoming"


def _resolve_relation_for_anchor(anchor: dict[str, Any], step: RelationStep) -> RelationStep:
    if step.rel_type != "SUPPLIES_TO" or step.direction != "auto":
        return step
    direction = _resolve_supply_direction(anchor)
    if direction == "outgoing":
        return RelationStep("SUPPLIES_TO", "outgoing", "major_customers", "major_customer")
    return RelationStep("SUPPLIES_TO", "incoming", "suppliers", "supplier")


def _resolve_branch_for_anchor(anchor: dict[str, Any], branch: BranchRankStep) -> BranchRankStep:
    relation = _resolve_relation_for_anchor(anchor, branch.relation)
    if relation is branch.relation:
        return branch
    kind = "major_customer" if relation.direction == "outgoing" else "supplier"
    return BranchRankStep(
        kind=kind,  # type: ignore[arg-type]
        relation=relation,
        rank=MetricRankStep(branch.rank.metric_id, alias="top_" + kind),
    )


def _edge_matches_relation(edge: dict[str, Any], relation: RelationStep, anchor_id: str | None = None) -> bool:
    if edge.get("rel_type") != relation.rel_type:
        return False
    if relation.direction == "incoming" and anchor_id:
        return str(edge.get("to_id") or "") == str(anchor_id)
    if relation.direction == "outgoing" and anchor_id:
        return str(edge.get("from_id") or "") == str(anchor_id)
    return True


def _anchor_from_seed(seed: Seed) -> dict[str, Any]:
    return {
        "id": seed.get("id") or "",
        "name": seed.get("name") or "",
        "key_type": seed.get("key_type"),
        "key_value": seed.get("key_value"),
        "corp_code": seed.get("key_value") if seed.get("key_type") == "corp_code" else "",
        "er_name": seed.get("key_value") if seed.get("key_type") == "er_name" else "",
    }


def _anchor_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate.get("corp_code"):
        key_type, key_value = "corp_code", candidate["corp_code"]
    else:
        key_type, key_value = "er_name", candidate.get("er_name") or candidate.get("name")
    return {
        "id": candidate.get("id") or "",
        "name": candidate.get("name") or "",
        "key_type": key_type,
        "key_value": key_value,
        "corp_code": candidate.get("corp_code") or "",
        "er_name": candidate.get("er_name") or "",
    }


def _norm_name(value: str) -> str:
    out = (value or "").lower()
    for token in ("주식회사", "(주)", "㈜", " ", "\t", "\n"):
        out = out.replace(token, "")
    return out


def _seed_order(seed: Seed, query: str) -> tuple[int, int, float]:
    qn = _norm_name(query)
    name = _norm_name(str(seed.get("name") or ""))
    key = _norm_name(str(seed.get("key_value") or ""))
    mentioned = bool((name and name in qn) or (key and key in qn))
    origin_rank = 0 if seed.get("origin") == "upstream" else 1
    return (0 if mentioned else 1, origin_rank, -float(seed.get("score") or 0.0))


def _common_anchor_seeds(org_seeds: list[Seed], query: str, min_count: int) -> list[Seed]:
    ordered = sorted(org_seeds, key=lambda s: _seed_order(s, query))
    mentioned = [s for s in ordered if _seed_order(s, query)[0] == 0]
    pool = mentioned if len(mentioned) >= min_count else ordered
    return pool[:min_count]


def _intersect_common_candidates(
    anchors: list[dict[str, Any]],
    step: RelationStep,
) -> list[dict[str, Any]]:
    per_anchor: list[tuple[dict[str, Any], dict[str, dict[str, Any]]]] = []
    for anchor in anchors:
        by_id = {cand["id"]: cand for cand in _relation_candidates(anchor, step) if cand.get("id")}
        per_anchor.append((anchor, by_id))

    if not per_anchor:
        return []

    common_ids = set(per_anchor[0][1].keys())
    for _, by_id in per_anchor[1:]:
        common_ids &= set(by_id.keys())

    common: list[dict[str, Any]] = []
    for cand_id in common_ids:
        first_cand = per_anchor[0][1][cand_id]
        merged = dict(first_cand)
        anchor_edges: list[dict[str, Any]] = []
        anchor_matches: list[dict[str, Any]] = []
        for anchor, by_id in per_anchor:
            cand = by_id[cand_id]
            edge = dict(cand["edge"])
            anchor_edges.append(edge)
            anchor_matches.append({
                "anchor": anchor,
                "edge": edge,
                "source": edge.get("source") or "",
                "chunk_id": edge.get("chunk_id") or "",
            })
        merged["edge"] = anchor_edges[0]
        merged["anchor_edges"] = anchor_edges
        merged["anchor_matches"] = anchor_matches
        common.append(merged)

    _attach_candidate_evidence(common)
    return common


def _run_branch_from_anchor(
    anchor: dict[str, Any],
    branch: BranchRankStep,
    year: int | None,
    *,
    exclude_ids: set[str] | None = None,
) -> dict[str, Any]:
    if branch.relation.rel_type == "SUPPLIES_TO" and branch.relation.direction == "auto":
        incoming = BranchRankStep(
            kind="supplier",
            relation=RelationStep("SUPPLIES_TO", "incoming", "suppliers", "supplier"),
            rank=MetricRankStep(branch.rank.metric_id, alias="top_supplier"),
        )
        outgoing = BranchRankStep(
            kind="major_customer",
            relation=RelationStep("SUPPLIES_TO", "outgoing", "major_customers", "major_customer"),
            rank=MetricRankStep(branch.rank.metric_id, alias="top_major_customer"),
        )
        results = [
            _run_branch_from_anchor(anchor, incoming, year, exclude_ids=exclude_ids),
            _run_branch_from_anchor(anchor, outgoing, year, exclude_ids=exclude_ids),
        ]

        def result_score(result: dict[str, Any]) -> tuple[bool, float, float, int]:
            selected = result.get("selected") or {}
            evidence = selected.get("evidence") or selected.get("edge", {}).get("evidence") or {}
            metric_value = _as_float(selected.get("metric", {}).get("value"), float("-inf"))
            if metric_value == 0.0 and selected.get("metric", {}).get("value") is None:
                metric_value = float("-inf")
            return (
                bool(selected),
                _as_float(evidence.get("confidence")),
                metric_value,
                len(result.get("unranked_confirmed") or []),
            )

        best = max(results, key=result_score)
        best["auto_direction_candidates"] = [
            {
                "kind": r.get("kind"),
                "relation": r.get("relation"),
                "selected": (r.get("selected") or {}).get("name"),
                "abstained": r.get("abstained"),
            }
            for r in results
        ]
        return best

    branch = _resolve_branch_for_anchor(anchor, branch)
    candidates = _relation_candidates(
        anchor,
        branch.relation,
        exclude_ids=exclude_ids or {str(anchor.get("id") or "")},
    )
    for cand in candidates:
        cand["edge"]["branch_kind"] = branch.kind
    _attach_candidate_evidence(candidates)
    policy = "operating_counterparty" if branch.kind in {"supplier", "major_customer"} else "default"
    ranked = _rank_candidates(candidates, branch.rank.metric_id, year, policy=policy)
    selected, unranked_confirmed = _select_supported(ranked, branch.relation.rel_type)
    return {
        "kind": branch.kind,
        "relation": branch.relation.__dict__,
        "rank": branch.rank.__dict__,
        "candidates": ranked,
        "selected": selected,
        "unranked_confirmed": unranked_confirmed,
        "abstained": selected is None,
        "evidence_floor": _evidence_floor(branch.relation.rel_type),
    }


def _run_branch(
    top_first: dict[str, Any],
    branch: BranchRankStep,
    year: int | None,
) -> dict[str, Any]:
    return _run_branch_from_anchor(
        _anchor_from_candidate(top_first),
        branch,
        year,
        exclude_ids={str(top_first.get("id") or "")},
    )


def _best_supported_branch(branches: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    supported: list[dict[str, Any]] = []
    for kind, branch in branches.items():
        selected = branch.get("selected") or {}
        evidence = selected.get("evidence") or selected.get("edge", {}).get("evidence") or {}
        if selected:
            supported.append({
                "kind": kind,
                "company": selected.get("name") or "",
                "confidence": evidence.get("confidence", 0.0),
                "level": evidence.get("level", "low"),
                "warnings": evidence.get("warnings") or [],
            })
    if not supported:
        return None
    supported.sort(key=lambda b: float(b.get("confidence") or 0.0), reverse=True)
    return supported[0]


def execute(plan: StructuredPlan, seeds: list[Seed], query: str) -> tuple[list[GraphHit], dict] | None:
    org_seeds = [s for s in seeds if s.get("label") == "organization"]
    if not org_seeds:
        return None
    if plan.kind == "multi_anchor_branch_rank":
        return _execute_multi_anchor_branch_rank(plan, org_seeds, query)
    if plan.kind == "single_anchor_branch_rank":
        for org_seed in sorted(org_seeds, key=lambda s: _seed_order(s, query)):
            result = _execute_single_anchor_branch_rank(plan, org_seed, query)
            if result:
                return result
        return None
    for org_seed in sorted(org_seeds, key=lambda s: _seed_order(s, query)):
        result = _execute_for_seed(plan, org_seed, query)
        if result:
            return result
    return None


def _execute_multi_anchor_branch_rank(
    plan: StructuredPlan,
    org_seeds: list[Seed],
    query: str,
) -> tuple[list[GraphHit], dict] | None:
    year = parse_year(query)
    anchor_seeds = _common_anchor_seeds(org_seeds, query, max(2, plan.common_anchor_min))
    if len(anchor_seeds) < max(2, plan.common_anchor_min):
        return None

    anchors = [_anchor_from_seed(seed) for seed in anchor_seeds]
    common_candidates = _intersect_common_candidates(anchors, plan.first_relation)
    first_ranked = _rank_candidates(
        common_candidates,
        plan.first_rank.metric_id,
        year,
        policy=plan.first_candidate_policy,
    )
    top_first, first_unranked_confirmed = _select_supported(first_ranked, plan.first_relation.rel_type)
    if not top_first:
        return None

    branches: dict[str, dict[str, Any]] = {}
    for branch in plan.branch_ranks:
        branches[branch.kind] = _run_branch(top_first, branch, year)

    hits: list[GraphHit] = []
    seen_nodes: set[str] = set()
    seen_rels: set[str] = set()

    def add_node(id_: str, name: str, role: str, score: float = 1.0) -> None:
        if not id_ or id_ in seen_nodes:
            return
        seen_nodes.add(id_)
        hits.append(_node_hit(id_, name, {"structured_role": role}, score))

    def add_rel(edge: dict[str, Any]) -> None:
        rel_id = f"rel:{edge['rel_type']}:{edge['from_id']}:{edge['to_id']}"
        if rel_id in seen_rels:
            return
        seen_rels.add(rel_id)
        evidence = edge.get("evidence") or {}
        score = 1.0 if evidence.get("level") == "high" else 0.75 if evidence.get("level") == "medium" else 0.55
        hits.append(_rel_hit(edge, score))

    for anchor in anchors:
        add_node(str(anchor.get("id") or ""), str(anchor.get("name") or ""), "common_anchor")
    add_node(str(top_first.get("id") or ""), str(top_first.get("name") or ""), "selected_common_supplier", 0.98)
    for edge in top_first.get("anchor_edges") or [top_first["edge"]]:
        add_rel(edge)

    for kind, branch in branches.items():
        selected = branch.get("selected")
        if not selected:
            continue
        add_node(str(selected.get("id") or ""), str(selected.get("name") or ""), "selected_" + kind, 0.92)
        add_rel(selected["edge"])

    answer_edges = list(top_first.get("anchor_edges") or [top_first["edge"]])
    for branch in branches.values():
        selected = branch.get("selected")
        if selected:
            answer_edges.append(selected["edge"])

    structured = {
        "mode": "structured",
        "year": year,
        "metric_label": _METRIC_LABEL.get(plan.first_rank.metric_id, plan.first_rank.metric_id),
        "plan": plan.to_dict(),
        "first_candidate_policy": plan.first_candidate_policy,
        "anchors": anchors,
        "first": {
            "relation": plan.first_relation.__dict__,
            "common_anchor_min": plan.common_anchor_min,
            "candidates": first_ranked,
            "selected": top_first,
            "unranked_confirmed": first_unranked_confirmed,
            "evidence_floor": _evidence_floor(plan.first_relation.rel_type),
        },
        "branches": branches,
        "best_supported_branch": _best_supported_branch(branches),
        "answer_edges": answer_edges,
        "quality_notes": [
            "common supplier candidates are intersected across anchors before ranking",
            "major_customer, related_party, and investment branches are executed independently",
            "edge evidence is scored from source/chunk availability and endpoint mentions",
        ],
    }
    meta = {
        "mode": "structured",
        "structured": structured,
        "patterns_run": ["structured_plan", *[s.get("op", "") for s in plan.steps]],
        "n_hits": len(hits),
        "fallback_used": False,
        "errors": [],
    }
    return hits, meta


def _execute_single_anchor_branch_rank(
    plan: StructuredPlan,
    org_seed: Seed,
    query: str,
) -> tuple[list[GraphHit], dict] | None:
    year = parse_year(query)
    anchor = _anchor_from_seed(org_seed)

    branches: dict[str, dict[str, Any]] = {}
    for branch in plan.branch_ranks:
        branch_result = _run_branch_from_anchor(
            anchor,
            branch,
            year,
            exclude_ids={str(anchor.get("id") or "")},
        )
        branches[branch_result["kind"]] = branch_result

    hits: list[GraphHit] = []
    seen_nodes: set[str] = set()
    seen_rels: set[str] = set()

    def add_node(id_: str, name: str, role: str, score: float = 1.0) -> None:
        if not id_ or id_ in seen_nodes:
            return
        seen_nodes.add(id_)
        hits.append(_node_hit(id_, name, {"structured_role": role}, score))

    def add_rel(edge: dict[str, Any]) -> None:
        rel_id = f"rel:{edge['rel_type']}:{edge['from_id']}:{edge['to_id']}"
        if rel_id in seen_rels:
            return
        seen_rels.add(rel_id)
        evidence = edge.get("evidence") or {}
        score = 1.0 if evidence.get("level") == "high" else 0.75 if evidence.get("level") == "medium" else 0.55
        hits.append(_rel_hit(edge, score))

    add_node(str(anchor.get("id") or ""), str(anchor.get("name") or ""), "anchor")
    answer_edges: list[dict[str, Any]] = []
    for kind, branch in branches.items():
        selected = branch.get("selected")
        if not selected:
            continue
        relation = RelationStep(**branch["relation"])
        if not _edge_matches_relation(selected["edge"], relation, str(anchor.get("id") or "")):
            continue
        add_node(str(selected.get("id") or ""), str(selected.get("name") or ""), "selected_" + kind, 0.92)
        add_rel(selected["edge"])
        answer_edges.append(selected["edge"])

    structured = {
        "mode": "structured",
        "year": year,
        "metric_label": _METRIC_LABEL.get(plan.first_rank.metric_id, plan.first_rank.metric_id),
        "plan": plan.to_dict(),
        "anchor": anchor,
        "branches": branches,
        "best_supported_branch": _best_supported_branch(branches),
        "answer_edges": answer_edges,
        "abstained": not bool(answer_edges),
        "abstain_reason": "" if answer_edges else "no branch candidate passed relation, evidence, and metric gates",
        "quality_notes": [
            "supplier, related_party, and investment branches are executed independently from the same anchor",
            "each branch is ranked by the same metric before answer assembly",
            "edge evidence is scored from source/chunk availability and endpoint mentions",
        ],
    }
    meta = {
        "mode": "structured",
        "structured": structured,
        "patterns_run": ["structured_plan", *[s.get("op", "") for s in plan.steps]],
        "n_hits": len(hits),
        "fallback_used": False,
        "errors": [],
    }
    return hits, meta


def _execute_for_seed(
    plan: StructuredPlan,
    org_seed: Seed,
    query: str,
) -> tuple[list[GraphHit], dict] | None:
    year = parse_year(query)
    anchor = _anchor_from_seed(org_seed)

    first_relation = _resolve_relation_for_anchor(anchor, plan.first_relation)
    first_candidates = _relation_candidates(anchor, first_relation)
    _attach_candidate_evidence(first_candidates)
    first_policy = plan.first_candidate_policy
    if first_relation.rel_type != "SUPPLIES_TO":
        first_policy = "default"
    first_ranked = _rank_candidates(
        first_candidates,
        plan.first_rank.metric_id,
        year,
        policy=first_policy,
    )
    top_first, first_unranked_confirmed = _select_supported(first_ranked, first_relation.rel_type)
    if not top_first:
        return None

    second_ranked: list[dict[str, Any]] = []
    top_second: dict[str, Any] | None = None
    resolved_second_relation: RelationStep | None = None
    if plan.kind == "two_hop_rank" and plan.second_relation and plan.second_rank:
        exclude = {top_first["id"]}
        if anchor.get("id"):
            exclude.add(str(anchor["id"]))
        resolved_second_relation = _resolve_relation_for_anchor(_anchor_from_candidate(top_first), plan.second_relation)
        second_candidates = _relation_candidates(
            _anchor_from_candidate(top_first),
            resolved_second_relation,
            exclude_ids=exclude,
        )
        _attach_candidate_evidence(second_candidates)
        second_ranked = _rank_candidates(second_candidates, plan.second_rank.metric_id, year)
        top_second, _ = _select_supported(second_ranked, resolved_second_relation.rel_type)

    hits: list[GraphHit] = [_node_hit(str(anchor.get("id") or ""), str(anchor.get("name") or ""), {"structured_role": "anchor"})]
    seen_nodes = {str(anchor.get("id") or "")}

    def add_answer_hit(cand: dict[str, Any] | None, role: str) -> None:
        if not cand:
            return
        cid = cand.get("id") or ""
        if cid and cid not in seen_nodes:
            seen_nodes.add(cid)
            hits.append(_node_hit(cid, cand.get("name") or "", {"structured_role": role}, 0.95))
        hits.append(_rel_hit(cand["edge"], 1.0))

    add_answer_hit(top_first, "selected_first")
    add_answer_hit(top_second, "selected_second")

    answer_edges = [top_first["edge"]]
    if top_second:
        answer_edges.append(top_second["edge"])

    structured = {
        "mode": "structured",
        "year": year,
        "metric_label": _METRIC_LABEL.get(plan.first_rank.metric_id, plan.first_rank.metric_id),
        "plan": plan.to_dict(),
        "first_candidate_policy": plan.first_candidate_policy,
        "anchor": anchor,
        "first": {
            "relation": first_relation.__dict__,
            "candidates": first_ranked,
            "selected": top_first,
            "unranked_confirmed": first_unranked_confirmed,
            "evidence_floor": _evidence_floor(first_relation.rel_type),
        },
        "second": {
            "relation": resolved_second_relation.__dict__ if resolved_second_relation else None,
            "candidates": second_ranked,
            "selected": top_second,
        } if plan.kind == "two_hop_rank" else None,
        "answer_edges": answer_edges,
    }
    meta = {
        "mode": "structured",
        "structured": structured,
        "patterns_run": ["structured_plan", *[s.get("op", "") for s in plan.steps]],
        "n_hits": len(hits),
        "fallback_used": False,
        "errors": [],
    }
    return hits, meta
