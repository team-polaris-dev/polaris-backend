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
    INDUCED_EDGE_SCORE,
    MAX_INDUCED_EDGES,
    STRUCTURED_CONFIRMED_RENDER_CAP,
    STRUCTURED_MIN_EVIDENCE,
    STRUCTURED_MIN_EVIDENCE_OPERATING,
)
from config.relations import (
    NETWORK_REL_TYPES,
    SUPPLY_NOISE_NAME_TERMS,
    corp_name_variants,
)
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
_TYPE_ATTESTED_RELS = {"RELATED_PARTY"}
# 출자현황·대주주현황은 DART 공시 '표'에서 와서 서술형 청크 본문이 없다(rcept_no 출처만 보유).
# 표의 rcept_no 출처 자체가 1차 근거이므로, 본문 청크/관계어를 요구하는 게이트 대신 출처 보유만
# 확인하는 별도 게이트로 분리한다(_candidate_supported 참고).
_SOURCE_ATTESTED_RELS = {"INVESTS_IN", "IS_MAJOR_SHAREHOLDER_OF", "IS_SUBSIDIARY_OF"}

# 엣지 근거 신뢰도 사다리(_score_edge_evidence). 출처+청크본문+양끝언급+관계어가 모두 있으면
# 최상(FULL), 단계적으로 약해진다. config 로 빼지 않는다 — 게이트 임계(STRUCTURED_MIN_EVIDENCE
# 등)는 config 지만 이 사다리 값 자체는 모듈 내부 판정 로직이라 함께 읽혀야 의미가 산다.
_EVIDENCE_CONF_FULL = 0.95         # 출처+청크본문+양끝언급+관계어
_EVIDENCE_CONF_ENDPOINTS = 0.8     # 출처+청크본문+양끝언급(관계어 누락)
_EVIDENCE_CONF_CHUNK_TEXT = 0.55   # 출처+청크본문(양끝 미언급)
_EVIDENCE_CONF_CHUNK_REF = 0.45    # 출처+청크참조(본문 못 찾음)
_EVIDENCE_CONF_SOURCE_ONLY = 0.35  # 출처만(청크 없음)
_EVIDENCE_CONF_NONE = 0.1          # 출처 없음
# level 경계 = 점수 가산 경계와 동일 임계. high≥0.8, medium≥0.55.
_EVIDENCE_LEVEL_HIGH_MIN = 0.8
_EVIDENCE_LEVEL_MEDIUM_MIN = 0.55
# 근거 level → 답 노드 hit score(높을수록 확신). 강근거=1.0, 중=0.75, 약=0.55.
_NODE_SCORE_HIGH = 1.0
_NODE_SCORE_MEDIUM = 0.75
_NODE_SCORE_LOW = 0.55
_ANSWER_HIT_DEFAULT_SCORE = 0.95   # add_answer_hit 기본(메트릭 랭킹 1위 등 명시 근거)

# SUPPLIES_TO 질의 시점 노이즈 게이트(_passes_noise_gate). 적재 과정에서 SUPPLIES_TO 에는
# corp_code 없는 외부/제품 노드와 운영 공급사가 아닌 금융기관(대주·인수단)이 섞인다. 엣지를
# 지우지 않고 랭킹 후보에서만 비파괴적으로 빼 되돌릴 수 있게 한다. 운영 거래상대만 남긴다.
# 노이즈 단어 목록은 config/relations.json 단어집(SUPPLY_NOISE_NAME_TERMS)이 SSOT.


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
    # 별칭 변형 규칙(에스케이↔SK 음역, 하이닉스 약칭)은 config/relations.json 단어집이 SSOT.
    return corp_name_variants(_norm_evidence_text(name))


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
        confidence = _EVIDENCE_CONF_FULL
    elif has_source and has_chunk_text and from_mentioned and to_mentioned:
        confidence = _EVIDENCE_CONF_ENDPOINTS
        warnings.append("chunk_names_relation_term_missing")
    elif has_source and has_chunk_text:
        confidence = _EVIDENCE_CONF_CHUNK_TEXT
        warnings.append("chunk_does_not_name_both_endpoints")
    elif has_source and has_chunk_ref:
        confidence = _EVIDENCE_CONF_CHUNK_REF
        warnings.append("chunk_reference_not_found")
    elif has_source:
        confidence = _EVIDENCE_CONF_SOURCE_ONLY
        warnings.append("document_source_without_chunk")
    else:
        confidence = _EVIDENCE_CONF_NONE
        warnings.append("missing_source")

    if rel_type in (_TYPE_ATTESTED_RELS | _SOURCE_ATTESTED_RELS) and confidence < _EVIDENCE_LEVEL_HIGH_MIN:
        warnings.append("weak_evidence_for_accounting_or_investment_relation")

    level = (
        "high" if confidence >= _EVIDENCE_LEVEL_HIGH_MIN
        else "medium" if confidence >= _EVIDENCE_LEVEL_MEDIUM_MIN
        else "low"
    )
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

    if evidence_confidence >= _EVIDENCE_LEVEL_HIGH_MIN:
        score += 1.0
        reasons.append("strong_edge_evidence")
    elif evidence_confidence >= _EVIDENCE_LEVEL_MEDIUM_MIN:
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

    # 정렬키 우선순위(질문이 "매출 상위"면 매출이 주 정렬키여야 한다):
    #  1) bucket>=0 — 페널티 받은 후보(허브·operating 정책의 지배구조 계열사)는 한 단계
    #     아래 tier 로 가라앉힌다. 단 페널티의 '정도'는 지표를 못 이긴다(coarse 강등).
    #  2) 지표 보유 여부 — 지표 없는 후보(매출 0/미보유 외국·제품 노드)는 지표 보유 후보 아래.
    #  3) 지표 값 — 사용자가 요청한 지표(매출 등) 내림차순. 이게 진짜 랭킹 차원.
    #  4) bucket 값 — 동률(같은 지표값)일 때만 근거·운영상대 품질로 미세 정렬.
    # 예전엔 bucket 이 1순위라 근거 좋은 매출 0원 후보(WNC·NVIDIA)가 매출 1.8조 후보를
    # 누르고 "매출 상위"의 맨 위에 올라오는 버그가 있었다 → 지표를 bucket 위로 올린다.
    ranked.sort(
        key=lambda c: (
            (c.get("policy", {}).get("bucket", 0) or 0) >= 0,
            c.get("metric", {}).get("value") is not None,
            c.get("metric", {}).get("value") or float("-inf"),
            c.get("policy", {}).get("bucket", 0) or 0,
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
    if rel_type in _SOURCE_ATTESTED_RELS:
        # 표 출처 관계: 청크 본문이 구조적으로 없어 conf 가 0.35(출처만)에 묶인다.
        # 출처(rcept_no) 보유만으로 게이트를 통과시켜 매출 랭킹으로 넘긴다.
        return _evidence_confidence(candidate) >= _EVIDENCE_CONF_SOURCE_ONLY
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


def _confirmed_render_candidates(
    unranked_confirmed: list[dict[str, Any]],
    relation: RelationStep,
    cap: int,
    anchor_id: str | None = None,
) -> list[dict[str, Any]]:
    """Evidence-confirmed relationships that have no comparable metric.

    They pass the evidence / type-attestation gate but carry no metric value, so
    they cannot win the metric ranking. Render them (capped, in ranked order) so
    the relationship network the question asked for is not blank when no
    metric-bearing candidate exists. The cap prevents hairballs.
    """
    out: list[dict[str, Any]] = []
    for cand in unranked_confirmed or []:
        if len(out) >= cap:
            break
        edge = cand.get("edge")
        if not edge or not _edge_matches_relation(edge, relation, anchor_id):
            continue
        out.append(cand)
    return out


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


def execute(plan: StructuredPlan, seeds: list[Seed], query: str) -> tuple[list[GraphHit], dict] | None:
    org_seeds = [s for s in seeds if s.get("label") == "organization"]
    if not org_seeds:
        return None
    if plan.kind == "multi_hop_chain":
        return _execute_multi_hop_chain(plan, org_seeds, query)
    if plan.kind == "multi_anchor_rank":
        return _execute_multi_anchor_rank(plan, org_seeds, query)
    if plan.kind == "community_member_rank":
        for org_seed in sorted(org_seeds, key=lambda s: _seed_order(s, query)):
            result = _execute_community_member_rank(plan, org_seed, query)
            if result:
                return result
        return None
    return None


def _chain_hop_node_score(depth: int) -> float:
    """홉 깊이가 깊을수록 노드 확신 점수를 단계적으로 낮춘다(1홉≈high, 이후 감쇠)."""
    return max(_NODE_SCORE_LOW, _NODE_SCORE_HIGH - 0.1 * depth)


def _passes_noise_gate(candidate: dict[str, Any], rel_type: str) -> bool:
    """SUPPLIES_TO 랭킹 진입 전 노이즈 후보를 비파괴적으로 거른다(질의 시점 게이트).

    SUPPLIES_TO 외 관계는 영향 없음. SUPPLIES_TO 는 (1) corp_code 없는 외부/제품 노드,
    (2) 금융기관(대주·인수단)을 제외한다 — 둘 다 운영 공급사가 아니며 매출 랭킹의 분모를
    오염시킨다. 엣지를 삭제하지 않으므로 게이트만 끄면 원상복구된다.
    """
    if rel_type != "SUPPLIES_TO":
        return True
    if not candidate.get("corp_code"):
        return False
    name = str(candidate.get("name") or "")
    return not any(term in name for term in SUPPLY_NOISE_NAME_TERMS)


def _gate_candidates(candidates: list[dict[str, Any]], rel_type: str) -> list[dict[str, Any]]:
    return [c for c in candidates if _passes_noise_gate(c, rel_type)]


def _candidate_rel_type(candidate: dict[str, Any]) -> str:
    return str(candidate.get("edge", {}).get("rel_type") or "")


def _gather_hop_candidates(
    anchor: dict[str, Any],
    hop: Any,
    *,
    exclude_ids: set[str],
) -> list[dict[str, Any]]:
    """홉의 모든 관계(hop.rel_steps())를 펼쳐 후보를 합집합으로 모은다(노드 id 기준 dedupe).

    '수혜'처럼 영향이 한 관계로만 흐르지 않는 복합 홉을 위해 공급·지분·지배·투자·특수관계
    후보를 한 풀로 합친다. 관계 배열의 순서 = LLM 선호: 같은 노드가 여러 관계로 닿으면 먼저
    나온 관계의 엣지를 대표로 남겨, 어떤 관계로 보여줄지를 LLM 이 정하게 한다(executor 가
    "지분이 공급을 이긴다" 같은 룰을 박지 않는다). 노이즈 게이트는 관계별 rel_type 으로 적용.
    단일 관계 홉이면 기존 _relation_candidates+노이즈게이트와 동일한 결과.
    """
    anchor_id = str(anchor.get("id") or "")
    excl = exclude_ids | {anchor_id}
    by_id: dict[str, dict[str, Any]] = {}
    for step in hop.rel_steps():
        relation = _resolve_relation_for_anchor(anchor, step)
        cands = _gate_candidates(
            _relation_candidates(anchor, relation, exclude_ids=excl),
            relation.rel_type,
        )
        for cand in cands:
            cid = str(cand.get("id") or "")
            if not cid or cid in by_id:
                continue
            by_id[cid] = cand
    return list(by_id.values())


def _candidate_is_fertile(
    candidate: dict[str, Any],
    next_hop: Any,
    year: int | None,
    *,
    exclude_ids: set[str],
) -> bool:
    """이 후보를 다음 홉 앵커로 삼았을 때 이어질 후속 후보가 1개 이상 있는지 전방탐색한다.

    막다른 길(예: 유일 상위가 지주사뿐이라 다음 홉이 끊기는 경우)을 피해, top_n 슬롯을
    실제로 체인이 이어질 수 있는 후보로 채우기 위함이다. 노이즈·근거 게이트를 다음 홉과
    동일하게 적용해, "전방에서도 채택 가능한가" 를 같은 기준으로 본다.
    """
    anchor = _anchor_from_candidate(candidate)
    cands = _gather_hop_candidates(anchor, next_hop, exclude_ids=exclude_ids)
    if not cands:
        return False
    _attach_candidate_evidence(cands)
    return any(_candidate_supported(c, _candidate_rel_type(c)) for c in cands)


def _select_hop_candidates(
    ranked_supported: list[dict[str, Any]],
    top_n: int,
    *,
    next_hop: Any | None,
    year: int | None,
    visited_ids: set[str],
) -> list[dict[str, Any]]:
    """메트릭 순서를 지키며 상위 top_n 을 고르되, 다음 홉이 남았으면 이어질 수 있는(fertile)
    후보를 우선 채운다. fertile 가 모자라면 막다른 후보로 슬롯을 메워 커버리지를 그리디보다
    줄이지 않는다(정직한 degrade). 마지막 홉이면 전방탐색 없이 그대로 상위 top_n.
    """
    cap = max(1, top_n)
    if next_hop is None:
        return ranked_supported[:cap]
    fertile: list[dict[str, Any]] = []
    infertile: list[dict[str, Any]] = []
    for cand in ranked_supported:
        if len(fertile) >= cap:
            break
        if _candidate_is_fertile(cand, next_hop, year, exclude_ids=visited_ids):
            fertile.append(cand)
        else:
            infertile.append(cand)
    selected = fertile[:cap]
    if len(selected) < cap:
        selected += infertile[: cap - len(selected)]
    return selected


def _execute_multi_hop_chain(
    plan: StructuredPlan,
    org_seeds: list[Seed],
    query: str,
) -> tuple[list[GraphHit], dict] | None:
    """LLM 이 계획한 다중홉 랭킹 체인(cutline)을 한 walker 로 실행한다.

    각 홉마다 현재 frontier 의 모든 앵커에서 hop.relation 후보를 뽑아 hop.rank 지표로
    줄세우고 근거 게이트를 통과한 상위 top_n 을 답으로 채택한다. 채택된 후보가 다음 홉의
    frontier(앵커)가 되어 "수혜의 수혜" 를 임의 깊이로 잇는다 — 단일 2홉 top-1 랭킹의
    한계를 top_n·N홉으로 일반화한다. 한 홉도 답을 못 내면 None → search 가
    abstain 으로 graceful degrade.
    """
    if not plan.hops:
        return None
    year = parse_year(query)

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
        score = _NODE_SCORE_HIGH if evidence.get("level") == "high" else _NODE_SCORE_MEDIUM if evidence.get("level") == "medium" else _NODE_SCORE_LOW
        hits.append(_rel_hit(edge, score))

    # 초기 frontier = 질문이 호명한 org 앵커들(순서: 질문 언급 우선).
    visited_ids: set[str] = set()
    frontier: list[dict[str, Any]] = []
    for seed in sorted(org_seeds, key=lambda s: _seed_order(s, query)):
        anchor = _anchor_from_seed(seed)
        aid = str(anchor.get("id") or "")
        if not aid or aid in visited_ids:
            continue
        add_node(aid, str(anchor.get("name") or ""), "anchor", _NODE_SCORE_HIGH)
        visited_ids.add(aid)
        frontier.append(anchor)
    if not frontier:
        return None

    answer_edges: list[dict[str, Any]] = []
    hop_results: list[dict[str, Any]] = []
    degraded = False
    for depth, hop in enumerate(plan.hops):
        next_hop = plan.hops[depth + 1] if depth + 1 < len(plan.hops) else None
        next_frontier: list[dict[str, Any]] = []
        selected_in_hop: set[str] = set()
        selected_names: list[str] = []
        for anchor in frontier:
            candidates = _gather_hop_candidates(anchor, hop, exclude_ids=visited_ids)
            if not candidates:
                continue
            _attach_candidate_evidence(candidates)
            ranked = _rank_candidates(candidates, hop.rank.metric_id, year, policy=hop.policy)
            supported = [c for c in ranked if _candidate_supported(c, _candidate_rel_type(c))]
            chosen = _select_hop_candidates(
                supported, hop.top_n, next_hop=next_hop, year=year, visited_ids=visited_ids,
            )
            if len(chosen) < max(1, hop.top_n):
                degraded = True
            for cand in chosen:
                cid = str(cand.get("id") or "")
                if not cid or cid in selected_in_hop:
                    continue
                selected_in_hop.add(cid)
                add_node(cid, str(cand.get("name") or ""), f"hop{depth + 1}_selected", _chain_hop_node_score(depth + 1))
                add_rel(cand["edge"])
                answer_edges.append(cand["edge"])
                selected_names.append(str(cand.get("name") or ""))
                if cid not in visited_ids:
                    next_frontier.append(_anchor_from_candidate(cand))
        for anchor in next_frontier:
            visited_ids.add(str(anchor.get("id") or ""))
        hop_results.append({
            "depth": depth + 1,
            "relation": hop.relation.__dict__,
            "relations": [r.__dict__ for r in hop.rel_steps()],
            "metric": hop.rank.metric_id,
            "metric_label": _METRIC_LABEL.get(hop.rank.metric_id, hop.rank.metric_id),
            "top_n": hop.top_n,
            "selected": selected_names,
        })
        frontier = next_frontier
        if not frontier:
            break

    if not answer_edges:
        return None

    structured = {
        "mode": "structured",
        "kind": "multi_hop_chain",
        "year": year,
        "plan": plan.to_dict(),
        "hops": hop_results,
        "answer_edges": answer_edges,
        "abstained": False,
        "degraded": degraded,
        "quality_notes": [
            "각 홉에서 관계 후보를 노이즈 게이트로 거른 뒤 지표로 줄세워 상위 top_n 을 다음 홉 앵커로 넘긴다",
            "비종단 홉은 전방탐색으로 이어질 수 있는(fertile) 후보를 우선 채워 막다른 길을 피한다",
            "SUPPLIES_TO 는 corp_code 없는 외부·제품 노드와 금융기관을 랭킹에서 비파괴적으로 제외한다",
            "엣지 근거는 출처/청크/양끝 언급으로 점수화한다",
            *(["데이터가 모자라 일부 홉이 top_n 미만으로 채워졌다(없는 관계를 지어내지 않음)"] if degraded else []),
        ],
    }
    meta = {
        "mode": "structured",
        "structured": structured,
        "patterns_run": ["structured_plan", "multi_hop_chain"],
        "n_hits": len(hits),
        "fallback_used": False,
        "errors": [],
    }
    return hits, meta


def _execute_multi_anchor_rank(
    plan: StructuredPlan,
    org_seeds: list[Seed],
    query: str,
) -> tuple[list[GraphHit], dict] | None:
    """공통 앵커 교집합 후보를 단일 지표로 줄세워 1위를 답한다.

    "둘 다와 거래하는 소재사 중 매출 1위" 류 — 둘 이상 앵커의 공통 거래상대를 교집합으로
    구한 뒤 지표로 줄세워 first 에서 멈춘다. 교집합 정확성이 핵심이라 text2cypher 로 뭉개지
    않고 결정적으로 보장한다. 교집합 후보가 근거/지표 게이트를 못 넘으면 None → search 가
    abstain 으로 graceful degrade.
    """
    year = parse_year(query)
    min_anchors = max(2, plan.common_anchor_min)
    anchor_seeds = _common_anchor_seeds(org_seeds, query, min_anchors)
    if len(anchor_seeds) < min_anchors:
        return None

    anchors = [_anchor_from_seed(seed) for seed in anchor_seeds]
    common_candidates = _intersect_common_candidates(anchors, plan.first_relation)
    ranked = _rank_candidates(
        common_candidates,
        plan.first_rank.metric_id,
        year,
        policy=plan.first_candidate_policy,
    )
    top_first, unranked_confirmed = _select_supported(ranked, plan.first_relation.rel_type)
    confirmed = (
        _confirmed_render_candidates(unranked_confirmed, plan.first_relation, STRUCTURED_CONFIRMED_RENDER_CAP)
        if not top_first
        else []
    )
    if not top_first and not confirmed:
        return None

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
        score = _NODE_SCORE_HIGH if evidence.get("level") == "high" else _NODE_SCORE_MEDIUM if evidence.get("level") == "medium" else _NODE_SCORE_LOW
        hits.append(_rel_hit(edge, score))

    for anchor in anchors:
        add_node(str(anchor.get("id") or ""), str(anchor.get("name") or ""), "common_anchor")

    answer_edges: list[dict[str, Any]] = []
    confirmed_edges: list[dict[str, Any]] = []
    if top_first:
        add_node(str(top_first.get("id") or ""), str(top_first.get("name") or ""), "selected_common_supplier", 0.98)
        for edge in top_first.get("anchor_edges") or [top_first["edge"]]:
            add_rel(edge)
            answer_edges.append(edge)
        # 1위만 노드로 내보내면 보고서가 "유일한 공통 공급사"로 오판한다. 근거·지표 게이트를
        # 통과한 나머지 교집합 후보도 관계망에 노출해 "N곳 중 1위"로 비교 서술하게 한다.
        # 답·1위 선정은 그대로 — 렌더링만 확장(상위 N개로 hairball 방지).
        runner_ups = [
            c for c in ranked
            if c is not top_first
            and _candidate_supported(c, plan.first_relation.rel_type)
            and c.get("metric", {}).get("value") is not None
        ]
        for cand in runner_ups[:STRUCTURED_CONFIRMED_RENDER_CAP]:
            add_node(str(cand.get("id") or ""), str(cand.get("name") or ""), "common_supplier", 0.85)
            for edge in cand.get("anchor_edges") or [cand["edge"]]:
                add_rel(edge)
    else:
        for cand in confirmed:
            add_node(str(cand.get("id") or ""), str(cand.get("name") or ""), "confirmed_common_supplier", 0.7)
            for edge in cand.get("anchor_edges") or [cand["edge"]]:
                add_rel(edge)
                confirmed_edges.append(edge)

    structured = {
        "mode": "structured",
        "kind": "multi_anchor_rank",
        "year": year,
        "metric_label": _METRIC_LABEL.get(plan.first_rank.metric_id, plan.first_rank.metric_id),
        "plan": plan.to_dict(),
        "first_candidate_policy": plan.first_candidate_policy,
        "anchors": anchors,
        "first": {
            "relation": plan.first_relation.__dict__,
            "common_anchor_min": plan.common_anchor_min,
            "candidates": ranked,
            "selected": top_first,
            "unranked_confirmed": unranked_confirmed,
            "evidence_floor": _evidence_floor(plan.first_relation.rel_type),
        },
        "answer_edges": answer_edges,
        "confirmed_edges": confirmed_edges,
        "abstained": not answer_edges and not confirmed_edges,
        "quality_notes": [
            "공통 앵커 교집합 후보를 단일 지표로 줄세워 1위만 답한다",
            "교집합·랭킹·근거게이트로 공통 거래상대를 결정적으로 확정한다",
            "엣지 근거는 출처/청크/양끝 언급으로 점수화한다",
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


def _community_for_anchor(anchor: dict[str, Any]) -> dict[str, Any] | None:
    """앵커가 속한 가장 큰 커뮤니티의 멤버 목록. 없거나 실패 시 None.

    앵커는 corp_code 또는 er_name 으로 들어온다(매처가 약칭을 er_name 노드로 해소하는 경우
    잦음). 커뮤니티 멤버는 corp_code 배열이라, 앵커 노드를 먼저 찾아 그 corp_code 로 군집을
    역조회한다.
    """
    try:
        match_clause, params = _anchor_match(anchor)
    except ValueError:
        return None
    cypher = (
        f"{match_clause} "
        "WITH anchor.corp_code AS cc WHERE cc IS NOT NULL "
        "MATCH (c:Community) WHERE cc IN c.members "
        "RETURN c.members AS members, c.member_names AS member_names, "
        "       c.cluster_id AS cluster_id, c.size AS size "
        "ORDER BY c.size DESC LIMIT 1"
    )
    try:
        with neo4j_driver.session() as session:
            row = session.run(cypher, **params).single()
    except Exception as exc:
        print(f"⚠️ [structured_executor] community lookup failed: {exc!r}")
        return None
    if not row:
        return None
    members = [str(m) for m in (row.get("members") or []) if m]
    if not members:
        return None
    return {
        "members": members,
        "member_names": [str(n) for n in (row.get("member_names") or [])],
        "cluster_id": row.get("cluster_id"),
        "size": int(row.get("size") or len(members)),
    }


def _member_network_edges(member_codes: list[str]) -> list[dict[str, Any]]:
    """군집 멤버 집합 *내부*의 회사↔회사 활성 망 엣지. 헤어볼 방지로 cap.

    근거(evidence)는 붙이지 않는다 — traverse/induced 경로와 동일하게 구조화 적재 관계는
    무근거로 패널 게이트(_path_from_hit)를 통과하고, 투자/특수관계는 텍스트(facts)로만 남는다.
    """
    codes = [c for c in dict.fromkeys(member_codes) if c]
    if len(codes) < 2:
        return []
    cypher = f"""
MATCH (a:Organization)-[r]->(b:Organization)
WHERE a.corp_code IN $codes AND b.corp_code IN $codes
  AND a <> b
  AND type(r) IN $network_rel_types
  AND coalesce(r.valid_to, '') = ''
  AND r.qc_disabled_at IS NULL
WITH a, b, r,
     {_org_id_expr('a')} AS from_id,
     {_org_id_expr('b')} AS to_id
RETURN type(r) AS rel_type, from_id, a.name AS from_name,
       to_id, b.name AS to_name, r.rcept_no AS source, r.chunk_id AS chunk_id
ORDER BY from_id ASC, to_id ASC, rel_type ASC
LIMIT $cap
"""
    try:
        with neo4j_driver.session() as session:
            rows = [
                r.data()
                for r in session.run(
                    cypher,
                    codes=codes,
                    network_rel_types=list(NETWORK_REL_TYPES),
                    cap=MAX_INDUCED_EDGES,
                )
            ]
    except Exception as exc:
        print(f"⚠️ [structured_executor] member network query failed: {exc!r}")
        return []
    edges: list[dict[str, Any]] = []
    for row in rows:
        from_id = str(row.get("from_id") or "")
        to_id = str(row.get("to_id") or "")
        rel_type = str(row.get("rel_type") or "")
        if not from_id or not to_id or not rel_type or from_id == to_id:
            continue
        edges.append({
            "rel_type": rel_type,
            "from_id": from_id,
            "from_name": str(row.get("from_name") or ""),
            "to_id": to_id,
            "to_name": str(row.get("to_name") or ""),
            "role": "",
            "source": str(row.get("source") or ""),
            "chunk_id": str(row.get("chunk_id") or ""),
        })
    return edges


def _execute_community_member_rank(
    plan: StructuredPlan,
    org_seed: Seed,
    query: str,
) -> tuple[list[GraphHit], dict] | None:
    """그룹/계열 군집 멤버를 노드 지표(매출 등)로 줄세워 1위를 답하고 군집 관계망을 시각화."""
    year = parse_year(query)
    anchor = _anchor_from_seed(org_seed)

    community = _community_for_anchor(anchor)
    if not community:
        return None
    members = community["members"]

    metric_id = plan.first_rank.metric_id
    metric_rows = _fetch_metric_values(members, metric_id, year)
    if not metric_rows:
        return None

    # corp_code → 표시명 (군집 멤버명 우선, 지표 행 corp_name 으로 보강)
    name_by_code: dict[str, str] = {}
    for code, nm in zip(members, community.get("member_names") or []):
        if code and nm:
            name_by_code[str(code)] = str(nm)
    for row in metric_rows:
        code = str(row.get("corp_code") or "")
        nm = str(row.get("corp_name") or "")
        if code and nm:
            name_by_code[code] = nm

    # 랭킹(이미 value DESC). 결정성: 동점이면 corp_code asc.
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
        return None
    ranked.sort(key=lambda m: (-m["value"], m["corp_code"]))
    winner = ranked[0]

    edges = _member_network_edges(members)

    hits: list[GraphHit] = []
    # (a) 1위 재무수치 — render._fmt_graph 가 account_id/value/bsns_year 로 읽어 수치를 노출
    hits.append({
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

    # (b) 관계망에 등장하는 멤버 노드 (고립 노드 방지: 엣지 끝점 + 1위만)
    endpoint_ids = {e["from_id"] for e in edges} | {e["to_id"] for e in edges}
    endpoint_ids.add(winner["corp_code"])
    seen_nodes: set[str] = set()
    for code in [winner["corp_code"], *members]:
        if code not in endpoint_ids or code in seen_nodes:
            continue
        seen_nodes.add(code)
        role = "selected_member" if code == winner["corp_code"] else "community_member"
        score = _NODE_SCORE_HIGH if code == winner["corp_code"] else _NODE_SCORE_LOW
        hits.append(_node_hit(code, name_by_code.get(code, code), {"structured_role": role}, score))

    # (c) 군집 내부 관계망 엣지
    seen_rels: set[str] = set()
    for edge in edges:
        rel_id = f"rel:{edge['rel_type']}:{edge['from_id']}:{edge['to_id']}"
        if rel_id in seen_rels:
            continue
        seen_rels.add(rel_id)
        hits.append(_rel_hit(edge, INDUCED_EDGE_SCORE))

    structured = {
        "mode": "structured",
        "kind": "community_member_rank",
        "year": year,
        "metric_label": _METRIC_LABEL.get(metric_id, metric_id),
        "plan": plan.to_dict(),
        "anchor": anchor,
        "community": {
            "cluster_id": community.get("cluster_id"),
            "size": community.get("size"),
            "member_count": len(members),
        },
        "members": ranked,
        "selected": winner,
        "answer_edges": [],
        "abstained": False,
        "rankable": True,
        "quality_notes": [
            "앵커가 속한 군집의 멤버를 노드 지표(연결재무 CFS/연간)로 줄세웠다",
            "1위는 fin_metric hit, 군집 관계망은 회사↔회사 망 엣지로 시각화한다",
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
