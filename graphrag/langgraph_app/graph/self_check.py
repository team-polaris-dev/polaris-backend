"""Graph 노드 자체 검증 메타 — Reflection 노드(다른 팀원)에 힌트 제공.

전체 답변 검증(키워드 커버리지/faithfulness)은 Reflection 담당.
여기서는 그래프 노드 입장의 "요청된 엔티티/관계/연도가 결과에 반영됐는지"
만 점검해 self_check dict 로 반환한다.
"""
from __future__ import annotations

from typing import Any


# requested_relations 표기 → fact_type 매핑 (Supervisor 가 어느 표기로 줄지 모름)
_REL_MAP: dict[str, str] = {
    "shareholder": "shareholder",
    "주주": "shareholder",
    "최대주주": "shareholder",
    "executive": "executive",
    "임원": "executive",
    "subsidiary": "subsidiary",
    "자회사": "subsidiary",
    "종속회사": "subsidiary",
    "investment": "investment",
    "투자": "investment",
    "supplies": "supplies_to",
    "supplies_to": "supplies_to",
    "공급": "supplies_to",
    "produces": "produces",
    "제품": "produces",
    "fin_metric": "fin_metric",
    "재무": "fin_metric",
}


def self_check(
    inp: dict[str, Any],
    facts: list[dict[str, Any]],
    chunk_ids: list[str],
) -> dict[str, Any]:
    req_ents = set(inp.get("entities") or [])
    req_rels = set(inp.get("requested_relations") or [])
    slots = inp.get("slots") or {}
    req_year = str(slots.get("year")) if slots.get("year") is not None else ""

    covered_ents: set[str] = set()
    for f in facts:
        for key in ("code", "sub_code", "b_code", "s_code"):
            v = f.get(key)
            if v in req_ents:
                covered_ents.add(v)

    fact_types = {f.get("type") for f in facts}
    covered_rels = {r for r in req_rels if _REL_MAP.get(r) in fact_types}

    year_covered = True
    if req_year:
        year_covered = any(str(f.get("year")) == req_year for f in facts)

    missing: list[str] = []
    miss_ents = sorted(req_ents - covered_ents)
    if miss_ents:
        missing.append(f"entities: {miss_ents}")
    miss_rels = sorted(req_rels - covered_rels)
    if miss_rels:
        missing.append(f"relations: {miss_rels}")
    if not year_covered:
        missing.append(f"year: {req_year}")
    if not facts and not chunk_ids:
        missing.append("no_results")

    return {
        "entities_requested": len(req_ents),
        "entities_covered": len(covered_ents),
        "relations_requested": len(req_rels),
        "relations_covered": len(covered_rels),
        "year_covered": year_covered,
        "missing": missing,
        "has_results": bool(facts or chunk_ids),
    }
