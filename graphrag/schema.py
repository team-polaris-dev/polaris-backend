"""GraphRAG 출력 계약 + legacy adapter.

GraphHit: 노드/관계/재무를 모두 동일 hit로 표현 (label로 분기)
GraphSearchOutput: 신규 키 (graph_hits / graph_seeds / graph_meta)
Seed: matcher → traverse로 넘어가는 시드 4튜플

adapt_to_legacy: 신규 hit → 기존 graph_facts/paths/provenance (셰임)
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict
try:
    from typing import NotRequired  # type: ignore[attr-defined]
except ImportError:  # Python 3.10
    from typing_extensions import NotRequired  # type: ignore[no-redef]


HitLabel = Literal[
    "organization", "person", "product", "technology",
    "fin_metric", "relationship",
]

KeyType = Literal[
    "corp_code", "er_name", "person_id", "product_id", "tech_id",
]


class Seed(TypedDict, total=False):
    label: HitLabel
    id: str            # display id (corp_code OR "org:" + er_name OR person_id ...)
    key_type: KeyType
    key_value: str     # Cypher 파라미터에 들어가는 실제 값
    name: str
    score: float
    origin: str        # "upstream" | "fulltext"


class GraphHit(TypedDict, total=False):
    id: str
    label: HitLabel
    name: str
    attrs: dict[str, Any]
    score: float
    source: NotRequired[str]
    seed_origin: NotRequired[str]


class GraphSearchOutput(TypedDict):
    graph_hits: list[GraphHit]
    graph_seeds: list[dict]
    graph_meta: dict


# ─────────────────────────────────────────────────────────────
# Legacy adapter (셰임)
# ─────────────────────────────────────────────────────────────

# label → legacy type 매핑 (UnifiedResult.type)
_HIT_TO_LEGACY_TYPE = {
    "organization": "organization",
    "person": "person",
    "product": "product",
    "technology": "technology",
    "fin_metric": "fin_metric",
    # relationship은 attrs.rel_type 따라 분기 (아래 _rel_legacy_type)
}

_REL_TO_LEGACY_TYPE = {
    "EXECUTIVE_OF": "executive",
    "IS_MAJOR_SHAREHOLDER_OF": "shareholder",
    "IS_SUBSIDIARY_OF": "subsidiary",
    "INVESTS_IN": "investment",
    "SUPPLIES_TO": "supply",
    "PRODUCES": "produces",
    "USES_TECH": "uses_tech",
    "RELATED_PARTY": "related_party",
    "INTERLOCKING_DIRECTORATE": "interlocking_directorate",
}


def _fact_from_hit(hit: GraphHit) -> dict:
    """GraphHit → legacy UnifiedResult dict.

    합성기/렌더러가 res.get('type'/'code'/'name'/'value'/'extra'/'source')만 보므로
    이 키만 정확히 채우면 됨.
    """
    label = hit.get("label", "")
    attrs = dict(hit.get("attrs", {}))
    source = hit.get("source", "")

    if label == "relationship":
        rel_type = attrs.get("rel_type", "")
        fact_type = _REL_TO_LEGACY_TYPE.get(rel_type, rel_type.lower() or "relationship")
        # name: "from → to" 형태 (기존 텍스트 합성기 호환)
        from_name = attrs.get("from_name") or attrs.get("from") or ""
        to_name = attrs.get("to_name") or attrs.get("to") or ""
        name = f"{from_name} → {to_name}" if from_name and to_name else hit.get("name", "")
        # value: 관계 종류별 의미 있는 정량 값
        if "qota_rt" in attrs and attrs["qota_rt"] is not None:
            value: Any = attrs["qota_rt"]
        elif "tier" in attrs:
            value = attrs.get("tier")
        elif "pos" in attrs:
            value = attrs.get("pos")
        else:
            value = rel_type
        code = attrs.get("from_id") or hit.get("id", "")
        return {
            "type": fact_type,
            "code": code,
            "name": name,
            "value": value,
            "extra": attrs,
            "source": source,
        }

    fact_type = _HIT_TO_LEGACY_TYPE.get(label, label)
    code = hit.get("id", "")
    name = hit.get("name", "")

    if label == "fin_metric":
        value: Any = attrs.get("value")
    else:
        value = name

    return {
        "type": fact_type,
        "code": code,
        "name": name,
        "value": value,
        "extra": attrs,
        "source": source,
    }


def _path_from_hit(hit: GraphHit) -> list[str] | None:
    """relationship hit → legacy graph_paths 트리플 [from, rel_type, to]."""
    if hit.get("label") != "relationship":
        return None
    attrs = hit.get("attrs", {})
    rel_type = attrs.get("rel_type") or ""
    from_name = attrs.get("from_name") or attrs.get("from") or ""
    to_name = attrs.get("to_name") or attrs.get("to") or ""
    if not (rel_type and from_name and to_name):
        return None
    return [from_name, rel_type, to_name]


def adapt_to_legacy(hits: list[GraphHit]) -> dict:
    """신규 graph_hits → 기존 graph_facts/paths/provenance."""
    facts: list[dict] = []
    paths: list[list[str]] = []
    sources: list[str] = []
    seen_sources: set[str] = set()

    for hit in hits:
        facts.append(_fact_from_hit(hit))

        path = _path_from_hit(hit)
        if path:
            paths.append(path)

        src = hit.get("source")
        if src and src not in seen_sources:
            sources.append(src)
            seen_sources.add(src)

    return {
        "facts": facts,
        "paths": paths,
        "provenance": sources,
    }
