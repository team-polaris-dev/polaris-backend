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

from config.relations import (
    REL_TO_LEGACY_TYPE as _REL_TO_LEGACY_TYPE,
    NETWORK_REL_TYPES as _NETWORK_REL_TYPES,
)
from config.graphrag import PANEL_MIN_EVIDENCE


_TYPE_ATTESTED_RELS = {"RELATED_PARTY", "INVESTS_IN"}


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

# _REL_TO_LEGACY_TYPE, _NETWORK_REL_TYPES 는 config.relations 에서 import(상단).
# 망(패널) 엣지 = 회사↔회사 사업관계만(is_network). 임원·제품·기술은 속성 → 망 제외.


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
    """relationship hit → legacy graph_paths 트리플 [from, rel_type, to].

    회사↔회사 사업관계(_NETWORK_REL_TYPES)만 망 엣지로. 제품/기술/임원/재무 관계는
    facts에는 남지만 그래프 패널에는 그리지 않는다 (속성 ≠ 관계).
    """
    if hit.get("label") != "relationship":
        return None
    attrs = hit.get("attrs", {})
    rel_type = attrs.get("rel_type") or ""
    if rel_type not in _NETWORK_REL_TYPES:
        return None
    if rel_type in _TYPE_ATTESTED_RELS and not attrs.get("evidence_relation_term_found"):
        return None
    evidence_confidence = attrs.get("evidence_confidence")
    if evidence_confidence is not None:
        try:
            if float(evidence_confidence) < PANEL_MIN_EVIDENCE:
                return None
        except (TypeError, ValueError):
            return None
    from_name = attrs.get("from_name") or attrs.get("from") or ""
    to_name = attrs.get("to_name") or attrs.get("to") or ""
    if not (rel_type and from_name and to_name):
        return None
    return [from_name, rel_type, to_name]


def adapt_to_legacy(hits: list[GraphHit]) -> dict:
    """신규 graph_hits → 기존 graph_facts/paths/provenance.

    path_sources/path_chunks 는 paths 와 행 단위로 정렬된다(같은 i = 같은 망 엣지).
    serialize.build_graph 가 엣지별 근거를 i 로 읽으므로 paths 와 길이가 반드시 같아야
    한다 — facts(전체 hit) 로 인덱싱하면 어긋난다(예전 버그). 그래서 같은 루프에서 동시 append.
    """
    facts: list[dict] = []
    paths: list[list[str]] = []
    path_sources: list[str] = []   # 문서 출처(rcept_no) — 모든 망 엣지
    path_chunks: list[str] = []    # 청크 출처(chunk_id) — 추출 엣지만(없으면 '')
    sources: list[str] = []
    seen_sources: set[str] = set()

    for hit in hits:
        facts.append(_fact_from_hit(hit))

        path = _path_from_hit(hit)
        if path:
            paths.append(path)
            path_sources.append(hit.get("source") or "")
            path_chunks.append((hit.get("attrs") or {}).get("chunk_id") or "")

        src = hit.get("source")
        if src and src not in seen_sources:
            sources.append(src)
            seen_sources.add(src)

    return {
        "facts": facts,
        "paths": paths,
        "provenance": sources,
        "path_sources": path_sources,
        "path_chunks": path_chunks,
    }
