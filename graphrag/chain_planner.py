"""다중홉 랭킹 체인(cutline) 플래너 — 공식 text2cypher 위에 얹는 value-add 튜닝 레이어.

'A가 오르면 수혜 볼 기업 N개, 거기서 또 수혜 볼 기업 N개'처럼 영향이 단계적으로 전파되는
질문을, 각 홉에서 관계 후보를 노드 지표(매출 등)로 줄세워 상위 top_n 을 다음 홉 앵커로
넘기는 재귀 랭킹으로 만든다. 라이브러리(Text2CypherRetriever)는 단일 read-only Cypher 한 개만
생성하므로 이 재귀 줄세우기는 엔진이 아니라 오케스트레이션 결정 → 도메인 레이어로 남긴다.

LLM 은 제한된 hops 스키마(관계·방향·지표·top_n)만 채운다. 절대 Cypher/SQL/자유텍스트를
만들지 않으며 무효 출력은 폐기(None)돼 호출자가 폴백한다. 실행은 structured_executor.execute
(plan.kind=='multi_hop_chain' → _execute_multi_hop_chain)가 맡는다.
"""
from __future__ import annotations

from typing import Any

from config.relations import CHAIN_RELATION_TYPES, metric_for_query
from graphrag.plan_schema import (
    HopStep,
    MetricId,
    MetricRankStep,
    RelationStep,
    StructuredPlan,
)

_RELATION_TYPES = frozenset(CHAIN_RELATION_TYPES)  # 허용 관계 = 단어집 SSOT(config.relations)
_DIRECTIONS = {"incoming", "outgoing", "undirected", "auto"}
_METRICS = {
    "ifrs-full_Revenue",
    "dart_OperatingIncomeLoss",
    "ifrs-full_ProfitLoss",
    "ifrs-full_Assets",
}
_FIRST_POLICIES = {"default", "operating_counterparty"}

_RELATION_DEFAULTS: dict[tuple[str, str], tuple[str, str]] = {
    ("SUPPLIES_TO", "incoming"): ("suppliers", "supplier"),
    ("SUPPLIES_TO", "outgoing"): ("buyers", "buyer"),
    ("SUPPLIES_TO", "auto"): ("supply_counterparties", "supply_counterparty"),
    ("RELATED_PARTY", "undirected"): ("related_parties", "related_party"),
    ("IS_MAJOR_SHAREHOLDER_OF", "incoming"): ("shareholders", "shareholder"),
    ("IS_MAJOR_SHAREHOLDER_OF", "outgoing"): ("shareholdings", "shareholding"),
    ("IS_SUBSIDIARY_OF", "incoming"): ("subsidiaries", "subsidiary"),
    ("IS_SUBSIDIARY_OF", "outgoing"): ("parents", "parent"),
    ("INVESTS_IN", "outgoing"): ("investees", "investee"),
    ("INVESTS_IN", "incoming"): ("investors", "investor"),
    ("INVESTS_IN", "undirected"): ("investment_parties", "investment"),
}

# 다중홉 체인(cutline) cue. "수혜의 수혜"처럼 영향이 단계적으로 퍼지는 질문 신호.
# 두 축을 모두 요구한다(_looks_chain): 영향이 *전파*되는 어휘(수혜/낙수…) + 그 전파가
# *반복*된다는 표지(또/거기서/그다음…). 전파어만 있으면 단일 홉 '수혜주'질문이므로 체인 아님.
_CHAIN_PROPAGATION_TERMS = (
    "수혜", "낙수", "파급", "연쇄", "타고", "한 다리", "한다리", "건너",
)
_CHAIN_RECURSION_TERMS = (
    "또", "거기서", "거기에", "그다음", "그 다음", "이어서", "다시", "연달아",
    "단계적", "재차", "그곳",
)


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(t in text for t in terms)


def _looks_chain(query: str) -> bool:
    """다중홉 랭킹 체인(cutline)인가. 전파 cue + 반복 표지를 모두 요구한다.

    "SK하이닉스가 오르면 수혜 볼 기업, 거기서 또 수혜 볼 기업 N개" 류. 전파어만 있으면
    단일 홉 '수혜주' 질문이라 체인이 아니다 → 반복 표지(또/거기서…)까지 있어야 LLM 에 위임.
    """
    q = " ".join((query or "").split())
    if not q:
        return False
    return _has_any(q, _CHAIN_PROPAGATION_TERMS) and _has_any(q, _CHAIN_RECURSION_TERMS)


def _relation_step(data: dict[str, Any] | None) -> RelationStep | None:
    if not isinstance(data, dict):
        return None
    rel = str(data.get("rel_type") or data.get("relation") or "").strip()
    direction = str(data.get("direction") or "").strip()
    if rel not in _RELATION_TYPES or direction not in _DIRECTIONS:
        return None
    default_alias, default_role = _RELATION_DEFAULTS.get((rel, direction), (rel.lower(), rel.lower()))
    alias = str(data.get("alias") or default_alias)
    role = str(data.get("role") or default_role)
    return RelationStep(rel, direction, alias, role)  # type: ignore[arg-type]


def _relation_steps(raw: Any) -> tuple[RelationStep, ...]:
    """relation 필드를 RelationStep 튜플로. dict 하나=단일, list=복합(합집합) 홉.

    무효 원소는 버린다. 하나도 못 만들면 빈 튜플(호출자가 폐기). 중복 (rel,direction)
    은 첫 등장만 남긴다.
    """
    items = raw if isinstance(raw, list) else [raw]
    out: list[RelationStep] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        step = _relation_step(item if isinstance(item, dict) else None)
        if step is None:
            continue
        key = (step.rel_type, step.direction)
        if key in seen:
            continue
        seen.add(key)
        out.append(step)
    return tuple(out)


def _metric_id(data: dict[str, Any], query: str) -> MetricId | None:
    """LLM 이 준 rank_metric 을 검증, 없으면 질의 어휘로 해소(config.relations 단어집 SSOT)."""
    metric = str(data.get("rank_metric") or data.get("metric_id") or "").strip()
    if metric in _METRICS:
        return metric  # type: ignore[return-value]
    return metric_for_query(query)  # type: ignore[return-value]


def _hop_step(data: dict[str, Any], query: str) -> HopStep | None:
    """체인 한 홉 JSON → HopStep. 관계/지표 미해소면 None(체인 전체 폐기)."""
    if not isinstance(data, dict):
        return None
    relations = _relation_steps(data.get("relation") or data)
    if not relations:
        return None
    relation = relations[0]
    metric_id = _metric_id(data, query)
    if not metric_id:
        return None
    try:
        top_n = int(data.get("top_n") or 3)
    except (TypeError, ValueError):
        top_n = 3
    top_n = max(1, min(top_n, 10))
    policy = str(data.get("policy") or data.get("first_candidate_policy") or "default")
    if policy not in _FIRST_POLICIES:
        policy = "default"
    return HopStep(
        relation=relation,
        rank=MetricRankStep(metric_id, alias="hop_" + relation.role),
        top_n=top_n,
        policy=policy,  # type: ignore[arg-type]
        relations=relations if len(relations) > 1 else (),
    )


def _coerce_chain_plan(data: dict[str, Any], query: str) -> StructuredPlan | None:
    """multi_hop_chain JSON → StructuredPlan. 홉이 2개 미만이거나 하나라도 무효면 None."""
    raw_hops = data.get("hops") or []
    if not isinstance(raw_hops, list) or not raw_hops:
        return None
    hops: list[HopStep] = []
    for raw_hop in raw_hops:
        hop = _hop_step(raw_hop, query) if isinstance(raw_hop, dict) else None
        if hop is None:
            return None
        hops.append(hop)
    if len(hops) < 2:
        return None  # 체인은 최소 2홉(수혜의 수혜) — 단일 홉은 일반 랭킹이 처리.
    return StructuredPlan(
        kind="multi_hop_chain",
        first_relation=hops[0].relation,
        first_rank=hops[0].rank,
        hops=hops,
        planner="llm",
        raw_reason=str(data.get("reason") or "LLM multi-hop chain"),
        steps=[
            {
                "op": "chain_hop",
                "relation": [r.rel_type for r in h.rel_steps()],
                "direction": [r.direction for r in h.rel_steps()],
                "metric": h.rank.metric_id,
                "top_n": h.top_n,
            }
            for h in hops
        ],
    )


def coerce_plan(data: dict[str, Any], query: str) -> StructuredPlan | None:
    """체인 LLM JSON 을 실행 가능한 multi_hop_chain StructuredPlan 으로 검증·변환.

    supported:false·kind 불일치·홉 무효는 None(호출자 폴백). 이 모듈은 체인만 다룬다.
    """
    if not data or data.get("supported") is False:
        return None
    if str(data.get("kind") or "").strip() != "multi_hop_chain":
        return None
    return _coerce_chain_plan(data, query)


def build_chain_plan(data: dict[str, Any], query: str) -> StructuredPlan | None:
    """통합 라우터용: hops 필드만으로 multi_hop_chain plan 빌드(kind 키 불요).

    router 의 통합 LLM 응답은 question_type='chain' + hops 형태라 coerce_plan 의 kind 검사
    대신 홉 검증만 적용한다. 홉이 없거나 무효면 None(라우터가 비-체인으로 폴백).
    """
    if not isinstance(data, dict):
        return None
    return _coerce_chain_plan(data, query)
