"""Deterministic planner for structured GraphRAG questions.

The planner intentionally emits only a tiny DSL. It does not generate Cypher.
Unsupported questions fall back to the existing Local/PPR search path.
"""
from __future__ import annotations

import re

from graphrag.plan_schema import BranchRankStep, MetricRankStep, RelationStep, StructuredPlan


_SUPPLY_IN = (
    "공급", "납품", "협력사", "거래처", "벤더", "제품을 공급", "부품을 공급",
    "장비를 공급", "소재를 공급",
)
_SUPPLY_OUT = ("고객", "매출처", "납품처", "구매처", "사가는", "판매하는")
_RELATED = ("특수관계", "관련된 회사", "관련 회사", "관계자", "계열거래", "관계기업")
_RANK = ("가장", "최고", "1위", "상위", "제일", "많은", "높은", "잘나가")
_SECOND_HOP = ("그 회사", "해당 기업", "해당 회사", "그회사", "관련된 회사중", "특수관계자 중")
_COMMON_ANCHOR = ("둘 다", "모두", "공통", "양사", "둘다")
_BRANCH_COMPARE = ("각각", "비교", "관계 타입", "관계타입", "근거")
_CUSTOMER_BRANCH = ("주요 매출처", "매출처", "고객", "납품처")
_RELATED_BRANCH = ("특수관계", "관련 회사", "관계자", "관계기업")
_INVESTMENT_BRANCH = ("투자 관계", "투자관계", "투자", "출자 관계", "출자관계", "출자")
_RELATION_TYPE_COMPARE = ("관계 유형", "관계유형", "관계 유형별", "관계유형별", "관계 타입", "관계타입")


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(t in text for t in terms)


def _metric_id(query: str) -> tuple[str, str] | None:
    """Return (metric_id, reason). Defaults '잘나가는' to revenue."""
    if "영업이익" in query:
        return "dart_OperatingIncomeLoss", "영업이익 기준 랭킹"
    if "순이익" in query or "당기순이익" in query:
        return "ifrs-full_ProfitLoss", "순이익 기준 랭킹"
    if "자산" in query or "규모" in query:
        return "ifrs-full_Assets", "자산 기준 랭킹"
    if "매출" in query or "수익" in query or "잘나가" in query:
        return "ifrs-full_Revenue", "매출액 기준 랭킹"
    return None


def _first_relation(query: str) -> tuple[RelationStep, str] | None:
    """Map relation phrases to a constrained traversal step."""
    if _has_any(query, _SUPPLY_IN):
        if _has_any(query, _SUPPLY_OUT) and not re.search(r"공급|협력사|벤더|납품", query):
            return (
                RelationStep("SUPPLIES_TO", "outgoing", "buyers", "buyer"),
                "고객/매출처 표현을 SUPPLIES_TO outgoing으로 해석",
            )
        return (
            RelationStep("SUPPLIES_TO", "incoming", "suppliers", "supplier"),
            "공급/협력사 표현을 SUPPLIES_TO incoming으로 해석",
        )
    if "투자" in query:
        return (
            RelationStep("INVESTS_IN", "outgoing", "investees", "investee"),
            "투자 표현을 INVESTS_IN outgoing으로 해석",
        )
    if "자회사" in query or "종속" in query:
        return (
            RelationStep("IS_SUBSIDIARY_OF", "incoming", "subsidiaries", "subsidiary"),
            "자회사 표현을 IS_SUBSIDIARY_OF incoming으로 해석",
        )
    if "주주" in query or "지분" in query or "대주주" in query:
        return (
            RelationStep("IS_MAJOR_SHAREHOLDER_OF", "incoming", "shareholders", "shareholder"),
            "주주/지분 표현을 IS_MAJOR_SHAREHOLDER_OF incoming으로 해석",
        )
    return None


def _second_relation(query: str) -> tuple[RelationStep, str] | None:
    if _has_any(query, _RELATED):
        return (
            RelationStep("RELATED_PARTY", "undirected", "related_parties", "related_party"),
            "관련/특수관계 표현을 RELATED_PARTY undirected로 해석",
        )
    return None


def _include_original_anchor_on_second(query: str) -> bool:
    return any(t in query for t in ("본인 포함", "자기 자신 포함", "원래 회사 포함"))


def _is_multi_anchor_branch_question(query: str) -> bool:
    """Detect common-counterparty questions that ask several relation branches."""
    return (
        _has_any(query, _COMMON_ANCHOR)
        and _has_any(query, _SUPPLY_IN)
        and _has_any(query, _CUSTOMER_BRANCH)
        and _has_any(query, _RELATED)
        and _has_any(query, _INVESTMENT_BRANCH)
        and _has_any(query, _BRANCH_COMPARE)
    )


def _is_single_anchor_branch_question(query: str) -> bool:
    """Detect one-anchor questions that ask separate relation-type branches."""
    return (
        not _has_any(query, _COMMON_ANCHOR)
        and _has_any(query, _SUPPLY_IN)
        and _has_any(query, _RELATED_BRANCH)
        and _has_any(query, _INVESTMENT_BRANCH)
        and (_has_any(query, _BRANCH_COMPARE) or _has_any(query, _RELATION_TYPE_COMPARE))
    )


def _branch_ranks(metric_id: str) -> list[BranchRankStep]:
    return [
        BranchRankStep(
            kind="major_customer",
            relation=RelationStep("SUPPLIES_TO", "outgoing", "major_customers", "major_customer"),
            rank=MetricRankStep(metric_id, alias="top_major_customer"),  # type: ignore[arg-type]
        ),
        BranchRankStep(
            kind="related_party",
            relation=RelationStep("RELATED_PARTY", "undirected", "related_parties", "related_party"),
            rank=MetricRankStep(metric_id, alias="top_related_party"),  # type: ignore[arg-type]
        ),
        BranchRankStep(
            kind="investment",
            relation=RelationStep("INVESTS_IN", "undirected", "investment_parties", "investment"),
            rank=MetricRankStep(metric_id, alias="top_investment"),  # type: ignore[arg-type]
        ),
    ]


def _single_anchor_branch_ranks(metric_id: str) -> list[BranchRankStep]:
    return [
        BranchRankStep(
            kind="supplier",
            relation=RelationStep("SUPPLIES_TO", "incoming", "suppliers", "supplier"),
            rank=MetricRankStep(metric_id, alias="top_supplier"),  # type: ignore[arg-type]
        ),
        BranchRankStep(
            kind="related_party",
            relation=RelationStep("RELATED_PARTY", "undirected", "related_parties", "related_party"),
            rank=MetricRankStep(metric_id, alias="top_related_party"),  # type: ignore[arg-type]
        ),
        BranchRankStep(
            kind="investment",
            relation=RelationStep("INVESTS_IN", "undirected", "investment_parties", "investment"),
            rank=MetricRankStep(metric_id, alias="top_investment"),  # type: ignore[arg-type]
        ),
    ]


def _first_candidate_policy(step: RelationStep) -> str:
    if step.rel_type == "SUPPLIES_TO" and step.direction == "incoming":
        return "operating_counterparty"
    return "default"


def plan(query: str) -> StructuredPlan | None:
    """Return a supported structured plan or None for local/PPR fallback."""
    q = " ".join((query or "").split())
    if not q:
        return None

    metric = _metric_id(q)
    first = _first_relation(q)
    if not metric or not first or not _has_any(q, _RANK):
        return None

    metric_id, metric_reason = metric
    first_step, first_reason = first

    if _is_multi_anchor_branch_question(q):
        branch_ranks = _branch_ranks(metric_id)
        steps: list[dict] = [
            {
                "op": "intersect_anchors",
                "relation": first_step.rel_type,
                "direction": first_step.direction,
                "min_anchors": 2,
                "as": first_step.alias,
            },
            {"op": "join_metric", "metric": metric_id, "target": first_step.alias},
            {"op": "argmax", "by": metric_id, "as": "top_" + first_step.role},
        ]
        for branch in branch_ranks:
            steps.extend([
                {
                    "op": "branch_traverse",
                    "branch": branch.kind,
                    "from": "top_" + first_step.role,
                    "relation": branch.relation.rel_type,
                    "direction": branch.relation.direction,
                    "as": branch.relation.alias,
                },
                {"op": "join_metric", "metric": metric_id, "target": branch.relation.alias},
                {"op": "argmax", "by": metric_id, "as": branch.rank.alias},
                {"op": "score_evidence", "target": branch.rank.alias},
            ])
        return StructuredPlan(
            kind="multi_anchor_branch_rank",
            first_relation=first_step,
            first_rank=MetricRankStep(metric_id, alias="top_" + first_step.role),  # type: ignore[arg-type]
            branch_ranks=branch_ranks,
            common_anchor_min=2,
            first_candidate_policy="default",
            raw_reason="; ".join([
                "공통 앵커 교집합 협력사 질문",
                first_reason,
                metric_reason,
                "매출처/특수관계자/투자 관계를 별도 branch로 비교",
            ]),
            steps=steps,
        )

    if _is_single_anchor_branch_question(q):
        branch_ranks = _single_anchor_branch_ranks(metric_id)
        steps: list[dict] = []
        for branch in branch_ranks:
            steps.extend([
                {
                    "op": "branch_traverse",
                    "branch": branch.kind,
                    "from": "anchor",
                    "relation": branch.relation.rel_type,
                    "direction": branch.relation.direction,
                    "as": branch.relation.alias,
                },
                {"op": "join_metric", "metric": metric_id, "target": branch.relation.alias},
                {"op": "argmax", "by": metric_id, "as": branch.rank.alias},
                {"op": "score_evidence", "target": branch.rank.alias},
            ])
        return StructuredPlan(
            kind="single_anchor_branch_rank",
            first_relation=branch_ranks[0].relation,
            first_rank=branch_ranks[0].rank,
            branch_ranks=branch_ranks,
            common_anchor_min=1,
            first_candidate_policy="default",
            raw_reason="; ".join([
                "단일 기준 기업의 관계 유형별 branch ranking 질문",
                first_reason,
                metric_reason,
                "공급/특수관계/투자 관계를 기준 기업에서 각각 독립 탐색",
            ]),
            steps=steps,
        )

    second = _second_relation(q)
    has_second = second is not None and _has_any(q, _SECOND_HOP)

    steps: list[dict] = [
        {"op": "traverse", "relation": first_step.rel_type, "direction": first_step.direction, "as": first_step.alias},
        {"op": "join_metric", "metric": metric_id, "target": first_step.alias},
        {"op": "argmax", "by": metric_id, "as": "top_" + first_step.role},
    ]
    raw_reasons = [first_reason, metric_reason]

    if has_second and second is not None:
        second_step, second_reason = second
        steps.extend([
            {"op": "traverse", "from": "top_" + first_step.role, "relation": second_step.rel_type, "direction": second_step.direction, "as": second_step.alias},
            {"op": "join_metric", "metric": metric_id, "target": second_step.alias},
            {"op": "argmax", "by": metric_id, "as": "top_" + second_step.role},
        ])
        raw_reasons.append(second_reason)
        return StructuredPlan(
            kind="two_hop_rank",
            first_relation=first_step,
            first_rank=MetricRankStep(metric_id, alias="top_" + first_step.role),  # type: ignore[arg-type]
            second_relation=second_step,
            second_rank=MetricRankStep(metric_id, alias="top_" + second_step.role),  # type: ignore[arg-type]
            first_candidate_policy=_first_candidate_policy(first_step),
            exclude_original_anchor_from_second=not _include_original_anchor_on_second(q),
            raw_reason="; ".join(raw_reasons),
            steps=steps,
        )

    return StructuredPlan(
        kind="single_hop_rank",
        first_relation=first_step,
        first_rank=MetricRankStep(metric_id, alias="top_" + first_step.role),  # type: ignore[arg-type]
        first_candidate_policy=_first_candidate_policy(first_step),
        raw_reason="; ".join(raw_reasons),
        steps=steps,
    )
