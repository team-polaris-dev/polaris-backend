"""LLM-assisted planner for structured GraphRAG questions.

The LLM only chooses among the constrained logical-plan fields defined in
plan_schema.py. It never writes Cypher or SQL; invalid output is discarded and
the deterministic planner/fallback search path continues.
"""
from __future__ import annotations

import json
import os
from typing import Any

from graphrag.plan_schema import (
    BranchKind,
    BranchRankStep,
    Direction,
    FirstCandidatePolicy,
    MetricId,
    MetricRankStep,
    RelationStep,
    RelationType,
    StructuredPlan,
)


_FALSE = {"0", "false", "False", "no", "NO", "off", "OFF"}
_ENABLED = os.environ.get("GRAPHRAG_LLM_PLANNER", "1") not in _FALSE

_RELATION_TYPES = {
    "SUPPLIES_TO",
    "RELATED_PARTY",
    "IS_MAJOR_SHAREHOLDER_OF",
    "IS_SUBSIDIARY_OF",
    "INVESTS_IN",
}
_DIRECTIONS = {"incoming", "outgoing", "undirected"}
_METRICS = {
    "ifrs-full_Revenue",
    "dart_OperatingIncomeLoss",
    "ifrs-full_ProfitLoss",
    "ifrs-full_Assets",
}
_FIRST_POLICIES = {"default", "operating_counterparty"}
_BRANCH_KINDS = {"major_customer", "related_party", "investment"}

_RELATION_DEFAULTS: dict[tuple[str, str], tuple[str, str]] = {
    ("SUPPLIES_TO", "incoming"): ("suppliers", "supplier"),
    ("SUPPLIES_TO", "outgoing"): ("buyers", "buyer"),
    ("RELATED_PARTY", "undirected"): ("related_parties", "related_party"),
    ("IS_MAJOR_SHAREHOLDER_OF", "incoming"): ("shareholders", "shareholder"),
    ("IS_MAJOR_SHAREHOLDER_OF", "outgoing"): ("shareholdings", "shareholding"),
    ("IS_SUBSIDIARY_OF", "incoming"): ("subsidiaries", "subsidiary"),
    ("IS_SUBSIDIARY_OF", "outgoing"): ("parents", "parent"),
    ("INVESTS_IN", "outgoing"): ("investees", "investee"),
    ("INVESTS_IN", "incoming"): ("investors", "investor"),
    ("INVESTS_IN", "undirected"): ("investment_parties", "investment"),
}

_RANK_TERMS = ("가장", "최고", "1위", "상위", "제일", "많은", "높은", "잘나가")
_RELATION_TERMS = (
    "거래", "공급", "납품", "협력", "벤더", "매출처", "고객", "관련",
    "특수관계", "관계자", "계열", "주주", "지분", "자회사", "종속", "투자",
)


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(t in text for t in terms)


def _looks_structured(query: str) -> bool:
    q = " ".join((query or "").split())
    return bool(q and _has_any(q, _RANK_TERMS) and _has_any(q, _RELATION_TERMS))


def _as_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    content = getattr(raw, "content", None)
    if isinstance(content, str):
        return json.loads(content)
    raise ValueError("LLM planner response is not JSON")


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


def _metric_id(data: dict[str, Any], query: str) -> MetricId | None:
    metric = str(data.get("rank_metric") or data.get("metric_id") or "").strip()
    if metric in _METRICS:
        return metric  # type: ignore[return-value]
    if "영업이익" in query:
        return "dart_OperatingIncomeLoss"
    if "순이익" in query or "당기순이익" in query:
        return "ifrs-full_ProfitLoss"
    if "자산" in query or "규모" in query:
        return "ifrs-full_Assets"
    if "매출" in query or "수익" in query or "잘나가" in query:
        return "ifrs-full_Revenue"
    return None


def _make_steps(
    kind: str,
    first: RelationStep,
    metric_id: str,
    second: RelationStep | None,
    branches: list[BranchRankStep] | None = None,
) -> list[dict]:
    steps: list[dict] = [
        {
            "op": "intersect_anchors" if kind == "multi_anchor_branch_rank" else "traverse",
            "relation": first.rel_type,
            "direction": first.direction,
            "as": first.alias,
        },
        {"op": "join_metric", "metric": metric_id, "target": first.alias},
        {"op": "argmax", "by": metric_id, "as": "top_" + first.role},
    ]
    if kind == "two_hop_rank" and second is not None:
        steps.extend([
            {"op": "traverse", "from": "top_" + first.role, "relation": second.rel_type, "direction": second.direction, "as": second.alias},
            {"op": "join_metric", "metric": metric_id, "target": second.alias},
            {"op": "argmax", "by": metric_id, "as": "top_" + second.role},
        ])
    if kind == "multi_anchor_branch_rank":
        for branch in branches or []:
            steps.extend([
                {
                    "op": "branch_traverse",
                    "branch": branch.kind,
                    "from": "top_" + first.role,
                    "relation": branch.relation.rel_type,
                    "direction": branch.relation.direction,
                    "as": branch.relation.alias,
                },
                {"op": "join_metric", "metric": metric_id, "target": branch.relation.alias},
                {"op": "argmax", "by": metric_id, "as": branch.rank.alias},
                {"op": "score_evidence", "target": branch.rank.alias},
            ])
    return steps


def _branch_rank(data: dict[str, Any], metric_id: MetricId) -> BranchRankStep | None:
    kind = str(data.get("kind") or data.get("branch") or "").strip()
    if kind not in _BRANCH_KINDS:
        return None
    relation = _relation_step(data.get("relation") or data.get("branch_relation"))
    if relation is None:
        return None
    return BranchRankStep(
        kind=kind,  # type: ignore[arg-type]
        relation=relation,
        rank=MetricRankStep(metric_id, alias="top_" + kind),
    )


def coerce_plan(data: dict[str, Any], query: str) -> StructuredPlan | None:
    """Validate LLM JSON and convert it into an executable StructuredPlan."""
    if not data or data.get("supported") is False:
        return None

    kind = str(data.get("kind") or "").strip()
    if kind not in {"single_hop_rank", "two_hop_rank", "multi_anchor_branch_rank"}:
        return None

    metric_id = _metric_id(data, query)
    first = _relation_step(data.get("first_relation"))
    if not metric_id or first is None:
        return None

    second = _relation_step(data.get("second_relation")) if kind == "two_hop_rank" else None
    if kind == "two_hop_rank" and second is None:
        return None
    branches: list[BranchRankStep] = []
    if kind == "multi_anchor_branch_rank":
        raw_branches = data.get("branch_relations") or data.get("branches") or []
        if not isinstance(raw_branches, list):
            return None
        for raw_branch in raw_branches:
            if isinstance(raw_branch, dict):
                branch = _branch_rank(raw_branch, metric_id)
                if branch:
                    branches.append(branch)
        needed = {"major_customer", "related_party", "investment"}
        if {b.kind for b in branches} != needed:
            return None

    first_policy = str(data.get("first_candidate_policy") or "default")
    if first_policy not in _FIRST_POLICIES:
        first_policy = "default"

    exclude_anchor = data.get("exclude_original_anchor_from_second")
    if exclude_anchor is None:
        exclude_anchor = kind == "two_hop_rank"

    steps = _make_steps(kind, first, metric_id, second, branches)
    return StructuredPlan(
        kind=kind,  # type: ignore[arg-type]
        first_relation=first,
        first_rank=MetricRankStep(metric_id, alias="top_" + first.role),
        second_relation=second,
        second_rank=MetricRankStep(metric_id, alias="top_" + second.role) if second else None,
        branch_ranks=branches,
        common_anchor_min=int(data.get("common_anchor_min") or (2 if kind == "multi_anchor_branch_rank" else 1)),
        first_candidate_policy=first_policy,  # type: ignore[arg-type]
        exclude_original_anchor_from_second=bool(exclude_anchor),
        planner="llm",
        raw_reason=str(data.get("reason") or "LLM constrained planner"),
        steps=steps,
    )


_SYSTEM_PROMPT = """너는 GraphRAG 질의를 제한된 실행 계획으로 바꾸는 semantic parser다.
너는 Cypher, SQL, 자유 텍스트 계획을 절대 만들지 않는다. 아래 허용값 중에서만 고른다.

허용 relation:
- SUPPLIES_TO: 공급/납품 관계. incoming = 후보가 기준 회사에 공급, outgoing = 기준 회사가 후보에 공급.
- RELATED_PARTY: 특수관계자/관련 회사. direction은 보통 undirected.
- IS_MAJOR_SHAREHOLDER_OF: 주주/지분 관계.
- IS_SUBSIDIARY_OF: 자회사/종속회사 관계.
- INVESTS_IN: 투자 관계.

허용 rank_metric:
- ifrs-full_Revenue: 매출액, 수익, 잘나가는 회사의 기본 기준.
- dart_OperatingIncomeLoss: 영업이익.
- ifrs-full_ProfitLoss: 순이익/당기순이익.
- ifrs-full_Assets: 자산/규모.

first_candidate_policy:
- default: 질문이 그룹/특수관계/지배 관계까지 포함해도 되는 경우.
- operating_counterparty: 거래처, 협력사, 공급사처럼 운영상 거래 상대를 묻는 경우.

출력은 JSON 객체 하나만 한다."""

_USER_TEMPLATE = """질문을 아래 JSON 스키마로만 변환하라.

{{
  "supported": true 또는 false,
  "kind": "single_hop_rank" 또는 "two_hop_rank" 또는 "multi_anchor_branch_rank",
  "first_relation": {{
    "rel_type": "SUPPLIES_TO|RELATED_PARTY|IS_MAJOR_SHAREHOLDER_OF|IS_SUBSIDIARY_OF|INVESTS_IN",
    "direction": "incoming|outgoing|undirected"
  }},
  "rank_metric": "ifrs-full_Revenue|dart_OperatingIncomeLoss|ifrs-full_ProfitLoss|ifrs-full_Assets",
  "second_relation": {{
    "rel_type": "SUPPLIES_TO|RELATED_PARTY|IS_MAJOR_SHAREHOLDER_OF|IS_SUBSIDIARY_OF|INVESTS_IN",
    "direction": "incoming|outgoing|undirected"
  }} 또는 null,
  "branch_relations": [
    {{
      "kind": "major_customer|related_party|investment",
      "relation": {{
        "rel_type": "SUPPLIES_TO|RELATED_PARTY|INVESTS_IN",
        "direction": "incoming|outgoing|undirected"
      }}
    }}
  ],
  "common_anchor_min": 2,
  "exclude_original_anchor_from_second": true,
  "first_candidate_policy": "default|operating_counterparty",
  "reason": "짧은 한국어 근거"
}}

규칙:
- 랭킹/최댓값/1위 질문이 아니면 supported=false.
- 두 번째로 '그 회사/해당 기업의 관련 회사 중'처럼 이어지면 two_hop_rank.
- 'A와 B 둘 다/공통 협력사'를 먼저 찾고, 그 회사의 매출처/특수관계자/투자 관계를 각각 비교하라고 하면 multi_anchor_branch_rank.
- '잘나가는'은 별도 지표가 없으면 매출액(ifrs-full_Revenue)으로 둔다.
- 2-hop에서는 원래 기준 회사가 다시 답으로 돌아오지 않도록 exclude_original_anchor_from_second=true.
- multi_anchor_branch_rank의 first_relation은 보통 SUPPLIES_TO incoming이며, first_candidate_policy는 보통 default다. 공통 공급사 교집합 자체가 운영 거래 필터다.
- multi_anchor_branch_rank의 branch_relations는 major_customer=SUPPLIES_TO outgoing, related_party=RELATED_PARTY undirected, investment=INVESTS_IN undirected 세 개를 모두 포함한다.

질문: {query}"""


def plan(query: str) -> StructuredPlan | None:
    """Return an LLM-derived constrained plan, or None when unsupported/invalid."""
    if not _ENABLED or not _looks_structured(query):
        return None

    from config.llm import json_llm  # noqa: PLC0415
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    raw = json_llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_USER_TEMPLATE.format(query=query)),
    ])
    data = _as_dict(raw)
    return coerce_plan(data, query)
