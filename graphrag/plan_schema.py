"""Structured GraphRAG query plan schema.

This is a constrained logical form for questions that should be answered by
deterministic graph traversal + metric ranking, not by broad PPR retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


RelationType = Literal[
    "SUPPLIES_TO",
    "RELATED_PARTY",
    "IS_MAJOR_SHAREHOLDER_OF",
    "IS_SUBSIDIARY_OF",
    "INVESTS_IN",
]

Direction = Literal["incoming", "outgoing", "undirected", "auto"]

MetricId = Literal[
    "ifrs-full_Revenue",
    "dart_OperatingIncomeLoss",
    "ifrs-full_ProfitLoss",
    "ifrs-full_Assets",
]

FirstCandidatePolicy = Literal[
    "default",
    "operating_counterparty",
]

BranchKind = Literal[
    "supplier",
    "major_customer",
    "related_party",
    "investment",
]


@dataclass(frozen=True)
class RelationStep:
    rel_type: RelationType
    direction: Direction
    alias: str
    role: str = ""


@dataclass(frozen=True)
class MetricRankStep:
    metric_id: MetricId
    order: Literal["desc"] = "desc"
    alias: str = "top"


@dataclass(frozen=True)
class BranchRankStep:
    kind: BranchKind
    relation: RelationStep
    rank: MetricRankStep


@dataclass(frozen=True)
class HopStep:
    """다중홉 랭킹 체인(cutline)의 한 단계.

    relation 으로 현재 frontier 에서 한 홉 펼치고, rank 지표로 줄세워 상위 top_n 을
    답으로 채택한다. 채택된 후보가 다음 홉의 앵커(frontier)가 된다. policy 는 후보
    버킷팅(operating_counterparty 면 지배 허브 강등)을 고른다.
    """

    relation: RelationStep
    rank: MetricRankStep
    top_n: int = 3
    policy: FirstCandidatePolicy = "default"


@dataclass(frozen=True)
class StructuredPlan:
    """A small executable plan for multi-hop ranking questions."""

    kind: Literal[
        "multi_anchor_rank",
        "community_member_rank",
        "multi_hop_chain",
    ]
    first_relation: RelationStep | None
    # multi_anchor_rank·community_member_rank 는 채운다. multi_hop_chain 은 hops 로 대신해 None.
    first_rank: MetricRankStep | None = None
    # multi_hop_chain 전용: 단계별 홉(관계+지표+top_n)을 순서대로. 다른 kind 는 빈 리스트.
    hops: list[HopStep] = field(default_factory=list)
    common_anchor_min: int = 1
    first_candidate_policy: FirstCandidatePolicy = "default"
    planner: Literal["deterministic", "llm"] = "deterministic"
    raw_reason: str = ""
    steps: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "first_relation": self.first_relation.__dict__ if self.first_relation else None,
            "first_rank": self.first_rank.__dict__ if self.first_rank else None,
            "hops": [
                {
                    "relation": h.relation.__dict__,
                    "rank": h.rank.__dict__,
                    "top_n": h.top_n,
                    "policy": h.policy,
                }
                for h in self.hops
            ],
            "common_anchor_min": self.common_anchor_min,
            "first_candidate_policy": self.first_candidate_policy,
            "planner": self.planner,
            "raw_reason": self.raw_reason,
            "steps": self.steps,
        }
