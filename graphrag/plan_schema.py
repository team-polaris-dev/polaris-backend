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

Direction = Literal["incoming", "outgoing", "undirected"]

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
class StructuredPlan:
    """A small executable plan for multi-hop ranking questions."""

    kind: Literal["single_hop_rank", "two_hop_rank", "multi_anchor_branch_rank"]
    first_relation: RelationStep
    first_rank: MetricRankStep
    second_relation: RelationStep | None = None
    second_rank: MetricRankStep | None = None
    branch_ranks: list[BranchRankStep] = field(default_factory=list)
    common_anchor_min: int = 1
    first_candidate_policy: FirstCandidatePolicy = "default"
    exclude_original_anchor_from_second: bool = False
    planner: Literal["deterministic", "llm"] = "deterministic"
    raw_reason: str = ""
    steps: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "first_relation": self.first_relation.__dict__,
            "first_rank": self.first_rank.__dict__,
            "second_relation": self.second_relation.__dict__ if self.second_relation else None,
            "second_rank": self.second_rank.__dict__ if self.second_rank else None,
            "branch_ranks": [
                {
                    "kind": b.kind,
                    "relation": b.relation.__dict__,
                    "rank": b.rank.__dict__,
                }
                for b in self.branch_ranks
            ],
            "common_anchor_min": self.common_anchor_min,
            "first_candidate_policy": self.first_candidate_policy,
            "exclude_original_anchor_from_second": self.exclude_original_anchor_from_second,
            "planner": self.planner,
            "raw_reason": self.raw_reason,
            "steps": self.steps,
        }
