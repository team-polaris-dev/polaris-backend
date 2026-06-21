"""Deterministic planner — text2cypher 로 대체 불가한 두 kind만 emit한다.

전체 통폐합 후, 일반 관계/구조/랭킹은 공식 Text2CypherRetriever(+ MariaDB SQL 랭킹
후처리)가 담당한다. 이 결정적 플래너는 그 표준 경로로 **대체할 수 없는** 두 가지만 남긴다:

- community_member_rank: Leiden 군집 멤버십은 관계가 아니라 노드 속성이라 관계
  화이트리스트(9 DOMAIN_RELS)로 표현할 수 없다. "삼성 계열사 중 매출 1위" 류.
- multi_anchor_rank: 둘 이상 앵커의 공통 거래상대 *교집합*을 단일 지표로 1위만 뽑는다.
  교집합 정확성이 핵심이라 결정적으로 보장한다.

이 둘은 search 에서 text2cypher 보다 먼저 실행된다(search.PRESERVED_DETERMINISTIC_KINDS).
지원하지 않는 질문은 None 을 돌려 text2cypher / PPR 경로로 폴백한다. Cypher 는 만들지 않는다.
"""
from __future__ import annotations

import re

from config.relations import GROUP_SCOPE_TERMS, has_rank_intent
from graphrag.plan_schema import BranchRankStep, MetricRankStep, RelationStep, StructuredPlan


_SUPPLY_IN = (
    "공급", "납품", "협력사", "거래처", "거래하는", "벤더", "공급사", "공급업체", "조달", "매입",
    "납품받", "제품을 공급", "부품을 공급", "장비를 공급", "소재를 공급",
)
_SUPPLY_OUT = ("고객", "고객사", "매출처", "납품처", "구매처", "사가는", "판매", "판매하는", "공급처")
_COMMON_ANCHOR = ("둘 다", "모두", "공통", "양사", "둘다", "동시에", "동시", "양쪽")
_BRANCH_COMPARE = ("각각", "비교", "관계 타입", "관계타입", "근거")
_CUSTOMER_BRANCH = ("주요 매출처", "매출처", "고객", "납품처")
_RELATED_BRANCH = ("특수관계", "관련 회사", "관계자", "관계기업")
_INVESTMENT_BRANCH = ("투자 관계", "투자관계", "투자", "출자 관계", "출자관계", "출자")
_RELATION_TYPE_COMPARE = ("관계 유형", "관계유형", "관계 유형별", "관계유형별", "관계 타입", "관계타입")


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(t in text for t in terms)


def _metric_id(query: str) -> tuple[str, str] | None:
    """Return (metric_id, reason). Defaults '잘나가는' to revenue.

    node.py(has_metric)·executor 가 같이 쓰는 지표 해소기 — 보존한다.
    """
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
        if _has_any(query, _SUPPLY_OUT) and not re.search(r"협력사|벤더|공급사|공급업체|납품받|매입|조달", query):
            return (
                RelationStep("SUPPLIES_TO", "outgoing", "buyers", "buyer"),
                "고객/매출처 표현을 SUPPLIES_TO outgoing으로 해석",
            )
        if query.count("공급") and not re.search(r"협력사|벤더|공급사|공급업체|납품받|매입|조달|고객|고객사|매출처|납품처|판매", query):
            return (
                RelationStep("SUPPLIES_TO", "auto", "supply_counterparties", "supply_counterparty"),
                "공급 표현이 방향을 특정하지 않아 SUPPLIES_TO auto로 해석",
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


def _is_multi_anchor_rank_question(query: str) -> bool:
    """공통 거래상대를 단일 지표로 1위만 뽑는 질문(branch 비교 없음).

    "삼성전자와 SK하이닉스가 동시에 거래하는 소재 회사 중 매출 1위" 류 — 둘 이상 앵커가
    공통으로 가진 관계 상대를 metric 으로 줄세운다. branch 비교어(각각/비교/근거·관계 유형별)가
    있으면 통폐합으로 사라진 branch 비교 질문이므로 text2cypher 로 폴백시킨다(여기선 제외).
    """
    return (
        _has_any(query, _COMMON_ANCHOR)
        and not _has_any(query, _BRANCH_COMPARE)
        and not _has_any(query, _RELATION_TYPE_COMPARE)
    )


def _relation_slots(query: str, metric_id: str) -> list[BranchRankStep]:
    """질문에 나온 관계 유형 슬롯. community 게이트(관계어 유무)와 multi_anchor 의 first
    관계 폴백(첫 슬롯)에만 쓴다 — branch plan 자체는 더 이상 emit 하지 않는다."""
    branches: list[BranchRankStep] = []
    if _has_any(query, _SUPPLY_IN) or _has_any(query, _CUSTOMER_BRANCH):
        relation, _ = _first_relation(query) or (
            RelationStep("SUPPLIES_TO", "auto", "supply_counterparties", "supply_counterparty"),
            "",
        )
        kind = "major_customer" if relation.direction == "outgoing" else "supplier"
        if relation.direction == "outgoing":
            relation = RelationStep("SUPPLIES_TO", "outgoing", "major_customers", "major_customer")
        elif relation.direction == "incoming":
            relation = RelationStep("SUPPLIES_TO", "incoming", "suppliers", "supplier")
        else:
            relation = RelationStep("SUPPLIES_TO", "auto", "supply_counterparties", "supply_counterparty")
        branches.append(
            BranchRankStep(
                kind=kind,  # type: ignore[arg-type]
                relation=relation,
                rank=MetricRankStep(metric_id, alias="top_" + kind),  # type: ignore[arg-type]
            )
        )
    if _has_any(query, _RELATED_BRANCH):
        branches.append(
            BranchRankStep(
                kind="related_party",
                relation=RelationStep("RELATED_PARTY", "undirected", "related_parties", "related_party"),
                rank=MetricRankStep(metric_id, alias="top_related_party"),  # type: ignore[arg-type]
            )
        )
    if _has_any(query, _INVESTMENT_BRANCH):
        branches.append(
            BranchRankStep(
                kind="investment",
                relation=RelationStep("INVESTS_IN", "undirected", "investment_parties", "investment"),
                rank=MetricRankStep(metric_id, alias="top_investment"),  # type: ignore[arg-type]
            )
        )
    seen: set[str] = set()
    out: list[BranchRankStep] = []
    for branch in branches:
        if branch.kind in seen:
            continue
        seen.add(branch.kind)
        out.append(branch)
    return out


def _first_candidate_policy(step: RelationStep) -> str:
    if step.rel_type == "SUPPLIES_TO" and step.direction == "incoming":
        return "operating_counterparty"
    return "default"


def plan(query: str) -> StructuredPlan | None:
    """보존 결정적 plan(community_member_rank / multi_anchor_rank) 또는 None(폴백)."""
    q = " ".join((query or "").split())
    if not q:
        return None

    metric = _metric_id(q)
    if not metric or not has_rank_intent(q):
        return None

    metric_id, metric_reason = metric
    relation_slots = _relation_slots(q, metric_id)
    first = _first_relation(q)
    if not first and not relation_slots:
        # "삼성 계열사 중 매출 1위" — 한 회사의 이웃이 아니라 앵커가 속한 그룹 군집
        # 전체를 노드 지표로 줄세우는 질문. 구체적 관계어가 없고 그룹 범위어만 있을 때.
        if _has_any(q, GROUP_SCOPE_TERMS):
            return StructuredPlan(
                kind="community_member_rank",
                first_relation=None,
                first_rank=MetricRankStep(metric_id, alias="top_member"),  # type: ignore[arg-type]
                raw_reason="; ".join([
                    "그룹/계열 군집 멤버 노드 지표 랭킹 질문",
                    metric_reason,
                    "앵커가 속한 커뮤니티 멤버를 노드 지표로 줄세움",
                ]),
                steps=[
                    {"op": "community_members", "from": "anchor"},
                    {"op": "join_metric", "metric": metric_id, "target": "community_members"},
                    {"op": "argmax", "by": metric_id, "as": "top_member"},
                ],
            )
        return None

    first_step, first_reason = first or (relation_slots[0].relation, "관계 유형에서 첫 관계 도출")

    if _is_multi_anchor_rank_question(q):
        # 공통 앵커 교집합 → 단일 지표 랭킹 1위. 교집합 정확성을 결정적으로 보장한다.
        steps = [
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
        return StructuredPlan(
            kind="multi_anchor_rank",
            first_relation=first_step,
            first_rank=MetricRankStep(metric_id, alias="top_" + first_step.role),  # type: ignore[arg-type]
            common_anchor_min=2,
            first_candidate_policy=_first_candidate_policy(first_step),
            raw_reason="; ".join([
                "공통 앵커 교집합 거래상대 단일 지표 랭킹 질문",
                first_reason,
                metric_reason,
                "branch 비교 없이 공통 후보를 지표로 줄세워 1위만 답한다",
            ]),
            steps=steps,
        )

    # 단일/2-hop·branch 랭킹은 text2cypher + SQL 랭킹이 흡수했다 → 폴백.
    return None
