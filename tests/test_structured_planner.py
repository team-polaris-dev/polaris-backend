"""결정적 planner.plan — 통폐합 후 보존 2종(community_member_rank/multi_anchor_rank)만 emit.

단일/2-hop 랭킹·2-hop 나열·branch 비교(단일/다중 앵커)는 공식 text2cypher(+ SQL 랭킹
후처리)가 흡수했으므로 planner 는 None 을 돌려 그 경로로 폴백한다. 여기선 (1) 보존 2종이
정확히 emit 되는지 (2) 폐기된 shape 들이 None 으로 폴백되는지만 검증한다.
"""
from graphrag.planner import plan


# ── 보존 kind 1: community_member_rank ───────────────────────────

def test_group_scope_revenue_question_becomes_community_member_rank():
    # "삼성 계열사 중 매출 1위" — 한 회사의 이웃이 아니라 앵커가 속한 군집(Leiden) 멤버를
    # 노드 지표로 줄세우는 질문. 관계어 없이 그룹 범위어만 있을 때 보존 결정 경로로 간다.
    out = plan("삼성 계열사 중 매출 1위는?")

    assert out is not None
    assert out.kind == "community_member_rank"
    assert out.first_relation is None
    assert out.first_rank.metric_id == "ifrs-full_Revenue"
    assert out.first_rank.alias == "top_member"
    assert out.steps[0]["op"] == "community_members"


# ── 보존 kind 2: multi_anchor_rank ───────────────────────────────

def test_common_trade_revenue_question_gets_multi_anchor_rank():
    # 공통 앵커 교집합 + 단일 지표 1위, branch 비교어 없음 → multi_anchor_rank(교집합 보장).
    q = "삼성전자와 SK하이닉스가 동시에 거래하는 소재 회사 중에서 매출액이 가장 높은 회사는?"

    out = plan(q)

    assert out is not None
    assert out.kind == "multi_anchor_rank"
    assert out.common_anchor_min == 2
    assert out.first_relation.rel_type == "SUPPLIES_TO"
    assert out.first_relation.direction == "incoming"
    assert out.first_rank.metric_id == "ifrs-full_Revenue"
    assert out.first_candidate_policy == "operating_counterparty"
    assert out.steps[0]["op"] == "intersect_anchors"
    assert out.steps[0]["min_anchors"] == 2


# ── 폐기된 shape → None 폴백(text2cypher + SQL 랭킹이 흡수) ─────────

def test_plain_local_graph_question_does_not_force_structured_plan():
    assert plan("SK하이닉스 주변 관계 보여줘") is None


def test_single_anchor_two_hop_rank_falls_back_to_none():
    # 단일 앵커 2-hop 랭킹(공급사 1위 → 그 회사 특수관계자 1위)은 text2cypher 가 흡수 → None.
    q = (
        "에스케이하이닉스에 제품을 공급하는 협력사 중 매출액이 가장 높은 기업과 "
        "해당 기업의 특수관계자 중 매출액이 가장 높은 기업은 어디입니까?"
    )
    assert plan(q) is None


def test_multi_anchor_branch_compare_falls_back_to_none():
    # 공통 앵커 + branch 비교어(각각/비교/근거/관계 타입)는 희귀 → 단일 text2cypher 관계답으로 degrade.
    q = (
        "SK하이닉스와 삼성전자 둘 다와 연결된 협력사를 찾고, 그중 매출액이 가장 높은 회사를 고른 다음, "
        "그 회사의 주요 매출처, 특수관계자, 투자 관계 중에서 매출액이 가장 큰 회사를 각각 하나씩 찾아서 "
        "어떤 관계 타입이 실제 근거에 가장 잘 맞는지 비교해줘."
    )
    assert plan(q) is None


def test_common_trade_branch_compare_falls_back_to_none():
    # 회귀 가드: 공통 앵커라도 branch 비교어가 있으면 multi_anchor_rank 가 가로채지 않고 None.
    q = (
        "삼성전자와 SK하이닉스 둘 다와 거래하는 협력사 중 매출 1위를 찾고, 그 회사의 "
        "주요 매출처·특수관계자·투자 관계 중 매출 1위를 각각 비교해줘."
    )
    assert plan(q) is None


def test_single_anchor_relation_type_compare_falls_back_to_none():
    q = (
        "SK하이닉스와 관련된 공급 관계, 특수관계, 투자 관계를 각각 따로 탐색해서 "
        "각 관계 타입별로 매출액이 가장 큰 회사를 하나씩 찾고, "
        "어떤 관계가 공시 근거상 가장 확실한지 비교해줘."
    )
    assert plan(q) is None
