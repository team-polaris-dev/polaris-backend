from graphrag.planner import plan


def test_supply_related_revenue_question_becomes_two_hop_plan():
    q = (
        "에스케이하이닉스에 제품을 공급하는 협력사 중 매출액이 가장 높은 기업과 "
        "해당 기업의 특수관계자 중 매출액이 가장 높은 기업은 어디입니까?"
    )

    out = plan(q)

    assert out is not None
    assert out.kind == "two_hop_rank"
    assert out.first_relation.rel_type == "SUPPLIES_TO"
    assert out.first_relation.direction == "incoming"
    assert out.first_rank.metric_id == "ifrs-full_Revenue"
    assert out.first_candidate_policy == "operating_counterparty"
    assert out.second_relation is not None
    assert out.second_relation.rel_type == "RELATED_PARTY"
    assert out.second_relation.direction == "undirected"
    assert out.exclude_original_anchor_from_second is True


def test_plain_local_graph_question_does_not_force_structured_plan():
    assert plan("SK하이닉스 주변 관계 보여줘") is None


def test_common_supplier_branch_compare_question_gets_multi_anchor_plan():
    q = (
        "SK하이닉스와 삼성전자 둘 다와 연결된 협력사를 찾고, 그중 매출액이 가장 높은 회사를 고른 다음, "
        "그 회사의 주요 매출처, 특수관계자, 투자 관계 중에서 매출액이 가장 큰 회사를 각각 하나씩 찾아서 "
        "어떤 관계 타입이 실제 근거에 가장 잘 맞는지 비교해줘."
    )

    out = plan(q)

    assert out is not None
    assert out.kind == "multi_anchor_branch_rank"
    assert out.common_anchor_min == 2
    assert out.first_relation.rel_type == "SUPPLIES_TO"
    assert out.first_relation.direction == "incoming"
    assert out.first_candidate_policy == "default"
    assert {b.kind for b in out.branch_ranks} == {"major_customer", "related_party", "investment"}
    by_kind = {b.kind: b for b in out.branch_ranks}
    assert by_kind["major_customer"].relation.rel_type == "SUPPLIES_TO"
    assert by_kind["major_customer"].relation.direction == "outgoing"
    assert by_kind["related_party"].relation.rel_type == "RELATED_PARTY"
    assert by_kind["investment"].relation.rel_type == "INVESTS_IN"
    assert by_kind["investment"].relation.direction == "undirected"


def test_common_trade_revenue_question_gets_multi_anchor_rank():
    # 공통 앵커 + 관계 + 랭킹, branch 비교어 없음 → 교집합 후 단일 지표 1위(multi_anchor_rank).
    q = "삼성전자와 SK하이닉스가 동시에 거래하는 소재 회사 중에서 매출액이 가장 높은 회사는?"

    out = plan(q)

    assert out is not None
    assert out.kind == "multi_anchor_rank"
    assert out.common_anchor_min == 2
    assert out.first_relation.rel_type == "SUPPLIES_TO"
    assert out.first_relation.direction == "incoming"
    assert out.first_rank.metric_id == "ifrs-full_Revenue"
    assert out.first_candidate_policy == "operating_counterparty"
    assert not out.branch_ranks
    assert out.steps[0]["op"] == "intersect_anchors"
    assert out.steps[0]["min_anchors"] == 2


def test_common_trade_branch_compare_stays_branch_rank():
    # 회귀 가드: branch 비교어가 있으면 새 kind 가 가로채지 않고 branch 플랜 유지.
    q = (
        "삼성전자와 SK하이닉스 둘 다와 거래하는 협력사 중 매출 1위를 찾고, 그 회사의 "
        "주요 매출처·특수관계자·투자 관계 중 매출 1위를 각각 비교해줘."
    )

    out = plan(q)

    assert out is not None
    assert out.kind == "multi_anchor_branch_rank"


def test_single_anchor_relation_type_compare_question_gets_branch_plan():
    q = (
        "SK하이닉스와 관련된 공급 관계, 특수관계, 투자 관계를 각각 따로 탐색해서 "
        "각 관계 타입별로 매출액이 가장 큰 회사를 하나씩 찾고, "
        "어떤 관계가 공시 근거상 가장 확실한지 비교해줘."
    )

    out = plan(q)

    assert out is not None
    assert out.kind == "single_anchor_branch_rank"
    assert out.common_anchor_min == 1
    assert out.first_candidate_policy == "default"
    assert {b.kind for b in out.branch_ranks} == {"supplier", "related_party", "investment"}
    by_kind = {b.kind: b for b in out.branch_ranks}
    assert by_kind["supplier"].relation.rel_type == "SUPPLIES_TO"
    assert by_kind["supplier"].relation.direction == "auto"
    assert by_kind["related_party"].relation.rel_type == "RELATED_PARTY"
    assert by_kind["related_party"].relation.direction == "undirected"
    assert by_kind["investment"].relation.rel_type == "INVESTS_IN"
    assert by_kind["investment"].relation.direction == "undirected"
    assert all(b.rank.metric_id == "ifrs-full_Revenue" for b in out.branch_ranks)


def test_single_anchor_two_branch_question_uses_only_requested_branches():
    q = (
        "동진쎄미켐을 기준으로 공급 관계와 특수관계 관계를 각각 탐색해서, "
        "각 관계 유형별 매출액이 가장 큰 회사를 찾고 실제 사업 관계로 보기 더 적합한 쪽을 설명해줘."
    )

    out = plan(q)

    assert out is not None
    assert out.kind == "single_anchor_branch_rank"
    assert {b.kind for b in out.branch_ranks} == {"supplier", "related_party"}
    by_kind = {b.kind: b for b in out.branch_ranks}
    assert by_kind["supplier"].relation.rel_type == "SUPPLIES_TO"
    assert by_kind["supplier"].relation.direction == "auto"
    assert by_kind["related_party"].relation.rel_type == "RELATED_PARTY"
    assert "investment" not in by_kind
