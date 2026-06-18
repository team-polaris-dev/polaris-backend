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
