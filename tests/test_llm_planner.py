from graphrag.llm_planner import coerce_plan


def test_llm_plan_json_is_coerced_to_structured_plan():
    data = {
        "supported": True,
        "kind": "two_hop_rank",
        "first_relation": {"rel_type": "SUPPLIES_TO", "direction": "incoming"},
        "rank_metric": "ifrs-full_Revenue",
        "second_relation": {"rel_type": "RELATED_PARTY", "direction": "undirected"},
        "exclude_original_anchor_from_second": True,
        "first_candidate_policy": "operating_counterparty",
        "reason": "거래 협력사를 매출 기준으로 고르고 관련 회사를 한 번 더 랭킹",
    }

    out = coerce_plan(data, "SK하이닉스와 거래하는 기업 중 가장 잘나가는 회사")

    assert out is not None
    assert out.planner == "llm"
    assert out.kind == "two_hop_rank"
    assert out.first_relation.rel_type == "SUPPLIES_TO"
    assert out.first_relation.direction == "incoming"
    assert out.first_candidate_policy == "operating_counterparty"
    assert out.second_relation is not None
    assert out.second_relation.rel_type == "RELATED_PARTY"
    assert out.exclude_original_anchor_from_second is True


def test_llm_plan_json_rejects_unknown_relation():
    data = {
        "supported": True,
        "kind": "single_hop_rank",
        "first_relation": {"rel_type": "FREEFORM_CYPHER", "direction": "incoming"},
        "rank_metric": "ifrs-full_Revenue",
    }

    assert coerce_plan(data, "질문") is None


def test_llm_plan_json_is_coerced_to_multi_anchor_branch_plan():
    data = {
        "supported": True,
        "kind": "multi_anchor_branch_rank",
        "first_relation": {"rel_type": "SUPPLIES_TO", "direction": "incoming"},
        "rank_metric": "ifrs-full_Revenue",
        "branch_relations": [
            {"kind": "major_customer", "relation": {"rel_type": "SUPPLIES_TO", "direction": "outgoing"}},
            {"kind": "related_party", "relation": {"rel_type": "RELATED_PARTY", "direction": "undirected"}},
            {"kind": "investment", "relation": {"rel_type": "INVESTS_IN", "direction": "undirected"}},
        ],
        "common_anchor_min": 2,
        "first_candidate_policy": "operating_counterparty",
        "reason": "공통 협력사 후 관계 타입별 branch 비교",
    }

    out = coerce_plan(data, "SK하이닉스와 삼성전자 둘 다와 연결된 협력사를 찾아 관계 타입을 비교해줘")

    assert out is not None
    assert out.kind == "multi_anchor_branch_rank"
    assert out.common_anchor_min == 2
    assert out.first_candidate_policy == "operating_counterparty"
    assert {b.kind for b in out.branch_ranks} == {"major_customer", "related_party", "investment"}


def test_llm_multi_anchor_plan_rejects_missing_branch():
    data = {
        "supported": True,
        "kind": "multi_anchor_branch_rank",
        "first_relation": {"rel_type": "SUPPLIES_TO", "direction": "incoming"},
        "rank_metric": "ifrs-full_Revenue",
        "branch_relations": [
            {"kind": "major_customer", "relation": {"rel_type": "SUPPLIES_TO", "direction": "outgoing"}},
        ],
    }

    assert coerce_plan(data, "공통 협력사 관계 타입 비교") is None
