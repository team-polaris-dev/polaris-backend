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


def test_llm_multi_anchor_plan_accepts_branch_subset():
    data = {
        "supported": True,
        "kind": "multi_anchor_branch_rank",
        "first_relation": {"rel_type": "SUPPLIES_TO", "direction": "incoming"},
        "rank_metric": "ifrs-full_Revenue",
        "branch_relations": [
            {"kind": "major_customer", "relation": {"rel_type": "SUPPLIES_TO", "direction": "outgoing"}},
        ],
    }

    out = coerce_plan(data, "공통 협력사 중 매출처만 비교")

    assert out is not None
    assert out.kind == "multi_anchor_branch_rank"
    assert {b.kind for b in out.branch_ranks} == {"major_customer"}


def test_llm_plan_json_is_coerced_to_single_anchor_branch_plan():
    data = {
        "supported": True,
        "kind": "single_anchor_branch_rank",
        "rank_metric": "ifrs-full_Revenue",
        "branch_relations": [
            {"kind": "supplier", "relation": {"rel_type": "SUPPLIES_TO", "direction": "incoming"}},
            {"kind": "related_party", "relation": {"rel_type": "RELATED_PARTY", "direction": "undirected"}},
            {"kind": "investment", "relation": {"rel_type": "INVESTS_IN", "direction": "undirected"}},
        ],
        "reason": "기준 회사에서 공급/특수관계/투자 branch를 각각 랭킹",
    }

    out = coerce_plan(
        data,
        "SK하이닉스와 관련된 공급 관계, 특수관계, 투자 관계를 각각 탐색해 관계 타입별 1위를 비교해줘",
    )

    assert out is not None
    assert out.kind == "single_anchor_branch_rank"
    assert out.common_anchor_min == 1
    assert out.first_relation.rel_type == "SUPPLIES_TO"
    assert out.first_relation.direction == "incoming"
    assert {b.kind for b in out.branch_ranks} == {"supplier", "related_party", "investment"}


def test_llm_single_anchor_plan_accepts_branch_subset():
    data = {
        "supported": True,
        "kind": "single_anchor_branch_rank",
        "rank_metric": "ifrs-full_Revenue",
        "branch_relations": [
            {"kind": "supplier", "relation": {"rel_type": "SUPPLIES_TO", "direction": "incoming"}},
            {"kind": "related_party", "relation": {"rel_type": "RELATED_PARTY", "direction": "undirected"}},
        ],
    }

    out = coerce_plan(data, "기준 회사의 공급 관계와 특수관계만 비교")

    assert out is not None
    assert out.kind == "single_anchor_branch_rank"
    assert {b.kind for b in out.branch_ranks} == {"supplier", "related_party"}


def test_llm_single_anchor_plan_accepts_auto_supply_direction():
    data = {
        "supported": True,
        "kind": "single_anchor_branch_rank",
        "rank_metric": "ifrs-full_Revenue",
        "branch_relations": [
            {"kind": "supplier", "relation": {"rel_type": "SUPPLIES_TO", "direction": "auto"}},
        ],
    }

    out = coerce_plan(data, "동진쎄미켐의 공급 관계 매출 1위를 찾아줘")

    assert out is not None
    assert out.branch_ranks[0].relation.direction == "auto"
