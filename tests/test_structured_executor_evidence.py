from graphrag import structured_executor
from graphrag.plan_schema import RelationStep
from graphrag.structured_executor import (
    _confirmed_render_candidates,
    _edge_matches_relation,
    _score_edge_evidence,
    _select_supported,
    _supply_direction_from_counts,
)


def test_supply_edge_with_endpoint_names_and_relation_term_is_high_confidence():
    edge = {
        "rel_type": "SUPPLIES_TO",
        "from_name": "동진쎄미켐",
        "to_name": "삼성전자(주)",
        "source": "20241114001783",
        "chunk_id": "c1",
    }
    chunks = {"c1": "반도체 소재의 주요 매출처는 삼성전자와 하이닉스 2개 회사이며 동진쎄미켐이 공급합니다."}

    evidence = _score_edge_evidence(edge, chunks)

    assert evidence["level"] == "high"
    assert evidence["confidence"] >= 0.8
    assert evidence["to_mentioned"] is True
    assert evidence["relation_term_found"] is True


def test_related_party_edge_without_endpoint_support_is_marked_weak():
    edge = {
        "rel_type": "RELATED_PARTY",
        "from_name": "동진쎄미켐",
        "to_name": "삼성전자(주)",
        "source": "20240314001382",
        "chunk_id": "c2",
    }
    chunks = {"c2": "사천동진쎄미켐과기유한공사와 Dongjin Sweden AB에 대한 지급보증 내역입니다."}

    evidence = _score_edge_evidence(edge, chunks)

    assert evidence["level"] == "medium"
    assert evidence["confidence"] < 0.8
    assert "chunk_does_not_name_both_endpoints" in evidence["warnings"]
    assert "weak_evidence_for_accounting_or_investment_relation" in evidence["warnings"]


def test_investment_edge_without_chunk_is_low_confidence():
    edge = {
        "rel_type": "INVESTS_IN",
        "from_name": "삼성전자(주)",
        "to_name": "동진쎄미켐",
        "source": "20260515001804",
        "chunk_id": "",
    }

    evidence = _score_edge_evidence(edge, {})

    assert evidence["level"] == "low"
    assert "document_source_without_chunk" in evidence["warnings"]
    assert "weak_evidence_for_accounting_or_investment_relation" in evidence["warnings"]


def test_operating_counterparty_policy_demotes_governance_hub(monkeypatch):
    candidates = [
        {
            "id": "sk",
            "corp_code": "001",
            "name": "SK(주)",
            "anchor_rels": ["RELATED_PARTY", "INVESTS_IN"],
            "graph_degree": 120,
            "edge": {},
            "evidence": {"confidence": 0.95, "level": "high", "relation_term_found": True},
        },
        {
            "id": "supplier",
            "corp_code": "002",
            "name": "동진쎄미켐",
            "anchor_rels": [],
            "graph_degree": 4,
            "edge": {},
            "evidence": {"confidence": 0.55, "level": "medium", "relation_term_found": True},
        },
    ]

    def fake_metric_values(corp_codes, account_id="ifrs-full_Revenue", year=None):
        return [
            {"corp_code": "001", "account_id": account_id, "value": "122000000000000", "bsns_year": 2025},
            {"corp_code": "002", "account_id": account_id, "value": "1200000000000", "bsns_year": 2025},
        ]

    monkeypatch.setattr(structured_executor, "_fetch_metric_values", fake_metric_values)

    default_ranked = structured_executor._rank_candidates(candidates, "ifrs-full_Revenue", 2025)
    operating_ranked = structured_executor._rank_candidates(
        candidates,
        "ifrs-full_Revenue",
        2025,
        policy="operating_counterparty",
    )

    assert default_ranked[0]["name"] == "SK(주)"
    assert operating_ranked[0]["name"] == "동진쎄미켐"
    assert "governance_linked" in operating_ranked[1]["policy"]["reasons"]
    assert "pure_operating_counterparty" in operating_ranked[0]["policy"]["reasons"]


def _operating_candidates():
    # 솔브레인: 룰이 related_party 로 강등하지만 매출은 더 큰 '실제 공급사'(LLM-as-judge 가
    # '과강등'으로 본 사례). 에스에프에이: 강등 없는 저매출 후보.
    return [
        {
            "id": "solbrain",
            "corp_code": "001",
            "name": "솔브레인 주식회사",
            "anchor_rels": ["RELATED_PARTY"],
            "graph_degree": 10,
            "edge": {},
            "evidence": {"confidence": 0.55, "level": "medium", "relation_term_found": True},
        },
        {
            "id": "sfa",
            "corp_code": "002",
            "name": "(주)에스에프에이반도체",
            "anchor_rels": [],
            "graph_degree": 4,
            "edge": {},
            "evidence": {"confidence": 0.55, "level": "medium", "relation_term_found": True},
        },
    ]


def _operating_metric_values(corp_codes, account_id="ifrs-full_Revenue", year=None):
    return [
        {"corp_code": "001", "account_id": account_id, "value": "923381966875", "bsns_year": 2025},
        {"corp_code": "002", "account_id": account_id, "value": "367389147993", "bsns_year": 2025},
    ]


def test_operating_counterparty_llm_override_lifts_real_supplier(monkeypatch):
    candidates = _operating_candidates()
    monkeypatch.setattr(structured_executor, "_fetch_metric_values", _operating_metric_values)
    monkeypatch.setattr(structured_executor, "_LLM_RANK_POLICY", True)
    monkeypatch.setattr(
        structured_executor,
        "_invoke_rank_llm",
        lambda anchors, relation_label, question, contested: {"솔브레인 주식회사": "operating"},
    )

    # 룰만으로는 솔브레인이 강등돼 매출 낮은 에스에프에이가 1위.
    rule_ranked = structured_executor._rank_candidates(
        candidates, "ifrs-full_Revenue", 2025, policy="operating_counterparty",
    )
    assert rule_ranked[0]["name"] == "(주)에스에프에이반도체"

    # 앵커 문맥이 주어지면 LLM 이 '실제 공급사'로 판정 → 강등 해제 → 매출 1위 복귀.
    llm_ranked = structured_executor._rank_candidates(
        candidates,
        "ifrs-full_Revenue",
        2025,
        policy="operating_counterparty",
        anchors=["삼성전자(주)", "삼성SDI(주)"],
        relation_label="SUPPLIES_TO",
        question="삼성전자와 삼성SDI가 둘 다 거래하는 공급사 중 매출이 가장 큰 곳은?",
    )
    assert llm_ranked[0]["name"] == "솔브레인 주식회사"
    assert "llm_operating" in llm_ranked[0]["policy"]["reasons"]


def test_operating_counterparty_llm_confirms_hub_keeps_demotion(monkeypatch):
    candidates = _operating_candidates()
    monkeypatch.setattr(structured_executor, "_fetch_metric_values", _operating_metric_values)
    monkeypatch.setattr(structured_executor, "_LLM_RANK_POLICY", True)
    monkeypatch.setattr(
        structured_executor,
        "_invoke_rank_llm",
        lambda anchors, relation_label, question, contested: {"솔브레인 주식회사": "hub"},
    )

    ranked = structured_executor._rank_candidates(
        candidates,
        "ifrs-full_Revenue",
        2025,
        policy="operating_counterparty",
        anchors=["삼성전자(주)", "삼성SDI(주)"],
        relation_label="SUPPLIES_TO",
    )
    assert ranked[0]["name"] == "(주)에스에프에이반도체"
    demoted = next(c for c in ranked if c["name"] == "솔브레인 주식회사")
    assert "llm_hub" in demoted["policy"]["reasons"]
    assert demoted["policy"]["bucket"] < 0


def test_operating_counterparty_llm_off_falls_back_to_rule(monkeypatch):
    candidates = _operating_candidates()
    monkeypatch.setattr(structured_executor, "_fetch_metric_values", _operating_metric_values)
    monkeypatch.setattr(structured_executor, "_LLM_RANK_POLICY", False)

    def _boom(*args, **kwargs):
        raise AssertionError("LLM must not be called when GRAPHRAG_LLM_RANK_POLICY is off")

    monkeypatch.setattr(structured_executor, "_invoke_rank_llm", _boom)

    ranked = structured_executor._rank_candidates(
        candidates,
        "ifrs-full_Revenue",
        2025,
        policy="operating_counterparty",
        anchors=["삼성전자(주)", "삼성SDI(주)"],
        relation_label="SUPPLIES_TO",
    )
    assert ranked[0]["name"] == "(주)에스에프에이반도체"


def test_select_supported_prefers_stronger_evidence_before_metric(monkeypatch):
    candidates = [
        {
            "id": "weak_big",
            "corp_code": "001",
            "name": "삼성전자",
            "edge": {},
            "evidence": {"confidence": 0.55, "level": "medium", "relation_term_found": False},
        },
        {
            "id": "strong_smaller",
            "corp_code": "002",
            "name": "동진홀딩스",
            "edge": {},
            "evidence": {"confidence": 0.95, "level": "high", "relation_term_found": True},
        },
    ]

    def fake_metric_values(corp_codes, account_id="ifrs-full_Revenue", year=None):
        return [
            {"corp_code": "001", "account_id": account_id, "value": "333000000000000", "bsns_year": 2025},
            {"corp_code": "002", "account_id": account_id, "value": "1000000000", "bsns_year": 2025},
        ]

    monkeypatch.setattr(structured_executor, "_fetch_metric_values", fake_metric_values)
    ranked = structured_executor._rank_candidates(candidates, "ifrs-full_Revenue", 2025)
    selected, unranked = _select_supported(ranked, "RELATED_PARTY")

    assert selected is not None
    assert selected["name"] == "동진홀딩스"
    assert unranked == []


def test_select_supported_requires_relation_term_for_related_party():
    ranked = [
        {
            "id": "co_mentioned_big",
            "name": "SK하이닉스",
            "metric": {"value": 333000000000000.0},
            "evidence": {"confidence": 0.8, "level": "high", "relation_term_found": False},
        },
        {
            "id": "attested_smaller",
            "name": "동진홀딩스",
            "metric": {"value": 1000000000.0},
            "evidence": {"confidence": 0.95, "level": "high", "relation_term_found": True},
        },
    ]

    selected, unranked = _select_supported(ranked, "RELATED_PARTY")

    assert selected is not None
    assert selected["name"] == "동진홀딩스"
    assert unranked == []


def test_select_supported_accepts_source_only_investment():
    # 출자(INVESTS_IN)는 표 출처 관계 → 본문 청크 없이 출처만(conf 0.35) 있어도 게이트 통과,
    # 매출 랭킹으로 1위 선정. (DART 출자현황 표는 서술형 청크가 구조적으로 없음.)
    ranked = [
        {
            "id": "big_source_only",
            "name": "삼성SDI",
            "metric": {"value": 13000000000000.0},
            "evidence": {"confidence": 0.35, "level": "low", "relation_term_found": False},
        },
        {
            "id": "smaller_source_only",
            "name": "동진쎄미켐",
            "metric": {"value": 1000000000000.0},
            "evidence": {"confidence": 0.35, "level": "low", "relation_term_found": False},
        },
    ]

    selected, unranked = _select_supported(ranked, "INVESTS_IN")

    assert selected is not None
    assert selected["name"] == "삼성SDI"
    assert unranked == []


def test_select_supported_rejects_sourceless_investment():
    # 출처조차 없으면(conf 0.1) 표 출처 관계라도 탈락 — 출처 보유가 최소 근거선.
    ranked = [
        {
            "id": "no_source",
            "name": "유령투자처",
            "metric": {"value": 999.0},
            "evidence": {"confidence": 0.1, "level": "low", "relation_term_found": False},
        },
    ]

    selected, unranked = _select_supported(ranked, "INVESTS_IN")

    assert selected is None
    assert unranked == []


def test_select_supported_keeps_numeric_floor_for_supply_relation():
    ranked = [
        {
            "id": "co_mentioned_customer",
            "name": "SK하이닉스",
            "metric": {"value": 333000000000000.0},
            "evidence": {"confidence": 0.8, "level": "high", "relation_term_found": False},
        },
    ]

    selected, unranked = _select_supported(ranked, "SUPPLIES_TO")

    assert selected is not None
    assert selected["name"] == "SK하이닉스"
    assert unranked == []


def test_select_supported_accepts_source_only_shareholder_relation():
    # 대주주(IS_MAJOR_SHAREHOLDER_OF)도 표 출처 관계 → 출처만(conf 0.35)으로 게이트 통과.
    # (대주주현황 표 역시 서술형 청크가 없어 conf 가 0.35 에 묶인다.)
    ranked = [
        {
            "id": "shareholder",
            "name": "삼성에스디에스",
            "metric": {"value": 13000000000000.0},
            "evidence": {"confidence": 0.35, "level": "low", "relation_term_found": False},
        },
    ]

    selected, unranked = _select_supported(ranked, "IS_MAJOR_SHAREHOLDER_OF")

    assert selected is not None
    assert selected["name"] == "삼성에스디에스"
    assert unranked == []


def test_select_supported_keeps_metricless_candidate_out_of_rank(monkeypatch):
    candidates = [
        {
            "id": "confirmed",
            "corp_code": "",
            "name": "해외관계사",
            "edge": {},
            "evidence": {"confidence": 0.95, "level": "high", "relation_term_found": True},
        },
    ]

    monkeypatch.setattr(structured_executor, "_fetch_metric_values", lambda *args, **kwargs: [])
    ranked = structured_executor._rank_candidates(candidates, "ifrs-full_Revenue", 2025)
    selected, unranked = _select_supported(ranked, "RELATED_PARTY")

    assert selected is None
    assert [c["name"] for c in unranked] == ["해외관계사"]


def test_supply_direction_uses_edge_count_majority():
    assert _supply_direction_from_counts(incoming_count=5, outgoing_count=2) == "incoming"
    assert _supply_direction_from_counts(incoming_count=1, outgoing_count=4) == "outgoing"
    assert _supply_direction_from_counts(incoming_count=3, outgoing_count=3) == "incoming"


def test_edge_matches_relation_direction():
    edge = {"rel_type": "SUPPLIES_TO", "from_id": "a", "to_id": "b"}

    assert _edge_matches_relation(edge, RelationStep("SUPPLIES_TO", "incoming", "suppliers"), "b")
    assert not _edge_matches_relation(edge, RelationStep("SUPPLIES_TO", "incoming", "suppliers"), "a")
    assert _edge_matches_relation(edge, RelationStep("SUPPLIES_TO", "outgoing", "buyers"), "a")
    assert not _edge_matches_relation(edge, RelationStep("SUPPLIES_TO", "outgoing", "buyers"), "b")


def test_confirmed_render_candidates_returns_matching_in_order():
    relation = RelationStep("RELATED_PARTY", "undirected", "related")
    confirmed = [
        {"id": "a", "name": "동진첨단소재", "edge": {"rel_type": "RELATED_PARTY", "from_id": "anchor", "to_id": "a"}},
        {"id": "b", "name": "이브이에스텍", "edge": {"rel_type": "RELATED_PARTY", "from_id": "anchor", "to_id": "b"}},
    ]

    out = _confirmed_render_candidates(confirmed, relation, cap=8, anchor_id="anchor")

    assert [c["id"] for c in out] == ["a", "b"]


def test_confirmed_render_candidates_respects_cap_in_order():
    relation = RelationStep("RELATED_PARTY", "undirected", "related")
    confirmed = [
        {"id": f"c{i}", "name": f"n{i}", "edge": {"rel_type": "RELATED_PARTY", "from_id": "anchor", "to_id": f"c{i}"}}
        for i in range(20)
    ]

    out = _confirmed_render_candidates(confirmed, relation, cap=8, anchor_id="anchor")

    assert [c["id"] for c in out] == [f"c{i}" for i in range(8)]


def test_confirmed_render_candidates_drops_relation_mismatch():
    relation = RelationStep("SUPPLIES_TO", "incoming", "suppliers")
    confirmed = [
        {"id": "ok", "name": "협력사", "edge": {"rel_type": "SUPPLIES_TO", "from_id": "ok", "to_id": "anchor"}},
        {"id": "wrong_type", "name": "투자사", "edge": {"rel_type": "INVESTS_IN", "from_id": "wrong_type", "to_id": "anchor"}},
        {"id": "wrong_dir", "name": "고객사", "edge": {"rel_type": "SUPPLIES_TO", "from_id": "anchor", "to_id": "wrong_dir"}},
    ]

    out = _confirmed_render_candidates(confirmed, relation, cap=8, anchor_id="anchor")

    assert [c["id"] for c in out] == ["ok"]


def test_confirmed_render_candidates_empty_when_none():
    relation = RelationStep("RELATED_PARTY", "undirected", "related")

    assert _confirmed_render_candidates([], relation, cap=8, anchor_id="anchor") == []
