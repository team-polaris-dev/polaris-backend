from graphrag import structured_executor
from graphrag.structured_executor import _score_edge_evidence


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
