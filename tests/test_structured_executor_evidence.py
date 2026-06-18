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
