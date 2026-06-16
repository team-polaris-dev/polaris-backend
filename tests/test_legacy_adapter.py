"""adapt_to_legacy 어댑터 단위 테스트.

GraphHit → 기존 graph_facts / graph_paths / graph_provenance 변환 정확성을 검증.
이슈 #17 result_check/gen이 graph_facts를 직접 봐서 셰임 출력이 정확해야 함.
"""
from __future__ import annotations

from graphrag.schema import adapt_to_legacy


def test_executive_hit_to_fact() -> None:
    """임원 hit → legacy fact type='executive', value=pos."""
    hits = [{
        "id": "rel:EXECUTIVE_OF:p_abc:00126380",
        "label": "relationship",
        "name": "노태문 → 삼성전자(주)",
        "attrs": {
            "rel_type": "EXECUTIVE_OF",
            "from_id": "p_abc",
            "from_name": "노태문",
            "to_id": "00126380",
            "to_name": "삼성전자(주)",
            "pos": "대표이사",
        },
        "score": 0.8,
        "source": "20260317000635",
    }]
    legacy = adapt_to_legacy(hits)

    assert len(legacy["facts"]) == 1
    f = legacy["facts"][0]
    assert f["type"] == "executive"
    assert f["name"] == "노태문 → 삼성전자(주)"
    assert f["value"] == "대표이사"
    assert f["source"] == "20260317000635"

    assert len(legacy["paths"]) == 1
    assert legacy["paths"][0] == ["노태문", "EXECUTIVE_OF", "삼성전자(주)"]

    assert legacy["provenance"] == ["20260317000635"]


def test_shareholder_qota_rt_carried_in_value() -> None:
    """주주 hit → value에 qota_rt 들어가야 함."""
    hits = [{
        "id": "rel:IS_MAJOR_SHAREHOLDER_OF:00126256:00126380",
        "label": "relationship",
        "name": "삼성생명 → 삼성전자(주)",
        "attrs": {
            "rel_type": "IS_MAJOR_SHAREHOLDER_OF",
            "from_id": "00126256",
            "from_name": "삼성생명",
            "to_id": "00126380",
            "to_name": "삼성전자(주)",
            "qota_rt": 8.51,
        },
        "score": 0.8,
        "source": "20260310002820",
    }]
    legacy = adapt_to_legacy(hits)
    assert legacy["facts"][0]["type"] == "shareholder"
    assert legacy["facts"][0]["value"] == 8.51


def test_supplier_role_preserved_in_path_direction() -> None:
    """supplier hit → path가 [supplier, SUPPLIES_TO, buyer] 방향 유지."""
    hits = [{
        "id": "rel:SUPPLIES_TO:00161383:00126380",
        "label": "relationship",
        "name": "한미반도체 → 삼성전자(주)",
        "attrs": {
            "rel_type": "SUPPLIES_TO",
            "from_id": "00161383",
            "from_name": "한미반도체",
            "to_id": "00126380",
            "to_name": "삼성전자(주)",
            "role": "supplier",
            "tier": 1,
        },
        "score": 0.8,
    }]
    legacy = adapt_to_legacy(hits)
    assert legacy["paths"][0] == ["한미반도체", "SUPPLIES_TO", "삼성전자(주)"]
    assert legacy["facts"][0]["extra"]["role"] == "supplier"


def test_empty_hits_yield_empty_keys_without_error() -> None:
    """빈 hits — KeyError 없이 빈 키 반환."""
    legacy = adapt_to_legacy([])
    assert legacy == {"facts": [], "paths": [], "provenance": []}


def test_fin_metric_node_hit() -> None:
    """fin_metric hit → fact value에 숫자 값, path 안 만들어짐."""
    hits = [{
        "id": "fm_abc",
        "label": "fin_metric",
        "name": "ifrs-full_Revenue",
        "attrs": {
            "account_id": "ifrs-full_Revenue",
            "value": 300875400000000.0,
            "unit": "KRW",
            "bsns_year": 2025,
            "metric_id": "fm_abc",
        },
        "score": 1.0,
        "source": "20260310002820",
    }]
    legacy = adapt_to_legacy(hits)
    assert legacy["facts"][0]["type"] == "fin_metric"
    assert legacy["facts"][0]["value"] == 300875400000000.0
    assert legacy["paths"] == []
    assert legacy["provenance"] == ["20260310002820"]
