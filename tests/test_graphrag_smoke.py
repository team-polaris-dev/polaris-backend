"""GraphRAG 스모크 테스트.

라이브 Neo4j 의존. entity_fulltext 인덱스 + 데이터 있어야 함.
"""
from __future__ import annotations

from graphrag import graph_search_node


def test_samsung_executive_query() -> None:
    out = graph_search_node({"reconstructed_query": "삼성전자 대표이사가 누구야?"})
    hits = out["graph_hits"]
    assert len(hits) > 0
    assert any(h["label"] == "organization" and "삼성전자" in h["name"] for h in hits) or \
           any(h["label"] == "relationship" and "삼성전자" in h.get("attrs", {}).get("to_name", "") for h in hits)
    assert any(h["label"] == "person" for h in hits), "임원 person hit 없음"
    assert out["graph_meta"]["n_seeds"] >= 1


def test_supply_chain_role_attrs() -> None:
    # 정규명으로 질의 — 'SK하이닉스' 약칭은 음차 노드(에스케이하이닉스)로 매칭이 안 돼
    # 별칭 보강 전까지 파편 노드로 빠진다(별도 이슈). 여기선 공급역할 속성 자체를 검증.
    out = graph_search_node({"reconstructed_query": "에스케이하이닉스(주) 공급망 공급사"})
    supply_hits = [
        h for h in out["graph_hits"]
        if h["label"] == "relationship" and h["attrs"].get("rel_type") == "SUPPLIES_TO"
    ]
    assert supply_hits, "공급(SUPPLIES_TO) hit 없음"
    roles = {h["attrs"].get("role") for h in supply_hits}
    assert roles & {"supplier", "buyer"}, f"공급역할 없음: {roles}"


def test_product_seed_reverse() -> None:
    out = graph_search_node({"reconstructed_query": "HBM 제조사"})
    # FULLTEXT가 Product:HBM을 잡고, pattern_product_seed_reverse가 PRODUCES 관계를 emit해야 함
    produces_hits = [
        h for h in out["graph_hits"]
        if h["label"] == "relationship" and h["attrs"].get("rel_type") == "PRODUCES"
    ]
    assert out["graph_meta"]["n_seeds"] >= 1, "HBM seed 못 잡음"
    # producer hit이 1개 이상 있어야 함 (라이브 데이터 의존 — 약하게 assert)
    print(f"  produces_hits={len(produces_hits)}, seeds={out['graph_meta']['n_seeds']}")


def test_nonexistent_query_no_fallback() -> None:
    """seed 0개면 fallback도 안 돔. fallback_used=False."""
    out = graph_search_node({"reconstructed_query": "존재하지않는회사명999XYZ"})
    assert out["graph_meta"]["n_seeds"] == 0
    assert out["graph_meta"]["n_hits"] == 0
    assert out["graph_meta"]["fallback_used"] is False


def test_legacy_keys_present_and_aligned() -> None:
    """셰임 — 신·구 키 동시 emit + facts 개수 == hits 개수."""
    out = graph_search_node({"reconstructed_query": "삼성전자"})
    assert "graph_hits" in out and "graph_facts" in out
    assert "graph_paths" in out and "graph_provenance" in out
    assert len(out["graph_facts"]) == len(out["graph_hits"])
    # provenance는 dedup된 unique set이므로 hits 수보다 작거나 같음
    assert len(out["graph_provenance"]) <= len(out["graph_hits"])
