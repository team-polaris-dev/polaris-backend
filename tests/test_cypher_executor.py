"""text-to-Cypher 실행기 — 오프라인(Neo4j/RDB monkeypatch).

검증된 Cypher 행을 주입해: 노드 role(anchor/bridge/sibling), 관계 hit(DOMAIN_RELS),
셰임 1:1(len(facts)==len(hits)), 결정성(이름 asc), dedup, 앵커 주입을 검증한다.
_run_cypher 와 _fetch_chunk_texts 만 monkeypatch — 네트워크 0.
"""
from __future__ import annotations

from config.relations import DOMAIN_RELS
from graphrag import cypher_executor
from graphrag.cypher_generator import GeneratedCypher
from graphrag.schema import adapt_to_legacy


_ANCHORS = [{"corp_code": "00126256", "name": "삼성생명"}]

_GENERATED = GeneratedCypher(
    cypher=(
        "MATCH (anchor:Organization)<-[r1:IS_MAJOR_SHAREHOLDER_OF]-(bridge:Organization)"
        "-[r2:IS_MAJOR_SHAREHOLDER_OF]->(sib:Organization) "
        "WHERE anchor.corp_code IN $anchors "
        "RETURN bridge AS from_name ORDER BY from_name ASC LIMIT 50"
    ),
    params={},
    reason="형제 계열사 조회",
)


def _rows_factory(captured=None):
    """bridge→anchor + bridge→형제 행. dup 형제 행으로 dedup 도 검증(이름 asc)."""
    def _rows(cypher, params):
        if captured is not None:
            captured["params"] = params
        return [
            # bridge(삼성물산) → anchor(삼성생명)
            {"from_id": "00126362", "from_name": "삼성물산", "from_role": "bridge",
             "to_id": "00126256", "to_name": "삼성생명", "to_role": "anchor",
             "rel_type": "IS_MAJOR_SHAREHOLDER_OF", "source": "A1", "chunk_id": ""},
            # bridge → sibling(삼성에스디에스)
            {"from_id": "00126362", "from_name": "삼성물산", "from_role": "bridge",
             "to_id": "00164742", "to_name": "삼성에스디에스", "to_role": "sibling",
             "to_stock_code": "018260",
             "rel_type": "IS_MAJOR_SHAREHOLDER_OF", "source": "S1", "chunk_id": ""},
            # bridge → sibling(삼성전자)
            {"from_id": "00126362", "from_name": "삼성물산", "from_role": "bridge",
             "to_id": "00126380", "to_name": "삼성전자", "to_role": "sibling",
             "to_stock_code": "005930",
             "rel_type": "IS_MAJOR_SHAREHOLDER_OF", "source": "S2", "chunk_id": ""},
            # 같은 형제(삼성전자) 중복 행 → dedup 되어야 함.
            {"from_id": "00126362", "from_name": "삼성물산", "from_role": "bridge",
             "to_id": "00126380", "to_name": "삼성전자", "to_role": "sibling",
             "to_stock_code": "005930",
             "rel_type": "IS_MAJOR_SHAREHOLDER_OF", "source": "S2", "chunk_id": ""},
        ]
    return _rows


def _patch(monkeypatch, captured=None):
    monkeypatch.setattr(cypher_executor, "_run_cypher", _rows_factory(captured))
    # 근거 텍스트 없이 오프라인으로 evidence 점수 로직만 실행.
    monkeypatch.setattr(cypher_executor, "_fetch_chunk_texts", lambda ids: {})


def test_emits_anchor_bridge_sibling_node_roles(monkeypatch):
    _patch(monkeypatch)
    hits, _ = cypher_executor.run(_GENERATED, _ANCHORS, "삼성생명 형제 계열사")

    orgs = [h for h in hits if h["label"] == "organization"]
    roles = {h["attrs"].get("structured_role") for h in orgs}
    assert roles == {"anchor", "bridge", "sibling"}
    # 앵커 + bridge(dedup 1) + 형제 2 = 4 노드.
    assert {h["name"] for h in orgs} == {"삼성생명", "삼성물산", "삼성에스디에스", "삼성전자"}
    siblings = [h for h in orgs if h["attrs"].get("structured_role") == "sibling"]
    assert all(h["attrs"].get("stock_code") for h in siblings)


def test_relationship_hits_use_domain_rels(monkeypatch):
    _patch(monkeypatch)
    hits, meta = cypher_executor.run(_GENERATED, _ANCHORS, "삼성생명 형제 계열사")

    rels = [h for h in hits if h["label"] == "relationship"]
    # bridge→anchor + bridge→sib1 + bridge→sib2 = 3 엣지(중복 형제 dedup).
    assert len(rels) == 3
    for h in rels:
        assert h["attrs"]["rel_type"] in DOMAIN_RELS
    assert meta["structured"]["kind"] == "text2cypher"
    assert meta["structured"]["abstained"] is False


def test_shame_one_to_one_contract(monkeypatch):
    _patch(monkeypatch)
    hits, _ = cypher_executor.run(_GENERATED, _ANCHORS, "삼성생명 형제 계열사")

    legacy = adapt_to_legacy(hits)
    # 셰임 1:1 — 모든 hit 가 fact 로 매핑.
    assert len(legacy["facts"]) == len(hits)
    # paths 는 panel 근거 게이트를 통과한 망 엣지만(행 단위 정렬 유지).
    assert all(len(p) == 3 for p in legacy["paths"])
    assert len(legacy["path_sources"]) == len(legacy["paths"])
    assert len(legacy["path_chunks"]) == len(legacy["paths"])


def test_source_only_edges_gated_from_panel(monkeypatch):
    # chunk 본문이 없어 출처만 있는 엣지는 근거 confidence 가 낮아 panel path 에서 제외(fail-closed).
    _patch(monkeypatch)
    hits, _ = cypher_executor.run(_GENERATED, _ANCHORS, "삼성생명 형제 계열사")
    assert adapt_to_legacy(hits)["paths"] == []


def test_strong_evidence_renders_panel_path(monkeypatch):
    # chunk 본문이 양끝 회사+관계어를 언급하면 강근거 → 망 path 가 실제로 렌더된다(게이트가 항상-0 아님).
    rows = [
        {"from_id": "00126362", "from_name": "삼성물산", "from_role": "bridge",
         "to_id": "00126256", "to_name": "삼성생명", "to_role": "anchor",
         "rel_type": "IS_MAJOR_SHAREHOLDER_OF", "source": "A1", "chunk_id": "c1"},
    ]
    monkeypatch.setattr(cypher_executor, "_run_cypher", lambda cypher, params: rows)
    monkeypatch.setattr(
        cypher_executor, "_fetch_chunk_texts",
        lambda ids: {"c1": "삼성물산은 삼성생명의 최대주주로서 지분을 보유한다"},
    )
    hits, _ = cypher_executor.run(_GENERATED, _ANCHORS, "삼성생명 최대주주")
    paths = adapt_to_legacy(hits)["paths"]
    assert paths == [["삼성물산", "IS_MAJOR_SHAREHOLDER_OF", "삼성생명"]]


def test_determinism_sibling_names_ascending(monkeypatch):
    _patch(monkeypatch)
    hits, _ = cypher_executor.run(_GENERATED, _ANCHORS, "삼성생명 형제 계열사")

    siblings = [
        h["name"] for h in hits
        if h["label"] == "organization" and h["attrs"].get("structured_role") == "sibling"
    ]
    assert siblings == sorted(siblings)
    assert siblings == ["삼성에스디에스", "삼성전자"]


def test_dedup_duplicate_sibling_collapsed(monkeypatch):
    _patch(monkeypatch)
    hits, _ = cypher_executor.run(_GENERATED, _ANCHORS, "삼성생명 형제 계열사")

    sib_ids = [
        h["id"] for h in hits
        if h["label"] == "organization" and h["attrs"].get("structured_role") == "sibling"
    ]
    assert len(sib_ids) == len(set(sib_ids)) == 2


def test_anchors_injected_into_params(monkeypatch):
    captured: dict = {}
    _patch(monkeypatch, captured)
    cypher_executor.run(_GENERATED, _ANCHORS, "삼성생명 형제 계열사")

    assert captured["params"]["anchors"] == ["00126256"]


def test_raw_seed_anchor_shape_yields_corp_code(monkeypatch):
    # 실제 search() 가 넘기는 raw Seed(key_type/key_value, corp_code 키 없음)에서도
    # corp_code 가 추출돼 $anchors 로 주입되는지(통합 계약).
    captured: dict = {}
    _patch(monkeypatch, captured)
    raw_seed = {"label": "organization", "id": "00126256",
                "key_type": "corp_code", "key_value": "00126256", "name": "삼성생명"}
    cypher_executor.run(_GENERATED, [raw_seed], "삼성생명 형제 계열사")

    assert captured["params"]["anchors"] == ["00126256"]


def test_zero_rows_returns_none(monkeypatch):
    monkeypatch.setattr(cypher_executor, "_run_cypher", lambda cypher, params: [])
    monkeypatch.setattr(cypher_executor, "_fetch_chunk_texts", lambda ids: {})
    assert cypher_executor.run(_GENERATED, _ANCHORS, "삼성생명 형제 계열사") is None


_SUPPLY_ANCHORS = [{"corp_code": "00164779", "name": "SK하이닉스"}]
_SUPPLY_GENERATED = GeneratedCypher(
    cypher=(
        "MATCH (supplier:Organization)-[r:SUPPLIES_TO]->(anchor:Organization) "
        "WHERE anchor.corp_code IN $anchors "
        "RETURN supplier.corp_code AS from_id, supplier.name AS from_name, "
        "anchor.corp_code AS to_id, anchor.name AS to_name, type(r) AS rel_type "
        "LIMIT 50"
    ),
    params={},
    reason="SK하이닉스 공급사",
)


def _supply_rows(_cypher, _params):
    return [
        # supplier(소재A) → anchor(SK하이닉스): 앵커가 to → role=supplier
        {"from_id": "00111111", "from_name": "소재A",
         "to_id": "00164779", "to_name": "SK하이닉스",
         "rel_type": "SUPPLIES_TO", "source": "S1", "chunk_id": ""},
        # anchor(SK하이닉스) → buyer(고객B): 앵커가 from → role=buyer
        {"from_id": "00164779", "from_name": "SK하이닉스",
         "to_id": "00222222", "to_name": "고객B",
         "rel_type": "SUPPLIES_TO", "source": "S2", "chunk_id": ""},
    ]


def test_supply_role_filled_from_anchor_direction(monkeypatch):
    # LLM 이 row.role 을 안 내보내도 앵커 기준 방향으로 supplier/buyer 결정적으로 보강.
    monkeypatch.setattr(cypher_executor, "_run_cypher", _supply_rows)
    monkeypatch.setattr(cypher_executor, "_fetch_chunk_texts", lambda ids: {})
    hits, _ = cypher_executor.run(_SUPPLY_GENERATED, _SUPPLY_ANCHORS, "SK하이닉스 공급망")

    rels = [h for h in hits if h["label"] == "relationship"]
    by_pair = {(h["attrs"]["from_name"], h["attrs"]["to_name"]): h["attrs"].get("role") for h in rels}
    assert by_pair[("소재A", "SK하이닉스")] == "supplier"
    assert by_pair[("SK하이닉스", "고객B")] == "buyer"


def test_supply_role_only_supplies_to_relation(monkeypatch):
    # SUPPLIES_TO 외 관계는 빈 role 유지(원본 보존). 다른 관계까지 라벨링하지 않는다.
    monkeypatch.setattr(cypher_executor, "_run_cypher", lambda c, p: [
        {"from_id": "00111111", "from_name": "주주A",
         "to_id": "00164779", "to_name": "SK하이닉스",
         "rel_type": "IS_MAJOR_SHAREHOLDER_OF", "source": "X", "chunk_id": ""},
    ])
    monkeypatch.setattr(cypher_executor, "_fetch_chunk_texts", lambda ids: {})
    hits, _ = cypher_executor.run(_SUPPLY_GENERATED, _SUPPLY_ANCHORS, "SK하이닉스 주주")
    rels = [h for h in hits if h["label"] == "relationship"]
    assert rels and rels[0]["attrs"].get("role") == ""
