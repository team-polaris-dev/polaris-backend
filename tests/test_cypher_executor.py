"""공식 retriever 행 → (hits, meta) 매퍼 — 오프라인(chunk fetch monkeypatch).

text2cypher.run_relationship_query 가 돌려준 행을 직접 주입해: 노드 role(anchor/
bridge/sibling), 관계 hit(DOMAIN_RELS), 셰임 1:1(len(facts)==len(hits)), 결정성
(이름 asc), dedup, 앵커 corp_code 추출(raw Seed 포함)을 검증한다. Cypher 실행은
라이브러리 책임이라 여기선 안 한다 — _fetch_chunk_texts 만 monkeypatch(네트워크 0).
"""
from __future__ import annotations

from config.relations import DOMAIN_RELS
from graphrag import cypher_executor
from graphrag.schema import adapt_to_legacy


_ANCHORS = [{"corp_code": "00126256", "name": "삼성생명"}]

_CYPHER = (
    "MATCH (anchor:Organization)<-[r1:IS_MAJOR_SHAREHOLDER_OF]-(bridge:Organization)"
    "-[r2:IS_MAJOR_SHAREHOLDER_OF]->(sib:Organization) "
    "WHERE anchor.corp_code IN ['00126256'] "
    "RETURN bridge AS from_name ORDER BY from_name ASC LIMIT 50"
)
_REASON = "형제 계열사 조회"


def _rows():
    """bridge→anchor + bridge→형제 행. dup 형제 행으로 dedup 도 검증(이름 asc)."""
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


def _patch(monkeypatch):
    # 근거 텍스트 없이 오프라인으로 evidence 점수 로직만 실행.
    monkeypatch.setattr(cypher_executor, "_fetch_chunk_texts", lambda ids: {})


def _run(rows=None, anchors=None, query="삼성생명 형제 계열사"):
    return cypher_executor.map_results(
        rows if rows is not None else _rows(),
        _CYPHER,
        anchors if anchors is not None else _ANCHORS,
        query,
        _REASON,
    )


def test_emits_anchor_bridge_sibling_node_roles(monkeypatch):
    _patch(monkeypatch)
    hits, _ = _run()

    orgs = [h for h in hits if h["label"] == "organization"]
    roles = {h["attrs"].get("structured_role") for h in orgs}
    assert roles == {"anchor", "bridge", "sibling"}
    # 앵커 + bridge(dedup 1) + 형제 2 = 4 노드.
    assert {h["name"] for h in orgs} == {"삼성생명", "삼성물산", "삼성에스디에스", "삼성전자"}
    siblings = [h for h in orgs if h["attrs"].get("structured_role") == "sibling"]
    assert all(h["attrs"].get("stock_code") for h in siblings)


def test_relationship_hits_use_domain_rels(monkeypatch):
    _patch(monkeypatch)
    hits, meta = _run()

    rels = [h for h in hits if h["label"] == "relationship"]
    # bridge→anchor + bridge→sib1 + bridge→sib2 = 3 엣지(중복 형제 dedup).
    assert len(rels) == 3
    for h in rels:
        assert h["attrs"]["rel_type"] in DOMAIN_RELS
    assert meta["structured"]["kind"] == "text2cypher"
    assert meta["structured"]["abstained"] is False
    assert meta["structured"]["cypher"] == _CYPHER
    assert meta["structured"]["reason"] == _REASON


def test_shame_one_to_one_contract(monkeypatch):
    _patch(monkeypatch)
    hits, _ = _run()

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
    hits, _ = _run()
    assert adapt_to_legacy(hits)["paths"] == []


def test_strong_evidence_renders_panel_path(monkeypatch):
    # chunk 본문이 양끝 회사+관계어를 언급하면 강근거 → 망 path 가 실제로 렌더된다(게이트가 항상-0 아님).
    rows = [
        {"from_id": "00126362", "from_name": "삼성물산", "from_role": "bridge",
         "to_id": "00126256", "to_name": "삼성생명", "to_role": "anchor",
         "rel_type": "IS_MAJOR_SHAREHOLDER_OF", "source": "A1", "chunk_id": "c1"},
    ]
    monkeypatch.setattr(
        cypher_executor, "_fetch_chunk_texts",
        lambda ids: {"c1": "삼성물산은 삼성생명의 최대주주로서 지분을 보유한다"},
    )
    hits, _ = _run(rows=rows, query="삼성생명 최대주주")
    paths = adapt_to_legacy(hits)["paths"]
    assert paths == [["삼성물산", "IS_MAJOR_SHAREHOLDER_OF", "삼성생명"]]


def test_determinism_sibling_names_ascending(monkeypatch):
    _patch(monkeypatch)
    hits, _ = _run()

    siblings = [
        h["name"] for h in hits
        if h["label"] == "organization" and h["attrs"].get("structured_role") == "sibling"
    ]
    assert siblings == sorted(siblings)
    assert siblings == ["삼성에스디에스", "삼성전자"]


def test_dedup_duplicate_sibling_collapsed(monkeypatch):
    _patch(monkeypatch)
    hits, _ = _run()

    sib_ids = [
        h["id"] for h in hits
        if h["label"] == "organization" and h["attrs"].get("structured_role") == "sibling"
    ]
    assert len(sib_ids) == len(set(sib_ids)) == 2


def test_anchor_code_extracted_into_meta(monkeypatch):
    _patch(monkeypatch)
    _, meta = _run()
    assert meta["structured"]["anchors"] == ["00126256"]


def test_raw_seed_anchor_shape_yields_corp_code(monkeypatch):
    # 실제 search() 가 넘기는 raw Seed(key_type/key_value, corp_code 키 없음)에서도
    # corp_code 가 추출돼 meta anchors 로 기록되는지(통합 계약).
    _patch(monkeypatch)
    raw_seed = {"label": "organization", "id": "00126256",
                "key_type": "corp_code", "key_value": "00126256", "name": "삼성생명"}
    _, meta = _run(anchors=[raw_seed])
    assert meta["structured"]["anchors"] == ["00126256"]


def test_zero_rows_returns_none(monkeypatch):
    _patch(monkeypatch)
    assert _run(rows=[]) is None


_SUPPLY_ANCHORS = [{"corp_code": "00164779", "name": "SK하이닉스"}]
_SUPPLY_CYPHER = (
    "MATCH (supplier:Organization)-[r:SUPPLIES_TO]->(anchor:Organization) "
    "WHERE anchor.corp_code IN ['00164779'] "
    "RETURN supplier.corp_code AS from_id, supplier.name AS from_name, "
    "anchor.corp_code AS to_id, anchor.name AS to_name, type(r) AS rel_type "
    "LIMIT 50"
)


def _supply_rows():
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


def _run_supply(rows, query):
    return cypher_executor.map_results(rows, _SUPPLY_CYPHER, _SUPPLY_ANCHORS, query, "")


def test_supply_role_filled_from_anchor_direction(monkeypatch):
    # LLM 이 row.role 을 안 내보내도 앵커 기준 방향으로 supplier/buyer 결정적으로 보강.
    monkeypatch.setattr(cypher_executor, "_fetch_chunk_texts", lambda ids: {})
    hits, _ = _run_supply(_supply_rows(), "SK하이닉스 공급망")

    rels = [h for h in hits if h["label"] == "relationship"]
    by_pair = {(h["attrs"]["from_name"], h["attrs"]["to_name"]): h["attrs"].get("role") for h in rels}
    assert by_pair[("소재A", "SK하이닉스")] == "supplier"
    assert by_pair[("SK하이닉스", "고객B")] == "buyer"


def test_supply_role_only_supplies_to_relation(monkeypatch):
    # SUPPLIES_TO 외 관계는 빈 role 유지(원본 보존). 다른 관계까지 라벨링하지 않는다.
    monkeypatch.setattr(cypher_executor, "_fetch_chunk_texts", lambda ids: {})
    rows = [
        {"from_id": "00111111", "from_name": "주주A",
         "to_id": "00164779", "to_name": "SK하이닉스",
         "rel_type": "IS_MAJOR_SHAREHOLDER_OF", "source": "X", "chunk_id": ""},
    ]
    hits, _ = _run_supply(rows, "SK하이닉스 주주")
    rels = [h for h in hits if h["label"] == "relationship"]
    assert rels and rels[0]["attrs"].get("role") == ""


# ── rank_results: text2cypher 위 SQL 재무 랭킹 후처리 ────────────────────
# 그래프(text2cypher)는 관계/존재만, 매출 1위 줄세우기는 MariaDB 결정적 SQL 이 담당한다.

_RANK_ANCHORS = ["00164779"]  # SK하이닉스 corp_code


def _supplier_hits():
    """map_results 가 낸 모양: 앵커 + 비앵커 공급사 노드 + 엣지 하나."""
    return [
        {"id": "00164779", "label": "organization", "name": "SK하이닉스",
         "attrs": {"structured_role": "anchor"}, "score": 1.0, "seed_origin": "structured"},
        {"id": "100", "label": "organization", "name": "동진쎄미켐",
         "attrs": {"structured_role": "supplier"}, "score": 0.75, "seed_origin": "structured"},
        {"id": "101", "label": "organization", "name": "한미반도체",
         "attrs": {"structured_role": "supplier"}, "score": 0.75, "seed_origin": "structured"},
        {"id": "102", "label": "organization", "name": "미니소재",
         "attrs": {"structured_role": "supplier"}, "score": 0.75, "seed_origin": "structured"},
        {"id": "rel:SUPPLIES_TO:100:00164779", "label": "relationship",
         "attrs": {"rel_type": "SUPPLIES_TO", "from_id": "100", "from_name": "동진쎄미켐",
                   "to_id": "00164779", "to_name": "SK하이닉스", "role": "supplier"}, "score": 0.75},
    ]


def _rank_meta():
    return {
        "mode": "structured",
        "structured": {"mode": "structured", "kind": "text2cypher", "abstained": False,
                       "answer_edges": [], "quality_notes": ["기존 노트"]},
        "patterns_run": ["text2cypher"],
        "n_hits": 5, "fallback_used": False, "errors": [],
    }


_RANK_METRIC = {"100": 5_000_000_000_000, "101": 3_000_000_000_000, "102": 100_000_000_000}


def _fake_metric_values(codes, account_id="ifrs-full_Revenue", year=None):
    return [{"corp_code": c, "account_id": account_id, "value": str(_RANK_METRIC[c]),
             "bsns_year": 2025, "unit": "KRW", "rcept_no": "f1", "corp_name": ""}
            for c in codes if c in _RANK_METRIC]


def test_rank_results_noop_without_rank_intent():
    # 랭킹 의도 없는 관계 질문 → hits/meta 동일 객체 그대로(additive no-op).
    hits, meta = _supplier_hits(), _rank_meta()
    out_hits, out_meta = cypher_executor.rank_results(hits, meta, "SK하이닉스 공급사", _RANK_ANCHORS)
    assert out_hits is hits and out_meta is meta


def test_rank_results_noop_without_metric():
    # 랭킹 의도는 있으나 지표 어휘가 없으면(무엇으로 줄세울지 불명) no-op.
    hits, meta = _supplier_hits(), _rank_meta()
    out_hits, out_meta = cypher_executor.rank_results(hits, meta, "공급사 중 가장 좋은 곳", _RANK_ANCHORS)
    assert out_hits is hits and out_meta is meta


def test_rank_results_ranks_suppliers_by_revenue(monkeypatch):
    monkeypatch.setattr(cypher_executor, "_fetch_metric_values", _fake_metric_values)
    hits, meta = _supplier_hits(), _rank_meta()
    out_hits, out_meta = cypher_executor.rank_results(
        hits, meta, "SK하이닉스 공급사 중 매출 1위", _RANK_ANCHORS)

    structured = out_meta["structured"]
    assert structured["rank_metric"] == "ifrs-full_Revenue"
    assert structured["rankable"] is True
    assert structured["selected"]["name"] == "동진쎄미켐"
    assert [m["name"] for m in structured["members"]] == ["동진쎄미켐", "한미반도체", "미니소재"]
    # 기존 노트 보존 + 후처리 노트 추가.
    assert structured["quality_notes"][0] == "기존 노트"

    # 1위 재무수치 hit 가 추가돼 답변이 수치를 노출(render._fmt_graph 계약).
    fin = [h for h in out_hits if h["label"] == "fin_metric"]
    assert len(fin) == 1
    assert fin[0]["attrs"]["account_id"] == "ifrs-full_Revenue"
    assert fin[0]["attrs"]["value"] == 5_000_000_000_000.0
    assert fin[0]["attrs"]["rank"] == 1

    # 1위 org 노드는 selected 로 강조, 나머지는 순위만 표시.
    winner = next(h for h in out_hits if h["label"] == "organization" and h["id"] == "100")
    assert winner["attrs"]["structured_role"] == "selected"
    assert winner["attrs"]["rank"] == 1 and winner["score"] == 1.0
    second = next(h for h in out_hits if h["id"] == "101")
    assert second["attrs"]["rank"] == 2 and second["attrs"]["structured_role"] == "supplier"


def test_rank_results_does_not_mutate_input(monkeypatch):
    monkeypatch.setattr(cypher_executor, "_fetch_metric_values", _fake_metric_values)
    hits, meta = _supplier_hits(), _rank_meta()
    cypher_executor.rank_results(hits, meta, "SK하이닉스 공급사 중 매출 1위", _RANK_ANCHORS)
    # 원본 hits 의 1위 노드는 건드리지 않는다(새 dict 반환).
    orig_winner = next(h for h in hits if h["id"] == "100")
    assert orig_winner["attrs"]["structured_role"] == "supplier"
    assert "rank" not in orig_winner["attrs"]
    assert meta["structured"].get("selected") is None


def test_rank_results_noop_without_nonanchor_candidate(monkeypatch):
    monkeypatch.setattr(cypher_executor, "_fetch_metric_values", _fake_metric_values)
    anchor_only = [{"id": "00164779", "label": "organization", "name": "SK하이닉스",
                    "attrs": {"structured_role": "anchor"}, "score": 1.0}]
    meta = _rank_meta()
    out_hits, out_meta = cypher_executor.rank_results(anchor_only, meta, "매출 1위", _RANK_ANCHORS)
    assert out_hits is anchor_only and out_meta is meta


def test_rank_results_noop_without_metric_rows(monkeypatch):
    # 후보는 있는데 매출 데이터가 0 → 억지 랭킹 금지, 관계 결과 보존.
    monkeypatch.setattr(cypher_executor, "_fetch_metric_values",
                        lambda codes, account_id="ifrs-full_Revenue", year=None: [])
    hits, meta = _supplier_hits(), _rank_meta()
    out_hits, out_meta = cypher_executor.rank_results(hits, meta, "매출 1위", _RANK_ANCHORS)
    assert out_hits is hits and out_meta is meta
