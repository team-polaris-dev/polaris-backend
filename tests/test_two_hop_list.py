"""지표 없는 2-hop 형제 계열사 나열 (two_hop_list) — 오프라인.

"삼성생명 최대주주가 지배하는 다른 상장 계열사" 류 질문이:
  1) planner 에서 two_hop_list 플랜으로 식별되고(지배주주어 + 지배/계열 대상, 랭크 의도 없음),
  2) executor 가 공통 지배주주(bridge)를 거쳐 형제 계열사를 나열하며 노드+망 엣지를 emit하고,
  3) 셰임 1:1 계약(len(facts)==len(hits))과 결정성(이름 asc)·dedup 을 지키며,
  4) listed_only 신호가 플랜·실행에 정확히 전달되는지
검증한다. Neo4j 는 _two_hop_list_rows monkeypatch 로 차단 — 네트워크 0.
"""
from __future__ import annotations

from graphrag import structured_executor
from graphrag.planner import plan
from graphrag.schema import adapt_to_legacy


# ── planner: 지배주주 형제 나열 → two_hop_list, 그 외는 기존 경로 보존 ────────────

def test_controlling_shareholder_listed_sibling_becomes_two_hop_list():
    out = plan("삼성생명 최대주주가 지배하는 다른 상장 계열사")

    assert out is not None
    assert out.kind == "two_hop_list"
    assert out.first_rank is None
    assert out.first_relation is not None
    assert out.first_relation.rel_type == "IS_MAJOR_SHAREHOLDER_OF"
    assert out.listed_only is True
    # to_dict() 는 first_rank=None 을 안전 직렬화하고 listed_only 를 싣는다(graph_meta).
    d = out.to_dict()
    assert d["first_rank"] is None
    assert d["listed_only"] is True


def test_two_hop_list_without_listed_keyword_is_not_listed_only():
    out = plan("삼성생명 최대주주가 보유한 다른 계열사")

    assert out is not None
    assert out.kind == "two_hop_list"
    assert out.listed_only is False


def test_two_hop_list_unlisted_keyword_does_not_set_listed_only():
    # '비상장'은 반대 의도 → listed_only 켜지지 않음.
    out = plan("삼성생명 최대주주가 지배하는 비상장 계열사")

    assert out is not None
    assert out.kind == "two_hop_list"
    assert out.listed_only is False


def test_bare_controlling_shareholder_lookup_is_not_two_hop_list():
    # 지배주주만 묻는 단일 조회(동사·대상 없음) → 나열 아님 → 구조화 플랜 없음(explore 로 degrade).
    assert plan("삼성생명 최대주주는 누구야") is None


def test_two_hop_list_yields_to_rank_path_when_rank_intent_present():
    # 같은 형제 범위라도 '매출 1위'처럼 랭크 의도가 있으면 나열이 아니라 랭킹 경로가 우선.
    out = plan("삼성생명 최대주주가 지배하는 계열사 중 매출 1위")
    assert out is None or out.kind != "two_hop_list"


# ── executor: bridge 경유 형제 나열 + 노드/망 엣지 + 결정성·dedup ─────────────

def _life_seed():
    return {
        "label": "organization", "id": "00126256", "key_type": "corp_code",
        "key_value": "00126256", "name": "삼성생명", "score": 1.0,
    }


def _life_rows_factory(captured=None):
    """이름 asc 정렬된 형제 행(쿼리 계약). dup sib 행을 끼워 dedup 도 검증."""
    def _rows(anchor, bridge_rel, *, listed_only):
        if captured is not None:
            captured["bridge_rel"] = bridge_rel
            captured["listed_only"] = listed_only
        return [
            {"bridge_id": "00126362", "bridge_name": "삼성물산",
             "b_id": "00164742", "b_name": "삼성에스디에스", "stock_code": "018260",
             "anchor_source": "A1", "anchor_chunk_id": "",
             "source": "S1", "chunk_id": ""},
            {"bridge_id": "00126362", "bridge_name": "삼성물산",
             "b_id": "00126380", "b_name": "삼성전자", "stock_code": "005930",
             "anchor_source": "A1", "anchor_chunk_id": "",
             "source": "S2", "chunk_id": ""},
            # 같은 형제(삼성전자)를 다른 bridge 행으로 한 번 더 → dedup 되어야 함.
            {"bridge_id": "00126362", "bridge_name": "삼성물산",
             "b_id": "00126380", "b_name": "삼성전자", "stock_code": "005930",
             "anchor_source": "A1", "anchor_chunk_id": "",
             "source": "S2", "chunk_id": ""},
        ]
    return _rows


def test_two_hop_list_lists_sibling_companies(monkeypatch):
    monkeypatch.setattr(structured_executor, "_two_hop_list_rows", _life_rows_factory())
    p = plan("삼성생명 최대주주가 지배하는 다른 상장 계열사")

    result = structured_executor._execute_two_hop_list(p, _life_seed(), "삼성생명 최대주주가 지배하는 다른 상장 계열사")

    assert result is not None
    _, meta = result
    structured = meta["structured"]
    assert structured["kind"] == "two_hop_list"
    assert structured["abstained"] is False
    assert structured["bridge_relation"] == "IS_MAJOR_SHAREHOLDER_OF"
    # 결정성: 쿼리 정렬(이름 asc)을 보존, dup 형제는 dedup.
    assert [s["name"] for s in structured["siblings"]] == ["삼성에스디에스", "삼성전자"]


def test_two_hop_list_emits_anchor_bridge_sibling_nodes(monkeypatch):
    monkeypatch.setattr(structured_executor, "_two_hop_list_rows", _life_rows_factory())
    p = plan("삼성생명 최대주주가 지배하는 다른 상장 계열사")
    hits, _ = structured_executor._execute_two_hop_list(p, _life_seed(), "삼성생명 최대주주가 지배하는 다른 상장 계열사")

    orgs = [h for h in hits if h["label"] == "organization"]
    roles = {h["attrs"].get("structured_role") for h in orgs}
    assert roles == {"anchor", "controlling_shareholder", "sibling"}
    # 앵커 + bridge(1, dedup) + 형제 2 = 4 노드.
    assert {h["name"] for h in orgs} == {"삼성생명", "삼성물산", "삼성에스디에스", "삼성전자"}
    siblings = [h for h in orgs if h["attrs"].get("structured_role") == "sibling"]
    assert all(h["attrs"].get("stock_code") for h in siblings)


def test_two_hop_list_emits_bridge_network_edges(monkeypatch):
    monkeypatch.setattr(structured_executor, "_two_hop_list_rows", _life_rows_factory())
    p = plan("삼성생명 최대주주가 지배하는 다른 상장 계열사")
    hits, _ = structured_executor._execute_two_hop_list(p, _life_seed(), "삼성생명 최대주주가 지배하는 다른 상장 계열사")

    rels = [h for h in hits if h["label"] == "relationship"]
    # bridge→anchor(1, dedup) + bridge→sib1 + bridge→sib2 = 3 엣지, 전부 대주주 관계.
    assert len(rels) == 3
    assert {h["attrs"]["rel_type"] for h in rels} == {"IS_MAJOR_SHAREHOLDER_OF"}
    pairs = {(h["attrs"]["from_name"], h["attrs"]["to_name"]) for h in rels}
    assert ("삼성물산", "삼성생명") in pairs
    assert ("삼성물산", "삼성에스디에스") in pairs
    assert ("삼성물산", "삼성전자") in pairs


def test_two_hop_list_shame_contract_and_panel_edges(monkeypatch):
    monkeypatch.setattr(structured_executor, "_two_hop_list_rows", _life_rows_factory())
    p = plan("삼성생명 최대주주가 지배하는 다른 상장 계열사")
    hits, _ = structured_executor._execute_two_hop_list(p, _life_seed(), "삼성생명 최대주주가 지배하는 다른 상장 계열사")

    legacy = adapt_to_legacy(hits)
    # 셰임 1:1 — 모든 hit 가 fact 로 매핑.
    assert len(legacy["facts"]) == len(hits)
    # 대주주 관계는 망 엣지(무근거로도 패널에 그려진다) → 3 path.
    assert len(legacy["paths"]) == 3
    assert all(len(p_) == 3 for p_ in legacy["paths"])
    assert len(legacy["path_sources"]) == len(legacy["paths"])


def test_two_hop_list_passes_listed_only_flag_to_query(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(structured_executor, "_two_hop_list_rows", _life_rows_factory(captured))
    p = plan("삼성생명 최대주주가 지배하는 다른 상장 계열사")
    structured_executor._execute_two_hop_list(p, _life_seed(), "삼성생명 최대주주가 지배하는 다른 상장 계열사")

    assert captured["listed_only"] is True
    assert captured["bridge_rel"] == "IS_MAJOR_SHAREHOLDER_OF"


def test_two_hop_list_none_when_no_sibling(monkeypatch):
    # 형제가 없으면 억지 나열 금지 → None (search 가 abstain 으로 degrade).
    monkeypatch.setattr(structured_executor, "_two_hop_list_rows", lambda *a, **k: [])
    p = plan("삼성생명 최대주주가 지배하는 다른 상장 계열사")
    assert structured_executor._execute_two_hop_list(p, _life_seed(), "삼성생명 최대주주가 지배하는 다른 상장 계열사") is None


def test_two_hop_list_dispatched_by_execute(monkeypatch):
    # 공개 진입점 execute() 가 kind 로 _execute_two_hop_list 에 분기하는지.
    monkeypatch.setattr(structured_executor, "_two_hop_list_rows", _life_rows_factory())
    p = plan("삼성생명 최대주주가 지배하는 다른 상장 계열사")
    result = structured_executor.execute(p, [_life_seed()], "삼성생명 최대주주가 지배하는 다른 상장 계열사")

    assert result is not None
    _, meta = result
    assert meta["structured"]["kind"] == "two_hop_list"
