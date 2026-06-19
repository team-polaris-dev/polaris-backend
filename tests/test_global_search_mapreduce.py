"""GraphRAG Global Search query-time map-reduce + DRIFT 결합 (오프라인).

Neo4j/LLM 을 monkeypatch 해 select→map→filter→sort→cap 과 graph 노드의 DRIFT
결합(로컬 graph_facts + 앵커 군집 community_results 동시 반환)을 검증한다.
"""
from __future__ import annotations

from graphrag import global_search as gs
from graphrag import node as graph_node


class _FakeMsg:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    """json_llm 대역. invoke(prompt) → .content (JSON 문자열)."""

    def __init__(self, content: str | None = None, raises: bool = False):
        self._content = content
        self._raises = raises

    def invoke(self, _prompt):
        if self._raises:
            raise RuntimeError("LLM 다운")
        return _FakeMsg(self._content)


def _community(cid: int, *, members, member_names=None, size=None, summary=None):
    return {
        "cluster_id": cid,
        "summary": summary or f"군집{cid} 요약",
        "size": size if size is not None else len(members),
        "members": list(members),
        "member_names": member_names or [f"회사{m}" for m in members],
        "anchor_names": member_names or [f"회사{m}" for m in members],
        "edge_dist": '{"INVESTS_IN": 3}',
    }


# ── select ─────────────────────────────────────────────────────────────

def test_select_uses_corp_code_intersection_when_anchored(monkeypatch):
    monkeypatch.setattr(gs, "GLOBAL_MAP_MAX_COMMUNITIES", 5)
    comms = [
        _community(0, members=["aaa", "bbb"]),
        _community(1, members=["ccc", "ddd"]),
        _community(2, members=["eee"]),
    ]
    selected = gs._select_communities(comms, "질문", anchor_corp_codes=["ddd"])
    assert [c["cluster_id"] for c in selected] == [1]


def test_select_empty_when_anchor_misses_all(monkeypatch):
    monkeypatch.setattr(gs, "GLOBAL_MAP_MAX_COMMUNITIES", 5)
    comms = [_community(0, members=["aaa"]), _community(1, members=["bbb"])]
    assert gs._select_communities(comms, "질문", anchor_corp_codes=["zzz"]) == []


def test_select_query_name_match_without_anchor(monkeypatch):
    monkeypatch.setattr(gs, "GLOBAL_MAP_MAX_COMMUNITIES", 5)
    comms = [
        _community(0, members=["a"], member_names=["동진쎄미켐"]),
        _community(1, members=["b"], member_names=["삼성전자"]),
    ]
    selected = gs._select_communities(comms, "동진쎄미켐 공급망", anchor_corp_codes=None)
    assert [c["cluster_id"] for c in selected] == [0]


def test_select_falls_back_to_all_then_caps(monkeypatch):
    monkeypatch.setattr(gs, "GLOBAL_MAP_MAX_COMMUNITIES", 2)
    comms = [_community(i, members=[f"m{i}"], member_names=[f"무관사{i}"]) for i in range(4)]
    selected = gs._select_communities(comms, "전혀무관한질문", anchor_corp_codes=None)
    assert [c["cluster_id"] for c in selected] == [0, 1]  # size desc 입력 보존 + cap


# ── map step (filter / sort / cap) ─────────────────────────────────────

def test_global_search_filters_sorts_and_caps(monkeypatch):
    monkeypatch.setattr(gs, "GLOBAL_MAP_REDUCE", True)
    monkeypatch.setattr(gs, "GLOBAL_MAP_MIN_SCORE", 1)
    monkeypatch.setattr(gs, "GLOBAL_MAP_MAX_COMMUNITIES", 5)
    comms = [
        _community(0, members=["x0"]),
        _community(1, members=["x1"]),
        _community(2, members=["x2"]),
        _community(3, members=["x3"]),
    ]
    monkeypatch.setattr(gs, "_load_communities", lambda: comms)
    scores = {0: 0, 1: 50, 2: 50, 3: 90}  # cid0 은 MIN 미만 → 폐기
    monkeypatch.setattr(gs, "_map_community", lambda q, c: (f"부분답{c['cluster_id']}", scores[c["cluster_id"]]))

    out = gs.global_search("질문", anchor_corp_codes=["x0", "x1", "x2", "x3"])

    # score desc, 동점(1,2)은 cluster_id asc → [3, 1, 2]; cid0 폐기.
    assert [u["code"] for u in out] == ["3", "1", "2"]
    assert out[0]["value"] == "부분답3"
    assert out[0]["extra"]["score"] == 90


def test_global_search_caps_mapped_results(monkeypatch):
    monkeypatch.setattr(gs, "GLOBAL_MAP_REDUCE", True)
    monkeypatch.setattr(gs, "GLOBAL_MAP_MIN_SCORE", 1)
    monkeypatch.setattr(gs, "GLOBAL_MAP_MAX_COMMUNITIES", 2)
    comms = [_community(i, members=[f"x{i}"]) for i in range(4)]
    monkeypatch.setattr(gs, "_load_communities", lambda: comms)
    monkeypatch.setattr(gs, "_map_community", lambda q, c: ("a", 100 - c["cluster_id"]))

    out = gs.global_search("질문", anchor_corp_codes=[f"x{i}" for i in range(4)])
    assert [u["code"] for u in out] == ["0", "1"]  # 상위 2개만


# ── _map_community 파싱 / 폴백 ──────────────────────────────────────────

def test_map_community_parses_and_clamps(monkeypatch):
    monkeypatch.setattr(gs, "json_llm", _FakeLLM(content='{"answer": "부분답", "score": 150}'))
    answer, score = gs._map_community("질문", _community(0, members=["a"]))
    assert answer == "부분답"
    assert score == 100  # 0~100 clamp


def test_map_community_falls_back_on_llm_error(monkeypatch):
    monkeypatch.setattr(gs, "json_llm", _FakeLLM(raises=True))
    comm = _community(7, members=["a"], summary="정적요약문")
    answer, score = gs._map_community("질문", comm)
    assert answer == "정적요약문"
    assert score == gs._FALLBACK_SCORE


def test_map_community_falls_back_on_bad_json(monkeypatch):
    monkeypatch.setattr(gs, "json_llm", _FakeLLM(content="이건 JSON 이 아님"))
    comm = _community(7, members=["a"], summary="정적요약문")
    answer, score = gs._map_community("질문", comm)
    assert answer == "정적요약문"
    assert score == gs._FALLBACK_SCORE


# ── graceful degradation ───────────────────────────────────────────────

def test_global_search_empty_when_no_communities(monkeypatch):
    monkeypatch.setattr(gs, "_load_communities", lambda: [])
    assert gs.global_search("질문") == []
    assert gs.global_search("질문", anchor_corp_codes=["a"]) == []


def test_global_search_static_fallback_when_reduce_off(monkeypatch):
    monkeypatch.setattr(gs, "GLOBAL_MAP_REDUCE", False)
    monkeypatch.setattr(gs, "GLOBAL_MAP_MAX_COMMUNITIES", 5)
    comm = _community(0, members=["a"], summary="정적요약")
    monkeypatch.setattr(gs, "_load_communities", lambda: [comm])

    def _must_not_call(*a, **k):
        raise AssertionError("REDUCE=0 인데 map 이 호출됨")

    monkeypatch.setattr(gs, "_map_community", _must_not_call)

    out = gs.global_search("질문", anchor_corp_codes=["a"])
    assert len(out) == 1
    assert out[0]["value"] == "정적요약"          # 정적 summary 그대로
    assert "score" not in out[0]["extra"]          # map 미수행 → 점수 없음


# ── DRIFT 결합 (graph 노드) ────────────────────────────────────────────

def test_anchor_corp_codes_extracts_corp_code_seeds_in_order():
    seeds = [
        {"key_type": "corp_code", "key_value": "00126380", "name": "삼성전자"},
        {"key_type": "er_name", "key_value": "org:동진", "name": "동진"},  # 제외
        {"key_type": "corp_code", "key_value": "00164742", "name": "SK"},
        {"key_type": "corp_code", "key_value": "00126380", "name": "삼성전자"},  # 중복 제외
    ]
    assert graph_node._anchor_corp_codes(seeds) == ["00126380", "00164742"]


def _fake_search_output(seeds):
    return {
        "graph_hits": [
            {"id": "00126380", "label": "organization", "name": "삼성전자", "attrs": {}, "score": 1.0},
        ],
        "graph_seeds": seeds,
        "graph_meta": {"n_seeds": len(seeds), "n_hits": 1, "fallback_used": False},
    }


def test_graph_node_drift_combines_local_and_community(monkeypatch):
    monkeypatch.setattr(graph_node, "_preflight", lambda: None)
    seeds = [{"key_type": "corp_code", "key_value": "00126380", "name": "삼성전자"}]
    monkeypatch.setattr(graph_node, "search", lambda q, upstream_seeds=None: _fake_search_output(seeds))

    captured = {}

    def _fake_global(query, anchor_corp_codes=None):
        captured["anchors"] = anchor_corp_codes
        return [{"type": "community", "code": "1", "name": "반도체군집", "value": "부분답", "extra": {"score": 80}, "source": "community:1"}]

    monkeypatch.setattr(graph_node, "global_search", _fake_global)

    out = graph_node.graph_search_node({"intent": "ctx", "reconstructed_query": "삼성전자 계열 구조"})

    assert "graph_facts" in out and out["graph_facts"]          # 로컬 사실 존재
    assert "community_results" in out                            # 글로벌 부분답 결합
    assert out["community_results"][0]["name"] == "반도체군집"
    assert captured["anchors"] == ["00126380"]                   # corp_code 앵커 전달


def test_graph_node_no_community_when_no_corp_code_anchor(monkeypatch):
    monkeypatch.setattr(graph_node, "_preflight", lambda: None)
    seeds = [{"key_type": "er_name", "key_value": "org:미해소", "name": "미해소사"}]
    monkeypatch.setattr(graph_node, "search", lambda q, upstream_seeds=None: _fake_search_output(seeds))

    def _must_not_call(*a, **k):
        raise AssertionError("앵커 없는데 global_search 호출됨")

    monkeypatch.setattr(graph_node, "global_search", _must_not_call)

    out = graph_node.graph_search_node({"intent": "ctx", "reconstructed_query": "미해소사"})
    assert "graph_facts" in out
    assert "community_results" not in out


def test_graph_node_global_intent_uses_unanchored_search(monkeypatch):
    captured = {}

    def _fake_global(query, anchor_corp_codes=None):
        captured["anchors"] = anchor_corp_codes
        captured["query"] = query
        return [{"type": "community", "code": "2", "name": "g", "value": "v", "extra": {}, "source": "community:2"}]

    monkeypatch.setattr(graph_node, "global_search", _fake_global)

    out = graph_node.graph_search_node({"intent": "global", "reconstructed_query": "반도체 업계 전반"})
    assert set(out.keys()) == {"community_results"}  # 로컬 검색 미수행
    assert out["community_results"][0]["code"] == "2"
    assert captured["anchors"] is None  # global 은 앵커 없이
