"""GraphMode 라우터 + 노드 분기 (오프라인).

llm_planner 의 결정적 프리필터(5모드)·fallback·LLM 검증과, graph 노드의 silent 봉인·
relation_only degrade(시드 관계망 채우기) 를 monkeypatch 로 검증한다. 네트워크 0.
프리필터로 해소되는 질의만 써서 LLM 미호출(결정성)을 함께 보장한다.
"""
from __future__ import annotations

import pytest

from config.relations import has_rank_intent
from graphrag import llm_planner as lp
from graphrag import node as graph_node
from graphrag.llm_planner import GraphMode


# ── 결정적 프리필터: 5모드 + 위임(None) + 노이즈 ───────────────────────────

@pytest.mark.parametrize(
    "query, has_anchor, has_metric, expected",
    [
        ("삼성전자 매출은?", True, True, GraphMode.SILENT),                  # 속성전용 → 침묵
        ("삼성 계열사 중 매출 가장 높은 곳", True, True, GraphMode.RELATION_RANK),  # 랭크+관계+노드지표
        ("동진쎄미켐 가장 큰 공급처", True, False, GraphMode.RELATION_ONLY),   # 랭크+관계, 지표없음 → degrade
        ("삼성전자 협력사", True, False, GraphMode.RELATION_EXPLORE),         # 관계만+앵커
        ("국내 대기업 큰 그림", False, False, GraphMode.MACRO),               # 앵커없음+매크로cue
    ],
)
def test_prefilter_resolves_five_modes(query, has_anchor, has_metric, expected):
    assert lp._prefilter_mode(query, has_anchor=has_anchor, has_metric=has_metric) is expected


def test_prefilter_defers_ambiguous_relation_without_anchor():
    # 관계를 묻지만 엔티티 미해소 → 결정 불가, LLM 에 위임(None).
    assert lp._prefilter_mode("협력사 알려줘", has_anchor=False, has_metric=False) is None


def test_prefilter_noise_degrades_to_explore_not_llm():
    # 신호 없음(앵커X·cue X·관계어 X) → EXPLORE 로 확정해 LLM 미호출.
    # 노드는 시드 0 으로 빈 로컬을 내며 안전 degrade (n_seeds=0 계약 보존).
    assert (
        lp._prefilter_mode("존재하지않는회사명999XYZ", has_anchor=False, has_metric=False)
        is GraphMode.RELATION_EXPLORE
    )


def test_prefilter_structured_without_metric_is_relation_only():
    # 금액 랭킹류(노드 지표 없음) → 억지 1위 금지, ONLY 로 degrade.
    assert (
        lp._prefilter_mode("가장 큰 공급처", has_anchor=True, has_metric=False)
        is GraphMode.RELATION_ONLY
    )


# ── #1 MACRO 가 퍼지 앵커보다 우선(업종어 회사명 매칭으로 가로채이던 버그) ──────

def test_prefilter_macro_cue_beats_fuzzy_anchor():
    # "반도체 업종 공급망 전반" — '반도체'가 회사명에 퍼지매칭돼 has_anchor 가 켜져도
    # 매크로 cue('업종','전반')가 우선 → MACRO (예전엔 RELATION_EXPLORE 로 샜다).
    assert (
        lp._prefilter_mode("반도체 업종 공급망 전반", has_anchor=True, has_metric=False)
        is GraphMode.MACRO
    )


def test_prefilter_no_macro_cue_with_anchor_stays_explore():
    # 매크로 cue 없는 관계 질문 + 앵커 → 여전히 EXPLORE(회귀 없음).
    assert (
        lp._prefilter_mode("삼성전자 협력사", has_anchor=True, has_metric=False)
        is GraphMode.RELATION_EXPLORE
    )


# ── #2 '최대'/'최다' 랭크어 + '최대주주' 복합어 가드 ─────────────────────────

@pytest.mark.parametrize(
    "query, expected",
    [
        ("가장 큰 곳", True),
        ("매출 최대 기업", True),
        ("최다 거래처", True),
        ("삼성전자 최대주주는?", False),   # 복합어 — 랭킹 아님
        ("최대 주주 알려줘", False),       # 띄어쓴 복합어
        ("대주주 현황", False),
        ("삼성전자 협력사", False),        # 랭크어 없음
    ],
)
def test_has_rank_intent_handles_choedaejuju_compound(query, expected):
    assert has_rank_intent(query) is expected


def test_prefilter_choedae_supplier_degrades_to_relation_only():
    # #2 핵심: "최대 공급처" 가 이제 랭크+관계로 잡혀 degrade(예전엔 EXPLORE 로 샜다).
    assert (
        lp._prefilter_mode("동진쎄미켐 최대 공급처", has_anchor=True, has_metric=False)
        is GraphMode.RELATION_ONLY
    )


def test_prefilter_choedaejuju_is_explore_not_only():
    # '최대주주' 는 랭킹이 아니라 지분구조 탐색 → EXPLORE(억지 degrade 회귀 방지).
    assert (
        lp._prefilter_mode("삼성전자 최대주주는?", has_anchor=True, has_metric=False)
        is GraphMode.RELATION_EXPLORE
    )


# ── _fallback_mode: 결정적 백스톱 ──────────────────────────────────────────

def test_fallback_silent_for_attribute_only():
    assert lp._fallback_mode("삼성전자 매출은?", has_anchor=True, has_metric=True) is GraphMode.SILENT


def test_fallback_macro_only_with_cue_when_no_anchor():
    assert lp._fallback_mode("업계 동향", has_anchor=False, has_metric=False) is GraphMode.MACRO


def test_fallback_explore_when_no_anchor_no_cue():
    # 앵커 미해소 + 매크로 cue 없음 → MACRO 가 아니라 EXPLORE(빈 로컬 degrade).
    # MACRO 면 global_search 가 강제돼 n_seeds=0 결과를 덮어쓰는 회귀가 난다.
    assert (
        lp._fallback_mode("존재하지않는회사명999XYZ", has_anchor=False, has_metric=False)
        is GraphMode.RELATION_EXPLORE
    )


# ── _coerce_mode: LLM 출력 검증 ────────────────────────────────────────────

def test_coerce_mode_accepts_valid_enum():
    data = {"supported": True, "mode": "relation_explore"}
    assert (
        lp._coerce_mode(data, "협력사 알려줘", has_anchor=True, has_metric=False)
        is GraphMode.RELATION_EXPLORE
    )


def test_coerce_mode_unsupported_falls_back():
    data = {"supported": False, "mode": "macro"}
    # supported:false → 결정적 fallback. 앵커 있음 → EXPLORE.
    assert (
        lp._coerce_mode(data, "협력사 알려줘", has_anchor=True, has_metric=False)
        is GraphMode.RELATION_EXPLORE
    )


def test_coerce_mode_invalid_enum_falls_back():
    data = {"supported": True, "mode": "freeform_cypher"}
    assert (
        lp._coerce_mode(data, "업계 동향", has_anchor=False, has_metric=False)
        is GraphMode.MACRO
    )


def test_coerce_mode_relation_rank_without_metric_degrades():
    data = {"supported": True, "mode": "relation_rank"}
    assert (
        lp._coerce_mode(data, "동진쎄미켐 가장 큰 공급처", has_anchor=True, has_metric=False)
        is GraphMode.RELATION_ONLY
    )


# ── plan_with_mode: LLM off / 다운 → degrade ───────────────────────────────

def test_plan_with_mode_llm_off_uses_fallback(monkeypatch):
    monkeypatch.setattr(lp, "_ENABLED", False)

    def _must_not_call(_q):
        raise AssertionError("LLM off 인데 호출됨")

    monkeypatch.setattr(lp, "_invoke_llm", _must_not_call)
    # 프리필터가 None 인 애매 질의도 LLM off → fallback(앵커 없음·cue 없음 → EXPLORE).
    mode, plan = lp.plan_with_mode("협력사 알려줘", has_anchor=False, has_metric=False)
    assert mode is GraphMode.RELATION_EXPLORE
    assert plan is None


def test_plan_with_mode_llm_down_degrades(monkeypatch):
    monkeypatch.setattr(lp, "_ENABLED", True)

    def _raise(_q):
        raise RuntimeError("LLM 다운")

    monkeypatch.setattr(lp, "_invoke_llm", _raise)
    mode, plan = lp.plan_with_mode("협력사 알려줘", has_anchor=True, has_metric=False)
    assert mode is GraphMode.RELATION_EXPLORE  # 앵커 있음 → 시드 관계망
    assert plan is None


def test_plan_with_mode_prefilter_skips_llm(monkeypatch):
    def _must_not_call(_q):
        raise AssertionError("프리필터로 해소됐는데 LLM 호출됨")

    monkeypatch.setattr(lp, "_invoke_llm", _must_not_call)
    mode, plan = lp.plan_with_mode("삼성전자 협력사", has_anchor=True, has_metric=False)
    assert mode is GraphMode.RELATION_EXPLORE
    assert plan is None


# ── 노드: silent 봉인 ──────────────────────────────────────────────────────

def test_node_silent_returns_anchor_stub(monkeypatch):
    monkeypatch.setattr(graph_node, "_preflight", lambda: None)
    monkeypatch.setattr(
        graph_node, "match",
        lambda q, upstream_seeds=None: [
            {"label": "organization", "id": "00126380", "key_value": "00126380", "name": "삼성전자", "score": 1.0}
        ],
    )

    def _must_not_call(*a, **k):
        raise AssertionError("silent 인데 search() 호출됨")

    monkeypatch.setattr(graph_node, "search", _must_not_call)

    out = graph_node.graph_search_node({"intent": "ctx", "reconstructed_query": "삼성전자 매출은?"})

    assert out["graph_meta"]["mode"] == "silent"
    assert len(out["graph_hits"]) == 1
    assert out["graph_hits"][0]["attrs"]["silent"] is True
    # 셰임 계약: graph_facts 길이 1 → empty_sources 게이트 통과(스텁).
    assert len(out["graph_facts"]) == len(out["graph_hits"])


def test_node_silent_empty_when_no_org_anchor(monkeypatch):
    monkeypatch.setattr(graph_node, "_preflight", lambda: None)
    monkeypatch.setattr(graph_node, "match", lambda q, upstream_seeds=None: [])

    def _must_not_call(*a, **k):
        raise AssertionError("silent 인데 search() 호출됨")

    monkeypatch.setattr(graph_node, "search", _must_not_call)

    out = graph_node.graph_search_node({"intent": "ctx", "reconstructed_query": "삼성전자 매출 얼마?"})
    assert out == {}  # 앵커 미해소 → 그래프 기여 없음(fail-closed)


def test_node_silent_match_failure_is_fail_closed(monkeypatch):
    monkeypatch.setattr(graph_node, "_preflight", lambda: None)

    def _raise(*a, **k):
        raise RuntimeError("matcher 다운")

    monkeypatch.setattr(graph_node, "match", _raise)
    out = graph_node.graph_search_node({"intent": "ctx", "reconstructed_query": "삼성전자 자산은?"})
    assert out == {}


# ── 노드: relation_only degrade (시드 관계망 채우기) ────────────────────────

def _abstain_output():
    seed = {
        "label": "organization", "id": "org:동진", "key_type": "er_name",
        "key_value": "org:동진", "name": "동진쎄미켐", "score": 1.0,
    }
    hit = {
        "id": "org:동진", "label": "organization", "name": "동진쎄미켐",
        "attrs": {"structured_role": "anchor", "abstained": True},
        "score": 1.0, "seed_origin": "structured",
    }
    return {
        "graph_hits": [hit],
        "graph_seeds": [seed],
        "graph_meta": {
            "mode": "structured",
            "structured": {"mode": "structured", "abstained": True, "answer_edges": []},
            "n_seeds": 1, "n_hits": 1, "fallback_used": False, "errors": [],
        },
    }


def _fallback_hits(_seed):
    return [
        {"id": "rel:1", "label": "relationship", "name": "공급",
         "attrs": {"rel_type": "SUPPLIES_TO", "src": "org:동진", "dst": "co:A"}, "score": 0.5},
        {"id": "co:A", "label": "organization", "name": "협력사A", "attrs": {}, "score": 0.5},
    ]


def test_node_relation_only_fills_seed_network_and_signals(monkeypatch):
    monkeypatch.setattr(graph_node, "_preflight", lambda: None)
    monkeypatch.setattr(graph_node, "search", lambda q, upstream_seeds=None: _abstain_output())
    monkeypatch.setattr(graph_node, "fallback_for", _fallback_hits)

    def _must_not_call(*a, **k):
        raise AssertionError("er_name 앵커(코드 없음)인데 global_search 호출됨")

    monkeypatch.setattr(graph_node, "global_search", _must_not_call)

    out = graph_node.graph_search_node(
        {"intent": "ctx", "reconstructed_query": "동진쎄미켐 가장 큰 공급처"}
    )

    # degrade 신호는 graph_meta 에만(셰임 계약 미파괴).
    assert out["graph_meta"]["rankable"] is False
    assert out["graph_meta"]["degrade_reason"]
    assert out["graph_meta"]["fallback_used"] is True
    # 시드 관계망이 채워졌다(앵커 스텁 + fallback_for hits).
    ids = {h["id"] for h in out["graph_hits"]}
    assert {"rel:1", "co:A"} <= ids
    # 셰임 1:1 계약 유지.
    assert len(out["graph_facts"]) == len(out["graph_hits"])


def test_node_relation_explore_assembles_without_degrade(monkeypatch):
    # 관계 탐색(랭크 없음·앵커 있음) → relation_explore, degrade 신호 없음.
    monkeypatch.setattr(graph_node, "_preflight", lambda: None)
    seeds = [{"key_type": "er_name", "key_value": "org:삼성", "label": "organization", "name": "삼성전자"}]
    out_search = {
        "graph_hits": [
            {"id": "rel:9", "label": "relationship", "name": "협력",
             "attrs": {"rel_type": "SUPPLIES_TO"}, "score": 0.7},
        ],
        "graph_seeds": seeds,
        "graph_meta": {"n_seeds": 1, "n_hits": 1, "fallback_used": False, "errors": []},
    }
    monkeypatch.setattr(graph_node, "search", lambda q, upstream_seeds=None: out_search)
    out = graph_node.graph_search_node({"intent": "ctx", "reconstructed_query": "삼성전자 협력사"})
    assert "rankable" not in out["graph_meta"]
    assert len(out["graph_facts"]) == len(out["graph_hits"])


def test_node_macro_without_anchor_routes_to_community(monkeypatch):
    monkeypatch.setattr(graph_node, "_preflight", lambda: None)
    # 앵커 미해소 매크로 → community map-reduce.
    monkeypatch.setattr(
        graph_node, "search",
        lambda q, upstream_seeds=None: {
            "graph_hits": [], "graph_seeds": [],
            "graph_meta": {"n_seeds": 0, "n_hits": 0, "fallback_used": False, "errors": []},
        },
    )
    monkeypatch.setattr(
        graph_node, "global_search",
        lambda q, anchor_corp_codes=None: [
            {"type": "community", "code": "1", "name": "g", "value": "v", "extra": {}, "source": "community:1"}
        ],
    )
    out = graph_node.graph_search_node({"intent": "ctx", "reconstructed_query": "반도체 업계 전반 큰 그림"})
    assert set(out.keys()) == {"community_results"}
    assert out["community_results"][0]["code"] == "1"


def test_node_macro_with_fuzzy_anchor_still_routes_to_community(monkeypatch):
    # #1 핵심: corp_code 앵커가 잡혔지만 회사명이 질의에 없으면(업종어 퍼지매칭) 명시 호명이
    # 아니므로 MACRO 유지 → community map-reduce. 예전엔 앵커가 있으면 EXPLORE 로 샜다.
    monkeypatch.setattr(graph_node, "_preflight", lambda: None)
    seeds = [{"key_type": "corp_code", "key_value": "00126380",
              "label": "organization", "name": "삼성디스플레이"}]
    monkeypatch.setattr(
        graph_node, "search",
        lambda q, upstream_seeds=None: {
            "graph_hits": [], "graph_seeds": seeds,
            "graph_meta": {"n_seeds": 1, "n_hits": 0, "fallback_used": False, "errors": []},
        },
    )
    monkeypatch.setattr(
        graph_node, "global_search",
        lambda q, anchor_corp_codes=None: [
            {"type": "community", "code": "9", "name": "g", "value": "v", "extra": {}, "source": "community:9"}
        ],
    )
    out = graph_node.graph_search_node(
        {"intent": "ctx", "reconstructed_query": "디스플레이 업종 전반 큰 그림"}
    )
    assert set(out.keys()) == {"community_results"}
    assert out["community_results"][0]["code"] == "9"


def test_node_macro_with_explicit_company_keeps_drift(monkeypatch):
    # 명시 호명("삼성전자")이면 매크로 cue 가 있어도 그 회사 관계망 + DRIFT 를 보존한다.
    # 구분 신호: _attach_communities 경유는 global_search 에 corp_code 앵커를 넘긴다(매크로
    # 조기반환은 앵커 없이 호출). captured anchors 로 DRIFT 경로를 탔음을 확인.
    monkeypatch.setattr(graph_node, "_preflight", lambda: None)
    seeds = [{"key_type": "corp_code", "key_value": "00126380",
              "label": "organization", "name": "삼성전자"}]
    monkeypatch.setattr(
        graph_node, "search",
        lambda q, upstream_seeds=None: {
            "graph_hits": [{"id": "00126380", "label": "organization", "name": "삼성전자",
                            "attrs": {}, "score": 1.0}],
            "graph_seeds": seeds,
            "graph_meta": {"n_seeds": 1, "n_hits": 1, "fallback_used": False, "errors": []},
        },
    )
    captured = {}

    def _fake_global(query, anchor_corp_codes=None):
        captured["anchors"] = anchor_corp_codes
        return [{"type": "community", "code": "0", "name": "삼성군집", "value": "부분답",
                 "extra": {"score": 70}, "source": "community:0"}]

    monkeypatch.setattr(graph_node, "global_search", _fake_global)

    out = graph_node.graph_search_node(
        {"intent": "ctx", "reconstructed_query": "삼성전자 시장 전반 큰 그림"}
    )
    assert "graph_facts" in out and out["graph_facts"]   # 명시 호명 → 로컬 관계망 보존
    assert "community_results" in out                      # DRIFT 결합
    assert captured["anchors"] == ["00126380"]            # _attach_communities 경유(앵커 전달)


# ── effective_query: 재구성이 그룹 군집 의도를 좁히면 원문 채택 (graph+gen 공유) ──

class _HumanMsg:
    """_last_human_text 가 보는 최소 메시지(type=='human', .content)."""
    type = "human"

    def __init__(self, content: str):
        self.content = content


def test_graph_query_reverts_when_reconstruction_narrows_group_scope():
    # 재구성이 "삼성 계열사"(군집 랭킹)를 "종속회사"(single_hop)로 좁히면 원문 채택.
    state = {
        "messages": [_HumanMsg("삼성 계열사 중 매출 가장 높은 곳은?")],
        "reconstructed_query": "삼성전자의 종속회사 중 매출액이 가장 높은 회사는 어디입니까?",
    }
    assert graph_node.effective_query(state) == "삼성 계열사 중 매출 가장 높은 곳은?"


def test_graph_query_keeps_reconstruction_when_group_intent_preserved():
    # 재구성이 군집 랭킹 의도를 유지하면(둘 다 community_member_rank) 재구성 채택.
    state = {
        "messages": [_HumanMsg("삼성 계열사 중 매출 가장 높은 곳은?")],
        "reconstructed_query": "삼성 그룹 계열사 중 매출액이 가장 높은 회사는?",
    }
    assert graph_node.effective_query(state) == "삼성 그룹 계열사 중 매출액이 가장 높은 회사는?"


def test_graph_query_keeps_reconstruction_for_genuine_subsidiary():
    # 원문이 애초에 자회사 랭킹(single_hop)이면 좁힘이 아니므로 재구성 그대로.
    state = {
        "messages": [_HumanMsg("삼성전자 자회사 중 매출 1위는?")],
        "reconstructed_query": "삼성전자(주)의 자회사 중 매출액이 가장 높은 회사는?",
    }
    assert graph_node.effective_query(state) == "삼성전자(주)의 자회사 중 매출액이 가장 높은 회사는?"


def test_graph_query_returns_reconstruction_when_no_original():
    # human 메시지 없음 → 재구성 그대로(폴백 경로 안전).
    state = {"messages": [], "reconstructed_query": "삼성전자 협력사"}
    assert graph_node.effective_query(state) == "삼성전자 협력사"
