"""통합 질문종류 분류기 router.classify — 오프라인.

결정적 프리필터(silent·보존 kind)는 LLM 미호출, 애매·체인·일반 모드는 LLM 1차 판정,
LLM off·다운·무효는 키워드 폴백으로 강등됨을 검증한다. _invoke_llm seam 만 monkeypatch —
네트워크 0. 핵심 회귀: 키워드 게이트(_looks_chain)가 막던 느슨한 표현도 LLM 이 chain 으로
분류함을 보인다.
"""
from __future__ import annotations

from graphrag import chain_planner, graph_mode, router


def _boom(_q):
    raise AssertionError("프리필터로 해소됐는데 LLM 이 호출됨")


def _two_hops():
    hop = {
        "relation": {"rel_type": "SUPPLIES_TO", "direction": "incoming"},
        "rank_metric": "ifrs-full_Revenue",
        "top_n": 3,
        "policy": "default",
    }
    return [dict(hop), dict(hop)]


# ── 결정적 프리필터: LLM 미호출 ────────────────────────────────────────────

def test_prefilter_silent_skips_llm(monkeypatch):
    monkeypatch.setattr(router, "_invoke_llm", _boom)
    r = router.classify("삼성전자 매출은?", has_metric=True)
    assert r.type == "silent"
    assert r.source == "prefilter"


def test_prefilter_preserved_kind_skips_llm(monkeypatch):
    monkeypatch.setattr(router, "_invoke_llm", _boom)
    r = router.classify("삼성 계열사 중 매출 가장 높은 곳", has_metric=True)
    assert r.type == "community_member_rank"
    assert r.source == "prefilter"
    assert r.plan is not None and r.plan.kind == "community_member_rank"


# ── LLM 1차 판정 ───────────────────────────────────────────────────────────

def test_llm_chain_extracts_hops(monkeypatch):
    monkeypatch.setattr(
        router, "_invoke_llm",
        lambda q: {"supported": True, "question_type": "chain",
                   "hops": _two_hops(), "reason": "수혜의 수혜"},
    )
    r = router.classify(
        "SK하이닉스가 오르면 수혜 기업, 그곳이 또 오르면 수혜 기업 3개", has_metric=True
    )
    assert r.type == "chain"
    assert r.source == "llm"
    assert r.plan is not None and r.plan.kind == "multi_hop_chain"
    assert len(r.plan.hops) == 2


def test_llm_relation_rank_demotes_to_only_without_metric(monkeypatch):
    monkeypatch.setattr(
        router, "_invoke_llm",
        lambda q: {"supported": True, "question_type": "relation_rank", "hops": []},
    )
    r = router.classify("동진쎄미켐 가장 큰 공급처", has_metric=False)
    assert r.type == "relation_only"


def test_llm_relation_explore_passthrough(monkeypatch):
    monkeypatch.setattr(
        router, "_invoke_llm",
        lambda q: {"supported": True, "question_type": "relation_explore", "hops": []},
    )
    r = router.classify("엔비디아 관련 회사 보여줘", has_metric=False)
    assert r.type == "relation_explore"
    assert r.source == "llm"


def test_llm_chain_with_invalid_hops_falls_back(monkeypatch):
    # chain 이라 했지만 홉 1개(무효) → None → 폴백(체인 아님).
    monkeypatch.setattr(
        router, "_invoke_llm",
        lambda q: {"supported": True, "question_type": "chain", "hops": [_two_hops()[0]]},
    )
    r = router.classify("엔비디아 협력사 보여줘", has_metric=False)
    assert r.type != "chain"
    assert r.source == "fallback"


def test_llm_invalid_type_falls_back(monkeypatch):
    monkeypatch.setattr(
        router, "_invoke_llm",
        lambda q: {"supported": True, "question_type": "freeform_cypher", "hops": []},
    )
    r = router.classify("삼성전자 협력사", has_metric=False)
    assert r.type == "relation_explore"
    assert r.source == "fallback"


# ── 폴백: LLM off / 다운 ───────────────────────────────────────────────────

def test_llm_off_uses_fallback(monkeypatch):
    monkeypatch.setattr(router, "_ENABLED", False)
    monkeypatch.setattr(router, "_invoke_llm", _boom)
    r = router.classify("삼성전자 협력사", has_metric=False)
    assert r.type == "relation_explore"
    assert r.source == "fallback"


def test_llm_down_uses_fallback(monkeypatch):
    monkeypatch.setattr(router, "_ENABLED", True)

    def _raise(_q):
        raise RuntimeError("LLM 다운")

    monkeypatch.setattr(router, "_invoke_llm", _raise)
    r = router.classify("삼성전자 협력사", has_metric=False)
    assert r.type == "relation_explore"
    assert r.source == "fallback"


# ── 핵심 회귀: 키워드 게이트가 막던 느슨한 표현도 LLM 이 chain 으로 ───────────

def test_loosened_chain_phrasing_reaches_llm_chain(monkeypatch):
    # _looks_chain 은 전파어(수혜/낙수…)+반복어(또/거기서…)를 모두 요구 → 아래 표현엔 전파어가
    # 없어 게이트가 막았다(단일 홉으로 샘). 통합 라우터는 프리필터가 가로채지 않고 LLM 에
    # 위임하므로, LLM 이 chain 으로 보면 chain 으로 분류된다.
    q = "엔비디아가 잘되면 다음으로 이득 보는 회사, 그 회사가 잘되면 이득 보는 3곳"
    assert chain_planner._looks_chain(q) is False  # 옛 키워드 게이트는 놓쳤다
    monkeypatch.setattr(
        router, "_invoke_llm",
        lambda _q: {"supported": True, "question_type": "chain",
                    "hops": _two_hops(), "reason": "단계적 수혜"},
    )
    r = router.classify(q, has_metric=True)
    assert r.type == "chain"
    assert r.source == "llm"


# ── 회귀 평가셋: 키워드 게이트가 가로채던 표현을 LLM 이 바르게 분류 ─────────────
# 통합 전엔 결정적 게이트가 이 질문들을 엉뚱한 모드로 확정해 LLM 이 호출조차 안 됐다.
# 통합 라우터는 프리필터(silent·보존 kind)만 fast-path 하고 나머지는 LLM 에 위임하므로,
# LLM 이 옳게 보면 옳게 분류된다. 아래는 라이브 스모크에서 실제로 회귀했던 두 케이스의
# 오프라인 계약 가드(프리필터가 가로채지 않고 위임함 + LLM 결과를 그대로 채택함).

def test_executive_question_delegates_to_llm_not_silent(monkeypatch):
    # "대표이사" 는 스칼라 속성이 아니라 인물 관계 → silent 로 프리필터되면 안 되고(위임),
    # LLM 이 relation_explore 로 보면 그대로 채택돼야 한다(예전 라이브 회귀: silent 로 침묵).
    q = "삼성전자 대표이사가 누구야?"
    assert graph_mode._is_silent(q) is False  # 프리필터가 silent 로 가로채지 않는다
    monkeypatch.setattr(
        router, "_invoke_llm",
        lambda _q: {"supported": True, "question_type": "relation_explore", "hops": []},
    )
    r = router.classify(q, has_metric=False)
    assert r.type == "relation_explore"
    assert r.source == "llm"


def test_product_manufacturer_delegates_to_llm_not_macro(monkeypatch):
    # 특정 제품의 제조사("HBM 제조사")는 산업 전반(macro)이 아니라 제품→생산기업 관계다.
    # 프리필터는 macro 를 확정하지 않으므로(위임), LLM 이 relation_explore 로 보면 채택된다
    # (예전 라이브 회귀: macro 로 새 community map-reduce 만 돌고 graph_hits 가 비었다).
    monkeypatch.setattr(
        router, "_invoke_llm",
        lambda _q: {"supported": True, "question_type": "relation_explore", "hops": []},
    )
    r = router.classify("HBM 제조사", has_metric=False)
    assert r.type == "relation_explore"
    assert r.source == "llm"
