"""search() text-to-Cypher 통합 seam — 오프라인.

_try_text2cypher 가 플래그·organization 앵커·생성/실행 결과에 따라 정확히 분기하고,
성공 시 구조화 조기반환과 동일한 모양(graph_hits/graph_seeds/graph_meta +
latency_ms/n_seeds/errors)으로 패키징하는지, 플래그 off·abstain 이면 폴백하도록
None 을 돌리는지 검증한다. match·생성기·실행기 seam 만 monkeypatch — 네트워크 0.
"""
from __future__ import annotations

from graphrag import search
from graphrag.cypher_generator import GeneratedCypher


def _org_seed():
    return {"label": "organization", "id": "00126256", "key_type": "corp_code",
            "key_value": "00126256", "name": "삼성생명", "score": 1.0}


def _gen():
    return GeneratedCypher(cypher="MATCH (anchor:Organization) WHERE anchor.corp_code "
                           "IN $anchors RETURN anchor.name AS from_name", params={}, reason="x")


def _meta():
    return {
        "mode": "structured",
        "structured": {"mode": "structured", "kind": "text2cypher", "abstained": False},
        "patterns_run": ["text2cypher"],
        "n_hits": 1,
        "fallback_used": False,
        "errors": [],
    }


def _hits():
    return [{"id": "x", "label": "organization", "name": "삼성전자",
             "attrs": {"structured_role": "sibling"}, "score": 1.0}]


# ── _try_text2cypher: 분기 ────────────────────────────────────────

def test_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(search, "TEXT2CYPHER_ENABLED", False)
    assert search._try_text2cypher("q", [_org_seed()], [], 0.0) is None


def test_no_org_seed_returns_none(monkeypatch):
    monkeypatch.setattr(search, "TEXT2CYPHER_ENABLED", True)
    product = {"label": "product", "id": "p1", "name": "매출채권"}
    assert search._try_text2cypher("q", [product], [], 0.0) is None


def test_generate_abstains_returns_none(monkeypatch):
    monkeypatch.setattr(search, "TEXT2CYPHER_ENABLED", True)
    monkeypatch.setattr(search, "generate_cypher", lambda q, anchors: None)
    assert search._try_text2cypher("q", [_org_seed()], [], 0.0) is None


def test_executor_none_returns_none(monkeypatch):
    monkeypatch.setattr(search, "TEXT2CYPHER_ENABLED", True)
    monkeypatch.setattr(search, "generate_cypher", lambda q, anchors: _gen())
    monkeypatch.setattr(search, "run_cypher", lambda g, anchors, q: None)
    assert search._try_text2cypher("q", [_org_seed()], [], 0.0) is None


def test_success_packages_structured_output(monkeypatch):
    monkeypatch.setattr(search, "TEXT2CYPHER_ENABLED", True)
    monkeypatch.setattr(search, "generate_cypher", lambda q, anchors: _gen())
    monkeypatch.setattr(search, "run_cypher", lambda g, anchors, q: (_hits(), _meta()))

    out = search._try_text2cypher("q", [_org_seed()], [], 0.0)
    assert out is not None
    assert out["graph_meta"]["structured"]["kind"] == "text2cypher"
    assert out["graph_meta"]["n_seeds"] == 1
    assert "latency_ms" in out["graph_meta"]
    assert out["graph_hits"] == _hits()


# ── search(): 진입점에서 우선 시도 / 폴백 ──────────────────────────

def test_search_uses_text2cypher_when_enabled(monkeypatch):
    monkeypatch.setattr(search, "match", lambda q, upstream_seeds=None: [_org_seed()])
    monkeypatch.setattr(search, "TEXT2CYPHER_ENABLED", True)
    monkeypatch.setattr(search, "generate_cypher", lambda q, anchors: _gen())
    monkeypatch.setattr(search, "run_cypher", lambda g, anchors, q: (_hits(), _meta()))

    out = search.search("삼성생명 형제 계열사")
    assert out["graph_meta"]["structured"]["kind"] == "text2cypher"


def test_search_flag_off_skips_generation(monkeypatch):
    # 플래그 off → 생성기 호출조차 안 됨(우선 시도 스킵). 호출되면 RuntimeError 로 잡는다.
    monkeypatch.setattr(search, "match", lambda q, upstream_seeds=None: [_org_seed()])
    monkeypatch.setattr(search, "TEXT2CYPHER_ENABLED", False)

    def _boom(*a, **k):
        raise RuntimeError("generate_cypher must not be called when flag is off")

    monkeypatch.setattr(search, "generate_cypher", _boom)
    # 폴백 경로(planner→structured/execute)는 결정적·offline. 호출 자체가 예외 없이 진행되면 통과.
    monkeypatch.setattr(search, "plan_structured", lambda q: None)
    monkeypatch.setattr(search, "plan_structured_llm", lambda q: None)
    monkeypatch.setattr(search, "expand_ppr", lambda seeds: ([], []))
    monkeypatch.setattr(search, "expand", lambda seeds: ([], []))
    monkeypatch.setattr(search, "fallback_for", lambda sd: [])

    out = search.search("삼성생명 형제 계열사")
    assert out["graph_meta"].get("structured", {}).get("kind") != "text2cypher"
