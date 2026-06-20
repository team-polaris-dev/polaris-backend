"""text-to-Cypher 생성기 — 오프라인(LLM/Neo4j monkeypatch).

flag off / supported=false / 정상 / EXPLAIN 실패 후 수리 / 수리 cap 초과를 검증한다.
_invoke_llm 과 _explain_ok 만 monkeypatch — 실제 LLM·Neo4j 호출 0.
"""
from __future__ import annotations

from config import graphrag as cfg
from graphrag import cypher_generator


_ANCHORS = [{"corp_code": "00126380", "name": "삼성전자"}]

_VALID_CYPHER = (
    "MATCH (anchor:Organization)-[r:SUPPLIES_TO]->(o:Organization) "
    "WHERE anchor.corp_code IN $anchors "
    "RETURN anchor.name AS from_name, o.name AS to_name, r.rcept_no AS source"
)


def _enable(monkeypatch):
    monkeypatch.setattr(cfg, "TEXT2CYPHER_ENABLED", True)
    monkeypatch.setattr(cfg, "TEXT2CYPHER_RESULT_LIMIT", 50)
    monkeypatch.setattr(cfg, "TEXT2CYPHER_REPAIR_MAX", 1)


def test_flag_off_returns_none(monkeypatch):
    monkeypatch.setattr(cfg, "TEXT2CYPHER_ENABLED", False)
    # flag off 면 LLM 을 호출조차 하지 않아야 한다.
    called = {"n": 0}

    def _llm(*a, **k):
        called["n"] += 1
        return {}

    monkeypatch.setattr(cypher_generator, "_invoke_llm", _llm)
    assert cypher_generator.generate("삼성전자 공급사", _ANCHORS) is None
    assert called["n"] == 0


def test_no_anchors_returns_none(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(cypher_generator, "_invoke_llm", lambda *a, **k: {})
    assert cypher_generator.generate("삼성전자 공급사", []) is None


def test_supported_false_returns_none(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(
        cypher_generator, "_invoke_llm",
        lambda *a, **k: {"supported": False, "cypher": "", "reason": "랭킹 질문"},
    )
    monkeypatch.setattr(cypher_generator, "_explain_ok", lambda *a, **k: True)
    assert cypher_generator.generate("매출 1위 회사", _ANCHORS) is None


def test_valid_cypher_returns_generated(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(
        cypher_generator, "_invoke_llm",
        lambda *a, **k: {"supported": True, "cypher": _VALID_CYPHER, "params": {}, "reason": "공급사 조회"},
    )
    monkeypatch.setattr(cypher_generator, "_explain_ok", lambda *a, **k: True)

    gen = cypher_generator.generate("삼성전자 공급사", _ANCHORS)
    assert gen is not None
    assert gen.reason == "공급사 조회"
    # $anchors 참조 보존(인라인 회사명 아님) + 결정성 주입.
    assert "$anchors" in gen.cypher
    assert "ORDER BY" in gen.cypher.upper()
    assert "LIMIT 50" in gen.cypher.upper()


def test_static_guard_failure_not_repaired(monkeypatch):
    _enable(monkeypatch)
    called = {"n": 0}

    def _llm(*a, **k):
        called["n"] += 1
        # 쓰기 절 포함 → 정적 가드 위반 → 수리 없이 즉시 None.
        return {"supported": True, "cypher": "MATCH (a:Organization) DELETE a", "reason": "x"}

    monkeypatch.setattr(cypher_generator, "_invoke_llm", _llm)
    monkeypatch.setattr(cypher_generator, "_explain_ok", lambda *a, **k: True)
    assert cypher_generator.generate("삼성전자 공급사", _ANCHORS) is None
    assert called["n"] == 1  # 정적 실패는 재호출(수리)하지 않는다.


def test_explain_fails_then_repair_succeeds(monkeypatch):
    _enable(monkeypatch)
    calls = {"n": 0}

    def _llm(query, repair_error=None):
        calls["n"] += 1
        return {"supported": True, "cypher": _VALID_CYPHER, "params": {}, "reason": "수리됨"}

    explain_results = iter([False, True])

    def _explain(cypher, params):
        return next(explain_results)

    monkeypatch.setattr(cypher_generator, "_invoke_llm", _llm)
    monkeypatch.setattr(cypher_generator, "_explain_ok", _explain)

    gen = cypher_generator.generate("삼성전자 공급사", _ANCHORS)
    assert gen is not None
    assert calls["n"] == 2  # 최초 + 수리 1회.


def test_explain_always_fails_returns_none_after_cap(monkeypatch):
    _enable(monkeypatch)
    calls = {"n": 0}

    def _llm(query, repair_error=None):
        calls["n"] += 1
        return {"supported": True, "cypher": _VALID_CYPHER, "params": {}, "reason": "여전히 실패"}

    monkeypatch.setattr(cypher_generator, "_invoke_llm", _llm)
    monkeypatch.setattr(cypher_generator, "_explain_ok", lambda *a, **k: False)

    assert cypher_generator.generate("삼성전자 공급사", _ANCHORS) is None
    # REPAIR_MAX=1 → 최초 + 수리 1회 = 2.
    assert calls["n"] == 2


def test_llm_down_returns_none(monkeypatch):
    _enable(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(cypher_generator, "_invoke_llm", _boom)
    assert cypher_generator.generate("삼성전자 공급사", _ANCHORS) is None
