"""Layer A: Neo4j 공식 Text2CypherRetriever 래퍼 — 오프라인(retriever/LLM monkeypatch).

flag off / 앵커 없음 / 정상 / 화이트리스트 위반 / retriever 예외 / 앵커 인젝션 차단 /
LLMBase 어댑터 content 매핑을 검증한다. 실제 Neo4j·LLM 호출 0.
"""
from __future__ import annotations

from types import SimpleNamespace

from config import graphrag as cfg
from graphrag import text2cypher as t2c


_WHITELISTED = (
    "MATCH (cand:Organization)-[r:SUPPLIES_TO]->(anchor:Organization) "
    "WHERE anchor.corp_code IN ['00126380'] "
    "RETURN coalesce(cand.corp_code, cand.er_name, cand.name) AS from_id, "
    "cand.name AS from_name, "
    "coalesce(anchor.corp_code, anchor.er_name, anchor.name) AS to_id, "
    "anchor.name AS to_name, type(r) AS rel_type, "
    "coalesce(r.rcept_no,'') AS source, coalesce(r.chunk_id,'') AS chunk_id, "
    "'supplier' AS from_role, 'anchor' AS to_role"
)


class _Rec:
    def __init__(self, d: dict) -> None:
        self._d = d

    def data(self) -> dict:
        return self._d


class _FakeRetriever:
    def __init__(self, cypher: str, rows: list[dict]) -> None:
        self._cypher = cypher
        self._rows = rows
        self.calls: list[tuple] = []

    def get_search_results(self, query_text, prompt_params=None):
        self.calls.append((query_text, prompt_params))
        return SimpleNamespace(
            metadata={"cypher": self._cypher},
            records=[_Rec(r) for r in self._rows],
        )


def _enable(monkeypatch):
    monkeypatch.setattr(cfg, "TEXT2CYPHER_ENABLED", True)


def test_flag_off_returns_none(monkeypatch):
    monkeypatch.setattr(cfg, "TEXT2CYPHER_ENABLED", False)
    called = {"n": 0}
    monkeypatch.setattr(t2c, "_build_retriever", lambda: called.__setitem__("n", called["n"] + 1))
    assert t2c.run_relationship_query("삼성전자 공급사", ["00126380"]) is None
    assert called["n"] == 0  # flag off 면 retriever 를 만들지도 않는다.


def test_no_anchors_returns_none(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(t2c, "_build_retriever", lambda: (_ for _ in ()).throw(AssertionError("unreachable")))
    assert t2c.run_relationship_query("삼성전자 공급사", []) is None


def test_valid_returns_rows_and_cypher(monkeypatch):
    _enable(monkeypatch)
    row = {"from_id": "x", "from_name": "동진", "to_id": "00126380", "to_name": "삼성전자",
           "rel_type": "SUPPLIES_TO", "source": "r1", "chunk_id": "c1",
           "from_role": "supplier", "to_role": "anchor"}
    fake = _FakeRetriever(_WHITELISTED, [row])
    monkeypatch.setattr(t2c, "_build_retriever", lambda: fake)

    out = t2c.run_relationship_query("삼성전자 공급사", ["00126380", "00126380"])
    assert out is not None
    rows, cypher = out
    assert rows == [row]
    assert cypher == _WHITELISTED
    # 앵커는 corp_code 리터럴 리스트로 prompt_params 에 주입된다(중복 제거).
    _, prompt_params = fake.calls[0]
    assert prompt_params == {"anchors": "['00126380']"}


def test_whitelist_violation_returns_none(monkeypatch):
    _enable(monkeypatch)
    bad = "MATCH (a:Person)-[r:OWNS]->(b:Organization) RETURN a.name AS from_name"
    monkeypatch.setattr(t2c, "_build_retriever", lambda: _FakeRetriever(bad, [{"x": 1}]))
    assert t2c.run_relationship_query("삼성전자 공급사", ["00126380"]) is None


def test_retriever_exception_returns_none(monkeypatch):
    _enable(monkeypatch)

    class _Boom:
        def get_search_results(self, **kwargs):
            raise RuntimeError("neo4j down")

    monkeypatch.setattr(t2c, "_build_retriever", lambda: _Boom())
    assert t2c.run_relationship_query("삼성전자 공급사", ["00126380"]) is None


def test_anchor_literal_blocks_injection():
    # corp_code 가 아닌 인젝션 토큰은 제거되고 영숫자(+하이픈) 코드만 리터럴로 통과.
    out = t2c._anchors_literal(["00126380", "bad'; DROP", "", "00126380", "12-34"])
    assert out == "['00126380', '12-34']"


def test_adapter_maps_content_to_llm_response(monkeypatch):
    sent = {}

    class _Stub:
        def invoke(self, x):
            sent["x"] = x
            return SimpleNamespace(content="MATCH (a:Organization) RETURN a")

    monkeypatch.setattr("config.llm.llm", _Stub(), raising=False)
    adapter = t2c._make_adapter()
    resp = adapter.invoke("프롬프트")
    assert resp.content == "MATCH (a:Organization) RETURN a"
    assert sent["x"] == "프롬프트"
