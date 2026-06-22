"""search() 공식 text-to-Cypher 통합 seam — 오프라인.

_try_text2cypher 가 플래그·organization 앵커·앵커 corp_code·생성/매핑 결과에 따라
정확히 분기하고, 성공 시 구조화 조기반환과 동일한 모양(graph_hits/graph_seeds/
graph_meta + latency_ms/n_seeds/errors)으로 패키징하는지, 플래그 off·abstain 이면
폴백하도록 None 을 돌리는지 검증한다. match·retriever·매퍼 seam 만 monkeypatch — 네트워크 0.
"""
from __future__ import annotations

from graphrag import cypher_executor, search


def _org_seed():
    return {"label": "organization", "id": "00126256", "key_type": "corp_code",
            "key_value": "00126256", "name": "삼성생명", "score": 1.0}


_CYPHER = ("MATCH (anchor:Organization) WHERE anchor.corp_code "
           "IN ['00126256'] RETURN anchor.name AS from_name")


def _produced():
    """run_relationship_query 반환 모양: (rows, cypher)."""
    rows = [{"from_id": "x", "from_name": "삼성전자", "to_id": "00126256",
             "to_name": "삼성생명", "rel_type": "IS_MAJOR_SHAREHOLDER_OF",
             "source": "r1", "chunk_id": "", "from_role": "sibling", "to_role": "anchor"}]
    return rows, _CYPHER


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


def test_no_anchor_code_returns_none(monkeypatch):
    # organization 앵커지만 corp_code 가 없으면(이름만) 앵커 인라인 불가 → 폴백.
    monkeypatch.setattr(search, "TEXT2CYPHER_ENABLED", True)
    nameless = {"label": "organization", "id": "삼성", "name": "삼성"}
    assert search._try_text2cypher("q", [nameless], [], 0.0) is None


def test_retriever_abstains_returns_none(monkeypatch):
    monkeypatch.setattr(search, "TEXT2CYPHER_ENABLED", True)
    monkeypatch.setattr(search, "run_relationship_query", lambda q, codes: None)
    assert search._try_text2cypher("q", [_org_seed()], [], 0.0) is None


def test_mapper_none_returns_none(monkeypatch):
    monkeypatch.setattr(search, "TEXT2CYPHER_ENABLED", True)
    monkeypatch.setattr(search, "run_relationship_query", lambda q, codes: _produced())
    monkeypatch.setattr(search, "map_results", lambda rows, cypher, anchors, q: None)
    assert search._try_text2cypher("q", [_org_seed()], [], 0.0) is None


def test_success_packages_structured_output(monkeypatch):
    monkeypatch.setattr(search, "TEXT2CYPHER_ENABLED", True)
    monkeypatch.setattr(search, "run_relationship_query", lambda q, codes: _produced())
    monkeypatch.setattr(search, "map_results", lambda rows, cypher, anchors, q: (_hits(), _meta()))

    out = search._try_text2cypher("q", [_org_seed()], [], 0.0)
    assert out is not None
    assert out["graph_meta"]["structured"]["kind"] == "text2cypher"
    assert out["graph_meta"]["n_seeds"] == 1
    assert "latency_ms" in out["graph_meta"]
    assert out["graph_hits"] == _hits()


def test_anchor_codes_passed_to_retriever(monkeypatch):
    # raw Seed(key_type/key_value) → corp_code 추출돼 retriever 에 코드 리스트로 전달.
    captured: dict = {}
    monkeypatch.setattr(search, "TEXT2CYPHER_ENABLED", True)

    def _capture(q, codes):
        captured["codes"] = codes
        return _produced()

    monkeypatch.setattr(search, "run_relationship_query", _capture)
    monkeypatch.setattr(search, "map_results", lambda rows, cypher, anchors, q: (_hits(), _meta()))
    search._try_text2cypher("q", [_org_seed()], [], 0.0)
    assert captured["codes"] == ["00126256"]


# ── search(): 진입점에서 우선 시도 / 폴백 ──────────────────────────

def _route_explore(q, **k):
    """관계/모드 질문 = 구조화 plan 없음 → search 가 text2cypher 로 흐른다."""
    return search.Route("relation_explore")


def test_search_uses_text2cypher_when_enabled(monkeypatch):
    monkeypatch.setattr(search, "match", lambda q, upstream_seeds=None: [_org_seed()])
    monkeypatch.setattr(search, "TEXT2CYPHER_ENABLED", True)
    monkeypatch.setattr(search, "classify", _route_explore)
    monkeypatch.setattr(search, "run_relationship_query", lambda q, codes: _produced())
    monkeypatch.setattr(search, "map_results", lambda rows, cypher, anchors, q: (_hits(), _meta()))

    out = search.search("삼성생명 형제 계열사")
    assert out["graph_meta"]["structured"]["kind"] == "text2cypher"


def test_search_runs_sql_rank_postep_on_text2cypher(monkeypatch):
    # 랭킹 질문이면 text2cypher 관계 결과 위에 MariaDB SQL 랭킹 후처리가 1위를 surface 한다.
    monkeypatch.setattr(search, "match", lambda q, upstream_seeds=None: [_org_seed()])
    monkeypatch.setattr(search, "TEXT2CYPHER_ENABLED", True)
    monkeypatch.setattr(search, "classify", _route_explore)
    monkeypatch.setattr(search, "run_relationship_query", lambda q, codes: _produced())
    ranked_hits = [
        {"id": "00126256", "label": "organization", "name": "삼성생명",
         "attrs": {"structured_role": "anchor"}, "score": 1.0},
        {"id": "100", "label": "organization", "name": "회사A",
         "attrs": {"structured_role": "supplier"}, "score": 0.75},
        {"id": "101", "label": "organization", "name": "회사B",
         "attrs": {"structured_role": "supplier"}, "score": 0.75},
    ]
    monkeypatch.setattr(search, "map_results", lambda rows, cypher, anchors, q: (ranked_hits, _meta()))
    monkeypatch.setattr(
        cypher_executor, "_fetch_metric_values",
        lambda codes, account_id="ifrs-full_Revenue", year=None: [
            {"corp_code": "100", "value": "900", "bsns_year": 2025,
             "unit": "KRW", "rcept_no": "f", "corp_name": ""},
            {"corp_code": "101", "value": "100", "bsns_year": 2025,
             "unit": "KRW", "rcept_no": "f", "corp_name": ""},
        ],
    )
    out = search.search("삼성생명 공급사 중 매출 1위")
    structured = out["graph_meta"]["structured"]
    assert structured["selected"]["name"] == "회사A"
    assert structured["rank_metric"] == "ifrs-full_Revenue"
    fin = [h for h in out["graph_hits"] if h["label"] == "fin_metric"]
    assert fin and fin[0]["attrs"]["value"] == 900.0


def test_chain_query_routes_to_chain_before_text2cypher(monkeypatch):
    # 다중홉 체인 질문("수혜의 수혜")은 라우터가 chain plan 을 실어 주면 text2cypher 단일 홉이
    # 가로채기 전에 chain 실행기로 라우팅돼야 한다. route.plan 이 있으면 run_relationship_query 는
    # 호출조차 되면 안 된다.
    monkeypatch.setattr(search, "match", lambda q, upstream_seeds=None: [_org_seed()])
    monkeypatch.setattr(search, "TEXT2CYPHER_ENABLED", True)

    chain_plan = object()  # 진위만 중요(execute_structured 가 받는 sentinel)
    monkeypatch.setattr(
        search, "classify",
        lambda q, **k: search.Route("chain", plan=chain_plan),
    )

    captured: dict = {}

    def _exec(plan, seeds, query):
        captured["plan"] = plan
        return _hits(), {"mode": "structured",
                         "structured": {"mode": "structured", "kind": "multi_hop_chain"},
                         "patterns_run": ["multi_hop_chain"], "n_hits": 1,
                         "fallback_used": False, "errors": []}

    monkeypatch.setattr(search, "execute_structured", _exec)

    def _boom(*a, **k):
        raise RuntimeError("text2cypher must not run when a chain plan exists")

    monkeypatch.setattr(search, "run_relationship_query", _boom)

    out = search.search("sk하이닉스가 오르면 수혜 기업, 그곳이 오르면 수혜 기업 3개")
    assert captured["plan"] is chain_plan
    assert out["graph_meta"]["structured"]["kind"] == "multi_hop_chain"


def test_search_flag_off_skips_generation(monkeypatch):
    # 플래그 off → retriever 호출조차 안 됨(우선 시도 스킵). 호출되면 RuntimeError 로 잡는다.
    monkeypatch.setattr(search, "match", lambda q, upstream_seeds=None: [_org_seed()])
    monkeypatch.setattr(search, "TEXT2CYPHER_ENABLED", False)

    def _boom(*a, **k):
        raise RuntimeError("run_relationship_query must not be called when flag is off")

    monkeypatch.setattr(search, "run_relationship_query", _boom)
    # 폴백 경로(구조화 plan 없음 → text2cypher 스킵 → PPR/expand/fallback)는 결정적·offline.
    monkeypatch.setattr(search, "classify", _route_explore)
    monkeypatch.setattr(search, "expand_ppr", lambda seeds: ([], []))
    monkeypatch.setattr(search, "expand", lambda seeds: ([], []))
    monkeypatch.setattr(search, "fallback_for", lambda sd: [])

    out = search.search("삼성생명 형제 계열사")
    assert out["graph_meta"].get("structured", {}).get("kind") != "text2cypher"
