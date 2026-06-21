"""Layer B: LLM-planned 다중홉 랭킹 체인(cutline) — 오프라인.

walker(_execute_multi_hop_chain)는 그래프/RDB 를 monkeypatch 로 대체해 검증한다(실호출 0).
chain_planner.coerce_plan 의 multi_hop_chain 분기와 _looks_chain 게이트도 순수 검증한다.

핵심 시나리오: "SK하이닉스가 오르면 수혜 볼 기업(top N), 거기서 또 수혜 볼 기업(top N)" —
각 홉에서 관계 후보를 매출로 줄세워 상위 top_n 을 다음 홉 앵커로 넘기는 N-hop 체인.
"""
from __future__ import annotations

from graphrag import chain_planner as cp
from graphrag import structured_executor as se
from graphrag.plan_schema import HopStep, MetricRankStep, RelationStep, StructuredPlan


_SK = "00164779"


def _cand(corp_code: str, name: str, anchor_id: str, anchor_name: str, conf: float = 0.8) -> dict:
    return {
        "id": corp_code,
        "corp_code": corp_code,
        "er_name": "",
        "name": name,
        "source": "r1",
        "chunk_id": "c1",
        "anchor_rels": [],
        "graph_degree": 3,
        "edge": {
            "rel_type": "SUPPLIES_TO",
            "from_id": corp_code,
            "from_name": name,
            "to_id": anchor_id,
            "to_name": anchor_name,
            "role": "supplier",
            "source": "r1",
            "chunk_id": "c1",
        },
        "evidence": {"confidence": conf, "level": "high" if conf >= 0.8 else "low",
                     "relation_term_found": conf >= 0.8},
    }


# anchor corp_code → 그 앵커에 공급하는(SUPPLIES_TO incoming) 후보들.
_GRAPH = {
    _SK: [
        _cand("100", "동진쎄미켐", _SK, "SK하이닉스"),
        _cand("101", "한미반도체", _SK, "SK하이닉스"),
        _cand("102", "미니소재", _SK, "SK하이닉스"),
    ],
    "100": [
        _cand("200", "디엔에프", "100", "동진쎄미켐"),
        _cand("201", "에이스소재", "100", "동진쎄미켐"),
    ],
    "101": [
        _cand("300", "부품테크", "101", "한미반도체"),
        _cand("301", "소형정밀", "101", "한미반도체"),
    ],
}

_METRIC = {
    "100": 5_000_000_000_000, "101": 3_000_000_000_000, "102": 100_000_000_000,
    "200": 800_000_000_000, "201": 200_000_000_000,
    "300": 600_000_000_000, "301": 100_000_000_000,
}


def _fake_relation_candidates(anchor, step, *, exclude_ids=None):
    key = anchor.get("corp_code") or anchor.get("id") or ""
    return [dict(r, edge=dict(r["edge"]), evidence=dict(r["evidence"])) for r in _GRAPH.get(key, [])]


def _fake_metric_values(corp_codes, account_id="ifrs-full_Revenue", year=None):
    return [
        {"corp_code": c, "account_id": account_id, "value": str(_METRIC[c]),
         "bsns_year": 2025, "unit": "KRW", "rcept_no": "f1", "corp_name": ""}
        for c in corp_codes if c in _METRIC
    ]


def _seed():
    return {
        "label": "organization", "id": _SK, "name": "SK하이닉스",
        "key_type": "corp_code", "key_value": _SK, "score": 1.0, "origin": "upstream",
    }


def _two_hop_plan(top_n: int = 2) -> StructuredPlan:
    relation = RelationStep("SUPPLIES_TO", "incoming", "suppliers", "supplier")
    rank = MetricRankStep("ifrs-full_Revenue")
    hop = HopStep(relation=relation, rank=rank, top_n=top_n)
    return StructuredPlan(
        kind="multi_hop_chain",
        first_relation=relation,
        first_rank=rank,
        hops=[hop, HopStep(relation=relation, rank=rank, top_n=top_n)],
    )


def _patch(monkeypatch):
    monkeypatch.setattr(se, "_relation_candidates", _fake_relation_candidates)
    monkeypatch.setattr(se, "_attach_candidate_evidence", lambda cands: None)
    monkeypatch.setattr(se, "_fetch_metric_values", _fake_metric_values)


def test_chain_walks_two_hops_top_n_per_hop(monkeypatch):
    _patch(monkeypatch)
    out = se.execute(_two_hop_plan(top_n=2), [_seed()], "SK하이닉스가 오르면 수혜 볼 기업, 거기서 또 수혜 볼 기업 2개")
    assert out is not None
    hits, meta = out
    structured = meta["structured"]
    assert structured["kind"] == "multi_hop_chain"

    hops = structured["hops"]
    assert len(hops) == 2
    # 홉1: 매출 desc 상위 2 (미니소재 탈락 = top_n 존중).
    assert hops[0]["selected"] == ["동진쎄미켐", "한미반도체"]
    assert "미니소재" not in hops[0]["selected"]
    # 홉2: 프런티어(동진→한미) 순서로 각 top_2.
    assert hops[1]["selected"] == ["디엔에프", "에이스소재", "부품테크", "소형정밀"]

    # 답 엣지 = 홉1(2) + 홉2(4).
    assert len(structured["answer_edges"]) == 6
    org_names = {h["name"] for h in hits if h.get("label") == "organization"}
    assert org_names == {"SK하이닉스", "동진쎄미켐", "한미반도체", "디엔에프", "에이스소재", "부품테크", "소형정밀"}


def test_chain_top_n_one_keeps_single_path(monkeypatch):
    _patch(monkeypatch)
    out = se.execute(_two_hop_plan(top_n=1), [_seed()], "SK하이닉스 수혜의 수혜")
    assert out is not None
    _, meta = out
    hops = meta["structured"]["hops"]
    assert hops[0]["selected"] == ["동진쎄미켐"]          # 매출 1위만
    assert hops[1]["selected"] == ["디엔에프"]            # 동진의 매출 1위 공급사만


def test_chain_abstains_when_no_candidate_supported(monkeypatch):
    # 모든 후보 근거가 바닥(conf 0.1 < SUPPLIES_TO 0.55) → 게이트 전멸 → None(graceful degrade).
    weak = {k: [_cand(c["corp_code"], c["name"], c["edge"]["to_id"], c["edge"]["to_name"], conf=0.1)
                for c in v] for k, v in _GRAPH.items()}
    monkeypatch.setattr(se, "_relation_candidates",
                        lambda anchor, step, *, exclude_ids=None: [dict(r, edge=dict(r["edge"]), evidence=dict(r["evidence"])) for r in weak.get(anchor.get("corp_code") or anchor.get("id") or "", [])])
    monkeypatch.setattr(se, "_attach_candidate_evidence", lambda cands: None)
    monkeypatch.setattr(se, "_fetch_metric_values", _fake_metric_values)
    assert se.execute(_two_hop_plan(), [_seed()], "SK하이닉스 수혜의 수혜") is None


def _patch_graph(monkeypatch, graph: dict, metric: dict):
    monkeypatch.setattr(
        se, "_relation_candidates",
        lambda anchor, step, *, exclude_ids=None: [
            dict(r, edge=dict(r["edge"]), evidence=dict(r["evidence"]))
            for r in graph.get(anchor.get("corp_code") or anchor.get("id") or "", [])
        ],
    )
    monkeypatch.setattr(se, "_attach_candidate_evidence", lambda cands: None)
    monkeypatch.setattr(
        se, "_fetch_metric_values",
        lambda corp_codes, account_id="ifrs-full_Revenue", year=None: [
            {"corp_code": c, "account_id": account_id, "value": str(metric[c]),
             "bsns_year": 2025, "unit": "KRW", "rcept_no": "f1", "corp_name": ""}
            for c in corp_codes if c in metric
        ],
    )


def test_chain_skips_dead_end_for_fertile_lower_rank(monkeypatch):
    # hop0 매출 1위(대기업A)는 상류가 없어 막다른 길, 2위(중견B)는 이어진다. 전방탐색이 없으면
    # top_1 이 대기업A 로 끊겨 1홉에서 죽는다(라이브 회귀: SK실트론 → 지주사 SK(주)뿐 → 단절).
    # 전방탐색 walker 는 이어질 수 있는 중견B 를 골라 2홉을 완성한다.
    graph = {
        _SK: [
            _cand("900", "대기업A", _SK, "SK하이닉스"),   # rev 9T, 막다른 길(상류 없음)
            _cand("901", "중견B", _SK, "SK하이닉스"),      # rev 2T, 이어짐
        ],
        "901": [_cand("910", "하위C", "901", "중견B")],
    }
    metric = {"900": 9_000_000_000_000, "901": 2_000_000_000_000, "910": 500_000_000_000}
    _patch_graph(monkeypatch, graph, metric)
    out = se.execute(_two_hop_plan(top_n=1), [_seed()], "SK하이닉스 수혜의 수혜")
    assert out is not None
    _, meta = out
    hops = meta["structured"]["hops"]
    assert hops[0]["selected"] == ["중견B"]   # 막다른 대기업A 가 아니라 이어지는 중견B
    assert hops[1]["selected"] == ["하위C"]


def test_noise_gate_excludes_bank_and_no_corp_code(monkeypatch):
    # 금융기관(○○은행)과 corp_code 없는 외부 노드는 SUPPLIES_TO 랭킹에서 비파괴적으로 빠진다.
    bank = _cand("700", "우리은행", _SK, "SK하이닉스")      # 금융기관 → 제외
    foreign = _cand("rambus", "Rambus", _SK, "SK하이닉스")  # corp_code 없음 → 제외
    foreign["corp_code"] = ""
    real = _cand("710", "정상소재", _SK, "SK하이닉스")       # 운영 공급사 → 유지
    metric = {"700": 9_000_000_000_000, "710": 1_000_000_000_000}  # 은행 매출이 커도 제외
    _patch_graph(monkeypatch, {_SK: [bank, foreign, real]}, metric)
    out = se.execute(_two_hop_plan(top_n=3), [_seed()], "SK하이닉스 수혜")
    assert out is not None
    _, meta = out
    hops = meta["structured"]["hops"]
    assert hops[0]["selected"] == ["정상소재"]
    assert "우리은행" not in hops[0]["selected"]
    assert "Rambus" not in hops[0]["selected"]


def test_coerce_chain_plan_builds_two_hops():
    data = {
        "supported": True,
        "kind": "multi_hop_chain",
        "hops": [
            {"relation": {"rel_type": "SUPPLIES_TO", "direction": "incoming"},
             "rank_metric": "ifrs-full_Revenue", "top_n": 2},
            {"relation": {"rel_type": "SUPPLIES_TO", "direction": "incoming"},
             "rank_metric": "ifrs-full_Revenue", "top_n": 3},
        ],
    }
    plan = cp.coerce_plan(data, "SK하이닉스 수혜의 수혜 매출 상위")
    assert plan is not None
    assert plan.kind == "multi_hop_chain"
    assert [h.top_n for h in plan.hops] == [2, 3]
    assert plan.hops[0].relation.rel_type == "SUPPLIES_TO"
    assert plan.first_relation == plan.hops[0].relation


def test_coerce_chain_plan_rejects_single_hop():
    data = {"kind": "multi_hop_chain", "hops": [
        {"relation": {"rel_type": "SUPPLIES_TO", "direction": "incoming"}, "rank_metric": "ifrs-full_Revenue"},
    ]}
    assert cp.coerce_plan(data, "q") is None


def test_coerce_chain_plan_rejects_invalid_hop():
    data = {"kind": "multi_hop_chain", "hops": [
        {"relation": {"rel_type": "SUPPLIES_TO", "direction": "incoming"}, "rank_metric": "ifrs-full_Revenue"},
        {"relation": {"rel_type": "BOGUS_REL", "direction": "incoming"}, "rank_metric": "ifrs-full_Revenue"},
    ]}
    assert cp.coerce_plan(data, "q") is None


def test_looks_chain_requires_propagation_and_recursion():
    assert cp._looks_chain("SK하이닉스가 오르면 수혜 볼 기업, 거기서 또 수혜 볼 기업 3개")
    assert cp._looks_chain("삼성전자 낙수효과로 수혜 보는 회사, 그다음 또 낙수 받는 회사")
    # 전파어만(반복 표지 없음) → 단일 홉 수혜주 질문이라 체인 아님.
    assert not cp._looks_chain("삼성전자 수혜주는?")
    # 체인 cue 자체가 없음.
    assert not cp._looks_chain("삼성전자 공급사 중 매출 1위")
