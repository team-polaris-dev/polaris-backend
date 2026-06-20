"""공통 거래상대 단일 지표 랭킹 (multi_anchor_rank) — 오프라인.

"삼성전자와 SK하이닉스가 동시에 거래하는 소재 회사 중 매출 1위" 류가:
  1) planner 에서 multi_anchor_rank 로 식별되고(공통 앵커 + 관계 + 랭킹, branch 비교 없음),
  2) executor 가 두 앵커 공급사를 교집합해 매출로 줄세워 1위(selected_common_supplier)를 emit 하며,
  3) 셰임 1:1 계약(len(facts)==len(hits))을 깨지 않는지 검증한다. Neo4j/RDB 는 monkeypatch — 네트워크 0.
"""
from __future__ import annotations

from graphrag import structured_executor
from graphrag.planner import plan
from graphrag.schema import adapt_to_legacy


_SAMSUNG = "00126380"
_HYNIX = "00164779"
_COMMON = "00111111"   # 소재A: 두 앵커에 모두 공급(교집합)
_ONLY_SS = "00122222"  # 소재B: 삼성에만
_ONLY_HX = "00133333"  # 소재C: 하이닉스에만

_Q = "삼성전자와 SK하이닉스가 동시에 거래하는 소재 회사 중에서 매출액이 가장 높은 회사는?"


def _cand(cid, name, anchor_id, anchor_name, chunk_id):
    return {
        "id": cid, "corp_code": cid, "er_name": "", "name": name,
        "source": "R" + cid, "chunk_id": chunk_id,
        "anchor_rels": [], "graph_degree": 3,
        "edge": {
            "rel_type": "SUPPLIES_TO",
            "from_id": cid, "from_name": name,
            "to_id": anchor_id, "to_name": anchor_name,
            "role": "supplier", "source": "R" + cid, "chunk_id": chunk_id,
        },
    }


def _relation_candidates(anchor, step, *, exclude_ids=None):
    if anchor.get("corp_code") == _SAMSUNG:
        return [_cand(_COMMON, "소재A", _SAMSUNG, "삼성전자", "c1"),
                _cand(_ONLY_SS, "소재B", _SAMSUNG, "삼성전자", "c2")]
    if anchor.get("corp_code") == _HYNIX:
        return [_cand(_COMMON, "소재A", _HYNIX, "SK하이닉스", "c3"),
                _cand(_ONLY_HX, "소재C", _HYNIX, "SK하이닉스", "c4")]
    return []


def _metric_rows(corp_codes, account_id="ifrs-full_Revenue", year=None):
    # _fetch_metric_values 계약: value DESC. 교집합(소재A)만 지표를 가진다.
    table = {_COMMON: "5000000000000"}
    return [
        {"corp_code": c, "corp_name": "소재A", "value": table[c],
         "bsns_year": 2024, "account_id": account_id, "unit": "KRW", "rcept_no": "M" + c}
        for c in corp_codes if c in table
    ]


def _chunk_texts(ids):
    # 양끝 회사 + 관계어(공급) 언급 → 강근거(FULL) → operating 게이트 통과.
    return {
        "c1": "소재A는 삼성전자에 부품을 공급한다",
        "c3": "소재A는 SK하이닉스에 소재를 공급한다",
    }


def _patch(monkeypatch):
    monkeypatch.setattr(structured_executor, "_relation_candidates", _relation_candidates)
    monkeypatch.setattr(structured_executor, "_fetch_metric_values", _metric_rows)
    monkeypatch.setattr(structured_executor, "_fetch_chunk_texts", _chunk_texts)


def _seeds():
    return [
        {"label": "organization", "id": _SAMSUNG, "key_type": "corp_code",
         "key_value": _SAMSUNG, "name": "삼성전자", "score": 1.0},
        {"label": "organization", "id": _HYNIX, "key_type": "corp_code",
         "key_value": _HYNIX, "name": "SK하이닉스", "score": 1.0},
    ]


def test_planner_routes_common_trade_to_multi_anchor_rank():
    out = plan(_Q)
    assert out is not None
    assert out.kind == "multi_anchor_rank"
    assert out.first_relation.rel_type == "SUPPLIES_TO"
    assert out.first_relation.direction == "incoming"


def test_executor_intersects_and_ranks_common_supplier(monkeypatch):
    _patch(monkeypatch)
    p = plan(_Q)
    result = structured_executor.execute(p, _seeds(), _Q)

    assert result is not None
    hits, meta = result
    structured = meta["structured"]
    assert structured["kind"] == "multi_anchor_rank"
    assert structured["abstained"] is False
    # 교집합 = 소재A 뿐(소재B/C 는 한쪽만) → 1위로 선택.
    assert structured["first"]["selected"]["name"] == "소재A"


def test_executor_emits_common_anchors_and_selected(monkeypatch):
    _patch(monkeypatch)
    p = plan(_Q)
    hits, _ = structured_executor.execute(p, _seeds(), _Q)

    orgs = [h for h in hits if h["label"] == "organization"]
    roles = {h["attrs"].get("structured_role") for h in orgs}
    assert "common_anchor" in roles
    assert "selected_common_supplier" in roles
    common = {h["name"] for h in orgs if h["attrs"].get("structured_role") == "common_anchor"}
    assert common == {"삼성전자", "SK하이닉스"}
    selected = [h for h in orgs if h["attrs"].get("structured_role") == "selected_common_supplier"]
    assert len(selected) == 1 and selected[0]["name"] == "소재A"


def test_executor_renders_both_anchor_edges(monkeypatch):
    _patch(monkeypatch)
    p = plan(_Q)
    hits, _ = structured_executor.execute(p, _seeds(), _Q)

    rels = [h for h in hits if h["label"] == "relationship"]
    # 1위 소재A → 두 앵커 각각으로 SUPPLIES_TO 엣지 2개.
    pairs = {(h["attrs"]["from_name"], h["attrs"]["to_name"]) for h in rels}
    assert pairs == {("소재A", "삼성전자"), ("소재A", "SK하이닉스")}
    assert all(h["attrs"]["rel_type"] == "SUPPLIES_TO" for h in rels)


_COMMON2 = "00144444"  # 소재D: 두 앵커에 모두 공급(교집합 러너업)


def _relation_candidates_two_common(anchor, step, *, exclude_ids=None):
    if anchor.get("corp_code") == _SAMSUNG:
        return [_cand(_COMMON, "소재A", _SAMSUNG, "삼성전자", "c1"),
                _cand(_COMMON2, "소재D", _SAMSUNG, "삼성전자", "c5")]
    if anchor.get("corp_code") == _HYNIX:
        return [_cand(_COMMON, "소재A", _HYNIX, "SK하이닉스", "c3"),
                _cand(_COMMON2, "소재D", _HYNIX, "SK하이닉스", "c6")]
    return []


def _metric_rows_two_common(corp_codes, account_id="ifrs-full_Revenue", year=None):
    table = {_COMMON: "5000000000000", _COMMON2: "3000000000000"}
    return [
        {"corp_code": c, "corp_name": "공통", "value": table[c],
         "bsns_year": 2024, "account_id": account_id, "unit": "KRW", "rcept_no": "M" + c}
        for c in corp_codes if c in table
    ]


def _chunk_texts_two_common(ids):
    return {
        "c1": "소재A는 삼성전자에 부품을 공급한다",
        "c3": "소재A는 SK하이닉스에 소재를 공급한다",
        "c5": "소재D는 삼성전자에 부품을 공급한다",
        "c6": "소재D는 SK하이닉스에 소재를 공급한다",
    }


def test_executor_renders_runner_up_common_suppliers(monkeypatch):
    # 교집합이 둘 이상이면 1위(selected_common_supplier) 외 러너업도 common_supplier 로
    # 관계망에 노출돼야 보고서가 "유일한 공통 공급사"로 오판하지 않는다.
    monkeypatch.setattr(structured_executor, "_relation_candidates", _relation_candidates_two_common)
    monkeypatch.setattr(structured_executor, "_fetch_metric_values", _metric_rows_two_common)
    monkeypatch.setattr(structured_executor, "_fetch_chunk_texts", _chunk_texts_two_common)
    p = plan(_Q)
    hits, _ = structured_executor.execute(p, _seeds(), _Q)

    orgs = [h for h in hits if h["label"] == "organization"]
    selected = [h for h in orgs if h["attrs"].get("structured_role") == "selected_common_supplier"]
    runners = [h for h in orgs if h["attrs"].get("structured_role") == "common_supplier"]
    assert len(selected) == 1 and selected[0]["name"] == "소재A"   # 매출 1위
    assert {h["name"] for h in runners} == {"소재D"}               # 러너업도 노출
    # 러너업 엣지도 양 앵커로 렌더 → 관계망에 8노드형 교집합 골격이 보인다.
    rels = [h for h in hits if h["label"] == "relationship"]
    pairs = {(h["attrs"]["from_name"], h["attrs"]["to_name"]) for h in rels}
    assert ("소재D", "삼성전자") in pairs and ("소재D", "SK하이닉스") in pairs


def test_executor_shame_contract(monkeypatch):
    _patch(monkeypatch)
    p = plan(_Q)
    hits, _ = structured_executor.execute(p, _seeds(), _Q)

    legacy = adapt_to_legacy(hits)
    # 셰임 1:1 — 모든 hit 가 fact 로 매핑.
    assert len(legacy["facts"]) == len(hits)


def test_executor_none_when_single_anchor(monkeypatch):
    # 앵커가 하나뿐이면 교집합 불가 → None(호출부 abstain graceful degrade).
    _patch(monkeypatch)
    p = plan(_Q)
    one = [_seeds()[0]]
    assert structured_executor.execute(p, one, _Q) is None
