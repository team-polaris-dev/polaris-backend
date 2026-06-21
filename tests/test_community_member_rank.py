"""그룹/계열 군집 멤버 노드 지표 랭킹 (community_member_rank) — 오프라인.

"삼성 계열사 중 매출 가장 높은 곳" 류 질문이:
  1) planner 에서 community_member_rank 플랜으로 식별되고(구체 관계어 없이 그룹 범위어만),
  2) executor 가 앵커 군집 멤버를 RDB 매출로 줄세워 1위(fin_metric) + 군집 관계망을 emit하며,
  3) 셰임 1:1 계약(len(facts)==len(hits))을 깨지 않는지
검증한다. Neo4j/RDB 는 monkeypatch 로 차단 — 네트워크 0.
"""
from __future__ import annotations

from graphrag import structured_executor
from graphrag.planner import plan
from graphrag.schema import adapt_to_legacy


# ── planner: 그룹 범위어 → community_member_rank, 구체 관계어 → 기존 경로 ──────

def test_group_scope_revenue_question_becomes_community_member_rank():
    out = plan("삼성 계열사 중 매출 가장 높은 곳")

    assert out is not None
    assert out.kind == "community_member_rank"
    assert out.first_relation is None
    assert out.first_rank.metric_id == "ifrs-full_Revenue"
    # to_dict() 는 first_relation=None 을 안전 직렬화해야 한다(graph_meta 에 실림).
    assert out.to_dict()["first_relation"] is None


def test_group_scope_operating_income_question_picks_operating_metric():
    out = plan("삼성 그룹사 중 영업이익 1위는?")

    assert out is not None
    assert out.kind == "community_member_rank"
    assert out.first_rank.metric_id == "dart_OperatingIncomeLoss"


def test_subsidiary_revenue_question_is_not_community():
    # 구체 관계어(자회사)가 있으면 한 회사의 이웃 랭킹 → text2cypher 가 흡수(None 폴백).
    # 핵심 가드: 그룹 군집 경로(community_member_rank)로 새지 않는다.
    out = plan("삼성전자 자회사 중 매출 1위")

    assert out is None


def test_group_scope_without_metric_is_not_structured():
    # 노드 지표어가 없으면 구조화 플랜 자체가 안 선다(금액·관계 신호 부재) → None.
    assert plan("삼성 계열사 보여줘") is None


# ── executor: 군집 멤버를 매출로 줄세워 1위 + 관계망 ─────────────────────────

def _samsung_community():
    return {
        "members": ["00126380", "00126362", "00164742"],
        "member_names": ["삼성전자", "삼성물산", "삼성에스디에스"],
        "cluster_id": 0,
        "size": 3,
    }


def _samsung_metric_rows(corp_codes, account_id="ifrs-full_Revenue", year=None):
    # value DESC (_fetch_metric_values 계약: ORDER BY value DESC).
    return [
        {"corp_code": "00126380", "corp_name": "삼성전자", "value": "333600000000000",
         "bsns_year": 2024, "account_id": account_id, "unit": "KRW", "rcept_no": "R1"},
        {"corp_code": "00126362", "corp_name": "삼성물산", "value": "40700000000000",
         "bsns_year": 2024, "account_id": account_id, "unit": "KRW", "rcept_no": "R2"},
        {"corp_code": "00164742", "corp_name": "삼성에스디에스", "value": "13900000000000",
         "bsns_year": 2024, "account_id": account_id, "unit": "KRW", "rcept_no": "R3"},
    ]


def _samsung_member_edges(member_codes):
    return [
        {"rel_type": "IS_SUBSIDIARY_OF", "from_id": "00126362", "from_name": "삼성물산",
         "to_id": "00126380", "to_name": "삼성전자", "role": "", "source": "R4", "chunk_id": ""},
        {"rel_type": "IS_MAJOR_SHAREHOLDER_OF", "from_id": "00126380", "from_name": "삼성전자",
         "to_id": "00164742", "to_name": "삼성에스디에스", "role": "", "source": "R5", "chunk_id": ""},
    ]


def _samsung_seed():
    return {
        "label": "organization", "id": "00126380", "key_type": "corp_code",
        "key_value": "00126380", "name": "삼성전자", "score": 1.0,
    }


def _patch_samsung(monkeypatch):
    monkeypatch.setattr(structured_executor, "_community_for_anchor", lambda _c: _samsung_community())
    monkeypatch.setattr(structured_executor, "_fetch_metric_values", _samsung_metric_rows)
    monkeypatch.setattr(structured_executor, "_member_network_edges", _samsung_member_edges)


def test_community_rank_picks_highest_revenue_member(monkeypatch):
    _patch_samsung(monkeypatch)
    p = plan("삼성 계열사 중 매출 가장 높은 곳")

    result = structured_executor._execute_community_member_rank(p, _samsung_seed(), "삼성 계열사 중 매출 가장 높은 곳")

    assert result is not None
    hits, meta = result
    structured = meta["structured"]
    assert structured["kind"] == "community_member_rank"
    assert structured["rankable"] is True
    assert structured["abstained"] is False
    # 1위 = 삼성전자(333.6조), 멤버는 value DESC.
    assert structured["selected"]["name"] == "삼성전자"
    assert [m["name"] for m in structured["members"]] == ["삼성전자", "삼성물산", "삼성에스디에스"]


def test_community_rank_emits_winner_fin_metric_hit(monkeypatch):
    _patch_samsung(monkeypatch)
    p = plan("삼성 계열사 중 매출 가장 높은 곳")
    hits, _ = structured_executor._execute_community_member_rank(p, _samsung_seed(), "삼성 계열사 중 매출 가장 높은 곳")

    fin = [h for h in hits if h["label"] == "fin_metric"]
    # render._fmt_graph 가 (year, account_id) 로 묶으므로 1위 하나만 emit.
    assert len(fin) == 1
    assert fin[0]["name"] == "삼성전자"
    assert fin[0]["attrs"]["rank"] == 1
    assert fin[0]["attrs"]["account_id"] == "ifrs-full_Revenue"
    assert fin[0]["attrs"]["value"] == 333600000000000.0


def test_community_rank_emits_member_network(monkeypatch):
    _patch_samsung(monkeypatch)
    p = plan("삼성 계열사 중 매출 가장 높은 곳")
    hits, _ = structured_executor._execute_community_member_rank(p, _samsung_seed(), "삼성 계열사 중 매출 가장 높은 곳")

    rels = [h for h in hits if h["label"] == "relationship"]
    orgs = [h for h in hits if h["label"] == "organization"]
    rel_types = {h["attrs"]["rel_type"] for h in rels}
    assert rel_types == {"IS_SUBSIDIARY_OF", "IS_MAJOR_SHAREHOLDER_OF"}
    # 엣지 끝점 + 1위 멤버가 노드로 나온다(고립 노드 없음).
    assert {h["name"] for h in orgs} == {"삼성전자", "삼성물산", "삼성에스디에스"}
    # 1위 멤버 노드는 selected_member 로 표시.
    selected = [h for h in orgs if h["attrs"].get("structured_role") == "selected_member"]
    assert len(selected) == 1 and selected[0]["name"] == "삼성전자"


def test_community_rank_shame_contract_and_panel_edges(monkeypatch):
    _patch_samsung(monkeypatch)
    p = plan("삼성 계열사 중 매출 가장 높은 곳")
    hits, _ = structured_executor._execute_community_member_rank(p, _samsung_seed(), "삼성 계열사 중 매출 가장 높은 곳")

    legacy = adapt_to_legacy(hits)
    # 셰임 1:1 — 모든 hit 가 fact 로 매핑.
    assert len(legacy["facts"]) == len(hits)
    # 지배구조 엣지(자회사·대주주)는 무근거로도 패널 망에 그려진다.
    assert len(legacy["paths"]) == 2
    assert all(len(p_) == 3 for p_ in legacy["paths"])
    assert len(legacy["path_sources"]) == len(legacy["paths"])


def test_community_rank_none_when_no_community(monkeypatch):
    monkeypatch.setattr(structured_executor, "_community_for_anchor", lambda _c: None)
    monkeypatch.setattr(structured_executor, "_fetch_metric_values", _samsung_metric_rows)
    monkeypatch.setattr(structured_executor, "_member_network_edges", _samsung_member_edges)
    p = plan("삼성 계열사 중 매출 가장 높은 곳")
    assert structured_executor._execute_community_member_rank(p, _samsung_seed(), "삼성 계열사 중 매출 가장 높은 곳") is None


def test_community_rank_none_when_no_metric_rows(monkeypatch):
    # 군집은 있으나 RDB 매출이 없으면 억지 랭킹 금지 → None (fail-closed).
    monkeypatch.setattr(structured_executor, "_community_for_anchor", lambda _c: _samsung_community())
    monkeypatch.setattr(structured_executor, "_fetch_metric_values", lambda *a, **k: [])
    monkeypatch.setattr(structured_executor, "_member_network_edges", _samsung_member_edges)
    p = plan("삼성 계열사 중 매출 가장 높은 곳")
    assert structured_executor._execute_community_member_rank(p, _samsung_seed(), "삼성 계열사 중 매출 가장 높은 곳") is None


def test_community_rank_resolves_er_name_anchor(monkeypatch):
    # 매처가 약칭을 er_name 노드로 해소해도(코드 없이) 군집 역조회로 1위를 낸다.
    _patch_samsung(monkeypatch)
    p = plan("삼성 계열사 중 매출 가장 높은 곳")
    er_seed = {"label": "organization", "id": "org:삼성물산", "key_type": "er_name",
               "key_value": "삼성물산", "name": "삼성물산", "score": 1.0}
    result = structured_executor._execute_community_member_rank(p, er_seed, "삼성 계열사 중 매출 가장 높은 곳")
    assert result is not None
    _, meta = result
    assert meta["structured"]["selected"]["name"] == "삼성전자"
