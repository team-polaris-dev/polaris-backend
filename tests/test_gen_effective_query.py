"""gen(render.generate_report_node) 이 effective_query 를 쓰는지 — 오프라인.

재구성이 "삼성 계열사"(그룹 군집 랭킹)를 "삼성전자의 종속회사"(single_hop)로 좁혀도,
gen 본문이 보는 [사용자 질문]은 원문이어야 한다 — 그래프(1위=그룹 전체 최댓값)와
답변이 어긋나지 않도록. LLM 은 capture 스텁으로 차단(네트워크 0).
"""
from __future__ import annotations

from nodes import render


class _HumanMsg:
    type = "human"

    def __init__(self, content: str):
        self.content = content


class _FakeReply:
    content = "보고서 본문"


class _CapturingLLM:
    """render.llm 대체 — invoke(messages) 를 가로채 마지막 호출을 보관."""
    def __init__(self):
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return _FakeReply()


def _human_prompt(cap: _CapturingLLM) -> str:
    # generate_report_node 는 [System, Human] 순으로 넘긴다.
    return cap.last_messages[-1].content


def test_gen_uses_original_when_reconstruction_narrows_group_scope(monkeypatch):
    cap = _CapturingLLM()
    monkeypatch.setattr(render, "llm", cap)

    state = {
        "messages": [_HumanMsg("삼성 계열사 중 매출 가장 높은 곳은?")],
        "reconstructed_query": "삼성전자의 종속회사 중 매출액이 가장 높은 회사는?",
        "rdb_results": [], "vec_results": [],
        "graph_facts": [], "graph_paths": [], "graph_provenance": [],
        "community_results": [],
    }
    out = render.generate_report_node(state)

    assert out["final_draft"] == "보고서 본문"
    prompt = _human_prompt(cap)
    # [사용자 질문] 은 원문(계열사) — 재구성의 '종속회사' 축소가 본문에 새지 않는다.
    assert "삼성 계열사 중 매출 가장 높은 곳은?" in prompt
    assert "종속회사" not in prompt.split("[분석 기초 자료]")[0]


def test_gen_keeps_reconstruction_for_genuine_subsidiary(monkeypatch):
    cap = _CapturingLLM()
    monkeypatch.setattr(render, "llm", cap)

    state = {
        "messages": [_HumanMsg("삼성전자 자회사 중 매출 1위는?")],
        "reconstructed_query": "삼성전자(주)의 자회사 중 매출액이 가장 높은 회사는?",
        "rdb_results": [], "vec_results": [],
        "graph_facts": [], "graph_paths": [], "graph_provenance": [],
        "community_results": [],
    }
    render.generate_report_node(state)

    prompt = _human_prompt(cap)
    # 애초에 자회사 랭킹(single_hop)이면 좁힘이 아니므로 재구성 그대로.
    assert "삼성전자(주)의 자회사 중 매출액이 가장 높은 회사는?" in prompt
