import operator
from typing import get_args, get_origin

from core.graph import route_after_intent
from core.state import AgentState
from langchain_core.messages import AIMessage, HumanMessage
from nodes.rag import generate_report_node, supervisor_node
from nodes.router import context_reconstruct_node, direct_response_node, router_node
from tool.text_to_sql import _build_prompt


def test_rag_intent_routes_to_context_reconstruction():
    assert route_after_intent({"intent": "rag"}) == "ctx"


def test_search_results_has_additive_reducer():
    annotation = AgentState.__annotations__["search_results"]
    assert get_origin(annotation) is not None
    assert operator.add in get_args(annotation)


def test_agent_state_keeps_reflection_flag():
    assert "is_sufficient" in AgentState.__annotations__


def test_supervisor_uses_rdb_only_for_current_agent_scope():
    assert supervisor_node({}) == {"search_plan": ["RDB"]}


def test_context_reconstruct_preserves_latest_human_question():
    out = context_reconstruct_node({
        "messages": [
            HumanMessage(content="이전 질문"),
            AIMessage(content="이전 답변"),
            HumanMessage(content="삼성전자 공시 몇 건이야?"),
        ]
    })
    assert out["reconstructed_query"] == "삼성전자 공시 몇 건이야?"


def test_router_can_route_direct_and_render_requests():
    assert router_node({"messages": [HumanMessage(content="안녕하세요")]}) == {"intent": "direct"}
    assert router_node({"messages": [HumanMessage(content="방금 답변 쉽게 말해줘")]}) == {"intent": "render"}


def test_direct_response_returns_ai_message():
    out = direct_response_node({"messages": [HumanMessage(content="안녕하세요")]})
    assert out["messages"][0].type == "ai"


def test_generate_report_uses_synthesized_info_instead_of_placeholder():
    out = generate_report_node({"synthesized_info": "[RDB 검색]\n결과 1건: {'cnt': 3}"})
    assert out["final_draft"].startswith("[RDB 검색]")
    assert "초안 리포트" not in out["final_draft"]


def test_text_to_sql_does_not_add_missing_run_id_filter_guide():
    prompt = _build_prompt("삼성전자 공시 제목", "20260607_0000_01", None)
    assert "run_id = '20260607_0000_01'" not in prompt
