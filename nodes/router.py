# nodes/router.py
from langchain_core.messages import AIMessage

from core.state import AgentState


def _last_human_text(state: AgentState) -> str:
    for msg in reversed(state.get("messages", []) or []):
        if getattr(msg, "type", None) == "human" and getattr(msg, "content", None):
            return str(msg.content)
    return ""


def router_node(state: AgentState):
    """Route: 의도 분류"""
    text = _last_human_text(state).strip()
    lowered = text.lower()
    if any(k in lowered for k in ("다시 써", "쉽게 말", "톤", "말투", "render")):
        return {"intent": "render"}
    if lowered in ("안녕", "안녕하세요", "hello", "hi"):
        return {"intent": "direct"}
    return {"intent": "rag"}

def direct_response_node(state: AgentState):
    """Direct: 단순 질문/잡담"""
    return {"messages": [AIMessage(content="안녕하세요. 무엇을 도와드릴까요?")]}

def context_reconstruct_node(state: AgentState):
    """Ctx: 질문 문맥 재구성"""
    return {"reconstructed_query": _last_human_text(state)}
