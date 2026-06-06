# nodes/router.py
from core.state import AgentState
from config.llm import llm

def router_node(state: AgentState):
    """Route: 의도 분류"""
    # ... 로직 ...
    return {"intent": "rag"}

def direct_response_node(state: AgentState):
    """Direct: 단순 질문/잡담"""
    return {"messages": ["안녕하세요!"]}

def context_reconstruct_node(state: AgentState):
    """Ctx: 질문 문맥 재구성"""
    return {"reconstructed_query": "명확해진 질문"}