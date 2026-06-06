# nodes/rag.py
from core.state import AgentState

def supervisor_node(state: AgentState):
    return {"search_plan": ["RDB", "Vector"]}

def rdb_search_node(state: AgentState):
    return {"search_results": ["RDB 검색 결과"]}

def vector_search_node(state: AgentState):
    return {"search_results": ["Vector 검색 결과"]}

def graph_search_node(state: AgentState):
    return {"search_results": ["Graph 검색 결과"]}

def synthesizer_node(state: AgentState):
    combined = "\n".join(state["search_results"])
    return {"synthesized_info": combined}

def reflection_node(state: AgentState):
    return {"is_sufficient": True}

def generate_report_node(state: AgentState):
    return {"final_draft": "초안 리포트"}