# core/graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# 1. 분리해둔 모듈들 불러오기
from core.state import AgentState
from nodes.memory import load_memory_node, save_memory_node
from nodes.router import router_node, direct_response_node, context_reconstruct_node
from nodes.rag import ( rdb_search_node, vector_search_node, 
                       graph_search_node, synthesizer_node, reflection_node)
from nodes.render import generate_report_node

# 2. 라우팅 조건 함수들 정의
def route_after_intent(state: AgentState):
    return state["intent"]

def route_search_plan(state: AgentState):
    return state["search_plan"]

def route_reflection(state: AgentState):
    return "gen" if state.get("is_sufficient") else "ctx"

# 3. 그래프 조립 시작
workflow = StateGraph(AgentState)

# ==========================================
# 모든 노드(Node)
# ==========================================
workflow.add_node("mem", load_memory_node)
workflow.add_node("route", router_node)
workflow.add_node("direct", direct_response_node)
workflow.add_node("ctx", context_reconstruct_node)

workflow.add_node("rdb", rdb_search_node)
workflow.add_node("vec", vector_search_node)
workflow.add_node("graph", graph_search_node)
workflow.add_node("syn", synthesizer_node)
workflow.add_node("reflect", reflection_node)
workflow.add_node("gen", generate_report_node)

workflow.add_node("save", save_memory_node)

# ==========================================
#  엣지(Edge) 연결하기
# ==========================================
# 진입점
workflow.add_edge(START, "mem")
workflow.add_edge("mem", "route")

# 의도 분류 후 라우팅 분기
workflow.add_conditional_edges(
    "route",
    route_after_intent,
    {
        "direct": "direct",
        "ctx": "ctx",
    }
)

# 단순 잡담은 답변 후 바로 종료
workflow.add_edge("direct", END)

workflow.add_edge("ctx", "rdb")
workflow.add_edge("ctx", "vec")
workflow.add_edge("ctx", "graph")

# 검색 결과들을 하나로 모으기 (합성)
workflow.add_edge("rdb", "syn")
workflow.add_edge("vec", "syn")
workflow.add_edge("graph", "syn")

workflow.add_edge("syn", "reflect")

# 충분성 검증 후 분기
workflow.add_conditional_edges(
    "reflect",
    route_reflection,
    {
        "ctx": "ctx", # 부족하면 다시 검색 계획 수립으로
        "gen": "gen"  # 충분하면 보고서 생성으로
    }
)

# 최종 렌더링 및 저장 흐름
workflow.add_edge("gen", "save")
workflow.add_edge("save", END)

# ==========================================
# 4. 컴파일 및 앱 내보내기 (메모리+스토어 장착)
# ==========================================
memory = MemorySaver()      # 단기 대화 기록용 (thread_id 기반)
store = InMemoryStore()     # 장기 선호도 저장용 (user_id 기반)

app = workflow.compile(
    checkpointer=memory,
    store=store
)