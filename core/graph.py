# core/graph.py
import time
import functools

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# 1. 분리해둔 모듈들 불러오기
from core.state import AgentState
from nodes.memory import load_memory_node, save_memory_node
from nodes.router import ( router_node, direct_response_node, context_reconstruct_node,
                          result_check_node, route_result_check)
from nodes.rag import rdb_search_node, vector_search_node, graph_search_node
from nodes.render import generate_report_node

# 2. 라우팅 조건 함수들 정의
def _timed(name: str, fn):
    """state-only 노드를 감싸 소요시간(초)을 node_timings[name] 에 기록한다.

    config/store 를 주입받는 memory 노드에는 쓰지 않는다(시그니처 단순 노드 전용).
    """
    @functools.wraps(fn)
    def wrapper(state: AgentState):
        t0 = time.perf_counter()
        out = fn(state)
        elapsed = time.perf_counter() - t0
        out = dict(out) if isinstance(out, dict) else {}
        out["node_timings"] = {name: elapsed}
        return out
    return wrapper


def route_after_intent(state: AgentState):
    return state["intent"]

def route_search_plan(state: AgentState):
    return state["search_plan"]

# 3. 그래프 조립 시작
workflow = StateGraph(AgentState)

# ==========================================
# 모든 노드(Node)
# ==========================================
workflow.add_node("mem", load_memory_node)
workflow.add_node("route", _timed("route", router_node))
workflow.add_node("direct", _timed("direct", direct_response_node))
workflow.add_node("ctx", _timed("ctx", context_reconstruct_node))

workflow.add_node("rdb", _timed("rdb", rdb_search_node))
workflow.add_node("vec", _timed("vec", vector_search_node))
workflow.add_node("graph", _timed("graph", graph_search_node))
workflow.add_node("result_check", _timed("result_check", result_check_node))
workflow.add_node("gen", _timed("gen", generate_report_node))

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

# 세 검색 결과가 모두 모이면 규칙 기반 충분성 체크로
workflow.add_edge("rdb", "result_check")
workflow.add_edge("vec", "result_check")
workflow.add_edge("graph", "result_check")

# 결과 충분성 검증 후 분기 (LLM 미사용 — AgentState 직접 체크)
workflow.add_conditional_edges(
    "result_check",
    route_result_check,
    {
        "gen": "gen",   # 셋 다 결과 있음 → 보고서 생성으로
        "end": "save",  # 하나라도 비어 있음 → 재질문 안내 후 종료(저장 경유)
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