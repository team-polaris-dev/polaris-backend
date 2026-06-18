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
from nodes.rag import (rdb_search_node, vector_search_node, graph_search_node)
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
    # global(매크로/업계)은 시드 순회가 아니라 미리 만든 커뮤니티 요약만 읽으므로
    # 질의 재구성(ctx, ~13s)이 불필요 → graph 로 직접 보내 오버헤드 제거.
    # graph 노드가 intent=global 일 때 reconstructed_query 없으면 원문 질문으로 폴백.
    intent = state["intent"]
    if intent == "global":
        return "graph"
    return intent  # "direct" | "ctx"

def route_after_ctx(state: AgentState):
    """ctx(재구성) 이후 분기.

    ctx 는 로컬(ctx intent) 질의만 거친다 — global 은 route 에서 graph 로 직행하므로
    여기 오지 않는다. vec 와 graph 를 병렬 팬아웃하고, rdb 는 graph 뒤에 순차로 돈다
    (rdb 가 graph 앵커 corp_code/rcept_no 를 state 로 받아 결정론 SQL 을 짜기 때문 —
    LangGraph 병렬 분기는 state 를 공유하지 않아 rdb 를 graph 와 동시에 돌리면 앵커가 빈다).
    """
    return ["vec", "graph"]


def route_after_graph(state: AgentState):
    """graph 이후 분기.

    - global(매크로/업계): rdb/vec 를 돌리지 않으므로 바로 result_check 로.
    - 로컬(ctx): graph 가 채운 앵커로 rdb 가 결정론 검색을 잇는다(graph → rdb).
    """
    return "result_check" if state.get("intent") == "global" else "rdb"

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
# defer=True: vec(1홉)·rdb(2홉)가 서로 다른 super-step 에 도착해 result_check 가
# 두 번 실행되던 문제를 막는다. 두 갈래가 모두 끝난 뒤 1회만 실행 → 중복 로그 제거 +
# rdb 미반영 상태의 '미성숙 불충분' 판정 방지. (global 경로는 graph 단독→그대로 1회)
workflow.add_node("result_check", _timed("result_check", result_check_node), defer=True)
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
        "graph": "graph",   # global: ctx 우회하고 Global Search 단독 실행
    }
)

# 단순 잡담은 답변 후 바로 종료
workflow.add_edge("direct", END)

# ctx 이후(로컬 질의): vec/graph 병렬 팬아웃. rdb 는 graph 뒤에 순차(앵커 의존).
# (LangGraph conditional edge 는 리스트 반환으로 병렬 분기 지원.)
workflow.add_conditional_edges(
    "ctx",
    route_after_ctx,
    ["vec", "graph"],
)

# graph 이후: 로컬은 rdb 로(앵커 전달), global 은 바로 result_check.
workflow.add_conditional_edges(
    "graph",
    route_after_graph,
    {
        "rdb": "rdb",
        "result_check": "result_check",
    }
)

# 검색 결과가 모이면(로컬은 rdb+vec, 글로벌은 graph 단독) 규칙 기반 충분성 체크로.
workflow.add_edge("rdb", "result_check")
workflow.add_edge("vec", "result_check")

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