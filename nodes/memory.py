import os
import time

from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore
from core.state import AgentState

def load_memory_node(state: AgentState, config: RunnableConfig, store: BaseStore):
    """
    Mem: 대화가 시작될 때 Store(장기 기억)에서 사용자 선호도를 불러옵니다.
    대화 기록(messages)은 Checkpointer가 이미 자동으로 state에 로드해줍니다.
    """
    # 1. config에서 사용자 ID 추출 (호출 시 넘겨준 값)
    user_id = config.get("configurable", {}).get("user_id", "default_user")

    # 2. Store에서 사용자 정보 조회 (namespace는 튜플, key는 문자열)
    namespace = ("user_profile", user_id)
    item = store.get(namespace, "preferences")

    # 3. 데이터가 있으면 가져오고, 없으면 기본값 세팅
    if item:
        user_prefs = item.value
    else:
        # DB에 저장된 선호도가 없을 경우의 초기 기본값
        user_prefs = {
            "tone": "친절하고 쉬운 설명", 
            "level": "초보자"
        }

    # 4. 조회한 데이터를 State에 업데이트 (+ 파이프라인 시작 시각 기록 — 총 소요시간 계산용)
    return {
        "user_preferences": user_prefs,
        "pipeline_started_at": time.perf_counter(),
        # ── 매 턴 근거 초기화 ──
        # 아래 필드들은 리듀서가 없어 "해당 노드가 실행될 때만" 덮어써진다. direct(비공시)
        # 경로처럼 검색 노드를 건너뛰는 턴에선 체크포인터에 남은 직전 공시 턴의 값이 그대로
        # 살아남아, has_required_evidence→True→serialize_state 로 옛 관계도/재무가 패널에 샌다.
        # 진입점(mem)에서 비워두면 ctx 경로는 이후 노드가 다시 채우고 direct 경로는 빈 채로 끝난다.
        # (messages 는 체크포인터가 관리하므로 건드리지 않아 대화 맥락엔 영향 없음)
        "rdb_results": [],
        "vec_results": [],
        "graph_facts": [],
        "community_results": [],
        "graph_paths": [],
        "graph_provenance": [],
        "graph_path_sources": [],
        "graph_path_chunks": [],
        "graph_hits": [],
        "graph_seeds": [],
        "graph_meta": {},
    }


def save_memory_node(state: AgentState, config: RunnableConfig, store: BaseStore):
    """
    Save: 대화 종료 전, 변경된 사용자 선호도나 장기 기억을 Store에 저장합니다.
    (예: 사용자가 "앞으로는 수치만 간략하게 말해줘"라고 요청해서 상태가 바뀌었을 경우)
    """
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    namespace = ("user_profile", user_id)

    # 1. 현재 State에 있는 최신 선호도를 가져옵니다.
    current_prefs = state.get("user_preferences", {})

    # 2. Store에 저장 (이후 다른 세션/thread_id로 접속해도 유지됨)
    store.put(namespace, "preferences", current_prefs)

    # 3. 파이프라인 종착점 — 검색 결과·노드별 소요시간·질문·총시간을 HTML 로 덤프.
    #    (덤프 실패가 응답 흐름을 막지 않게 방어)
    try:
        started = state.get("pipeline_started_at")
        total = (time.perf_counter() - started) if started else None
        from nodes.router import _dump_resultcheck_html
        path = _dump_resultcheck_html(state, total_elapsed=total)
        # file:// URI 는 VSCode/Windows Terminal 등에서 Ctrl+클릭으로 바로 열린다.
        print(f"   📄 ResultCheck 상세 덤프 (Ctrl+클릭): {path.resolve().as_uri()}")
        # POLARIS_RESULTCHECK_OPEN=1 이면 매 실행마다 기본 브라우저로 자동 오픈.
        if os.environ.get("POLARIS_RESULTCHECK_OPEN") in ("1", "true", "True"):
            import webbrowser
            webbrowser.open(path.resolve().as_uri())
    except Exception as exc:
        print(f"   ⚠️ ResultCheck HTML 덤프 실패: {exc}")

    # 이 노드에서는 상태를 더 이상 변형하지 않으므로 빈 딕셔너리 반환
    return {}