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

    # 4. 조회한 데이터를 State에 업데이트
    return {"user_preferences": user_prefs}


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

    # 이 노드에서는 상태를 더 이상 변형하지 않으므로 빈 딕셔너리 반환
    return {}