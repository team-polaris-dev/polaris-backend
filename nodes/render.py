from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState

# 앞서 설정한 LLM 객체 불러오기 (테스트 시에는 FakeLLM이, 실전엔 ChatOpenAI가 작동합니다)
from config.llm import llm 

def render_node(state: AgentState):
    """
    Render: 생성된 초안(final_draft) 또는 이전 대화 내용을 
    사용자 선호도(user_preferences)에 맞춰 최종 톤으로 변환합니다.
    """
    # 1. State에서 필요한 데이터 추출
    draft = state.get("final_draft", "")
    prefs = state.get("user_preferences", {})
    
    # 만약 RAG를 거치지 않고 "방금 답변 더 쉽게 말해줘"라고 라우팅되어 넘어온 경우라면,
    # 이전 AI의 마지막 답변을 draft로 사용합니다.
    if not draft and len(state["messages"]) > 1:
        draft = state["messages"][-2].content if state["messages"][-1].type == "human" else state["messages"][-1].content

    # 기본 선호도 설정 (Store에 값이 없을 경우를 대비)
    tone = prefs.get("tone", "친절하고 상세한")
    level = prefs.get("level", "일반인")

    # 2. 톤 변환을 위한 시스템 프롬프트 작성
    system_prompt = f"""당신은 사용자의 맞춤형 AI 어시스턴트입니다.
제공된 [원본 텍스트]를 다음 사용자의 성향에 맞게 완벽하게 재작성(Render)하세요.

[사용자 맞춤 설정]
- 설명 난이도: {level}
- 말투 및 톤: {tone}

[규칙]
1. 원본 텍스트의 핵심 정보(수치, 사실관계 등)는 절대 누락하거나 왜곡하지 마세요.
2. 지정된 '말투 및 톤'과 '설명 난이도'에 철저히 맞춰서 문장을 재구성하세요.
3. 불필요한 서론(예: "알겠습니다, 다시 작성해 드릴게요") 없이 바로 변환된 결과물만 출력하세요.
"""

    human_prompt = f"[원본 텍스트]\n{draft}"

    # 3. LLM 호출하여 렌더링 진행
    messages_to_llm = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    # 💡 팁: 만약 테스트 환경이라 config/llm.py에서 FakeLLM을 쓰고 있다면,
    # FakeLLM이 응답을 반환할 것이고, 실제 OpenAI 연결 시에는 톤이 변환된 텍스트가 나옵니다.
    response = llm.invoke(messages_to_llm)

    # 4. 최종 변환된 메시지를 State의 messages 배열에 추가하도록 반환
    return {"messages": [response]}