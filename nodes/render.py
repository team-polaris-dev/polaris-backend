from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.state import AgentState

# config.llm 의 llm 은 이미 생성된 ApimakerLLM 인스턴스다(호출 X).
from config.llm import llm

def generate_report_node(state: AgentState):
    """
    Gen (Generate & Render): 
    검색 및 취합된 정보(synthesized_info)를 바탕으로, 
    사용자 선호도(user_preferences)에 맞춰 최종 분석 보고서를 생성합니다.
    """
    # 1. State에서 필요한 데이터 추출 (syn 노드에서 만든 취합 정보)
    # 만약 RAG 흐름을 정상적으로 탔다면 synthesized_info가 존재합니다.
    draft_info = state.get("synthesized_info", "")
    prefs = state.get("user_preferences", {})
    
    # 💡 엣지 케이스 처리: 
    # RAG 검색 없이 "방금 한 말 좀 더 쉽게 설명해줘"라고 직접 넘어온 경우,
    # 이전 AI의 마지막 답변을 재구성의 재료로 사용합니다.
    if not draft_info and len(state.get("messages", [])) > 1:
        messages = state["messages"]
        # 마지막 메시지가 사용자라면, 그 앞의 AI 답변을 가져옴
        if messages[-1].type == "human":
            draft_info = messages[-2].content
        else:
            draft_info = messages[-1].content

    # 기본 선호도 설정 (Store에 값이 없을 경우를 대비)
    tone = prefs.get("tone", "전문적이고 신뢰감 있는")
    level = prefs.get("level", "일반 투자자")

    # 2. 정보 병합 및 톤 변환을 동시에 수행하는 시스템 프롬프트 (Qwen2.5 최적화)
    system_prompt = f"""당신은 기업 공시 및 재무 분석을 도와주는 전문적이고 친절한 AI 어시스턴트입니다.
제공된 [분석 기초 자료]를 활용하여, 다음 사용자의 성향에 맞춘 완벽한 최종 답변(보고서)을 작성하세요.

[사용자 맞춤 설정]
- 설명 난이도: {level}
- 말투 및 톤: {tone}

[작성 규칙]
1. [분석 기초 자료]에 포함된 핵심 정보(수치, 사실관계, 기업명 등)는 절대 누락하거나 왜곡하지 마세요.
2. 지정된 '말투 및 톤'과 '설명 난이도'에 철저히 맞춰서 문장을 자연스럽게 구성하세요.
3. 가독성을 높이기 위해 필요하다면 마크다운(글머리 기호, 굵은 글씨 등)을 적극적으로 활용하세요.
4. "알겠습니다", "요청하신 보고서입니다" 같은 불필요한 서론 없이, 곧바로 최종 결과물만 출력하세요.
"""

    human_prompt = f"[분석 기초 자료]\n{draft_info}"

    # 3. LLM 호출하여 렌더링 진행
    messages_to_llm = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    print(f"📝 [Gen Node] 최종 보고서 작성 중... (타겟: {level}, 톤: {tone})")
    # ApimakerLLM.invoke 는 AIMessage 를 반환하므로 .content 로 문자열만 꺼낸다.
    response_text = llm.invoke(messages_to_llm).content

    # 4. 최종 변환된 메시지를 State의 messages 배열에 추가하고,
    # final_draft 키에도 명시적으로 저장하여 상태 업데이트
    return {
        "messages": [AIMessage(content=response_text)],
        "final_draft": response_text,
    }