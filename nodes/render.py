from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.state import AgentState
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

    # 사용자 질문 — 이게 빠지면 LLM 이 질문과 무관한 일반 기업소개 보고서를 짓는다.
    question = state.get("reconstructed_query") or ""
    if not question:
        for msg in reversed(state.get("messages", []) or []):
            if getattr(msg, "type", "") == "human":
                question = str(msg.content)
                break

    # 2. 정보 병합 및 톤 변환을 동시에 수행하는 시스템 프롬프트
    system_prompt = f"""당신은 기업 공시 및 재무 분석을 도와주는 전문적이고 친절한 AI 어시스턴트입니다.
[분석 기초 자료]만을 근거로 [사용자 질문]에 직접 답하는 최종 답변을 작성하세요.

[사용자 맞춤 설정]
- 설명 난이도: {level}
- 말투 및 톤: {tone}

[작성 규칙]
1. 답변의 중심은 반드시 [사용자 질문]에 대한 직접적인 답이어야 합니다.
   질문과 무관한 일반 기업 소개·연혁·홍보성 서사로 흐르지 마세요.
2. [분석 기초 자료]에 있는 사실·수치·관계만 사용하세요. 자료에 없는 내용
   (사전 학습 지식, 추정·재계산한 수치)을 추가하지 마세요.
3. 자료가 질문의 일부를 못 다루면 그 부분은 "자료에 없다"고 솔직하게 밝히세요.
4. [분석 기초 자료] 중 질문과 관련된 핵심 정보(수치, 사실관계, 기업명, 관계)는
   누락·왜곡하지 마세요. 특히 [관계망 데이터 (Graph)]의 'A → B' 관계는
   방향(누가 누구에게)을 정확히 유지하세요.
5. 지정된 '말투 및 톤'과 '설명 난이도'에 맞춰 자연스럽게 구성하고,
   가독성을 위해 마크다운을 적극 활용하세요.
6. "알겠습니다", "요청하신 보고서입니다" 같은 서론 없이 곧바로 최종 결과물만 출력하세요.
"""

    human_prompt = f"[사용자 질문]\n{question}\n\n[분석 기초 자료]\n{draft_info}"

    # 3. LLM 호출하여 렌더링 진행
    messages_to_llm = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    print(f"📝 [Gen Node] 최종 보고서 작성 중... (타겟: {level}, 톤: {tone})")

    # 커스텀 LLM(LLM 기반)은 문자열을 반환하므로 AIMessage 로 감싸 state 에 넣는다.

    response_text = llm.invoke(messages_to_llm).content

    # 4. 최종 변환된 메시지를 State의 messages 배열에 추가하고,
    # final_draft 키에도 명시적으로 저장하여 상태 업데이트
    return {
        "messages": [AIMessage(content=response_text)],
        "final_draft": response_text,
    }