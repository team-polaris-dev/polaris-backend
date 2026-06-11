import json
from langchain_core.messages import SystemMessage

# 가정: core.state 와 config.llm 은 사용자 환경에 맞게 임포트됨
from core.state import AgentState
from config.llm import llm, json_llm  # json_llm 추가 임포트

# ---------------------------------------------------------
# 노드 구현
# ---------------------------------------------------------
def router_node(state: AgentState):
    """Route: 의도 분류"""
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "direct"}

    # Ollama 환경: Pydantic with_structured_output 대신 json_llm과 명시적 프롬프트 사용
    system_prompt = SystemMessage(
        content=(
            "당신은 공시 분석 서비스의 라우터입니다. "
            "사용자의 마지막 메시지와 대화 맥락을 분석하여, "
            "기업의 공시 정보, 재무제표, 관계사 등 데이터 검색이 필요하면 'ctx', "
            "단순한 인사, 감사 표시, 검색이 필요 없는 잡담이면 'direct'로 분류하세요. "
            "반드시 아래의 JSON 형식으로만 응답해야 합니다.\n"
            '{"intent": "ctx"} 또는 {"intent": "direct"}'
        )
    )
    
    response = json_llm.invoke([system_prompt] + messages)
    
    try:
        # LLM이 반환한 JSON 문자열을 딕셔너리로 파싱
        parsed_response = json.loads(response.content)
        intent = parsed_response.get("intent", "ctx")  # 파싱 성공했으나 intent 키가 없으면 ctx
    except json.JSONDecodeError:
        print(f"⚠️ [Router Node] JSON 파싱 실패. 원본 응답: {response.content}")
        print("기본값(ctx)으로 폴백합니다.")
        intent = "ctx"
        
    print(f"🧭 [Router Node] 의도 분류: {intent==('ctx') and 'RAG' or 'DIRECT'}")
    return {"intent": intent}

def direct_response_node(state: AgentState):
    """Direct: 단순 질문/잡담"""
    messages = state.get("messages", [])
    
    system_prompt = SystemMessage(
        content=(
            "당신은 친절하고 전문적인 공시 분석 서비스 AI 어시스턴트입니다. "
            "사용자의 가벼운 인사나 잡담에 자연스럽게 대답해주세요. "
            "답변 끝에는 '특정 기업의 공시나 재무 정보가 궁금하시다면 언제든 물어보세요!'와 같이 "
            "본래 서비스의 목적을 부드럽게 안내해 주어도 좋습니다."
        )
    )
    
    print("💬 [Direct Node] 단순 응답 생성 중...")
    response = llm.invoke([system_prompt] + messages)
    
    # 상태의 messages 리스트에 AI의 응답을 추가하여 반환
    return {"messages": [response]}

from langchain_core.messages import SystemMessage
from core.state import AgentState
from config.llm import llm

def context_reconstruct_node(state: AgentState):
    """Ctx: 질문 문맥 재구성 (메모리 활용)"""
    messages = state.get("messages", [])
    
        
    # 대화 메모리가 존재할 경우 대명사 치환 지시
    system_prompt = SystemMessage(
        content=(
            "당신은 공시 분석 시스템의 질의 재구성(Query Reconstruction) 전문가입니다. "
            "사용자가 '그 회사', '거기', '이전 내용' 같은 대명사나 주어를 생략한 표현을 사용했을 때, "
            "반드시 함께 제공된 이전 대화 기록(메모리)을 추적하여 해당 대명사가 지칭하는 정확한 기업명이나 핵심 키워드를 찾아내세요. "
            "이후, 이전 맥락을 모르는 사람이나 DB 검색기라도 완벽하게 이해할 수 있도록 독립적인 하나의 질문으로 다시 작성하세요. "
            "부연 설명이나 인사말 없이 오직 '재구성된 질문' 자체만 텍스트로 출력해야 합니다."
        )
    )
    
    print("✏️ [Context Node] 메모리를 참조하여 대명사/문맥 해석 중...")
    
    # SystemMessage와 이전 대화 기록(messages)을 모두 LLM에 전달하여 문맥 파악
    response = llm.invoke([system_prompt] + messages)
    reconstructed_query = response.content.strip()
    
    print(f"   -> 원본 질의: {messages[-1].content}")
    print(f"   -> 재구성됨: {reconstructed_query}")
    
    return {"reconstructed_query": reconstructed_query}