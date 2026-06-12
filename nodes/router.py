import json
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from core.state import AgentState
# llm/json_llm 은 이미 생성된 ApimakerLLM 인스턴스(호출 X).
from config.llm import llm, json_llm

# 2-2. JSON 자동 파서 생성
parser = JsonOutputParser()

# 2-3. 문자열 기반 프롬프트 템플릿 생성
router_prompt = PromptTemplate(
    template="""당신은 공시 분석 서비스의 라우터입니다.
사용자의 메시지와 대화 맥락을 분석하여, 기업의 공시 정보, 재무제표, 관계사 등 데이터 검색이 필요하면 'ctx', 단순한 인사, 감사 표시, 검색이 필요 없는 잡담이면 'direct'로 분류하세요.
설명이나 마크다운 없이 반드시 아래의 JSON 형식으로만 응답해야 합니다.
{{"intent": "ctx"}} 또는 {{"intent": "direct"}}

[대화 기록]
{chat_history}

결과 JSON:""",
    input_variables=["chat_history"]
)

# ApimakerLLM 은 Runnable 이 아니므로 LCEL 파이프(|) 대신 직접 연결한다.
# 의도 분류는 JSON 응답이 필요하므로 json_llm 을 사용한다.
def _run_router(chat_history: str) -> dict:
    prompt_text = router_prompt.format(chat_history=chat_history)
    return parser.invoke(json_llm.invoke(prompt_text))


# ==========================================
# 3. 라우터 노드 함수 적용
# ==========================================
def router_node(state: dict): # AgentState 대신 dict 타입 힌트 (환경에 맞게 수정)
    """Route: 의도 분류"""
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "direct"}

    # 1. 메시지 객체 리스트를 하나의 텍스트로 합칩니다.
    # (예: "human: 안녕\nai: 안녕하세요!")
    chat_history_text = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in messages if hasattr(msg, 'type') and hasattr(msg, 'content')]
    )
    
    try:
        # 2. 프롬프트 완성 -> json_llm 호출 -> JSON 파싱
        parsed_response = _run_router(chat_history_text)
        intent = parsed_response.get("intent", "ctx")
        
    except Exception as e:
        # JSONDecodeError 등 체인 실행 중 오류 발생 시 폴백 처리
        print(f"⚠️ [Router Node] 체인 실행/파싱 실패. 에러: {e}")
        print("기본값(ctx)으로 폴백합니다.")
        intent = "ctx"
        
    print(f"🧭 [Router Node] 의도 분류: {'RAG' if intent == 'ctx' else 'DIRECT'}")
    return {"intent": intent}

def direct_response_node(state: AgentState):
   
    response = "공시 관련 질문만 답변할 수 있습니다. 특정 기업의 공시나 재무 정보가 궁금하시다면 언제든 물어보세요!"
    
    return {"messages": [response]}

str_parser = StrOutputParser()

# 2-2. 문자열 기반 프롬프트 템플릿 생성
reconstruct_prompt = PromptTemplate(
    template="""당신은 공시 분석 시스템의 질의 재구성(Query Reconstruction) 전문가입니다.
사용자가 '그 회사', '거기', '이전 내용' 같은 대명사나 주어를 생략한 표현을 사용했을 때, 반드시 함께 제공된 이전 대화 기록(메모리)을 추적하여 해당 대명사가 지칭하는 정확한 기업명이나 핵심 키워드를 찾아내세요.
이후, 이전 맥락을 모르는 사람이나 DB 검색기라도 완벽하게 이해할 수 있도록 독립적인 하나의 질문으로 다시 작성하세요.
부연 설명이나 인사말 없이 오직 '재구성된 질문' 자체만 텍스트로 출력해야 합니다.

[대화 기록]
{chat_history}

재구성된 질문:""",
    input_variables=["chat_history"]
)

# 문맥 재구성은 일반 텍스트 응답 → llm 사용, AIMessage.content 를 문자열로 파싱.
def _run_reconstruct(chat_history: str) -> str:
    prompt_text = reconstruct_prompt.format(chat_history=chat_history)
    return str_parser.invoke(llm.invoke(prompt_text))


def context_reconstruct_node(state: AgentState):
    """Ctx: 질문 문맥 재구성 (메모리 활용)"""
    messages = state.get("messages", [])
    
    print("✏️ [Context Node] 문장 재구성 중...")
    
    # 1. 메시지 객체들을 하나의 텍스트로 합치기
    chat_history_text = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in messages if hasattr(msg, 'type') and hasattr(msg, 'content')]
    )
    
    # 2. 프롬프트 완성 -> llm 호출 -> 문자열로 파싱
    reconstructed_query = _run_reconstruct(chat_history_text).strip()
    
    # 원본 질의 추출 (안전하게 처리)
    original_query = messages[-1].content if messages else "None"
    
    print(f"   -> 원본 질의: {original_query}")
    print(f"   -> 재구성됨: {reconstructed_query}")
    
    return {"reconstructed_query": reconstructed_query}