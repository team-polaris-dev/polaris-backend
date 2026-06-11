# nodes/rag.py
import json
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import AgentState
from config.llm import json_llm



def rdb_search_node(state: AgentState):
    # 분리된 키인 rdb_results 사용 + 통일된 dict 규격 적용
    return {
        "rdb_results": [
            {
                "type": "rdb_row",
                "code": "CORP001",
                "name": "매출액",
                "value": "100억",
                "extra": {"currency": "KRW"},
                "source": "rcept_no_111"
            }
        ]
    }

def vector_search_node(state: AgentState):
    # 분리된 키인 vec_results 사용 + 통일된 dict 규격 적용
    return {
        "vec_results": [
            {
                "type": "vec_chunk",
                "code": "CORP001",
                "name": "사업 개요",
                "value": "당사는 AI 솔루션을 개발하는...",
                "extra": {"similarity_score": 0.92},
                "source": "chunk_id_222"
            }
        ]
    }

def graph_search_node(state: AgentState):
    # 분리된 키인 graph_facts 사용 + 통일된 dict 규격 적용
    return {
        "graph_facts": [
            {
                "type": "subsidiary",
                "code": "CORP002",
                "name": "자회사A",
                "value": "지분율 100%",
                "extra": {"relation": "owns"},
                "source": "rcept_no_333"
            }
        ],
        "graph_paths": [["CORP001", "owns", "CORP002"]],   # 멀티홉 경로 mock
        "graph_provenance": ["rcept_no_333"]               # 근거 rcept_no
    }

def synthesizer_node(state: AgentState):
    """
    Syn: 3개의 DB(RDB, Vector, Graph)에서 검색된 결과를 하나의 텍스트로 병합합니다.
    """
    print("🧩 [Syn Node] 검색 결과 취합 중...")
    
    rdb_results = state.get("rdb_results", [])
    vec_results = state.get("vec_results", [])
    graph_facts = state.get("graph_facts", [])
    
    synthesized_text = ""
    
    if rdb_results:
        synthesized_text += "### [정형 데이터 (RDB)]\n"
        for res in rdb_results:
            # 새로운 dict 규격에 맞춰 문자열 조립
            synthesized_text += f"- 항목: {res.get('name')}, 값: {res.get('value')} (출처: {res.get('source')})\n"
            
    if vec_results:
        synthesized_text += "\n### [비정형 문서 (Vector)]\n"
        for res in vec_results:
            synthesized_text += f"- 내용: {res.get('value')} (출처: {res.get('source')})\n"
            
    if graph_facts:
        synthesized_text += "\n### [관계망 데이터 (Graph)]\n"
        for res in graph_facts:
            synthesized_text += f"- 관계: {res.get('name')} - {res.get('value')} (출처: {res.get('source')})\n"

    # 검색된 결과가 전혀 없을 경우에 대한 방어 로직
    if not synthesized_text.strip():
        synthesized_text = "검색된 데이터가 없습니다."

    return {"synthesized_info": synthesized_text}

def reflection_node(state: AgentState):
    """
    Reflect: 취합된 정보가 사용자의 질문에 답변하기에 충분한지 자체 검증합니다.
    """
    query = state.get("reconstructed_query", "")
    info = state.get("synthesized_info", "")
    current_retry = state.get("retry_count", 0)
    
    # Ollama 환경: 명시적인 JSON 반환 프롬프트 구성
    system_prompt = SystemMessage(
        content=(
            "당신은 공시 분석 데이터 검증(Reflection) 전문가입니다. "
            "사용자의 질문에 답하기 위해 검색된 [취합된 정보]가 충분한지 엄격하게 평가하세요.\n\n"
            "[평가 기준]\n"
            "1. 질문에서 요구하는 핵심 수치, 기업명, 사실관계가 취합된 정보에 존재하는가?\n"
            "2. 정보가 부족해서 사용자가 원하는 형태의 답변을 생성할 수 없는가?\n\n"
            "반드시 아래 JSON 형식으로만 응답하세요:\n"
            '{"is_sufficient": true 또는 false, "reason": "충분한 이유 또는 누락된 정보 설명"}'
        )
    )
    
    human_prompt = HumanMessage(
        content=f"[질문]\n{query}\n\n[취합된 정보]\n{info}"
    )
    
    print("🔍 [Reflect Node] 데이터 충분성 자체 검증 중...")
    response = json_llm.invoke([system_prompt, human_prompt])
    
    try:
        parsed_response = json.loads(response.content)
        is_sufficient = parsed_response.get("is_sufficient", True) # 기본값은 True로 두어 무한루프 방지
        reason = parsed_response.get("reason", "검증 완료")
    except json.JSONDecodeError:
        print("⚠️ [Reflect Node] JSON 파싱 실패. 강제로 검증 통과(True) 처리합니다.")
        is_sufficient = True
        reason = "파싱 오류"
        
    print(f"   -> 검증 결과: {'통과✅' if is_sufficient else '불충분❌'} (사유: {reason})")
    
    # 정보가 충분하면 바로 gen 노드로 갈 수 있도록 상태 업데이트
    if not is_sufficient and current_retry >= 1: # 0, 1, 2 (총 2번 실패 시)
        print("🛑 [Reflect Node] 최대 재시도 횟수(2회)에 도달했습니다. 누락된 정보가 있더라도 최종 답변 생성을 강제 진행합니다.")
        return {
            "is_sufficient": True, # 강제로 True를 주어 gen 노드로 보내기
            "retry_count": current_retry + 1
        }
    
    # 정상적으로 충분한 경우
    if is_sufficient:
        return {
            "is_sufficient": True,
            "retry_count": current_retry # 통과 시 카운트 유지
        }
    
    # 💡 정보가 불충분할 경우:
    # 다음 턴의 Ctx 노드가 '무엇이 부족한지' 알 수 있도록 내부 피드백 메시지를 State에 추가합니다.
    feedback_message = SystemMessage(
        content=(
            f"[자체 검증 시스템 알림] 이전 검색 결과가 질문을 해결하기에 불충분했습니다. "
            f"누락된 원인: {reason} "
            f"이 피드백을 반영하여, 다른 키워드를 사용하거나 검색 범위를 넓히는 방향으로 질문을 다시 재구성하세요."
        )
    )
    
    return {
        "is_sufficient": False,
        "messages": [feedback_message],  # 이 메시지가 추가되어 Ctx 노드로 돌아갑니다.
        "retry_count": current_retry + 1
    }
