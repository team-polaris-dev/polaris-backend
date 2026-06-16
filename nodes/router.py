import json
import os
from functools import lru_cache
from pathlib import Path
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from core.state import AgentState
from config.llm import llm, json_llm

# DB 스키마/단어집 JSON. 파일 통째로 읽어 재구성 프롬프트에 그대로 주입한다.
# (POLARIS_GLOSSARY_PATH 로 경로 override, 기본 config/glossary.json)
_GLOSSARY_PATH = Path(
    os.environ.get("POLARIS_GLOSSARY_PATH")
    or Path(__file__).resolve().parent.parent / "config" / "glossary.json"
)


@lru_cache(maxsize=1)
def _load_glossary_text() -> str:
    """glossary.json 원문을 그대로 반환(프로세스 1회 캐시).

    파일이 없거나 읽기 실패하면 빈 문자열 — 프롬프트에서 단어집만 빠지고
    파이프라인은 정상 동작한다. JSON 갱신 시 반영하려면 프로세스 재시작.
    """
    try:
        return _GLOSSARY_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""


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

def _run_router(chat_history: str) -> dict:
    prompt_text = router_prompt.format(chat_history=chat_history)
    return parser.invoke(json_llm.invoke(prompt_text))


# ==========================================
# 3. 라우터 노드 함수 적용
# ==========================================
def router_node(state: dict):
    """Route: 의도 분류"""
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "direct"}

    chat_history_text = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in messages if hasattr(msg, 'type') and hasattr(msg, 'content')]
    )

    try:
        parsed_response = _run_router(chat_history_text)
        intent = parsed_response.get("intent", "ctx")

    except Exception as e:
        print(f"⚠️ [Router Node] 파싱 실패. 에러: {e}")
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
이 시스템은 RDB / Vector / Graph(Neo4j) 검색기를 함께 사용합니다. 사용자의 구어체 질문을 검색기가 잘 이해하도록, 아래 [DB 스키마 및 단어집]을 참고해 표준 용어가 드러나는 독립적인 질문으로 다시 작성하세요.

[작성 규칙]
1. '그 회사', '거기', '이전 내용' 같은 대명사·생략된 주어는 [대화 기록]을 추적해 정확한 기업명·키워드로 치환하세요.
2. 사용자의 일상어를 단어집의 표준 용어로 함께 풀어 쓰세요. 예) "돈 얼마 벌었어?" → "매출액", "사장" → "대표이사(임원)", "연간 실적" → "사업보고서(연간)". 단, 사용자가 명시하지 않은 연도·항목·기준을 임의로 지어내지는 마세요.
3. 이전 맥락을 모르는 사람이나 DB 검색기라도 완벽하게 이해할 수 있는 하나의 독립적인 질문으로 작성하세요.
4. 부연 설명이나 인사말 없이 오직 '재구성된 질문' 자체만 텍스트로 출력하세요.

[DB 스키마 및 단어집 (JSON)]
{glossary}

[대화 기록]
{chat_history}

재구성된 질문:""",
    input_variables=["chat_history", "glossary"]
)

def _run_reconstruct(chat_history: str) -> str:
    prompt_text = reconstruct_prompt.format(
        chat_history=chat_history,
        glossary=_load_glossary_text(),
    )
    return str_parser.invoke(llm.invoke(prompt_text))


def context_reconstruct_node(state: AgentState):
    """Ctx: 질문 문맥 재구성 (메모리 활용)"""
    messages = state.get("messages", [])

    print("✏️ [Context Node] 문장 재구성 중...")

    chat_history_text = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in messages if hasattr(msg, 'type') and hasattr(msg, 'content')]
    )

    reconstructed_query = _run_reconstruct(chat_history_text).strip()
    
    # 원본 질의 추출 (안전하게 처리)
    original_query = messages[-1].content if messages else "None"
    
    print(f"   -> 원본 질의: {original_query}")
    print(f"   -> 재구성됨: {reconstructed_query}")

    return {"reconstructed_query": reconstructed_query}


#======================================================================
# Result Check 노드 — 규칙 기반(LLM 미사용) 검색 결과 충분성 체크
#======================================================================
# 디버그 표시용 — 모든 검색 소스의 State 키 → 사용자에게 보여줄 한국어 명칭.
_SOURCE_LABELS: dict[str, str] = {
    "rdb_results": "정형 데이터(재무·공시 수치)",
    "vec_results": "문서 본문(공시 원문)",
    "graph_facts": "관계망 데이터(임원·주주·계열사 등)",
}

# gen 으로 통과하려면 결과가 비어 있으면 안 되는(필수) 소스.
#    "vec_results" 줄을 주석 처리하면 vec 가 비어도 통과한다.
_REQUIRED_SOURCES: list[str] = [
    "rdb_results",
    "vec_results",   #  각 디비 불안정 시 주석 하여 테스트 
    "graph_facts",
]


def empty_sources(state: AgentState) -> list[str]:
    """필수 소스 중 결과가 비어 있는 것들의 사용자용 명칭 목록을 반환한다.

    _REQUIRED_SOURCES 에 든 키만 검사한다(주석으로 vec 를 빼면 검사 제외).
    하나라도 비어 있으면 result_check 가 사용자에게 재질문을 유도한다.
    """
    return [
        _SOURCE_LABELS.get(key, key)
        for key in _REQUIRED_SOURCES
        if not state.get(key)
    ]


def route_result_check(state: AgentState) -> str:
    """result_check 분기: 필수 소스가 모두 있으면 'gen', 하나라도 비면 'end'."""
    return "end" if empty_sources(state) else "gen"


def result_check_node(state: AgentState):
    """Result Check: LLM 없이 AgentState 의 검색 결과 유무만 규칙 기반으로 점검한다.

    필수 소스(_REQUIRED_SOURCES)가 모두 채워져 있으면 그대로 통과시켜 gen 노드가
    포매팅하게 하고, 하나라도 비어 있으면 어떤 검색에서 결과가 없었는지 명시하며
    더 구체적인 질문을 요청하는 답변을 만들어 END 로 종료한다.
    """
    print("🔎 [ResultCheck Node] 검색 결과 충분성 점검 중...")

    # 디버깅: 각 DB에서 무엇이 조회됐는지 건수 + 첫 항목 샘플로 한눈에 확인.
    # (필수 여부와 무관하게 세 소스 모두 표시 — vec 가 0건인지 바로 보이게)
    for key, label in _SOURCE_LABELS.items():
        rows = state.get(key) or []
        sample = ""
        if rows:
            first = rows[0]
            sample = f" | 예: {first.get('name')} / {first.get('value')}"
        print(f"   - {label}: {len(rows)}건{sample}")

    empties = empty_sources(state)

    # 필수 소스가 모두 있으면 통과 — gen 으로.
    if not empties:
        print("   -> 통과✅ (필수 검색 결과 모두 존재)")
        return {}

    # 하나라도 비어 있으면 어떤 소스가 비었는지 명시하고 재질문을 유도한다.
    print(f"   -> 불충분❌ (결과 없음: {', '.join(empties)})")
    empty_text = ", ".join(empties)
    guidance = (
        f"{empty_text}에서 검색 결과를 찾지 못했습니다. "
        "기업명, 연도, 항목(예: 매출액, 자회사 등)을 포함해 좀 더 구체적으로 "
        "질문해 주시면 더 정확한 답변을 드릴 수 있습니다."
    )
    return {
        "messages": [AIMessage(content=guidance)],
        "final_draft": guidance,
    }