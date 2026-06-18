# main.py
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import time
from typing import Any, List

# 작성해둔 LangGraph 컴파일 객체(app)를 불러옵니다.
from core.graph import app

# 적재 파이프라인 관리자 콘솔 + 챗봇 통계
from routers.admin import router as admin_router
from services.pipeline_jobs import init_pipeline_tables, sweep_stale_jobs
from services.chat_logging import init_chat_tables
from core.serialize import serialize_state
from core.digest import build_evidence_digest
from nodes.router import empty_sources
from tool import chat_store
from tool.vector_store import warmup as _vector_warmup

@asynccontextmanager
async def lifespan(_api: "FastAPI"):
    # 부팅 시 1회: 운영 메타 테이블 보장 + 죽은 워커가 남긴 잡 정리 + 챗봇 통계 테이블
    init_pipeline_tables()
    sweep_stale_jobs()
    init_chat_tables()
    # 벡터 검색 콜드스타트(chunk 인덱스 17만 행 + BM25 빌드, ~2분) 제거.
    # 블로킹하면 기동·헬스체크가 지연되므로(§9 sync 핸들러 불변식) 백그라운드 스레드로 데운다.
    # 데우는 중 들어온 첫 검색은 _load_chunk_index 의 락에서 대기했다가 캐시를 공유한다.
    threading.Thread(target=_vector_warmup, daemon=True, name="vector-warmup").start()
    yield


# FastAPI 애플리케이션 초기화
api = FastAPI(
    title="AI Agent API",
    description="LangGraph 기반 멀티 에이전트 RAG 서비스 + 적재 파이프라인 콘솔",
    version="1.1.0",
    lifespan=lifespan,
)
# CORS — 개발(vite :5173) + 운영(nginx 동일출처). 관리자 토큰 헤더 허용.
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 적재 파이프라인 관리자 라우터 (/api/admin/*)
api.include_router(admin_router)
# 1. Request DTO 정의 (클라이언트가 보낼 데이터)
class ChatRequest(BaseModel):
    user_id: str
    message: str
    thread_id: str = "default_thread" # 대화 세션 유지를 위한 키값


# 로그인/회원가입 — 사용자이름만 받아 간단히 처리
class LoginRequest(BaseModel):
    username: str


class LoginResponse(BaseModel):
    user_id: str
    display_name: str
    is_new: bool


# 사이드바 세션 목록 항목
class SessionItem(BaseModel):
    session_id: str
    title: str
    preview: str = ""
    message_count: int = 0
    last_at: str = ""

# 2. Response DTO 정의 (서버가 응답할 데이터)
class GraphNode(BaseModel):
    id: str
    label: str
    category: str = "기업"


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str = ""
    label: str = ""
    rcept_no: str = ""


class GraphData(BaseModel):
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []


class DocumentItem(BaseModel):
    rcept_no: str = ""
    chunk_id: str = ""
    corp_name: str = ""
    title: str = ""
    doc_type: str = ""
    date: str = ""
    summary: str = ""
    section_path: str = ""
    year: Any = None
    score: Any = None
    text: str = ""
    source_kind: str = ""


class FinancialMetric(BaseModel):
    label: str = ""
    value: float = 0.0
    unit: str = ""


class FinancialGroup(BaseModel):
    corp_name: str = ""
    year: Any = None
    unit: str = ""
    metrics: List[FinancialMetric] = []


class ChatResponse(BaseModel):
    response: str
    intent: str
    # 저장된 어시스턴트 메시지 id — 프론트가 이 id 로 digest 를 나중에 요청한다.
    message_id: int = 0
    # 우측 패널이 자동으로 펼칠 탭 힌트: 'graph' | 'documents' | 'none'
    panel: str = "none"
    graph: GraphData = GraphData()
    documents: List[DocumentItem] = []
    financials: List[FinancialGroup] = []
    # 우측 패널 '원본 문서' 탭 상단의 LLM 통합 근거 정리본(마크다운). 없으면 빈 문자열.
    digest: str = ""


# 세션 복원용 메시지 항목 — 우측 패널(관계도/원본문서)까지 함께 복원한다.
class HistoryMessage(BaseModel):
    message_id: int
    role: str
    content: str
    intent: str = ""
    created_at: str = ""
    panel: str = "none"
    graph: GraphData = GraphData()
    documents: List[DocumentItem] = []
    financials: List[FinancialGroup] = []
    digest: str = ""

# 3-0. 로그인 / 회원가입 — 사용자이름만 입력. 처음이면 자동 가입.
@api.post("/api/login", response_model=LoginResponse)
def login_endpoint(request: LoginRequest):
    try:
        result = chat_store.login_or_signup(request.username)
        return LoginResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"로그인 처리 중 오류 발생: {str(e)}")

# 3. POST 엔드포인트 구현
# sync def — app.invoke()가 블로킹이라 async def 로 두면 요청 내내 이벤트 루프가
# 정지한다(동시 요청·graphrag 의 자기 자신 /health 호출이 전부 타임아웃 → 500).
# sync 핸들러는 FastAPI 가 스레드풀에서 돌려 루프가 살아있다.
@api.get("/api/sessions", response_model=List[SessionItem])
def sessions_endpoint(user_id: str):
    try:
        return [SessionItem(**s) for s in chat_store.list_sessions(user_id)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"세션 목록 조회 오류: {str(e)}")


# 3-2. 세션 복원용 — 특정 세션의 메시지 기록
@api.get("/api/sessions/{session_id}/messages", response_model=List[HistoryMessage])
def session_messages_endpoint(session_id: str):
    try:
        return [HistoryMessage(**m) for m in chat_store.list_messages(session_id)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"메시지 기록 조회 오류: {str(e)}")


# 3-3. POST 엔드포인트 구현
@api.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        # 대화 기록 영속화 — 세션 보장 + 사용자 메시지 저장
        # (DB 가 꺼져 있어도 챗봇 자체는 동작하도록 기록 실패는 무시한다)
        try:
            chat_store.ensure_session(request.thread_id, request.user_id)
            chat_store.save_message(
                request.thread_id, request.user_id, "user", request.message
            )
        except Exception as log_err:
            print(f"⚠️ 사용자 메시지 기록 실패(무시): {log_err}")

        # LangGraph에 전달할 초기 상태(메시지) 세팅
        inputs = {"messages": [HumanMessage(content=request.message)]}

        # Checkpointer(메모리)가 대화 기록을 찾을 수 있도록 thread_id 설정
        config = {"configurable": {"thread_id": request.thread_id}}

        # 그래프 실행 (invoke는 최종 결과만 반환합니다. 스트리밍이 필요하면 stream 사용)
        # 응답 지연(latency) 측정용 시작 시각
        started_at = time.perf_counter()
        result = app.invoke(inputs, config)
        latency_ms = int((time.perf_counter() - started_at) * 1000)

        # 그래프의 최종 상태에서 필요한 데이터 추출
        final_message = result["messages"][-1].content
        final_intent = result.get("intent", "unknown")

        # 최종 state → 우측 패널용 그래프/원본문서 페이로드
        # 단, result_check 가 재질문을 요청한 턴(필수 검색 소스 일부가 비어 END)에는
        # 그래프·우측 패널을 열지 않는다 — "결과 못 찾았다"는 답변과 패널이 모순되지
        # 않도록. 판정은 result_check 와 동일하게 empty_sources(필수 소스) 로 한다.
        # 참고: 글로벌(매크로/업계) 턴은 rdb/vec/graph 가 비어 community_results 만 채우므로
        # empty_sources 가 True → 우측 패널은 닫힌다(의도된 동작). 답변 본문(response)은
        # gen 이 community_results 로 생성하므로 아래 분기와 무관하게 그대로 반환된다.
        if empty_sources(result):
            panel_data = {"graph": {"nodes": [], "edges": []}, "documents": [], "financials": [], "panel": "none"}
        else:
            panel_data = serialize_state(result)

        # 우측 '원본 문서' 탭 상단의 통합 근거 정리(digest)는 LLM 1회 추가 호출이라
        # 동기로 만들면 답변이 그만큼 늦는다. 여기서 만들지 않고 답변을 먼저 반환한 뒤,
        # 프론트가 message_id 로 POST /api/chat/digest 를 호출해 나중에 채운다.
        panel_data["digest"] = ""

        # 어시스턴트 응답 기록 — 우측 패널 데이터(관계도/원본문서)도 함께 저장한다.
        # 스키마 변경 없이, 비어 있던 search_plan(longtext) 컬럼에 JSON 으로 보관해
        # 세션 재진입 시 패널 버튼을 그대로 복원할 수 있게 한다.
        # 나중 digest 계산을 위해 reconstructed_query 도 패널 JSON 에 보관한다
        # (digest 엔드포인트가 message_id 로 이 행을 읽어 문서·질문을 복원).
        message_id = 0
        try:
            message_id = chat_store.save_message(
                request.thread_id,
                request.user_id,
                "assistant",
                final_message,
                intent=final_intent,
                latency_ms=latency_ms,
                # 패널(관계도/문서)에 도구 사용 목록(search_plan)을 같은 JSON 으로 보관한다.
                # 통계의 tool_usage 가 이 한 행만 읽으므로 별도 log_turn 적재가 필요 없다
                # (이전엔 log_turn 이 같은 턴을 한 번 더 INSERT 해 카운트가 2배였음).
                panel={
                    **panel_data,
                    "tools": result.get("search_plan"),
                    "reconstructed_query": result.get("reconstructed_query") or request.message,
                },
            ) or 0
        except Exception as log_err:
            print(f"⚠️ 응답 기록 실패(무시): {log_err}")

        return ChatResponse(
            response=final_message,
            intent=final_intent,
            message_id=message_id,
            panel=panel_data["panel"],
            graph=panel_data["graph"],
            documents=panel_data["documents"],
            financials=panel_data.get("financials", []),
            digest="",
        )

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"에이전트 처리 중 오류 발생: {str(e)}")


# 3-4. 핵심 사실 정리(digest) — 답변 반환 후 프론트가 별도로 호출(지연 분리).
class DigestRequest(BaseModel):
    message_id: int


class DigestResponse(BaseModel):
    digest: str = ""


@api.post("/api/chat/digest", response_model=DigestResponse)
def chat_digest_endpoint(request: DigestRequest):
    """저장된 어시스턴트 메시지(message_id)의 문서로 통합 근거 정리본을 만들어 영속화·반환.

    /api/chat 가 digest 없이 답변을 먼저 보내고, 프론트가 답변 렌더 후 이 엔드포인트를
    호출한다. 이미 계산돼 저장돼 있으면(세션 재방문 등) LLM 재호출 없이 캐시를 돌려준다.
    """
    try:
        panel = chat_store.get_message_panel(request.message_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"메시지 조회 오류: {str(e)}")
    if panel is None:
        raise HTTPException(status_code=404, detail="메시지를 찾을 수 없습니다.")

    documents = panel.get("documents") or []
    if not documents:
        return DigestResponse(digest="")
    cached = panel.get("digest") or ""
    if cached:
        return DigestResponse(digest=cached)

    digest = build_evidence_digest(panel.get("reconstructed_query") or "", documents)
    if digest:
        try:
            chat_store.set_message_digest(request.message_id, digest)
        except Exception as e:
            print(f"⚠️ digest 저장 실패(무시): {e}")
    return DigestResponse(digest=digest)


# 직접 실행할 때를 위한 엔트리포인트
if __name__ == "__main__":
    # uvicorn 웹 서버를 통해 FastAPI 앱 실행
    uvicorn.run("main:api", host="0.0.0.0", port=8000, reload=True)