# main.py
import time
from typing import Any, List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

# 작성해둔 LangGraph 컴파일 객체(app)를 불러옵니다.
from core.graph import app
from core.serialize import serialize_state
from tool import chat_store

# FastAPI 애플리케이션 초기화
api = FastAPI(
    title="AI Agent API",
    description="LangGraph 기반 멀티 에이전트 RAG 서비스",
    version="1.0.0"
)
# 🌟 2. 반드시 api = FastAPI() 바로 아래에 위치해야 합니다!
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], # "OPTIONS"를 포함한 모든 메서드를 허용한다는 뜻
    allow_headers=["*"],
)
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


class ChatResponse(BaseModel):
    response: str
    intent: str
    # 우측 패널이 자동으로 펼칠 탭 힌트: 'graph' | 'documents' | 'none'
    panel: str = "none"
    graph: GraphData = GraphData()
    documents: List[DocumentItem] = []


# 세션 복원용 메시지 항목 — 우측 패널(관계도/원본문서)까지 함께 복원한다.
# GraphData/DocumentItem 을 기본값으로 쓰므로 두 클래스 정의 이후에 둔다.
class HistoryMessage(BaseModel):
    message_id: int
    role: str
    content: str
    intent: str = ""
    created_at: str = ""
    panel: str = "none"
    graph: GraphData = GraphData()
    documents: List[DocumentItem] = []

# 3-0. 로그인 / 회원가입 — 사용자이름만 입력. 처음이면 자동 가입.
@api.post("/api/login", response_model=LoginResponse)
async def login_endpoint(request: LoginRequest):
    try:
        result = chat_store.login_or_signup(request.username)
        return LoginResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"로그인 처리 중 오류 발생: {str(e)}")


# 3-1. 사이드바용 — 사용자의 대화 세션 목록
@api.get("/api/sessions", response_model=List[SessionItem])
async def sessions_endpoint(user_id: str):
    try:
        return [SessionItem(**s) for s in chat_store.list_sessions(user_id)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"세션 목록 조회 오류: {str(e)}")


# 3-2. 세션 복원용 — 특정 세션의 메시지 기록
@api.get("/api/sessions/{session_id}/messages", response_model=List[HistoryMessage])
async def session_messages_endpoint(session_id: str):
    try:
        return [HistoryMessage(**m) for m in chat_store.list_messages(session_id)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"메시지 기록 조회 오류: {str(e)}")


# 3-3. POST 엔드포인트 구현
@api.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
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
        panel_data = serialize_state(result)

        # 어시스턴트 응답 기록 — 우측 패널 데이터(관계도/원본문서)도 함께 저장한다.
        # 스키마 변경 없이, 비어 있던 search_plan(longtext) 컬럼에 JSON 으로 보관해
        # 세션 재진입 시 패널 버튼을 그대로 복원할 수 있게 한다.
        # 아울러 에이전트 처리 메타(응답 지연·검증 결과·재시도 횟수)도 함께 기록한다.
        try:
            chat_store.save_message(
                request.thread_id,
                request.user_id,
                "assistant",
                final_message,
                intent=final_intent,
                latency_ms=latency_ms,
                is_sufficient=result.get("is_sufficient"),
                retry_count=result.get("retry_count"),
                panel=panel_data,
            )
        except Exception as log_err:
            print(f"⚠️ 응답 기록 실패(무시): {log_err}")

        return ChatResponse(
            response=final_message,
            intent=final_intent,
            panel=panel_data["panel"],
            graph=panel_data["graph"],
            documents=panel_data["documents"],
        )

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"에이전트 처리 중 오류 발생: {str(e)}")

# 직접 실행할 때를 위한 엔트리포인트
if __name__ == "__main__":
    # uvicorn 웹 서버를 통해 FastAPI 앱 실행
    uvicorn.run("main:api", host="0.0.0.0", port=8000, reload=True)