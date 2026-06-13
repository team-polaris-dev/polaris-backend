# main.py
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# 작성해둔 LangGraph 컴파일 객체(app)를 불러옵니다.
from core.graph import app

# 적재 파이프라인 관리자 콘솔 + 챗봇 통계
from routers.admin import router as admin_router
from services.pipeline_jobs import init_pipeline_tables, sweep_stale_jobs
from services.chat_logging import init_chat_tables


@asynccontextmanager
async def lifespan(_api: "FastAPI"):
    # 부팅 시 1회: 운영 메타 테이블 보장 + 죽은 워커가 남긴 잡 정리 + 챗봇 통계 테이블
    init_pipeline_tables()
    sweep_stale_jobs()
    init_chat_tables()
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

# 2. Response DTO 정의 (서버가 응답할 데이터)
class ChatResponse(BaseModel):
    response: str
    intent: str

# 3. POST 엔드포인트 구현
# sync def — app.invoke()가 블로킹이라 async def 로 두면 요청 내내 이벤트 루프가
# 정지한다(동시 요청·graphrag 의 자기 자신 /health 호출이 전부 타임아웃 → 500).
# sync 핸들러는 FastAPI 가 스레드풀에서 돌려 루프가 살아있다.
@api.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    import time as _time

    started = _time.perf_counter()
    try:
        # LangGraph에 전달할 초기 상태(메시지) 세팅
        inputs = {"messages": [HumanMessage(content=request.message)]}

        # Checkpointer(메모리)가 대화 기록을 찾을 수 있도록 thread_id 설정
        config = {"configurable": {"thread_id": request.thread_id}}

        # 그래프 실행 (invoke는 최종 결과만 반환합니다. 스트리밍이 필요하면 stream 사용)
        result = app.invoke(inputs, config)

        # 그래프의 최종 상태에서 필요한 데이터 추출
        final_message = result["messages"][-1].content
        final_intent = result.get("intent", "unknown")

        # 통계용 대화 로깅 (best-effort — 실패해도 응답엔 영향 없음)
        _log_chat_turn(request, result, final_message, final_intent,
                       latency_ms=int((_time.perf_counter() - started) * 1000))

        return ChatResponse(
            response=final_message,
            intent=final_intent
        )

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"에이전트 처리 중 오류 발생: {str(e)}")


def _log_chat_turn(request, result, final_message, final_intent, latency_ms):
    """대화 한 턴을 통계 테이블에 적재. 어떤 예외도 삼켜서 채팅 흐름을 막지 않는다."""
    try:
        from services.chat_logging import log_turn

        log_turn(
            user_id=request.user_id,
            session_id=request.thread_id,
            user_message=request.message,
            assistant_message=final_message,
            intent=final_intent,
            search_plan=result.get("search_plan"),
            is_sufficient=result.get("is_sufficient"),
            retry_count=result.get("retry_count"),
            latency_ms=latency_ms,
        )
    except Exception:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning("chat turn logging failed", exc_info=True)

# 직접 실행할 때를 위한 엔트리포인트
if __name__ == "__main__":
    # uvicorn 웹 서버를 통해 FastAPI 앱 실행
    uvicorn.run("main:api", host="0.0.0.0", port=8000, reload=True)