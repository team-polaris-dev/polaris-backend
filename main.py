# main.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# 작성해둔 LangGraph 컴파일 객체(app)를 불러옵니다.
from core.graph import app 

# FastAPI 애플리케이션 초기화
api = FastAPI(
    title="AI Agent API",
    description="LangGraph 기반 멀티 에이전트 RAG 서비스",
    version="1.0.0"
)

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
@api.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # LangGraph에 전달할 초기 상태(메시지) 세팅
        inputs = {"messages": [HumanMessage(content=request.message)]}
        
        # Checkpointer(메모리)가 대화 기록을 찾을 수 있도록 thread_id 설정
        config = {"configurable": {"thread_id": request.thread_id, "user_id": request.user_id}}
        
        # 그래프 실행 (invoke는 최종 결과만 반환합니다. 스트리밍이 필요하면 stream 사용)
        result = app.invoke(inputs, config)
        
        # 그래프의 최종 상태에서 필요한 데이터 추출
        final_message = result["messages"][-1].content
        final_intent = result.get("intent", "unknown")
        
        return ChatResponse(
            response=final_message,
            intent=final_intent
        )
        
    except Exception as e:
  
        raise HTTPException(status_code=500, detail=f"에이전트 처리 중 오류 발생: {str(e)}")

# 직접 실행할 때를 위한 엔트리포인트
if __name__ == "__main__":
    # uvicorn 웹 서버를 통해 FastAPI 앱 실행
    uvicorn.run("main:api", host="0.0.0.0", port=8000, reload=True)
