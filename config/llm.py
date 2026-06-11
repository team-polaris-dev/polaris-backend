import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

# Gemini 사용을 위한 LangChain 패키지 임포트
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 환경 변수에서 설정값 가져오기 (없을 경우 기본값 세팅)

MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# 1. 기본 LLM (문맥 재구성, 일반 응답, 결과 취합 등 일반적인 대화 및 요약용)
# RAG 파이프라인의 안정성을 위해 온도는 낮게 설정합니다.
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.1,
)

# 2. 구조화된 출력 전용 LLM (라우터 노드 등 JSON 형태의 엄격한 출력이 필요할 때)
# Gemini 1.5 모델은 JSON 모드를 지원하므로 response_mime_type을 설정해 줍니다.
json_llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.0,
    response_mime_type= "application/json"
)

print(f"✅ LLM 연결 준비 완료: {MODEL_NAME} (Google Gemini)")