# config/llm.py — 전역 LLM 객체. 운영=Ollama, 테스트/오프라인=FakeLLM 폴백.
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage

load_dotenv()

<<<<<<< HEAD
# ==========================================
# [운영 환경용] 실제 오픈AI 모델 (현재는 주석 처리)
# ==========================================
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# .env 파일 로드
load_dotenv()

# 환경 변수에서 설정값 가져오기 (없을 경우 기본값 세팅)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "gemma4:latest")

# 1. 기본 LLM (문맥 재구성, 일반 응답, 결과 취합 등 일반적인 대화 및 요약용)
# RAG 파이프라인의 안정성을 위해 온도는 낮게 설정합니다.
llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=MODEL_NAME,
    temperature=0.1,
)

# 2. 구조화된 출력 전용 LLM (라우터 노드 등 JSON 형태의 엄격한 출력이 필요할 때)
# Qwen2.5 모델은 JSON 모드를 꽤 잘 지원하므로 라우팅 에러가 적을 것입니다.
json_llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=MODEL_NAME,
    temperature=0.0,
    format="json"
)

print(f"✅ LLM 연결 준비 완료: {MODEL_NAME} (서버: {OLLAMA_BASE_URL})")

# ==========================================
# [테스트 환경용] API 키가 필요 없는 가짜 모델
# ==========================================
# class FakeLLM:
#     def invoke(self, *args, **kwargs):
#         print("[System] FakeLLM이 호출되었습니다.")
#         # LangChain의 기본 메시지 형태로 반환하여 다른 노드에서 에러가 나지 않게 합니다.
#         return AIMessage(content="API 호출 없이 생성된 임시 텍스트입니다.")

# # 앱 전체에서는 이 가짜 객체를 llm으로 사용합니다.
# llm = FakeLLM()
=======

class FakeLLM:
    """API/서버 없이 동작하는 더미 LLM (테스트·오프라인용)."""

    def invoke(self, *args, **kwargs):
        print("[System] FakeLLM이 호출되었습니다.")
        return AIMessage(content="API 호출 없이 생성된 임시 텍스트입니다.")


def _build_llm():
    if os.getenv("USE_FAKE_LLM", "").lower() in ("1", "true", "yes"):
        return FakeLLM()
    try:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=os.getenv("OLLAMA_BASE", "http://localhost:11434"),
            model=os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:14b"),
            temperature=0,
        )
    except Exception as e:  # 패키지 미설치/초기화 실패 → 폴백
        print(f"[System] Ollama LLM 초기화 실패 → FakeLLM 폴백: {e}")
        return FakeLLM()


# 앱 전역에서 이 객체를 llm 으로 사용한다.
llm = _build_llm()
>>>>>>> refs/rewritten/dev-2
