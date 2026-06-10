# config/llm.py — 전역 LLM 객체. 운영=Ollama, 테스트/오프라인=FakeLLM 폴백.
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage

load_dotenv()


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
