# config/llm.py
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

load_dotenv()

# ==========================================
# [운영 환경용] 실제 오픈AI 모델 (현재는 주석 처리)
# ==========================================
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ==========================================
# [테스트 환경용] API 키가 필요 없는 가짜 모델
# ==========================================
class FakeLLM:
    def invoke(self, *args, **kwargs):
        print("[System] FakeLLM이 호출되었습니다.")
        # LangChain의 기본 메시지 형태로 반환하여 다른 노드에서 에러가 나지 않게 합니다.
        return AIMessage(content="API 호출 없이 생성된 임시 텍스트입니다.")

# 앱 전체에서는 이 가짜 객체를 llm으로 사용합니다.
llm = FakeLLM()