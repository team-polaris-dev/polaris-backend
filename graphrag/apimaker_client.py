"""graphrag LLM 어댑터 — 백엔드 본체와 같은 in-process LLM(config.llm)에 위임.

구 구현은 별도 apimaker HTTP 서버(기본 127.0.0.1:8000)를 전제한 테스트 시절
자산이었는데, 이 환경에선 그 서버가 없어(8000 = polaris 백엔드 자신) 의도분류가
항상 rule fallback 으로 degrade 됐다. 2026-06-12 폐기하고 config/llm.py 의
ApimakerLLM(in-process Gemini CLI) 단일 경로로 통합 — LLM 접속정보 이중 관리 금지.

공개 표면(호출자 intent_classifier·text_to_cypher 는 무수정 호환):
    ApimakerClient(model=...).chat(system_prompt, user_prompt) -> str
    NoApimakerAvailable  — 어댑터 초기화 실패 (호출자가 rule fallback 으로 degrade)
    ApimakerError        — 호출 실패 (호출자가 재시도/fallback 결정)
"""
from __future__ import annotations


class NoApimakerAvailable(Exception):
    """in-process LLM 초기화 실패. 초기화 단계에서만 발생."""


class ApimakerError(Exception):
    """LLM 호출 실패. 호출자가 재시도/fallback 결정."""


class ApimakerClient:
    """구 HTTP 클라이언트와 같은 표면, 내부는 config.llm 의 in-process Gemini.

    graphrag 의 호출은 모두 JSON 출력을 기대하므로 json_mode=True 로 생성한다
    (JSON 지시문 추가 + 응답에서 JSON 블록 추출).
    """

    def __init__(
        self,
        base_url: str | None = None,  # 구 시그니처 호환용 — 더 이상 의미 없음
        model: str | None = None,
        provider: str | None = None,
        timeout: float | None = None,
    ) -> None:
        try:
            from config.llm import ApimakerLLM
        except Exception as e:  # noqa: BLE001 — import 실패는 전부 degrade 신호
            raise NoApimakerAvailable(f"config.llm import 실패: {e}") from e

        kwargs: dict = {"json_mode": True}
        if model:
            kwargs["model"] = model
        if provider:
            kwargs["provider"] = provider
        if timeout:
            kwargs["timeout"] = timeout
        self._llm = ApimakerLLM(**kwargs)
        self.provider = self._llm.provider
        self.model = self._llm.model or f"{self.provider}-cli-default"

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """system + user → 모델 응답 텍스트(JSON 추출됨). 실패 시 ApimakerError."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            resp = self._llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            return str(resp.content)
        except Exception as e:  # noqa: BLE001 — 타임아웃 포함 전부 호출자 fallback 으로
            raise ApimakerError(f"in-process LLM 호출 실패: {e}") from e
