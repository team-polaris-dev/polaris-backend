import asyncio
import threading
from typing import Any, List, Optional

from langchain_core.language_models.llms import LLM

# apimaker 를 HTTP 서버로 띄우지 않고, 같은 프로세스 안에서 라이브러리로 직접 사용한다.
from apimaker import AgentOptions, AgentService


class _BackgroundLoop:
    """프로세스 전역에서 단일 백그라운드 이벤트 루프를 운영하는 헬퍼.

    LangChain 의 ``LLM._call`` 은 동기지만 apimaker ``AgentService`` 는 async 이고,
    내부적으로 claude CLI 서브프로세스를 띄운다. FastAPI 의 실행 중인 이벤트 루프
    안에서 동기로 호출되더라도(=app.invoke 경로) 안전하게 코루틴을 돌리기 위해,
    전용 데몬 스레드에서 별도의 이벤트 루프를 영구 실행하고
    ``run_coroutine_threadsafe`` 로 작업을 넘긴다.

    (Windows 의 서브프로세스 지원을 위해 기본 정책의 ProactorEventLoop 를 쓴다.)
    """

    _lock = threading.Lock()
    _instance: "Optional[_BackgroundLoop]" = None

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="apimaker-llm-loop",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def instance(cls) -> "_BackgroundLoop":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def run(self, coro: Any) -> Any:
        # 호출 스레드(이미 실행 중인 이벤트 루프 포함)를 블록한 채 결과를 기다린다.
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()


async def _ask_once(
    provider: str,
    model: str,
    system_prompt: str,
    reasoning_effort: str,
    prompt: str,
) -> str:
    """1회용 세션으로 한 번 질의하고 응답 문자열을 반환한다.

    매 호출마다 세션을 새로 열고 닫아 대화 메모리를 초기화한다
    (기존 HTTP 방식과 동일한 stateless 의미).
    """
    service = AgentService()
    try:
        session = await service.start_session(
            provider,
            AgentOptions(
                model=model,
                system_prompt=system_prompt,
                reasoning_effort=reasoning_effort,
            ),
        )
        result = await service.send_message(session.session_id, prompt)
        return result.response
    finally:
        await service.close_all()


class llm(LLM):
    """apimaker ``AgentService`` 를 in-process 로 직접 호출하는 LangChain LLM.

    별도의 HTTP 서버(``uvicorn apimaker.api``)를 띄울 필요 없이, 같은 프로세스
    안에서 claude CLI 세션을 열어 응답을 받는다.
    """

    provider: str = "claude"
    model_name: str = "claude-haiku-4-5"
    system_prompt: str = "Answer tersely."
    reasoning_effort: str = "high"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        return _BackgroundLoop.instance().run(
            _ask_once(
                self.provider,
                self.model_name,
                self.system_prompt,
                self.reasoning_effort,
                prompt,
            )
        )

    @property
    def _llm_type(self) -> str:
        return "apimaker_inprocess"
