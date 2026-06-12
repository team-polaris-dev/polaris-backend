import asyncio

import json
import os
import re
import shutil
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage

load_dotenv()

# ========================================================================
# apimaker 를 직접 import 해 in-process 로 쓰는 LLM 어댑터
# ------------------------------------------------------------------------
# 별도 HTTP 서버(uvicorn) 를 띄우지 않고, apimaker 의 AgentService 를 이 프로세스
# 안에서 직접 호출한다. Gemini 프로바이더는 로컬 `gemini` CLI 를 서브프로세스로
# 띄워 OAuth(구독) 인증으로 응답을 받는다 (API 키 불필요).
#
# 전제:
#   1. 로컬 `gemini` CLI 설치 + 1회 로그인 (Google AI Pro/Ultra 계정)
#   2. apimaker 코어 의존성(fastapi/httpx/pydantic)은 백엔드 env 에 이미 존재
#      (claude/codex SDK 는 지연 import 라 미설치여도 무방 — Gemini 만 쓰면 됨)
#
# 동기↔비동기 브리지:
#   백엔드 노드는 llm.invoke() 를 동기로 호출하지만, main.py 의 async 엔드포인트가
#   app.invoke() 를 블로킹 실행하므로 호출 스레드에는 uvicorn 이벤트 루프가 돈다.
#   그 안에서 asyncio.run() 은 불가하므로, 전용 백그라운드 스레드의 이벤트 루프에
#   코루틴을 제출(run_coroutine_threadsafe)해 결과를 받는다.
# ========================================================================

# --- apimaker 패키지를 import 가능하게 (repo 내부 src 레이아웃) ---
_DEFAULT_SRC = Path(__file__).resolve().parent.parent / "apimaker-project" / "src"
_APIMAKER_SRC = Path(os.environ.get("APIMAKER_SRC_PATH", str(_DEFAULT_SRC)))
if _APIMAKER_SRC.is_dir() and str(_APIMAKER_SRC) not in sys.path:
    sys.path.insert(0, str(_APIMAKER_SRC))

from apimaker import AgentOptions, AgentService  # noqa: E402


# --- Windows: gemini CLI 런처 보정 (apimaker-project 는 수정하지 않음) ---
# npm 전역설치 gemini 는 Windows 에서 gemini.cmd/.ps1 형태라, apimaker 가 쓰는
# asyncio.create_subprocess_exec("gemini") 가 gemini.exe 를 못 찾아 실패한다.
# 셸(.cmd) 우회는 프롬프트(한글/JSON/특수문자) 인용 문제가 있으므로, 더 견고하게
# `node <gemini.js>` 를 직접 실행하도록 명령어 빌더(_gemini_cli_args)만 몽키패치한다.
def _resolve_gemini_launcher() -> list[str] | None:
    """['<node>', '<gemini.js>'] 반환. 해석 실패 시 None(원래 'gemini' 유지)."""
    js = os.environ.get("GEMINI_JS")
    node = os.environ.get("GEMINI_NODE") or shutil.which("node")
    if not js:
        which = shutil.which("gemini")
        if which:
            base = Path(which).parent
            for cand in (
                base / "node_modules/@google/gemini-cli/bundle/gemini.js",
                base / "node_modules/@google/gemini-cli/dist/index.js",
            ):
                if cand.is_file():
                    js = str(cand)
                    break
    if js and node and Path(js).is_file():
        return [node, js]
    return None


def _patch_gemini_for_windows() -> None:
    if sys.platform != "win32":
        return
    launcher = _resolve_gemini_launcher()
    if not launcher:
        return  # 못 찾으면 원본 유지 → apimaker 가 명확한 설치 안내 에러를 던짐
    import apimaker.providers.gemini as _gem

    _orig_args = _gem._gemini_cli_args

    def _patched_args(session, prompt):
        args = _orig_args(session, prompt)
        if args and args[0] == "gemini":
            return launcher + args[1:]
        return args

    _gem._gemini_cli_args = _patched_args


_patch_gemini_for_windows()


# 환경 변수 (하드코딩 금지 — .env 기반)
APIMAKER_PROVIDER = os.environ.get("APIMAKER_PROVIDER", "gemini")
# 모델 미지정 시 gemini CLI 기본값 사용. 지정 시 -m 로 전달 (예: gemini-2.5-flash).
APIMAKER_MODEL = os.environ.get("APIMAKER_MODEL") or os.environ.get("GEMINI_MODEL_NAME") or None
APIMAKER_TIMEOUT = float(os.environ.get("APIMAKER_TIMEOUT", "180"))

_JSON_DIRECTIVE = (
    "위 지시에 따라, 코드펜스(```)나 부연 설명 없이 유효한 JSON 객체 하나만 출력하세요."
)


# ------------------------------------------------------------------ async 브리지
class _BackgroundLoop:
    """동기 코드에서 코루틴을 실행하기 위한 전용 백그라운드 이벤트 루프(싱글톤).

    호출 스레드에 이미 이벤트 루프가 돌고 있어도 안전하게 동작한다.
    Windows 는 서브프로세스(gemini CLI) 지원을 위해 ProactorEventLoop 가 필요하다.
    """

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        with self._lock:
            if self._loop is None:
                if sys.platform == "win32":
                    loop = asyncio.ProactorEventLoop()
                else:
                    loop = asyncio.new_event_loop()
                threading.Thread(
                    target=loop.run_forever, daemon=True, name="apimaker-loop"
                ).start()
                self._loop = loop
        return self._loop

    def run(self, coro, timeout: float | None = None):
        loop = self._ensure()
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout)


_RUNNER = _BackgroundLoop()
_SERVICE = AgentService()  # 프로세스 전역 단일 서비스 (프로바이더 캐시)


# ------------------------------------------------------------------ 응답/유틸
def _split_messages(model_input) -> tuple[str | None, str]:
    """invoke 입력(list[Message]|str)을 (system_prompt, user_prompt)로 평탄화.

    - SystemMessage 들은 합쳐 system_prompt 로 분리(AgentOptions.system_prompt).
    - 나머지(Human/AI/문자열)는 한 프롬프트로 합친다. 멀티턴이면 역할 라벨을 붙인다.
    """
    if isinstance(model_input, str):
        return None, model_input

    systems: list[str] = []
    convo: list[tuple[str, str]] = []
    for m in model_input:
        if isinstance(m, str):
            convo.append(("user", m))
            continue
        mtype = getattr(m, "type", None)
        content = getattr(m, "content", None)
        content = str(content) if content is not None else str(m)
        if mtype == "system":
            systems.append(content)
        elif mtype == "ai":
            convo.append(("assistant", content))
        else:  # human / unknown
            convo.append(("user", content))

    system_text = "\n\n".join(s for s in systems if s) or None

    if len(convo) == 1 and convo[0][0] == "user":
        user_text = convo[0][1]
    else:
        lines = []
        for role, content in convo:
            lines.append(f"{'User' if role == 'user' else 'Assistant'}: {content}")
        user_text = "\n".join(lines)
    return system_text, user_text


def _extract_json(text: str) -> str:
    """LLM 응답에서 JSON 객체만 추출 (코드펜스/앞뒤 잡설 제거)."""
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.IGNORECASE | re.DOTALL)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text.strip()


class _StructuredWrapper:
    """with_structured_output(Schema) 결과. invoke(prompt) -> Schema 인스턴스."""

    def __init__(self, parent: "ApimakerLLM", schema):
        self._parent = parent
        self._schema = schema

    def invoke(self, model_input):
        keys = list(getattr(self._schema, "model_fields", {}).keys())
        keys_hint = ", ".join(f'"{k}"' for k in keys) if keys else "스키마 필드"
        system_text, user_text = _split_messages(model_input)
        user_text = (
            f"{user_text}\n\n다음 키를 가진 JSON 객체 하나만 출력하세요: {keys_hint}.\n"
            f"{_JSON_DIRECTIVE}"
        )
        raw = self._parent._call(system_text, user_text)
        data = json.loads(_extract_json(raw))
        return self._schema(**data)


class ApimakerLLM:
    """apimaker AgentService(in-process) 경유 LLM. 기존 ChatModel 최소 표면만 구현.

    지원: invoke(list[Message]|str) -> AIMessage(.content), with_structured_output(Schema).
    각 invoke 는 stateless — 세션을 만들고 한 번 보내고 닫는다(맥락은 messages 로 전달).
    """

    def __init__(
        self,
        *,
        provider: str = APIMAKER_PROVIDER,
        model: str | None = APIMAKER_MODEL,
        json_mode: bool = False,
        system_prompt: str | None = None,
        timeout: float = APIMAKER_TIMEOUT,
    ):
        self.provider = provider
        self.model = model
        self.json_mode = json_mode
        self.system_prompt = system_prompt
        self.timeout = timeout

    async def _one_shot(self, system_text: str | None, user_text: str) -> str:
        """apimaker 세션 1회 왕복 (start -> send -> close)."""
        options = AgentOptions(model=self.model, system_prompt=system_text)
        session = await _SERVICE.start_session(self.provider, options)
        try:
            result = await _SERVICE.send_message(session.session_id, user_text)
            return result.response
        finally:
            try:
                await _SERVICE.close_session(session.session_id)
            except Exception:
                pass  # 세션 정리 실패는 무시 (메모리 세션이라 프로세스 종료 시 소멸)

    def _call(self, system_text: str | None, user_text: str) -> str:
        merged_system = "\n\n".join(s for s in (self.system_prompt, system_text) if s) or None
        return _RUNNER.run(self._one_shot(merged_system, user_text), timeout=self.timeout)

    def invoke(self, model_input) -> AIMessage:
        system_text, user_text = _split_messages(model_input)
        if self.json_mode:
            user_text = f"{user_text}\n\n{_JSON_DIRECTIVE}"
        text = self._call(system_text, user_text)
        if self.json_mode:
            text = _extract_json(text)
        return AIMessage(content=text)

    def with_structured_output(self, schema):
        return _StructuredWrapper(self, schema)


# 1. 기본 LLM (문맥 재구성, 일반 응답, 결과 취합 등)
llm = ApimakerLLM(json_mode=False)

# 2. 구조화 출력 전용 LLM (라우터/리플렉션 등 JSON 강제)
json_llm = ApimakerLLM(json_mode=True)

print(
    f"✅ LLM 연결 준비 완료: provider={APIMAKER_PROVIDER}"
    f"{f', model={APIMAKER_MODEL}' if APIMAKER_MODEL else ''} (apimaker in-process)"
)
