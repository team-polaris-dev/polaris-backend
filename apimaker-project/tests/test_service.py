from __future__ import annotations

import asyncio
from typing import Any

import pytest

from apimaker.errors import SessionNotFoundError, UnsupportedProviderError
from apimaker.service import AgentService, _default_provider_factories
from apimaker.types import AgentOptions


class FakeProvider:
    name = "fake"

    def __init__(self) -> None:
        self.closed = False
        self.shutdown_called = False
        self.last_options: AgentOptions | None = None

    async def start_session(self, options: AgentOptions) -> list[str]:
        self.last_options = options
        return []

    async def send_message(self, session: list[str], prompt: str) -> str:
        session.append(prompt)
        if len(session) == 2 and "42" in session[0] and "8" in prompt:
            return "50"
        return f"turn {len(session)}"

    async def close_session(self, session: Any) -> None:
        self.closed = True

    async def shutdown(self) -> None:
        self.shutdown_called = True


def run(coro):
    return asyncio.run(coro)


def test_service_keeps_session_context() -> None:
    fake = FakeProvider()
    service = AgentService({"claude": lambda: fake})

    async def scenario() -> None:
        session = await service.start_session("claude")
        first = await service.send_message(session.session_id, "내가 좋아하는 숫자는 42야.")
        second = await service.send_message(session.session_id, "그 숫자에 8을 더하면?")

        assert first.response == "turn 1"
        assert second.response == "50"
        await service.close_session(session.session_id)
        assert fake.closed is True

    run(scenario())


def test_service_rejects_unknown_provider() -> None:
    service = AgentService({"claude": FakeProvider})

    async def scenario() -> None:
        with pytest.raises(UnsupportedProviderError):
            await service.start_session("codex")

    run(scenario())


def test_service_rejects_unknown_session() -> None:
    service = AgentService({"claude": FakeProvider})

    async def scenario() -> None:
        with pytest.raises(SessionNotFoundError):
            await service.send_message("missing", "hello")

    run(scenario())


def test_service_rejects_empty_prompt() -> None:
    service = AgentService({"claude": FakeProvider})

    async def scenario() -> None:
        session = await service.start_session("claude")
        with pytest.raises(ValueError):
            await service.send_message(session.session_id, "   ")

    run(scenario())


def test_default_provider_factories_include_gemini() -> None:
    factories = _default_provider_factories()

    assert sorted(factories) == ["claude", "codex", "gemini"]
