"""Provider-neutral session service."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from apimaker.errors import SessionNotFoundError, UnsupportedProviderError
from apimaker.providers.base import AgentProvider
from apimaker.providers.claude import ClaudeProvider
from apimaker.providers.codex import CodexProvider
from apimaker.providers.gemini import GeminiProvider
from apimaker.types import AgentOptions, AgentResponse, SessionInfo

ProviderFactory = Callable[[], AgentProvider]


@dataclass
class _StoredSession:
    provider_name: str
    provider: AgentProvider
    handle: Any


class AgentService:
    """In-memory session manager for local development."""

    def __init__(
        self,
        provider_factories: Mapping[str, ProviderFactory] | None = None,
    ) -> None:
        self._provider_factories = dict(provider_factories or _default_provider_factories())
        self._providers: dict[str, AgentProvider] = {}
        self._sessions: dict[str, _StoredSession] = {}

    async def start_session(
        self,
        provider: str,
        options: AgentOptions | None = None,
    ) -> SessionInfo:
        normalized_provider = provider.lower().strip()
        provider_impl = self._get_provider(normalized_provider)
        handle = await provider_impl.start_session(options or AgentOptions())
        session_id = str(uuid4())
        self._sessions[session_id] = _StoredSession(
            provider_name=normalized_provider,
            provider=provider_impl,
            handle=handle,
        )
        return SessionInfo(session_id=session_id, provider=normalized_provider)

    async def send_message(self, session_id: str, prompt: str) -> AgentResponse:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"unknown session id: {session_id}")

        response = await session.provider.send_message(session.handle, prompt)
        return AgentResponse(
            session_id=session_id,
            provider=session.provider_name,
            response=response,
        )

    async def close_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            raise SessionNotFoundError(f"unknown session id: {session_id}")
        await session.provider.close_session(session.handle)

    async def close_all(self) -> None:
        session_ids = list(self._sessions)
        for session_id in session_ids:
            await self.close_session(session_id)
        for provider in self._providers.values():
            await provider.shutdown()

    def _get_provider(self, provider: str) -> AgentProvider:
        if provider not in self._provider_factories:
            supported = ", ".join(sorted(self._provider_factories))
            raise UnsupportedProviderError(
                f"unsupported provider '{provider}'. Supported providers: {supported}"
            )
        if provider not in self._providers:
            self._providers[provider] = self._provider_factories[provider]()
        return self._providers[provider]


def _default_provider_factories() -> dict[str, ProviderFactory]:
    return {
        "claude": ClaudeProvider,
        "codex": CodexProvider,
        "gemini": GeminiProvider,
    }
