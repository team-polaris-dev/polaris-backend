"""Provider protocol used by AgentService."""

from __future__ import annotations

from typing import Any, Protocol

from apimaker.types import AgentOptions


class AgentProvider(Protocol):
    """Minimal async contract shared by Claude, Codex, and tests."""

    name: str

    async def start_session(self, options: AgentOptions) -> Any:
        """Create a provider-specific session handle."""

    async def send_message(self, session: Any, prompt: str) -> str:
        """Send a prompt to an existing provider session."""

    async def close_session(self, session: Any) -> None:
        """Close a provider-specific session handle."""

    async def shutdown(self) -> None:
        """Release provider-level resources."""
