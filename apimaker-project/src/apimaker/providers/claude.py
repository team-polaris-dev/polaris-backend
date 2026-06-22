"""Claude Agent SDK provider."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from apimaker.errors import InvalidProviderOptionError, ProviderRuntimeError
from apimaker.types import AgentOptions


@dataclass
class ClaudeSession:
    raw_client: Any
    client: Any


class ClaudeProvider:
    """Persistent Claude session backed by claude-agent-sdk."""

    name = "claude"

    async def start_session(self, options: AgentOptions) -> ClaudeSession:
        _validate_claude_options(options)

        if options.strip_anthropic_api_key:
            os.environ.pop("ANTHROPIC_API_KEY", None)

        try:
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
        except ImportError as exc:
            raise ProviderRuntimeError(
                "claude-agent-sdk is not installed. Install project dependencies first."
            ) from exc

        option_kwargs = _claude_option_kwargs(options)

        raw_client = ClaudeSDKClient(options=ClaudeAgentOptions(**option_kwargs))
        try:
            active_client = await raw_client.__aenter__()
        except Exception as exc:  # pragma: no cover - depends on local Claude auth/runtime
            raise ProviderRuntimeError(f"failed to start Claude session: {exc}") from exc

        return ClaudeSession(raw_client=raw_client, client=active_client or raw_client)

    async def send_message(self, session: ClaudeSession, prompt: str) -> str:
        try:
            from claude_agent_sdk import AssistantMessage, TextBlock
        except ImportError as exc:
            raise ProviderRuntimeError(
                "claude-agent-sdk is not installed. Install project dependencies first."
            ) from exc

        chunks: list[str] = []
        try:
            await session.client.query(prompt)
            async for msg in session.client.receive_response():
                if isinstance(msg, AssistantMessage):
                    chunks.extend(
                        block.text
                        for block in msg.content
                        if isinstance(block, TextBlock)
                    )
        except Exception as exc:  # pragma: no cover - depends on local Claude auth/runtime
            raise ProviderRuntimeError(f"Claude turn failed: {exc}") from exc

        return "".join(chunks).strip()

    async def close_session(self, session: ClaudeSession) -> None:
        try:
            await session.raw_client.__aexit__(None, None, None)
        except Exception as exc:  # pragma: no cover - cleanup best effort
            raise ProviderRuntimeError(f"failed to close Claude session: {exc}") from exc

    async def shutdown(self) -> None:
        return None


def _validate_claude_options(options: AgentOptions) -> None:
    if options.sandbox is not None:
        raise InvalidProviderOptionError(
            "sandbox is mapped for Codex in this wrapper. For Claude SDK sandbox "
            "objects, use provider_options['sandbox']."
        )
    if options.dangerously_bypass_approvals_and_sandbox is not None:
        raise InvalidProviderOptionError(
            "dangerously_bypass_approvals_and_sandbox is Codex-only."
        )
    if options.dangerously_bypass_hook_trust is not None:
        raise InvalidProviderOptionError("dangerously_bypass_hook_trust is Codex-only.")
    if options.codex.has_values():
        raise InvalidProviderOptionError("codex_options can only be used with provider='codex'.")
    if options.reasoning_effort == "minimal":
        raise InvalidProviderOptionError(
            "Claude effort supports low, medium, high, xhigh, and max; not minimal."
        )


def _claude_option_kwargs(options: AgentOptions) -> dict[str, Any]:
    # Raw escape hatches are loaded first. Explicit wrapper fields below win on
    # conflict so stable API fields remain predictable.
    option_kwargs: dict[str, Any] = dict(options.provider_options)
    option_kwargs.update(options.extra)

    # Keep the original local-development behavior: if the caller does not pick
    # a Claude permission mode, run without interactive permission prompts.
    option_kwargs["permission_mode"] = options.permission_mode or "bypassPermissions"

    _put(option_kwargs, "model", options.model)
    _put(option_kwargs, "cwd", options.cwd)
    _put(option_kwargs, "add_dirs", options.add_dirs)
    _put(option_kwargs, "system_prompt", options.system_prompt)
    _put(option_kwargs, "tools", options.tools)
    _put(option_kwargs, "allowed_tools", options.allowed_tools)
    _put(option_kwargs, "disallowed_tools", options.disallowed_tools)
    _put(option_kwargs, "mcp_servers", options.mcp_servers)
    _put(option_kwargs, "max_turns", options.max_turns)
    _put(option_kwargs, "max_budget_usd", options.max_budget_usd)
    _put(option_kwargs, "env", dict(options.env) if options.env else None)
    _put(option_kwargs, "output_format", options.output_format)

    # Wrapper reasoning_effort maps directly to ClaudeAgentOptions.effort.
    _put(option_kwargs, "effort", options.reasoning_effort)

    claude = options.claude
    _put(option_kwargs, "fallback_model", claude.fallback_model)
    _put(option_kwargs, "betas", claude.betas)
    _put(option_kwargs, "max_thinking_tokens", claude.max_thinking_tokens)
    _put(option_kwargs, "thinking", claude.thinking)
    _put(option_kwargs, "permission_prompt_tool_name", claude.permission_prompt_tool_name)
    _put(option_kwargs, "cli_path", claude.cli_path)
    _put(option_kwargs, "settings", claude.settings)
    _put(option_kwargs, "max_buffer_size", claude.max_buffer_size)
    _put(option_kwargs, "user", claude.user)
    _put(option_kwargs, "include_partial_messages", claude.include_partial_messages)
    _put(option_kwargs, "include_hook_events", claude.include_hook_events)
    _put(option_kwargs, "fork_session", claude.fork_session)
    _put(option_kwargs, "setting_sources", claude.setting_sources)
    _put(option_kwargs, "skills", claude.skills)
    _put(option_kwargs, "plugins", claude.plugins)
    _put(option_kwargs, "agents", claude.agents)
    _put(option_kwargs, "hooks", claude.hooks)
    _put(option_kwargs, "enable_file_checkpointing", claude.enable_file_checkpointing)
    _put(option_kwargs, "session_store_flush", claude.session_store_flush)
    _put(option_kwargs, "load_timeout_ms", claude.load_timeout_ms)
    _put(option_kwargs, "task_budget", claude.task_budget)
    _put(option_kwargs, "extra_args", dict(claude.extra_args) if claude.extra_args else None)

    return option_kwargs


def _put(target: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if value == [] or value == {}:
        return
    target[key] = value
