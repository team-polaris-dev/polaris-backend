"""Codex SDK provider."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from typing import Any

from apimaker.errors import InvalidProviderOptionError, ProviderRuntimeError
from apimaker.types import AgentOptions


@dataclass
class CodexSession:
    options: AgentOptions
    thread: Any | None = None
    cli_session_id: str | None = None


class CodexProvider:
    """Persistent Codex thread backed by openai-codex, with CLI fallback."""

    name = "codex"

    def __init__(self) -> None:
        self._raw_client: Any | None = None
        self._client: Any | None = None

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from openai_codex import AsyncCodex
        except ImportError as exc:
            raise ProviderRuntimeError(
                "openai-codex is not installed. Install project dependencies first."
            ) from exc

        self._raw_client = AsyncCodex()
        try:
            active_client = await self._raw_client.__aenter__()
        except Exception as exc:  # pragma: no cover - depends on local Codex auth/runtime
            raise ProviderRuntimeError(f"failed to start Codex client: {exc}") from exc

        self._client = active_client or self._raw_client
        return self._client

    async def start_session(self, options: AgentOptions) -> CodexSession:
        _validate_codex_options(options)

        if _requires_cli_fallback(options):
            return CodexSession(options=options)

        try:
            from openai_codex import Sandbox
        except ImportError as exc:
            return CodexSession(options=options)

        client = await self._ensure_client()
        kwargs: dict[str, Any] = {}
        if options.model:
            kwargs["model"] = options.model
        kwargs["sandbox"] = _sandbox_value(Sandbox, options.sandbox or "workspace_write")

        if options.cwd:
            raise ProviderRuntimeError(
                "cwd is not wired for the Codex Python provider in this sample. "
                "Run the API server from the intended workspace instead."
            )

        try:
            thread = await client.thread_start(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on local Codex auth/runtime
            raise ProviderRuntimeError(f"failed to start Codex thread: {exc}") from exc

        return CodexSession(options=options, thread=thread)

    async def send_message(self, session: CodexSession, prompt: str) -> str:
        if session.thread is None:
            return await _run_codex_cli(session, prompt)

        try:
            result = await session.thread.run(prompt)
        except Exception as exc:  # pragma: no cover - depends on local Codex auth/runtime
            raise ProviderRuntimeError(f"Codex turn failed: {exc}") from exc

        final_response = getattr(result, "final_response", None)
        return str(final_response if final_response is not None else result).strip()

    async def close_session(self, session: CodexSession) -> None:
        return None

    async def shutdown(self) -> None:
        if self._raw_client is None:
            return None
        try:
            await self._raw_client.__aexit__(None, None, None)
        except Exception as exc:  # pragma: no cover - cleanup best effort
            raise ProviderRuntimeError(f"failed to close Codex client: {exc}") from exc
        finally:
            self._raw_client = None
            self._client = None


def _sandbox_value(sandbox_cls: Any, sandbox: str) -> Any:
    mapping = {
        "read_only": "read_only",
        "workspace_write": "workspace_write",
        "full_access": "full_access",
    }
    try:
        return getattr(sandbox_cls, mapping[sandbox])
    except (AttributeError, KeyError) as exc:
        raise ProviderRuntimeError(f"unsupported Codex sandbox: {sandbox}") from exc


async def _run_codex_cli(session: CodexSession, prompt: str) -> str:
    args = _codex_cli_args(session, prompt)
    process_env = os.environ.copy()
    process_env.update(session.options.env)
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=session.options.cwd,
        env=process_env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate(input=prompt.encode("utf-8"))
    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")

    if process.returncode != 0:
        detail = stderr_text.strip() or stdout_text.strip()
        raise ProviderRuntimeError(f"codex CLI failed: {detail}")

    final_message = ""
    for line in stdout_text.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "thread.started":
            thread_id = event.get("thread_id")
            if isinstance(thread_id, str):
                session.cli_session_id = thread_id

        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            text = item.get("text")
            if isinstance(text, str):
                final_message = text

    return final_message.strip() or stdout_text.strip()


def _codex_cli_args(session: CodexSession, prompt: str) -> list[str]:
    options = session.options
    if session.cli_session_id:
        args = ["codex", "exec", "resume", "--json"]
        args.extend(_codex_resume_flags(options))
        args.extend([session.cli_session_id, "-"])
        return args

    args = ["codex", "exec", "--json"]
    args.extend(_codex_initial_flags(options))
    args.append("-")
    return args


def _validate_codex_options(options: AgentOptions) -> None:
    if options.permission_mode is not None:
        raise InvalidProviderOptionError("permission_mode is Claude-only.")
    if options.max_budget_usd is not None:
        raise InvalidProviderOptionError("max_budget_usd is Claude-only.")
    if options.tools is not None:
        raise InvalidProviderOptionError("tools is Claude-only in this wrapper.")
    if options.allowed_tools:
        raise InvalidProviderOptionError("allowed_tools is Claude-only in this wrapper.")
    if options.disallowed_tools:
        raise InvalidProviderOptionError("disallowed_tools is Claude-only in this wrapper.")
    if options.mcp_servers is not None:
        raise InvalidProviderOptionError(
            "mcp_servers should be passed through codex_options.config_overrides "
            "or provider_options for Codex."
        )
    if options.output_format is not None:
        raise InvalidProviderOptionError(
            "output_format is not directly supported by codex exec. Use "
            "codex_options.output_schema with a schema file path."
        )
    if options.max_turns is not None:
        raise InvalidProviderOptionError("max_turns is not supported by codex exec.")
    if options.claude.has_values():
        raise InvalidProviderOptionError(
            "claude_options can only be used with provider='claude'."
        )
    if options.reasoning_effort == "max":
        raise InvalidProviderOptionError(
            "Codex model_reasoning_effort supports minimal, low, medium, high, and xhigh; not max."
        )


def _requires_cli_fallback(options: AgentOptions) -> bool:
    return any(
        [
            options.add_dirs,
            options.cwd,
            options.system_prompt,
            options.reasoning_effort,
            options.provider_options,
            options.extra,
            options.env,
            options.dangerously_bypass_approvals_and_sandbox is not None,
            options.dangerously_bypass_hook_trust is not None,
            options.codex.has_values(),
        ]
    )


def _codex_initial_flags(options: AgentOptions) -> list[str]:
    args: list[str] = []
    if options.model:
        args.extend(["--model", options.model])
    if options.codex.profile:
        args.extend(["--profile", options.codex.profile])
    if options.cwd:
        args.extend(["--cd", options.cwd])
    for directory in options.add_dirs:
        args.extend(["--add-dir", directory])
    sandbox = _codex_cli_sandbox(options.sandbox or "workspace_write")
    args.extend(["--sandbox", sandbox])
    args.extend(_codex_common_flags(options, include_initial_only=True))
    return args


def _codex_resume_flags(options: AgentOptions) -> list[str]:
    args: list[str] = []
    if options.model:
        args.extend(["--model", options.model])
    if options.codex.profile:
        args.extend(["--profile", options.codex.profile])
    args.extend(_codex_common_flags(options, include_initial_only=False))
    return args


def _codex_common_flags(options: AgentOptions, *, include_initial_only: bool) -> list[str]:
    args: list[str] = []
    codex = options.codex

    if codex.strict_config:
        args.append("--strict-config")
    if codex.skip_git_repo_check:
        args.append("--skip-git-repo-check")
    if codex.ephemeral:
        args.append("--ephemeral")
    if codex.ignore_user_config:
        args.append("--ignore-user-config")
    if codex.ignore_rules:
        args.append("--ignore-rules")
    if codex.output_schema:
        args.extend(["--output-schema", codex.output_schema])
    if codex.output_last_message:
        args.extend(["--output-last-message", codex.output_last_message])
    if options.dangerously_bypass_approvals_and_sandbox:
        args.append("--dangerously-bypass-approvals-and-sandbox")
    if options.dangerously_bypass_hook_trust:
        args.append("--dangerously-bypass-hook-trust")

    for feature in codex.enabled_features:
        args.extend(["--enable", feature])
    for feature in codex.disabled_features:
        args.extend(["--disable", feature])
    for key, value in _codex_config_overrides(options).items():
        args.extend(["--config", f"{key}={_toml_literal(value)}"])

    # Some flags only exist on the initial `codex exec` command in the current
    # CLI help output. Session resume keeps the original execution context.
    if include_initial_only:
        if codex.oss:
            args.append("--oss")
        if codex.local_provider:
            args.extend(["--local-provider", codex.local_provider])
        if codex.color:
            args.extend(["--color", codex.color])
        for image in codex.images:
            args.extend(["--image", image])

    return args


def _codex_config_overrides(options: AgentOptions) -> dict[str, Any]:
    # Raw config is loaded first; typed fields below win to keep the documented
    # wrapper contract deterministic.
    config: dict[str, Any] = dict(options.provider_options)
    config.update(options.extra)
    config.update(options.codex.config_overrides)

    if options.reasoning_effort:
        config["model_reasoning_effort"] = options.reasoning_effort
    if options.system_prompt:
        config["developer_instructions"] = options.system_prompt
    if options.codex.approval_policy:
        config["approval_policy"] = options.codex.approval_policy
    if options.codex.web_search:
        config["web_search"] = options.codex.web_search
    if options.codex.personality:
        config["personality"] = options.codex.personality
    if options.codex.model_provider:
        config["model_provider"] = options.codex.model_provider

    return config


def _codex_cli_sandbox(sandbox: str) -> str:
    mapping = {
        "read_only": "read-only",
        "workspace_write": "workspace-write",
        "full_access": "danger-full-access",
    }
    try:
        return mapping[sandbox]
    except KeyError as exc:
        raise ProviderRuntimeError(f"unsupported Codex sandbox: {sandbox}") from exc


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if value is None:
        return '""'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        items = [
            f"{key} = {_toml_literal(item_value)}"
            for key, item_value in value.items()
        ]
        return "{ " + ", ".join(items) + " }"
    return json.dumps(str(value))
