"""Gemini CLI provider using local Google sign-in authentication."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os

from apimaker.errors import InvalidProviderOptionError, ProviderRuntimeError
from apimaker.types import AgentOptions


@dataclass
class GeminiSession:
    options: AgentOptions
    history: list[tuple[str, str]] = field(default_factory=list)


class GeminiProvider:
    """Gemini turns backed by the local `gemini` CLI OAuth session."""

    name = "gemini"

    async def start_session(self, options: AgentOptions) -> GeminiSession:
        _validate_gemini_options(options)
        return GeminiSession(options=options)

    async def send_message(self, session: GeminiSession, prompt: str) -> str:
        contextual_prompt = _prompt_with_history(session, prompt)
        response = await _run_gemini_cli(session, contextual_prompt)
        session.history.append((prompt, response))
        return response

    async def close_session(self, session: GeminiSession) -> None:
        return None

    async def shutdown(self) -> None:
        return None


async def _run_gemini_cli(session: GeminiSession, prompt: str) -> str:
    args = _gemini_cli_args(session, prompt)
    process_env = os.environ.copy()
    process_env.update(session.options.env)

    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=session.options.cwd,
            env=process_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ProviderRuntimeError(
            "gemini CLI is not installed or is not on PATH. Install @google/gemini-cli "
            "and run `gemini` once to sign in with Google."
        ) from exc

    stdout, stderr = await process.communicate()
    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")

    if process.returncode != 0:
        detail = stderr_text.strip() or stdout_text.strip()
        raise ProviderRuntimeError(f"gemini CLI failed: {detail}")

    return _parse_gemini_json_response(stdout_text)


def _gemini_cli_args(session: GeminiSession, prompt: str) -> list[str]:
    options = session.options
    final_prompt = prompt
    if options.system_prompt:
        final_prompt = f"{options.system_prompt}\n\n{prompt}"

    args = ["gemini"]
    if options.model:
        args.extend(["-m", options.model])
    args.extend(["-p", final_prompt, "--output-format", "json"])
    if options.max_turns is not None:
        args.extend(["--max-turns", str(options.max_turns)])
    return args


def _prompt_with_history(session: GeminiSession, prompt: str) -> str:
    if not session.history:
        return prompt

    lines = ["Previous conversation:"]
    for user_message, assistant_message in session.history:
        lines.append(f"User: {user_message}")
        lines.append(f"Assistant: {assistant_message}")
    lines.append("")
    lines.append(f"User: {prompt}")
    return "\n".join(lines)


def _parse_gemini_json_response(stdout_text: str) -> str:
    try:
        payload = json.loads(stdout_text)
    except json.JSONDecodeError as exc:
        raise ProviderRuntimeError(
            f"gemini CLI returned non-JSON output: {stdout_text.strip()}"
        ) from exc

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("details") or error
        raise ProviderRuntimeError(f"gemini CLI failed: {message}")
    if error:
        raise ProviderRuntimeError(f"gemini CLI failed: {error}")

    response = payload.get("response")
    if not isinstance(response, str):
        raise ProviderRuntimeError("gemini CLI JSON output did not include a response string.")
    return response.strip()


def _validate_gemini_options(options: AgentOptions) -> None:
    if options.claude.has_values():
        raise InvalidProviderOptionError(
            "claude_options can only be used with provider='claude'."
        )
    if options.codex.has_values():
        raise InvalidProviderOptionError(
            "codex_options can only be used with provider='codex'."
        )
    if options.permission_mode is not None:
        raise InvalidProviderOptionError("permission_mode is Claude-only.")
    if options.max_budget_usd is not None:
        raise InvalidProviderOptionError("max_budget_usd is Claude-only.")
    if options.sandbox is not None:
        raise InvalidProviderOptionError(
            "sandbox is not supported by this Gemini CLI wrapper yet."
        )
    if options.add_dirs:
        raise InvalidProviderOptionError(
            "add_dirs is not supported by this Gemini CLI wrapper yet."
        )
    if options.output_format is not None:
        raise InvalidProviderOptionError(
            "output_format is fixed to Gemini CLI JSON in this wrapper."
        )
    if options.tools is not None or options.allowed_tools or options.disallowed_tools:
        raise InvalidProviderOptionError(
            "tool allowlists are not supported by this Gemini CLI wrapper yet."
        )
    if options.mcp_servers is not None:
        raise InvalidProviderOptionError(
            "mcp_servers are not supported by this Gemini CLI wrapper yet."
        )
    if options.reasoning_effort is not None:
        raise InvalidProviderOptionError(
            "reasoning_effort is not supported by Gemini CLI headless mode."
        )
    if options.provider_options or options.extra:
        raise InvalidProviderOptionError(
            "provider_options are not supported by this Gemini CLI wrapper yet."
        )

    api_key_names = {"GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"}
    configured_keys = sorted(api_key_names.intersection(options.env))
    if configured_keys:
        joined = ", ".join(configured_keys)
        raise InvalidProviderOptionError(
            f"Gemini provider uses local CLI OAuth, not API-key auth; remove {joined}."
        )
