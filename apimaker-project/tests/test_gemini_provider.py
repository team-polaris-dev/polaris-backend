from __future__ import annotations

import asyncio

import pytest

from apimaker.errors import InvalidProviderOptionError, ProviderRuntimeError
from apimaker.providers.gemini import (
    GeminiProvider,
    GeminiSession,
    _gemini_cli_args,
    _parse_gemini_json_response,
    _validate_gemini_options,
)
from apimaker.types import AgentOptions, ClaudeOptions, CodexOptions


def test_gemini_cli_args_use_oauth_cli_with_json_output() -> None:
    options = AgentOptions(
        model="gemini-2.5-pro",
        cwd="/tmp/work",
        system_prompt="be precise",
        max_turns=3,
    )

    args = _gemini_cli_args(GeminiSession(options=options), "explain this")

    assert args == [
        "gemini",
        "-m",
        "gemini-2.5-pro",
        "-p",
        "be precise\n\nexplain this",
        "--output-format",
        "json",
        "--max-turns",
        "3",
    ]


def test_gemini_json_response_returns_response_field() -> None:
    text = '{"response": "hello from gemini", "stats": {"latency": 10}}'

    assert _parse_gemini_json_response(text) == "hello from gemini"


def test_gemini_json_response_reports_error_field() -> None:
    text = '{"error": {"message": "not logged in"}}'

    with pytest.raises(ProviderRuntimeError, match="not logged in"):
        _parse_gemini_json_response(text)


def test_gemini_validation_rejects_other_provider_options() -> None:
    with pytest.raises(InvalidProviderOptionError, match="claude_options"):
        _validate_gemini_options(
            AgentOptions(claude=ClaudeOptions(max_thinking_tokens=1000))
        )

    with pytest.raises(InvalidProviderOptionError, match="codex_options"):
        _validate_gemini_options(
            AgentOptions(codex=CodexOptions(profile="ci"))
        )

    with pytest.raises(InvalidProviderOptionError, match="API-key"):
        _validate_gemini_options(
            AgentOptions(env={"GEMINI_API_KEY": "secret"})
        )


def test_gemini_provider_replays_history_between_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    async def fake_run(session: GeminiSession, prompt: str) -> str:
        prompts.append(prompt)
        return f"answer {len(prompts)}"

    monkeypatch.setattr("apimaker.providers.gemini._run_gemini_cli", fake_run)

    async def scenario() -> None:
        provider = GeminiProvider()
        session = await provider.start_session(AgentOptions())

        first = await provider.send_message(session, "remember 42")
        second = await provider.send_message(session, "add 8")

        assert first == "answer 1"
        assert second == "answer 2"

    asyncio.run(scenario())
    assert prompts[0] == "remember 42"
    assert "User: remember 42" in prompts[1]
    assert "Assistant: answer 1" in prompts[1]
    assert prompts[1].endswith("User: add 8")
