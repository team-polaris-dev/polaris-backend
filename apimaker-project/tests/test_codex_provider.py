from __future__ import annotations

import pytest

from apimaker.errors import InvalidProviderOptionError
from apimaker.providers.codex import (
    CodexSession,
    _codex_cli_args,
    _toml_literal,
    _validate_codex_options,
)
from apimaker.types import AgentOptions, CodexOptions, ClaudeOptions


def test_codex_initial_cli_args_include_expanded_options() -> None:
    options = AgentOptions(
        model="gpt-test",
        cwd="/tmp/work",
        add_dirs=["/tmp/extra"],
        sandbox="workspace_write",
        system_prompt="follow local rules",
        reasoning_effort="high",
        env={"EXAMPLE": "1"},
        dangerously_bypass_hook_trust=True,
        codex=CodexOptions(
            profile="ci",
            strict_config=True,
            skip_git_repo_check=True,
            ignore_rules=True,
            output_schema="schema.json",
            approval_policy="never",
            web_search="disabled",
            enabled_features=["shell_snapshot"],
            disabled_features=["memories"],
            config_overrides={"model_verbosity": "low"},
        ),
        provider_options={"service_tier": "fast"},
    )

    args = _codex_cli_args(CodexSession(options=options), "hello")

    assert args[:3] == ["codex", "exec", "--json"]
    assert ["--model", "gpt-test"] == [args[args.index("--model")], args[args.index("--model") + 1]]
    assert "--strict-config" in args
    assert "--skip-git-repo-check" in args
    assert "--ignore-rules" in args
    assert "--dangerously-bypass-hook-trust" in args
    assert "--enable" in args
    assert "--disable" in args
    assert "--add-dir" in args
    assert "--config" in args
    joined = " ".join(args)
    assert 'model_reasoning_effort="high"' in joined
    assert 'developer_instructions="follow local rules"' in joined
    assert 'approval_policy="never"' in joined
    assert 'web_search="disabled"' in joined
    assert 'model_verbosity="low"' in joined
    assert 'service_tier="fast"' in joined
    assert args[-1] == "hello"


def test_codex_resume_args_omit_initial_only_flags() -> None:
    options = AgentOptions(
        model="gpt-test",
        cwd="/tmp/work",
        add_dirs=["/tmp/extra"],
        codex=CodexOptions(profile="ci", images=["a.png"], color="never"),
    )

    args = _codex_cli_args(
        CodexSession(options=options, cli_session_id="session-1"),
        "again",
    )

    assert args[:4] == ["codex", "exec", "resume", "--json"]
    assert "--profile" in args
    assert "--cd" not in args
    assert "--add-dir" not in args
    assert "--image" not in args
    assert "--color" not in args
    assert args[-2:] == ["session-1", "again"]


def test_codex_validation_rejects_provider_mismatches() -> None:
    with pytest.raises(InvalidProviderOptionError):
        _validate_codex_options(
            AgentOptions(claude=ClaudeOptions(max_thinking_tokens=1000))
        )

    with pytest.raises(InvalidProviderOptionError):
        _validate_codex_options(AgentOptions(reasoning_effort="max"))


def test_toml_literal_formats_common_values() -> None:
    assert _toml_literal("hello") == '"hello"'
    assert _toml_literal(True) == "true"
    assert _toml_literal(["a", 1]) == '["a", 1]'
