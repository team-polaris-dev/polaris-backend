from __future__ import annotations

import pytest

from apimaker.errors import InvalidProviderOptionError
from apimaker.providers.claude import _claude_option_kwargs, _validate_claude_options
from apimaker.types import AgentOptions, ClaudeOptions, CodexOptions


def test_claude_option_kwargs_include_expanded_options() -> None:
    options = AgentOptions(
        model="claude-test",
        cwd="/tmp/work",
        add_dirs=["/tmp/extra"],
        system_prompt="be precise",
        permission_mode="default",
        reasoning_effort="xhigh",
        max_turns=5,
        max_budget_usd=1.5,
        allowed_tools=["Read"],
        disallowed_tools=["Bash"],
        env={"A": "B"},
        output_format={"type": "json"},
        claude=ClaudeOptions(
            fallback_model="claude-fallback",
            max_thinking_tokens=4096,
            include_partial_messages=True,
            extra_args={"--debug": None},
        ),
        provider_options={"load_timeout_ms": 12345, "permission_mode": "plan"},
    )

    kwargs = _claude_option_kwargs(options)

    assert kwargs["model"] == "claude-test"
    assert kwargs["cwd"] == "/tmp/work"
    assert kwargs["add_dirs"] == ["/tmp/extra"]
    assert kwargs["system_prompt"] == "be precise"
    assert kwargs["permission_mode"] == "default"
    assert kwargs["effort"] == "xhigh"
    assert kwargs["max_turns"] == 5
    assert kwargs["max_budget_usd"] == 1.5
    assert kwargs["allowed_tools"] == ["Read"]
    assert kwargs["disallowed_tools"] == ["Bash"]
    assert kwargs["env"] == {"A": "B"}
    assert kwargs["output_format"] == {"type": "json"}
    assert kwargs["fallback_model"] == "claude-fallback"
    assert kwargs["max_thinking_tokens"] == 4096
    assert kwargs["include_partial_messages"] is True
    assert kwargs["extra_args"] == {"--debug": None}
    assert kwargs["load_timeout_ms"] == 12345


def test_claude_validation_rejects_provider_mismatches() -> None:
    with pytest.raises(InvalidProviderOptionError):
        _validate_claude_options(AgentOptions(sandbox="workspace_write"))

    with pytest.raises(InvalidProviderOptionError):
        _validate_claude_options(AgentOptions(reasoning_effort="minimal"))

    with pytest.raises(InvalidProviderOptionError):
        _validate_claude_options(
            AgentOptions(codex=CodexOptions(approval_policy="never"))
        )
