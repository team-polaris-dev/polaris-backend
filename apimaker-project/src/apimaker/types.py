"""Shared types for direct and HTTP agent calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

ProviderName = Literal["claude", "codex", "gemini"]
SandboxName = Literal["read_only", "workspace_write", "full_access"]
ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
CodexApprovalPolicy = Literal["untrusted", "on-request", "never"]
CodexWebSearch = Literal["cached", "live", "disabled"]
CodexColor = Literal["always", "never", "auto"]
ClaudePermissionMode = Literal[
    "default",
    "acceptEdits",
    "plan",
    "bypassPermissions",
    "dontAsk",
    "auto",
]
ClaudeSessionStoreFlush = Literal["batched", "eager"]


@dataclass(frozen=True)
class ClaudeOptions:
    """Claude-only session options.

    These names mirror ClaudeAgentOptions where possible. JSON-compatible values
    are exposed here; Python callback/object options such as can_use_tool,
    stderr, debug_stderr, and session_store remain direct-Python extension
    points rather than HTTP API fields.
    """

    fallback_model: str | None = None
    betas: list[str] = field(default_factory=list)

    # Thinking controls are Claude-specific. reasoning_effort maps to Claude's
    # effort field, while max_thinking_tokens and thinking are passed directly
    # to the SDK for users who need lower-level control.
    max_thinking_tokens: int | None = None
    thinking: Mapping[str, Any] | None = None

    permission_prompt_tool_name: str | None = None
    cli_path: str | None = None
    settings: str | None = None
    max_buffer_size: int | None = None
    user: str | None = None
    include_partial_messages: bool | None = None
    include_hook_events: bool | None = None
    fork_session: bool | None = None
    setting_sources: list[Literal["user", "project", "local"]] | None = None
    skills: list[str] | Literal["all"] | None = None

    # These are advanced SDK structures. They are intentionally typed as
    # JSON-like mappings/lists so the wrapper can expose them without importing
    # provider SDK classes into the public API schema.
    plugins: list[Mapping[str, Any]] = field(default_factory=list)
    agents: Mapping[str, Any] | None = None
    hooks: Mapping[str, Any] | None = None
    enable_file_checkpointing: bool | None = None
    session_store_flush: ClaudeSessionStoreFlush | None = None
    load_timeout_ms: int | None = None
    task_budget: Mapping[str, Any] | None = None
    extra_args: Mapping[str, str | None] = field(default_factory=dict)

    def has_values(self) -> bool:
        return self != ClaudeOptions()


@dataclass(frozen=True)
class CodexOptions:
    """Codex-only session options.

    Most values map to `codex exec` flags or `-c key=value` config overrides.
    Risky switches are exposed because this project is a local development
    wrapper, but they are never enabled unless explicitly requested.
    """

    profile: str | None = None
    oss: bool | None = None
    local_provider: str | None = None
    strict_config: bool | None = None

    # This sample defaults to True so it works in non-git scratch directories.
    # Set False when you want Codex's normal git-repository safety check.
    skip_git_repo_check: bool | None = True

    ephemeral: bool | None = None
    ignore_user_config: bool | None = None
    ignore_rules: bool | None = None
    output_schema: str | None = None
    output_last_message: str | None = None
    color: CodexColor | None = None
    images: list[str] = field(default_factory=list)

    # These map to Codex config keys through `-c`, not direct CLI flags for
    # `codex exec` in the currently installed CLI.
    approval_policy: CodexApprovalPolicy | None = None
    web_search: CodexWebSearch | None = None
    personality: str | None = None
    model_provider: str | None = None

    enabled_features: list[str] = field(default_factory=list)
    disabled_features: list[str] = field(default_factory=list)
    config_overrides: Mapping[str, Any] = field(default_factory=dict)

    def has_values(self) -> bool:
        return self != CodexOptions()


@dataclass(frozen=True)
class AgentOptions:
    """Provider-neutral options accepted when opening a session.

    Top-level fields represent the common wrapper contract. Provider-specific
    details live in claude/codex, and provider_options is a raw escape hatch for
    advanced users who need to pass a provider-native key that is not yet modeled.
    Explicit typed fields take precedence over provider_options on conflicts.
    """

    model: str | None = None
    cwd: str | None = None
    add_dirs: list[str] = field(default_factory=list)
    system_prompt: str | None = None
    sandbox: SandboxName | None = None
    permission_mode: ClaudePermissionMode | None = None
    strip_anthropic_api_key: bool = True
    reasoning_effort: ReasoningEffort | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    output_format: Mapping[str, Any] | None = None
    tools: list[str] | str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    mcp_servers: Mapping[str, Any] | str | None = None
    env: Mapping[str, str] = field(default_factory=dict)

    # Dangerous Codex CLI switches. They are useful for externally sandboxed
    # automation, but they remove important runtime protections. The API only
    # applies them when the request explicitly sets them.
    dangerously_bypass_approvals_and_sandbox: bool | None = None
    dangerously_bypass_hook_trust: bool | None = None

    claude: ClaudeOptions = field(default_factory=ClaudeOptions)
    codex: CodexOptions = field(default_factory=CodexOptions)

    # Raw provider-native escape hatch. Claude receives these as
    # ClaudeAgentOptions kwargs. Codex receives them as config overrides.
    provider_options: Mapping[str, Any] = field(default_factory=dict)

    # Backward-compatible direct-Python escape hatch. New API code should prefer
    # provider_options, claude, or codex.
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionInfo:
    """Public session metadata returned by the service/API."""

    session_id: str
    provider: str


@dataclass(frozen=True)
class AgentResponse:
    """Final text response from a provider turn."""

    session_id: str
    provider: str
    response: str
