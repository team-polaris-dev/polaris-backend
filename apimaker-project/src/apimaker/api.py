"""FastAPI wrapper around AgentService."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from apimaker.errors import (
    InvalidProviderOptionError,
    ProviderRuntimeError,
    SessionNotFoundError,
    UnsupportedProviderError,
)
from apimaker.service import AgentService
from apimaker.types import (
    AgentOptions,
    ClaudeOptions,
    ClaudePermissionMode,
    ClaudeSessionStoreFlush,
    CodexApprovalPolicy,
    CodexColor,
    CodexOptions,
    CodexWebSearch,
    ReasoningEffort,
    SandboxName,
)


class ClaudeOptionsRequest(BaseModel):
    """HTTP-visible Claude-only options with OpenAPI descriptions."""

    model_config = ConfigDict(extra="forbid")

    fallback_model: str | None = Field(
        default=None,
        description="Claude-only fallback model passed to ClaudeAgentOptions.",
    )
    betas: list[str] = Field(
        default_factory=list,
        description="Claude-only beta feature flags, for example context window betas.",
    )
    max_thinking_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Claude-only cap for thinking tokens. Codex has no direct equivalent.",
    )
    thinking: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Claude-only raw thinking config. It is passed to the SDK as-is and "
            "may fail if it does not match the installed SDK's expected shape."
        ),
    )
    permission_prompt_tool_name: str | None = Field(
        default=None,
        description="Claude-only tool name used for permission prompts.",
    )
    cli_path: str | None = Field(
        default=None,
        description="Claude-only path to a specific claude CLI executable.",
    )
    settings: str | None = Field(
        default=None,
        description="Claude-only settings path/string passed to ClaudeAgentOptions.",
    )
    max_buffer_size: int | None = Field(
        default=None,
        ge=1,
        description="Claude-only SDK stream buffer size.",
    )
    user: str | None = Field(
        default=None,
        description="Claude-only user identifier forwarded to the SDK.",
    )
    include_partial_messages: bool | None = Field(
        default=None,
        description="Claude-only flag to include partial streaming messages.",
    )
    include_hook_events: bool | None = Field(
        default=None,
        description="Claude-only flag to include hook events in the response stream.",
    )
    fork_session: bool | None = Field(
        default=None,
        description="Claude-only flag to fork the selected session.",
    )
    setting_sources: list[Literal["user", "project", "local"]] | None = Field(
        default=None,
        description="Claude-only setting source allowlist.",
    )
    skills: list[str] | Literal["all"] | None = Field(
        default=None,
        description="Claude-only skill selection. Use 'all' or a list of skill names.",
    )
    plugins: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Claude-only plugin configs passed to the SDK as JSON-like objects.",
    )
    agents: dict[str, Any] | None = Field(
        default=None,
        description="Claude-only custom agent definitions passed to the SDK as-is.",
    )
    hooks: dict[str, Any] | None = Field(
        default=None,
        description="Claude-only hook definitions passed to the SDK as-is.",
    )
    enable_file_checkpointing: bool | None = Field(
        default=None,
        description="Claude-only file checkpointing switch.",
    )
    session_store_flush: ClaudeSessionStoreFlush | None = Field(
        default=None,
        description="Claude-only session store flush mode: batched or eager.",
    )
    load_timeout_ms: int | None = Field(
        default=None,
        ge=1,
        description="Claude-only SDK load timeout in milliseconds.",
    )
    task_budget: dict[str, Any] | None = Field(
        default=None,
        description="Claude-only task budget object passed to the SDK as-is.",
    )
    extra_args: dict[str, str | None] = Field(
        default_factory=dict,
        description="Claude-only extra CLI args forwarded through ClaudeAgentOptions.",
    )

    def to_dataclass(self) -> ClaudeOptions:
        return ClaudeOptions(**self.model_dump())


class CodexOptionsRequest(BaseModel):
    """HTTP-visible Codex-only options with OpenAPI descriptions."""

    model_config = ConfigDict(extra="forbid")

    profile: str | None = Field(
        default=None,
        description="Codex-only profile loaded from CODEX_HOME/<profile>.config.toml.",
    )
    oss: bool | None = Field(
        default=None,
        description="Codex-only switch for local open-source provider mode.",
    )
    local_provider: str | None = Field(
        default=None,
        description="Codex-only local provider name, for example ollama or lmstudio.",
    )
    strict_config: bool | None = Field(
        default=None,
        description="Codex-only strict config validation switch.",
    )
    skip_git_repo_check: bool | None = Field(
        default=True,
        description=(
            "Codex-only safety check override. Defaults to true for this sample so "
            "non-git scratch directories work; set false to keep Codex's git check."
        ),
    )
    ephemeral: bool | None = Field(
        default=None,
        description="Codex-only flag to avoid persisting session rollout files.",
    )
    ignore_user_config: bool | None = Field(
        default=None,
        description="Codex-only flag to skip CODEX_HOME config.toml for controlled runs.",
    )
    ignore_rules: bool | None = Field(
        default=None,
        description="Codex-only flag to skip execpolicy rules.",
    )
    output_schema: str | None = Field(
        default=None,
        description="Codex-only path to a JSON schema file for final structured output.",
    )
    output_last_message: str | None = Field(
        default=None,
        description="Codex-only path where the CLI writes the final message.",
    )
    color: CodexColor | None = Field(
        default=None,
        description="Codex-only color mode. Usually unnecessary when --json is used.",
    )
    images: list[str] = Field(
        default_factory=list,
        description="Codex-only image paths attached to the first CLI prompt.",
    )
    approval_policy: CodexApprovalPolicy | None = Field(
        default=None,
        description=(
            "Codex-only approval policy. This maps to config key approval_policy "
            "for codex exec because the current exec CLI has no direct flag."
        ),
    )
    web_search: CodexWebSearch | None = Field(
        default=None,
        description="Codex-only web_search config override: cached, live, or disabled.",
    )
    personality: str | None = Field(
        default=None,
        description="Codex-only communication style/personality config override.",
    )
    model_provider: str | None = Field(
        default=None,
        description="Codex-only model provider id, passed as config model_provider.",
    )
    enabled_features: list[str] = Field(
        default_factory=list,
        description="Codex-only feature flags passed as repeated --enable values.",
    )
    disabled_features: list[str] = Field(
        default_factory=list,
        description="Codex-only feature flags passed as repeated --disable values.",
    )
    config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Codex-only raw config overrides. Each key/value becomes -c key=value. "
            "Typed fields take precedence on conflicts."
        ),
    )

    def to_dataclass(self) -> CodexOptions:
        return CodexOptions(**self.model_dump())


class SessionCreateRequest(BaseModel):
    """Create a provider-backed persistent session."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(
        min_length=1,
        description="Provider to use for this session: claude, codex, or gemini.",
    )
    model: str | None = Field(
        default=None,
        description="Model override for the selected provider.",
    )
    cwd: str | None = Field(
        default=None,
        description="Working directory for provider tools/CLI execution.",
    )
    add_dirs: list[str] = Field(
        default_factory=list,
        description="Additional writable directories where supported by the provider.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Session-level system prompt/instructions where supported.",
    )
    sandbox: SandboxName | None = Field(
        default=None,
        description=(
            "Wrapper sandbox level. Codex maps this to read-only, workspace-write, "
            "or danger-full-access. Use Claude raw provider_options for Claude SDK "
            "SandboxSettings objects."
        ),
    )
    permission_mode: ClaudePermissionMode | None = Field(
        default=None,
        description="Claude-only permission mode. Defaults to bypassPermissions if omitted.",
    )
    strip_anthropic_api_key: bool = Field(
        default=True,
        description="Claude helper: remove ANTHROPIC_API_KEY so Claude subscription auth is used.",
    )
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        description=(
            "Provider-neutral reasoning effort. Claude supports low/medium/high/xhigh/max; "
            "Codex supports minimal/low/medium/high/xhigh through model_reasoning_effort."
        ),
    )
    max_turns: int | None = Field(
        default=None,
        ge=1,
        description="Provider max turns for one run/session when supported.",
    )
    max_budget_usd: float | None = Field(
        default=None,
        gt=0,
        description="Claude-only maximum budget in USD.",
    )
    output_format: dict[str, Any] | None = Field(
        default=None,
        description="Provider output format/schema object where supported.",
    )
    tools: list[str] | str | None = Field(
        default=None,
        description="Tool preset or explicit tool list where supported by Claude.",
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="Allowed tool names where supported by the provider.",
    )
    disallowed_tools: list[str] = Field(
        default_factory=list,
        description="Disallowed tool names where supported by the provider.",
    )
    mcp_servers: dict[str, Any] | str | None = Field(
        default=None,
        description="MCP server config object or config path where supported.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Environment variables forwarded to the provider process where supported. "
            "Gemini uses local CLI OAuth; do not pass API-key environment variables."
        ),
    )
    dangerously_bypass_approvals_and_sandbox: bool | None = Field(
        default=None,
        description=(
            "Dangerous Codex CLI switch. Bypasses approvals and sandboxing. "
            "Only set this inside an externally hardened environment."
        ),
    )
    dangerously_bypass_hook_trust: bool | None = Field(
        default=None,
        description=(
            "Dangerous Codex CLI switch. Runs hooks without persisted hook trust. "
            "Only use for automation that already vets hook sources."
        ),
    )
    claude_options: ClaudeOptionsRequest = Field(
        default_factory=ClaudeOptionsRequest,
        description="Claude-only advanced options.",
    )
    codex_options: CodexOptionsRequest = Field(
        default_factory=CodexOptionsRequest,
        description="Codex-only advanced options.",
    )
    provider_options: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Raw provider-native escape hatch. Claude receives these as SDK kwargs; "
            "Codex receives them as -c config overrides. Typed fields win on conflict."
        ),
    )


class SessionCreateResponse(BaseModel):
    session_id: str
    provider: str


class MessageRequest(BaseModel):
    prompt: str = Field(min_length=1, description="Prompt to send to this existing session.")


class MessageResponse(BaseModel):
    session_id: str
    provider: str
    response: str


def create_app(agent_service: AgentService | None = None) -> FastAPI:
    service = agent_service or AgentService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await service.close_all()

    app = FastAPI(title="apimaker agent API", version="0.1.0", lifespan=lifespan)
    app.state.agent_service = service

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/sessions", response_model=SessionCreateResponse)
    async def create_session(request: SessionCreateRequest) -> SessionCreateResponse:
        options = AgentOptions(
            model=request.model,
            cwd=request.cwd,
            add_dirs=request.add_dirs,
            system_prompt=request.system_prompt,
            sandbox=request.sandbox,
            permission_mode=request.permission_mode,
            strip_anthropic_api_key=request.strip_anthropic_api_key,
            reasoning_effort=request.reasoning_effort,
            max_turns=request.max_turns,
            max_budget_usd=request.max_budget_usd,
            output_format=request.output_format,
            tools=request.tools,
            allowed_tools=request.allowed_tools,
            disallowed_tools=request.disallowed_tools,
            mcp_servers=request.mcp_servers,
            env=request.env,
            dangerously_bypass_approvals_and_sandbox=(
                request.dangerously_bypass_approvals_and_sandbox
            ),
            dangerously_bypass_hook_trust=request.dangerously_bypass_hook_trust,
            claude=request.claude_options.to_dataclass(),
            codex=request.codex_options.to_dataclass(),
            provider_options=request.provider_options,
        )
        try:
            session = await service.start_session(request.provider, options)
        except UnsupportedProviderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except InvalidProviderOptionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ProviderRuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return SessionCreateResponse(
            session_id=session.session_id,
            provider=session.provider,
        )

    @app.post("/sessions/{session_id}/messages", response_model=MessageResponse)
    async def send_message(
        session_id: str,
        request: MessageRequest,
    ) -> MessageResponse:
        try:
            result = await service.send_message(session_id, request.prompt)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ProviderRuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return MessageResponse(
            session_id=result.session_id,
            provider=result.provider,
            response=result.response,
        )

    @app.delete("/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str) -> None:
        try:
            await service.close_session(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ProviderRuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


app = create_app()
