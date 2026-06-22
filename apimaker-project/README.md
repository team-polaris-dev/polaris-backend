# apimaker

Development-only examples for using Claude, Codex, and Gemini as persistent Python
sessions, either directly from Python or through a small local FastAPI wrapper.

## Install

```bash
uv sync --dev
```

Claude, Codex, and Gemini expect their local CLIs to be installed and logged in
when using subscription-backed local auth:

```bash
claude --version
codex --version
codex login
gemini
```

For Gemini, start `gemini` once and choose Sign in with Google. Use the Google
account tied to your Google AI Pro/Ultra or Gemini Code Assist entitlement. The
Gemini provider intentionally uses the local CLI OAuth cache and rejects
`GEMINI_API_KEY`, `GOOGLE_API_KEY`, and service-account env vars in per-session
`env`.

Install provider SDKs only when you need them:

```bash
uv sync --extra claude --dev
uv sync --extra codex-sdk --dev
```

The Codex provider falls back to `codex exec/resume --json` when `openai-codex`
is not installed. This is useful on platforms where the beta SDK runtime wheel is
not available yet.

## Direct Python Call

This path keeps everything inside the Python process. It is the simplest choice
for local development and internal experiments.

```bash
uv run python examples/direct_call.py --provider claude
uv run python examples/direct_call.py --provider codex
uv run python examples/direct_call.py --provider gemini
```

The example opens one session/thread, sends two prompts, and checks whether the
second turn remembers the first.

Use `AgentService` directly when integrating into another Python module:

```python
import asyncio

from apimaker import AgentOptions, AgentService
from apimaker.types import ClaudeOptions, CodexOptions


async def main() -> None:
    service = AgentService()

    session = await service.start_session(
        "claude",
        AgentOptions(
            model="claude-sonnet-4-5",
            system_prompt="Answer tersely.",
            reasoning_effort="high",
            claude=ClaudeOptions(max_thinking_tokens=4096),
        ),
    )

    try:
        first = await service.send_message(session.session_id, "내가 좋아하는 숫자는 42야.")
        second = await service.send_message(session.session_id, "그 숫자에 8을 더하면?")
        print(first.response)
        print(second.response)
    finally:
        await service.close_all()


asyncio.run(main())
```

Codex uses the same service API. On this Linux environment, `openai-codex` may
not install because the beta runtime wheel is unavailable, so the provider uses
the installed `codex` CLI fallback:

```python
session = await service.start_session(
    "codex",
    AgentOptions(
        model="gpt-5.5",
        sandbox="workspace_write",
        reasoning_effort="high",
        codex=CodexOptions(
            approval_policy="on-request",
            web_search="disabled",
        ),
    ),
)
```

Gemini also uses the same service API, but it shells out to the local `gemini`
CLI in headless JSON mode instead of using API-key SDK auth:

```python
session = await service.start_session(
    "gemini",
    AgentOptions(
        model="gemini-2.5-pro",
        system_prompt="Answer tersely.",
    ),
)
```

The Gemini CLI headless process is started per turn. To keep the apimaker
session contract useful, the provider replays prior user/assistant turns into
the next prompt.

## Local API Wrapper

Start the local API server:

```bash
uv run uvicorn apimaker.api:app --host 127.0.0.1 --port 8000 --reload
```

Call it from another process:

```bash
uv run python examples/api_client.py --provider claude
uv run python examples/api_client.py --provider codex
uv run python examples/api_client.py --provider gemini
```

The HTTP shape is intentionally small:

```http
POST /sessions
POST /sessions/{session_id}/messages
DELETE /sessions/{session_id}
GET /health
```

### API usage with curl

Create a Claude session:

```bash
curl -sS http://127.0.0.1:8000/sessions \
  -H 'content-type: application/json' \
  -d '{
    "provider": "claude",
    "model": "claude-sonnet-4-5",
    "system_prompt": "Answer tersely.",
    "reasoning_effort": "high",
    "allowed_tools": ["Read"],
    "claude_options": {
      "max_thinking_tokens": 4096
    }
  }'
```

Create a Codex session:

```bash
curl -sS http://127.0.0.1:8000/sessions \
  -H 'content-type: application/json' \
  -d '{
    "provider": "codex",
    "model": "gpt-5.5",
    "cwd": "/home/me/project",
    "sandbox": "workspace_write",
    "reasoning_effort": "high",
    "system_prompt": "Follow the repository conventions.",
    "codex_options": {
      "approval_policy": "on-request",
      "web_search": "disabled",
      "enabled_features": ["shell_snapshot"],
      "config_overrides": {
        "model_verbosity": "low"
      }
    }
  }'
```

Create a Gemini session:

```bash
curl -sS http://127.0.0.1:8000/sessions \
  -H 'content-type: application/json' \
  -d '{
    "provider": "gemini",
    "model": "gemini-2.5-pro",
    "system_prompt": "Answer tersely."
  }'
```

The response contains a `session_id`:

```json
{
  "session_id": "0df7f505-e3b4-4d25-9ad2-8fc4e0b73f38",
  "provider": "claude"
}
```

Send multiple messages to the same session to keep context:

```bash
SESSION_ID=0df7f505-e3b4-4d25-9ad2-8fc4e0b73f38

curl -sS "http://127.0.0.1:8000/sessions/$SESSION_ID/messages" \
  -H 'content-type: application/json' \
  -d '{"prompt":"내가 좋아하는 숫자는 42야. 알겠다고만 답해."}'

curl -sS "http://127.0.0.1:8000/sessions/$SESSION_ID/messages" \
  -H 'content-type: application/json' \
  -d '{"prompt":"그 숫자에 8을 더하면? 숫자만."}'
```

Close the session when done:

```bash
curl -X DELETE "http://127.0.0.1:8000/sessions/$SESSION_ID"
```

### Options reference

The generated OpenAPI schema at `/docs` and `/openapi.json` includes field
descriptions. The main option groups are:

| Group | Fields |
|---|---|
| Common | `model`, `cwd`, `add_dirs`, `system_prompt`, `reasoning_effort`, `env` |
| Claude-focused | `permission_mode`, `tools`, `allowed_tools`, `disallowed_tools`, `mcp_servers`, `max_turns`, `max_budget_usd`, `output_format`, `claude_options` |
| Claude thinking | `claude_options.max_thinking_tokens`, `claude_options.thinking`, top-level `reasoning_effort` |
| Codex-focused | `sandbox`, `codex_options.profile`, `codex_options.approval_policy`, `codex_options.web_search`, `codex_options.personality`, `codex_options.model_provider`, `codex_options.enabled_features`, `codex_options.disabled_features` |
| Codex raw config | `codex_options.config_overrides`, `provider_options` |
| Gemini-focused | `model`, `cwd`, `system_prompt`, `max_turns` through the local `gemini` CLI |
| Dangerous | `dangerously_bypass_approvals_and_sandbox`, `dangerously_bypass_hook_trust`, `sandbox: "full_access"`, `codex_options.approval_policy: "never"` |

Provider-specific mismatches fail explicitly instead of being silently ignored.
For example, `claude_options.max_thinking_tokens` is Claude-only, and Codex
`reasoning_effort` accepts `minimal`, `low`, `medium`, `high`, and `xhigh`, not
`max`.

Use `provider_options` only as a local-development escape hatch. For Claude it
is merged into `ClaudeAgentOptions`; for Codex it becomes `codex exec -c`
configuration overrides. Gemini rejects `provider_options` for now because the
wrapper is intentionally limited to local CLI OAuth behavior. Typed fields win
over `provider_options` on conflicts.

### Error behavior

The wrapper intentionally fails fast when an option does not apply to the chosen
provider:

| Status | Meaning |
|---|---|
| `400` | Unknown provider or provider-specific option mismatch |
| `404` | Unknown `session_id` |
| `422` | Invalid request shape, enum value, or empty prompt |
| `502` | Claude/Codex/Gemini SDK or CLI failed at runtime |

## Development Scope

This project is for local development and testing. Sessions are stored in memory
and disappear when the Python process exits. Do not expose this API wrapper to
untrusted users or use a personal subscription login as a production backend.
Move to provider API-key based auth and add real user auth, persistence, rate
limits, and audit logging before production use.

The Codex CLI fallback passes `--skip-git-repo-check` because this sample
project may be used outside a normal Git checkout. Keep the server bound to
`127.0.0.1` and use normal sandbox modes while experimenting.

## Tests

The automated tests use fake providers, so they do not require Claude/Codex/Gemini
login or model calls.

```bash
uv run pytest -p no:cacheprovider
```
