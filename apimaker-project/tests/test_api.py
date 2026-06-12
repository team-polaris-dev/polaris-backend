from __future__ import annotations

from fastapi.testclient import TestClient

from apimaker.api import create_app
from apimaker.providers.codex import CodexProvider
from apimaker.service import AgentService
from tests.test_service import FakeProvider


def test_health() -> None:
    app = create_app(AgentService({"claude": FakeProvider}))
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_session_message_flow() -> None:
    app = create_app(AgentService({"claude": FakeProvider}))
    with TestClient(app) as client:
        created = client.post("/sessions", json={"provider": "claude"})
        assert created.status_code == 200
        session_id = created.json()["session_id"]

        first = client.post(
            f"/sessions/{session_id}/messages",
            json={"prompt": "내가 좋아하는 숫자는 42야."},
        )
        assert first.status_code == 200
        assert first.json()["response"] == "turn 1"

        second = client.post(
            f"/sessions/{session_id}/messages",
            json={"prompt": "그 숫자에 8을 더하면?"},
        )
        assert second.status_code == 200
        assert second.json()["response"] == "50"

        deleted = client.delete(f"/sessions/{session_id}")
        assert deleted.status_code == 204


def test_missing_session_returns_404() -> None:
    app = create_app(AgentService({"claude": FakeProvider}))
    with TestClient(app) as client:
        response = client.post(
            "/sessions/missing/messages",
            json={"prompt": "hello"},
        )
        assert response.status_code == 404


def test_session_create_accepts_expanded_options() -> None:
    fake = FakeProvider()
    app = create_app(AgentService({"claude": lambda: fake}))
    with TestClient(app) as client:
        response = client.post(
            "/sessions",
            json={
                "provider": "claude",
                "model": "claude-test",
                "cwd": "/tmp",
                "system_prompt": "be terse",
                "reasoning_effort": "high",
                "allowed_tools": ["Read"],
                "claude_options": {
                    "max_thinking_tokens": 2048,
                    "include_partial_messages": True,
                },
                "provider_options": {"load_timeout_ms": 12345},
            },
        )

        assert response.status_code == 200
        assert fake.last_options is not None
        assert fake.last_options.model == "claude-test"
        assert fake.last_options.reasoning_effort == "high"
        assert fake.last_options.claude.max_thinking_tokens == 2048
        assert fake.last_options.provider_options["load_timeout_ms"] == 12345


def test_codex_rejects_claude_only_options() -> None:
    app = create_app(AgentService({"codex": CodexProvider}))
    with TestClient(app) as client:
        response = client.post(
            "/sessions",
            json={"provider": "codex", "max_budget_usd": 1.0},
        )

        assert response.status_code == 400
        assert "max_budget_usd is Claude-only" in response.json()["detail"]


def test_openapi_includes_option_descriptions() -> None:
    app = create_app(AgentService({"claude": FakeProvider}))
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    schemas = schema["components"]["schemas"]
    session_schema = schemas["SessionCreateRequest"]
    assert "reasoning effort" in session_schema["properties"]["reasoning_effort"]["description"]
    assert "Dangerous Codex CLI switch" in session_schema["properties"][
        "dangerously_bypass_approvals_and_sandbox"
    ]["description"]
