"""apimaker HTTP 클라이언트 — graphrag 의 모든 LLM 호출은 이 클라이언트를 통해 나간다.

apimaker(=Claude/Codex CLI 세션 영속 래퍼) 는 Claude CLI 의 구독 인증을 쓰므로
ANTHROPIC_API_KEY 가 필요 없다. 서버 unreachable → NoApimakerAvailable raise →
호출자(intent_classifier·text_to_cypher) 가 규칙 fallback 으로 degrade.

HTTP shape (___test/src/apimaker/README.md):
    POST   /sessions                              -> {session_id, provider}
    POST   /sessions/{session_id}/messages        -> {response}
    DELETE /sessions/{session_id}
    GET    /health

graphrag 의 LLM 호출은 모두 stateless 분류·생성(JSON 출력). 세션 재사용 시
이전 턴이 입력 토큰으로 누적돼 분류 결정성을 흐릴 수 있어 **호출당 fresh
session** 정책. localhost HTTP 3 roundtrip 비용은 감수.

설정:
    APIMAKER_URL          기본 http://127.0.0.1:8000
    APIMAKER_MODEL        기본 claude-haiku-4-5  (Haiku=빠른 분류용)
    APIMAKER_PROVIDER     기본 claude
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class NoApimakerAvailable(Exception):
    """apimaker 서버 미가동 / /health 실패. 초기화 단계에서만 발생."""


class ApimakerError(Exception):
    """apimaker 가 4xx/5xx 또는 네트워크 에러. 호출자가 재시도/fallback 결정."""


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        snippet = e.read().decode("utf-8", "replace")[:200]
        raise ApimakerError(f"POST {url} -> {e.code}: {snippet}") from e
    except urllib.error.URLError as e:
        raise ApimakerError(f"POST {url} network: {e.reason}") from e


def _delete(url: str, timeout: float) -> None:
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
    except (urllib.error.HTTPError, urllib.error.URLError):
        pass  # 세션 누수는 apimaker 종료 시 close_all 로 정리됨


class ApimakerClient:
    """한 번의 LLM 호출 = 세션 생성 → 메시지 1회 → 세션 삭제."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("APIMAKER_URL", "http://127.0.0.1:8000")
        ).rstrip("/")
        self.model = model or os.environ.get("APIMAKER_MODEL", "claude-haiku-4-5")
        self.provider = provider or os.environ.get("APIMAKER_PROVIDER", "claude")
        self.timeout = timeout

        # health check — 서버 죽었으면 빠르게 NoApimakerAvailable
        health_url = f"{self.base_url}/health"
        try:
            with urllib.request.urlopen(health_url, timeout=min(self.timeout, 3.0)) as r:
                if r.status != 200:
                    raise NoApimakerAvailable(f"/health -> {r.status}")
        except urllib.error.HTTPError as e:
            raise NoApimakerAvailable(f"/health -> {e.code}") from e
        except urllib.error.URLError as e:
            raise NoApimakerAvailable(
                f"apimaker unreachable at {self.base_url}: {e.reason}"
            ) from e

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """system + user → 모델 응답 텍스트. 실패 시 ApimakerError."""
        created = _post_json(
            f"{self.base_url}/sessions",
            {
                "provider": self.provider,
                "model": self.model,
                "system_prompt": system_prompt,
            },
            timeout=self.timeout,
        )
        sid = created.get("session_id")
        if not sid:
            raise ApimakerError(f"start_session: missing session_id in {created!r}")

        try:
            msg = _post_json(
                f"{self.base_url}/sessions/{sid}/messages",
                {"prompt": user_prompt},
                timeout=self.timeout,
            )
            return str(msg.get("response") or "")
        finally:
            _delete(f"{self.base_url}/sessions/{sid}", timeout=self.timeout)
