"""Text-to-Cypher — 템플릿 미매칭 의도에 한해 LLM 으로 Cypher 생성.

POLARIS 불변규칙(숫자·관계는 결정론) 위반 방지를 위해:
  1) 화이트리스트 템플릿 우선 (intent_router) — 거기서 hit 하면 호출 안 됨.
  2) 여기서 생성한 Cypher 는 cypher_guard 통과 시만 실행.
  3) 실패 시 1회 retry (에러 메시지 첨부), 그래도 실패하면 호출자가 anchor_chunks 안전망.

LLM 호출은 apimaker(=Claude CLI 세션 영속 래퍼) 를 통해 나간다.
apimaker 서버 unreachable → NoLLMAvailable raise.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ...apimaker_client import ApimakerClient, ApimakerError, NoApimakerAvailable


class NoLLMAvailable(Exception):
    pass


class CypherGenError(Exception):
    pass


_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "text_to_cypher.md"
_prompt_cache: str | None = None


def _load_prompt() -> str:
    global _prompt_cache
    if _prompt_cache is None:
        _prompt_cache = _PROMPT_PATH.read_text(encoding="utf-8")
    return _prompt_cache


def _parse_json_block(text: str) -> dict[str, Any]:
    """LLM 응답에서 JSON 객체 추출. ```json fence 또는 raw object."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = fence.group(1) if fence else text
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise CypherGenError(f"no JSON object in response: {text[:200]}")
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        raise CypherGenError(f"invalid JSON: {e}; raw={raw[:200]}") from e


class TextToCypher:
    """apimaker 경유 LLM. 호출 실패는 NoLLMAvailable 통일."""

    def __init__(self, model: str | None = None):
        try:
            self.client = ApimakerClient(model=model)
        except NoApimakerAvailable as e:
            raise NoLLMAvailable(str(e)) from e
        self.model = self.client.model

    def generate(
        self,
        query: str,
        entities: list[str],
        slots: dict[str, Any],
        prior_error: str | None = None,
    ) -> tuple[str, dict[str, Any], str]:
        """(cypher, params, rationale) 반환. 실패 시 CypherGenError."""
        user_msg = (
            f"질문: {query}\n"
            f"entities (corp_codes): {entities}\n"
            f"slots: {slots}\n"
        )
        if prior_error:
            user_msg += f"\n이전 시도 에러: {prior_error}\n다른 접근으로 재시도하라."

        try:
            text = self.client.chat(_load_prompt(), user_msg)
        except ApimakerError as e:
            raise CypherGenError(f"apimaker call failed: {e}") from e

        obj = _parse_json_block(text)

        cypher = (obj.get("cypher") or "").strip()
        params = obj.get("params") or {}
        rationale = obj.get("rationale") or ""

        if not cypher:
            raise CypherGenError(f"empty cypher: {rationale}")
        if not isinstance(params, dict):
            raise CypherGenError(f"params not dict: {type(params)}")

        return cypher, params, rationale
