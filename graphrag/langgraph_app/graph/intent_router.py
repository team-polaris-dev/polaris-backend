"""Cypher 템플릿 매칭 라우터.

cypher_templates.yaml 을 로드하고 (intent, query triggers, entities, slots)
로 적합한 템플릿 1개를 선택한다. 매칭 실패 → (None, None, None) → 호출자가
text_to_cypher fallback 으로 이동.

매칭 순서는 YAML 작성 순서 (insertion-ordered dict). 우선순위 조정은 YAML 에서.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_CATALOG_DIR = Path(__file__).parent.parent / "catalogs"
# 작성 순서: 일반 템플릿(1~3) → 온톨로지(4). stage 필터가 있어 순서는 안전망 역할.
_CATALOG_FILES = ["cypher_templates.yaml", "ontology_rules.yaml"]
_catalog_cache: dict[str, dict[str, Any]] | None = None


def load_catalog() -> dict[str, dict[str, Any]]:
    global _catalog_cache
    if _catalog_cache is None:
        merged: dict[str, dict[str, Any]] = {}
        for fname in _CATALOG_FILES:
            path = _CATALOG_DIR / fname
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                merged.update(yaml.safe_load(f) or {})
        _catalog_cache = merged
    return _catalog_cache


def _build_params(
    tpl: dict[str, Any],
    entities: list[str],
    slots: dict[str, Any],
) -> dict[str, Any]:
    """템플릿 cypher 에 등장하는 $param 들을 entities/slots 로 채움."""
    cypher = tpl.get("cypher", "")
    params: dict[str, Any] = {}
    if "$corp_code" in cypher and entities:
        params["corp_code"] = entities[0]
    if "$corp_codes" in cypher:
        params["corp_codes"] = entities
    if "$year" in cypher and slots.get("year") is not None:
        params["year"] = int(slots["year"])
    if "$account_id" in cypher and slots.get("account_id"):
        params["account_id"] = slots["account_id"]
    if "$relation_type" in cypher and slots.get("relation_type"):
        params["relation_type"] = slots["relation_type"]
    return params


def match_template(
    intent: str | None,
    query: str,
    entities: list[str],
    slots: dict[str, Any],
    stage: int | None = None,
    catalog: dict[str, dict[str, Any]] | None = None,
) -> tuple[str | None, str | None, dict[str, Any] | None, str | None]:
    """매칭 성공 시 (template_key, cypher, params, fact_type), 실패 시 (None,None,None,None).

    intent = 12의도(04_graphrag.md §4). stage = 5단계 사다리(생략 시 단계 무시).
    템플릿은 작성 순서대로 순회 — 멀티홉(2)·교차(3)가 단답(1)보다 위라 먼저 잡힌다.
    stage 가 주어지면 1차로 stage 일치 템플릿만, 못 찾으면 stage 무시하고 재시도(LLM stage 오판 보정).
    """
    cat = catalog if catalog is not None else load_catalog()

    for require_stage in (True, False):
        for key, tpl in cat.items():
            allowed_intents = tpl.get("intents") or tpl.get("intent") or []
            if allowed_intents and intent not in allowed_intents:
                continue

            if require_stage and stage is not None and "stage" in tpl:
                if tpl["stage"] != stage:
                    continue

            triggers = tpl.get("triggers") or []
            if triggers and not any(t in query for t in triggers):
                continue

            if len(entities) < int(tpl.get("required_entities", 0)):
                continue

            req_slots = tpl.get("required_slots") or []
            if any(s not in slots or slots.get(s) in (None, "") for s in req_slots):
                continue

            params = _build_params(tpl, entities, slots)
            return key, tpl["cypher"], params, tpl.get("fact_type")

        if stage is None:
            break  # stage 없으면 1회만

    return None, None, None, None
