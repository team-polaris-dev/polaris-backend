"""DB 단어집 로더 — reconstruct_prompt 주입용.

단어집은 data/vocab.json 하나로 통합됨 (엔티티 + 재무계정 + 관계 + 코드 + 스키마).
재생성:
    python -m pipeline_scripts.graph.dump_vocab

이 모듈은 vocab.json을 읽어 제공하는 얇은 로더다. 하드코딩 사전 없음.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_VOCAB_PATH = Path(__file__).resolve().parents[1] / "data" / "vocab.json"

_EMPTY: dict[str, Any] = {
    "version": "missing",
    "stats": {},
    "schema_summary": "",
    "fin_accounts": {},
    "relation_predicates": {},
    "reprt_codes": {},
    "fs_div": {},
    "organization": [],
    "person": [],
    "product": [],
    "technology": [],
}


def _load() -> dict[str, Any]:
    if not _VOCAB_PATH.exists():
        return _EMPTY
    with _VOCAB_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


GLOSSARY: dict[str, Any] = _load()


def _format_alias_map(title: str, mapping: dict[str, list[str]]) -> list[str]:
    """{코드: [한글별칭...]} → 'DB값 ← 한글, 한글' 줄들. (Gemini가 한글→DB값 역매핑)"""
    out = [title]
    for code, aliases in mapping.items():
        out.append(f"- {code} ← {', '.join(aliases)}")
    return out


def format_for_prompt(top_n: int = 200) -> str:
    """reconstruct_prompt에 끼워넣을 단어집 + 스키마 + 용어 사전 문자열.

    구성:
      1. 스키마 요약
      2. 재무 계정 사전 (매출 → ifrs-full_Revenue)
      3. 관계 술어 사전 (자회사 → IS_SUBSIDIARY_OF)
      4. 보고서 코드 / 연결·별도 사전
      5. 자주 등장하는 엔티티 (degree 내림차순, 상위 top_n)
    """
    g = GLOSSARY
    lines = [g.get("schema_summary", ""), ""]

    lines += _format_alias_map(
        "[재무 계정 — 한글을 이 account_id로 치환]", g.get("fin_accounts", {}))
    lines.append("")
    lines += _format_alias_map(
        "[관계 술어 — 한글을 이 관계타입으로 치환]", g.get("relation_predicates", {}))
    lines.append("")
    lines += _format_alias_map(
        "[보고서 종류 — 한글을 reprt_code로 치환]", g.get("reprt_codes", {}))
    lines.append("")
    lines += _format_alias_map(
        "[재무제표 구분 — 한글을 fs_div로 치환]", g.get("fs_div", {}))
    lines.append("")

    lines.append("[자주 등장하는 엔티티 — degree 내림차순]")
    for cat in ("organization", "person", "product", "technology"):
        items = g.get(cat, [])[:top_n]
        if not items:
            continue
        names = ", ".join(e["name"] for e in items)
        lines.append(f"- {cat}: {names}")
    return "\n".join(lines)
