"""Cypher 가드 — Text-to-Cypher 결과 안전성 검증.

POLARIS 불변규칙: 숫자·관계는 DB 결정론. LLM 생성 Cypher는 통과 시만 실행.

검증 순서:
  1) 금지 키워드(쓰기·관리 명령) 사전 거부
  2) 사용 라벨 화이트리스트 검사
  3) LIMIT 미존재 시 자동 주입 (기본 100)
  4) EXPLAIN dry-run (Neo4j 가 실제 실행 없이 스키마·문법 검증, 비용 0)

성공 시 정규화된 cypher 문자열 반환. 실패 시 GuardError raise.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j import Session


class GuardError(Exception):
    pass


_FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|DELETE|SET|REMOVE|DROP|DETACH|LOAD\s+CSV)\b"
    r"|CALL\s+(apoc\.create|db\.create|dbms\.|apoc\.refactor|apoc\.merge)",
    re.IGNORECASE,
)
_LABEL_RE = re.compile(r":([A-Z][A-Za-z_][A-Za-z0-9_]*)")
_LIMIT_RE = re.compile(r"\bLIMIT\b", re.IGNORECASE)

ALLOWED_LABELS = {
    "Organization",
    "Person",
    "FinMetric",
    "FilingDocument",
    "Chunk",
    "Product",
    "Technology",
}
# 03_neo4j.md 의 관계 라벨도 동일 정규식에 잡히지만 의도적으로 화이트리스트에 포함
ALLOWED_RELS = {
    "EXECUTIVE_OF",
    "IS_MAJOR_SHAREHOLDER_OF",
    "IS_SUBSIDIARY_OF",
    "INVESTS_IN",
    "HAS_METRIC",
    "DERIVED_FROM",
    "PRODUCES",
    "USES_TECH",
    "SUPPLIES_TO",
    "RELATED_PARTY",
    "reports",
    "has_chunk",
    "hasActor",
    "hasObject",
    # 파생 (derived_by='rule') — 03_neo4j.md §2-3
    "CONTROLS_INDIRECTLY",
    "INTERLOCKING_DIRECTORATE",
}

DEFAULT_LIMIT = 100


def validate_cypher(
    cypher: str,
    session: "Session | None" = None,
    params: dict | None = None,
) -> str:
    """검증·정규화된 Cypher 반환. 실패 시 GuardError.

    params 를 주면 EXPLAIN dry-run 에 바인딩해 "parameter missing" 경고를 피한다
    (플랜만 생성 — 실제 실행 아님).
    """
    if not cypher or not cypher.strip():
        raise GuardError("empty cypher")

    if _FORBIDDEN.search(cypher):
        raise GuardError("write/admin operation detected")

    used = set(_LABEL_RE.findall(cypher))
    unknown = used - ALLOWED_LABELS - ALLOWED_RELS
    if unknown:
        raise GuardError(f"unknown labels/rels: {sorted(unknown)}")

    normalized = cypher.strip().rstrip(";")
    if not _LIMIT_RE.search(normalized):
        normalized = f"{normalized}\nLIMIT {DEFAULT_LIMIT}"

    if session is not None:
        try:
            session.run("EXPLAIN " + normalized, **(params or {})).consume()
        except Exception as e:  # neo4j.exceptions.Neo4jError 포함
            raise GuardError(f"EXPLAIN failed: {e}") from e

    return normalized
