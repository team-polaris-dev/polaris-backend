"""text-to-Cypher 정적 가드레일 — 순수 함수, Neo4j/LLM 없음.

read-only 거부(CREATE/MERGE/DELETE/SET/DETACH), whitelist 거부(잘못된 라벨/관계),
결정성 주입(ORDER BY+LIMIT append/보존, rand 거부)을 검증한다.
"""
from __future__ import annotations

import pytest

from config.relations import DOMAIN_RELS
from graphrag.cypher_generator import (
    _ensure_deterministic,
    _is_read_only,
    _uses_only_whitelisted,
)


_PLAIN_MATCH = (
    "MATCH (anchor:Organization)-[r:SUPPLIES_TO]->(o:Organization) "
    "WHERE anchor.corp_code IN $anchors "
    "RETURN anchor.name AS from_name, o.name AS to_name"
)


# ── read-only ────────────────────────────────────────────────

def test_plain_match_is_read_only():
    assert _is_read_only(_PLAIN_MATCH) is True


@pytest.mark.parametrize("clause", [
    "MATCH (a:Organization) CREATE (b:Organization) RETURN a",
    "MATCH (a:Organization) MERGE (a)-[:SUPPLIES_TO]->(a) RETURN a",
    "MATCH (a:Organization) DELETE a",
    "MATCH (a:Organization) SET a.name = 'x' RETURN a",
    "MATCH (a:Organization) DETACH DELETE a",
    "MATCH (a:Organization) REMOVE a.name RETURN a",
    "MATCH (a:Organization) FOREACH (x IN [1] | SET a.name = 'x') RETURN a",
    "LOAD CSV FROM 'file:///x.csv' AS row RETURN row",
])
def test_write_clauses_rejected(clause):
    assert _is_read_only(clause) is False


def test_read_only_case_insensitive():
    assert _is_read_only("match (a:Organization) create (b) return a") is False


def test_write_keyword_as_substring_not_rejected():
    # 'creates' 같은 식별자 내부 부분일치는 단어경계 검사로 통과해야 한다.
    q = "MATCH (a:Organization) WHERE a.name = 'creates inc' RETURN a.name AS from_name"
    assert _is_read_only(q) is True


# ── whitelist ────────────────────────────────────────────────

def test_whitelisted_label_and_rel_accepted():
    rel = DOMAIN_RELS[0]
    q = (
        f"MATCH (a:Organization)-[r:{rel}]->(b:Organization) "
        "RETURN a.name AS from_name, b.name AS to_name"
    )
    assert _uses_only_whitelisted(q) is True


def test_all_domain_rels_accepted():
    for rel in DOMAIN_RELS:
        q = f"MATCH (a:Organization)-[r:{rel}]->(b:Organization) RETURN a"
        assert _uses_only_whitelisted(q) is True, rel


def test_bad_label_rejected():
    q = "MATCH (a:Person)-[r:SUPPLIES_TO]->(b:Organization) RETURN a"
    assert _uses_only_whitelisted(q) is False


def test_bad_rel_type_rejected():
    q = "MATCH (a:Organization)-[r:OWNS]->(b:Organization) RETURN a"
    assert _uses_only_whitelisted(q) is False


# ── determinism ──────────────────────────────────────────────

def test_order_by_and_limit_appended_when_absent():
    out = _ensure_deterministic(_PLAIN_MATCH, 50)
    assert out is not None
    assert "ORDER BY" in out.upper()
    assert "LIMIT 50" in out.upper()


def test_existing_order_by_and_limit_preserved():
    q = _PLAIN_MATCH + " ORDER BY o.name DESC LIMIT 7"
    out = _ensure_deterministic(q, 50)
    assert out is not None
    # 기존 ORDER BY DESC / LIMIT 7 이 유지되고 중복 주입되지 않음.
    assert out.upper().count("ORDER BY") == 1
    assert out.upper().count("LIMIT") == 1
    assert "LIMIT 7" in out.upper()
    assert "DESC" in out.upper()


def test_rand_rejected():
    q = _PLAIN_MATCH + " ORDER BY rand() LIMIT 5"
    assert _ensure_deterministic(q, 50) is None


def test_random_uuid_rejected():
    q = _PLAIN_MATCH.replace("$anchors", "[randomUUID()]")
    assert _ensure_deterministic(q, 50) is None
