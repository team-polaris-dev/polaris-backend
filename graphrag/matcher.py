"""질의 → 엔티티 시드 매칭 (Neo4j FULLTEXT INDEX 단일).

흐름:
1. reconstructed_seeds(앞단 동봉 시) → ID 직접 사용
2. 없으면 escape_lucene(query) → entity_fulltext 인덱스 조회

산출: list[Seed]  (label, id, key_type, key_value, name, score, origin)
"""
from __future__ import annotations

import re
from typing import Iterable

from tool.graph_client import neo4j_driver
from graphrag.schema import Seed


# Lucene 특수문자 — 그대로 던지면 파싱 에러
_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')


def escape_lucene(q: str) -> str:
    """Lucene 쿼리 문자열에서 예약 토큰을 escape."""
    if not q:
        return ""
    return _LUCENE_SPECIAL.sub(r"\\\1", q)


_FULLTEXT_CYPHER = """
CALL db.index.fulltext.queryNodes('entity_fulltext', $q) YIELD node, score
WITH node, score WHERE score > $threshold
RETURN labels(node)[0] AS raw_label,
       CASE
         WHEN node:Organization THEN
           CASE WHEN node.corp_code IS NOT NULL THEN node.corp_code
                ELSE 'org:' + coalesce(node.er_name, node.name) END
         WHEN node:Person THEN node.person_id
         WHEN node:Product THEN node.product_id
         WHEN node:Technology THEN node.tech_id
       END AS id,
       CASE
         WHEN node:Organization AND node.corp_code IS NOT NULL THEN 'corp_code'
         WHEN node:Organization THEN 'er_name'
         WHEN node:Person THEN 'person_id'
         WHEN node:Product THEN 'product_id'
         WHEN node:Technology THEN 'tech_id'
       END AS key_type,
       CASE
         WHEN node:Organization AND node.corp_code IS NOT NULL THEN node.corp_code
         WHEN node:Organization THEN coalesce(node.er_name, node.name)
         WHEN node:Person THEN node.person_id
         WHEN node:Product THEN node.product_id
         WHEN node:Technology THEN node.tech_id
       END AS key_value,
       node.name AS name,
       score
ORDER BY score DESC
LIMIT $limit
"""


_LABEL_MAP = {
    "Organization": "organization",
    "Person": "person",
    "Product": "product",
    "Technology": "technology",
}


def match_fulltext(query: str, *, threshold: float = 0.5, limit: int = 20) -> list[Seed]:
    """FULLTEXT 단일 호출. 빈 질의면 빈 리스트."""
    q = escape_lucene((query or "").strip())
    if not q:
        return []

    seeds: list[Seed] = []
    with neo4j_driver.session() as s:
        rows = s.run(
            _FULLTEXT_CYPHER,
            q=q,
            threshold=threshold,
            limit=limit,
        ).data()

    for row in rows:
        label = _LABEL_MAP.get(row["raw_label"], row["raw_label"].lower())
        if not row.get("id") or not row.get("key_value"):
            continue
        seeds.append(Seed(
            label=label,
            id=row["id"],
            key_type=row["key_type"],
            key_value=row["key_value"],
            name=row.get("name") or "",
            score=float(row.get("score") or 0.0),
            origin="fulltext",
        ))
    return seeds


def match_upstream(upstream_ids: Iterable[str]) -> list[Seed]:
    """앞단이 동봉한 식별자 → Seed 직접 변환.

    upstream_ids는 vocab.json의 id 형식과 동일:
      - "00126380" (corp_code)
      - "org:<er_name>"
      - "p_<sha1>" (person_id 접두사)
      - 기타 product_id/tech_id
    """
    if not upstream_ids:
        return []

    # ID 패턴으로 1차 분류, DB로 확정
    by_kind: dict[str, list[str]] = {
        "corp_code": [],
        "er_name": [],
        "person_id": [],
        "product_id": [],
        "tech_id": [],
    }
    for raw in upstream_ids:
        if not raw:
            continue
        if raw.startswith("org:"):
            by_kind["er_name"].append(raw[4:])
        elif raw.startswith("p_"):
            by_kind["person_id"].append(raw)
        elif raw.isdigit() and len(raw) == 8:
            by_kind["corp_code"].append(raw)
        else:
            # product_id/tech_id 후보 — DB에서 확인 필요. 두 키 모두 시도.
            by_kind["product_id"].append(raw)
            by_kind["tech_id"].append(raw)

    seeds: list[Seed] = []
    with neo4j_driver.session() as s:
        if by_kind["corp_code"]:
            rows = s.run(
                "MATCH (o:Organization) WHERE o.corp_code IN $codes "
                "RETURN o.corp_code AS id, o.corp_code AS key_value, o.name AS name",
                codes=by_kind["corp_code"],
            ).data()
            for r in rows:
                seeds.append(Seed(
                    label="organization", id=r["id"], key_type="corp_code",
                    key_value=r["key_value"], name=r["name"] or "",
                    score=1.0, origin="upstream",
                ))

        if by_kind["er_name"]:
            rows = s.run(
                "MATCH (o:Organization) WHERE o.er_name IN $names "
                "RETURN 'org:' + o.er_name AS id, o.er_name AS key_value, o.name AS name",
                names=by_kind["er_name"],
            ).data()
            for r in rows:
                seeds.append(Seed(
                    label="organization", id=r["id"], key_type="er_name",
                    key_value=r["key_value"], name=r["name"] or "",
                    score=1.0, origin="upstream",
                ))

        if by_kind["person_id"]:
            rows = s.run(
                "MATCH (p:Person) WHERE p.person_id IN $ids "
                "RETURN p.person_id AS id, p.person_id AS key_value, p.name AS name",
                ids=by_kind["person_id"],
            ).data()
            for r in rows:
                seeds.append(Seed(
                    label="person", id=r["id"], key_type="person_id",
                    key_value=r["key_value"], name=r["name"] or "",
                    score=1.0, origin="upstream",
                ))

        # product/tech는 ID 형식이 겹칠 수 있어 두 라벨 다 조회
        if by_kind["product_id"]:
            rows = s.run(
                "MATCH (n:Product) WHERE n.product_id IN $ids "
                "RETURN n.product_id AS id, n.product_id AS key_value, n.name AS name",
                ids=by_kind["product_id"],
            ).data()
            for r in rows:
                seeds.append(Seed(
                    label="product", id=r["id"], key_type="product_id",
                    key_value=r["key_value"], name=r["name"] or "",
                    score=1.0, origin="upstream",
                ))

        if by_kind["tech_id"]:
            rows = s.run(
                "MATCH (n:Technology) WHERE n.tech_id IN $ids "
                "RETURN n.tech_id AS id, n.tech_id AS key_value, n.name AS name",
                ids=by_kind["tech_id"],
            ).data()
            for r in rows:
                seeds.append(Seed(
                    label="technology", id=r["id"], key_type="tech_id",
                    key_value=r["key_value"], name=r["name"] or "",
                    score=1.0, origin="upstream",
                ))

    # 중복 id 제거
    seen: set[str] = set()
    deduped: list[Seed] = []
    for sd in seeds:
        if sd["id"] in seen:
            continue
        seen.add(sd["id"])
        deduped.append(sd)
    return deduped


def match(query: str, upstream_seeds: Iterable[str] | None = None,
          *, threshold: float = 0.5, limit: int = 20) -> list[Seed]:
    """Public entry — upstream 우선, 없으면 FULLTEXT."""
    seeds = match_upstream(upstream_seeds or [])
    if seeds:
        return seeds
    return match_fulltext(query, threshold=threshold, limit=limit)
