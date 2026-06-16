"""질의 → 엔티티 시드 매칭 (Neo4j FULLTEXT INDEX 단일).

흐름:
1. reconstructed_seeds(앞단 동봉 시) → ID 직접 사용
2. 없으면 escape_lucene(query) → entity_fulltext 인덱스 조회

산출: list[Seed]  (label, id, key_type, key_value, name, score, origin)
"""
from __future__ import annotations

import re
from typing import Iterable

from config.graphrag import (
    FULLTEXT_POOL,
    FULLTEXT_THRESHOLD,
    MAX_SEEDS,
    SEED_SCORE_BAND,
)
from tool.graph_client import neo4j_driver
from graphrag.schema import Seed


# Lucene 특수문자 — 그대로 던지면 파싱 에러
_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')


def escape_lucene(q: str) -> str:
    """Lucene 쿼리 문자열에서 예약 토큰을 escape."""
    if not q:
        return ""
    return _LUCENE_SPECIAL.sub(r"\\\1", q)


# ── 과매칭 억제 (튠 파라미터는 config.graphrag) ───────────────────
# cjk analyzer 는 "에스케이하이닉스" 를 bigram 으로 쪼개 SK 계열 수십 곳에 매칭한다.
# 상위 N개를 그대로 시드로 쓰면 ego 그래프가 계열 합집합(헤어볼)이 된다. 그래서
# 결과를 (1) 정규화 정확매칭 우선 (2) top score 밴드컷 (3) 시드 상한 으로 좁힌다.
_SUFFIX_RE = re.compile(
    r"\(주\)|㈜|\(유\)|㈜|주식회사|유한회사|\(재\)|,?\s*inc\.?|,?\s*co\.?,?\s*ltd\.?",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    """회사명 정규화 — 법인 접미사·공백 제거 후 소문자. 정확매칭 판정용."""
    return re.sub(r"\s+", "", _SUFFIX_RE.sub("", s or "")).lower()


def _select(rows: list[dict], query: str, *, max_seeds: int, band: float) -> list[dict]:
    """FULLTEXT 풀(score 내림차순)에서 시드를 좁힌다.

    유지 조건(OR): ① 후보 정규화 이름이 질의에 그대로 포함(사용자가 그 이름을 직접 침)
                  ② top score 대비 band 이상(약칭으로 들어온 근접 매치 보존)
    둘 다 아니면 버린다(= 계열 bigram 부분일치 노이즈). 최대 max_seeds 개.
    """
    if not rows:
        return []
    qn = _norm(query)
    top = float(rows[0].get("score") or 0.0)
    cut = top * band
    chosen: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        nm = _norm(r.get("name") or "")
        strong = bool(nm) and nm in qn
        near = float(r.get("score") or 0.0) >= cut
        if not (strong or near):
            continue
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        chosen.append(r)
        if len(chosen) >= max_seeds:
            break
    return chosen


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


def match_fulltext(
    query: str,
    *,
    threshold: float = FULLTEXT_THRESHOLD,
    limit: int = FULLTEXT_POOL,
    max_seeds: int = MAX_SEEDS,
    band: float = SEED_SCORE_BAND,
) -> list[Seed]:
    """FULLTEXT 단일 호출 → 과매칭 억제(_select) → Seed 리스트. 빈 질의면 빈 리스트.

    limit 은 FULLTEXT 후보 풀 크기, max_seeds 는 최종 시드 상한이다.
    """
    q = escape_lucene((query or "").strip())
    if not q:
        return []

    with neo4j_driver.session() as s:
        rows = s.run(
            _FULLTEXT_CYPHER,
            q=q,
            threshold=threshold,
            limit=limit,
        ).data()

    # 유효 행만 남기고 과매칭 억제
    rows = [r for r in rows if r.get("id") and r.get("key_value")]
    rows = _select(rows, query, max_seeds=max_seeds, band=band)

    seeds: list[Seed] = []
    for row in rows:
        label = _LABEL_MAP.get(row["raw_label"], row["raw_label"].lower())
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
          *, threshold: float = FULLTEXT_THRESHOLD, limit: int = FULLTEXT_POOL) -> list[Seed]:
    """Public entry — upstream 우선, 없으면 FULLTEXT."""
    seeds = match_upstream(upstream_seeds or [])
    if seeds:
        return seeds
    return match_fulltext(query, threshold=threshold, limit=limit)
