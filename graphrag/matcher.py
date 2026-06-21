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
# 회사명 정규화·일반어 판정은 SSOT(config.entities). _norm 은 정확매칭 판정용 별칭.
from config.entities import (
    GENERIC_ORG_TERMS,
    LABEL_MAP as _LABEL_MAP,
    is_generic_org,
    normalize_corp_name as _norm,
)
from config.relations import METRIC_TERMS
from tool.graph_client import neo4j_driver
from graphrag.schema import Seed


# Lucene 특수문자 — 그대로 던지면 파싱 에러
_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')
_EXACT_CORP_ALIASES = {
    "sk하이닉스": "00164779",
    "에스케이하이닉스": "00164779",
    "하이닉스": "00164779",
    "삼성전자": "00126380",
    "samsungelectronics": "00126380",
}


def escape_lucene(q: str) -> str:
    """Lucene 쿼리 문자열에서 예약 토큰을 escape."""
    if not q:
        return ""
    return _LUCENE_SPECIAL.sub(r"\\\1", q)


# 토큰 끝 조사(은/는/이/가 …) — 불용어 정확매칭을 위해 제거한다("매출은" → "매출").
_LINK_JOSA = ("으로", "은", "는", "이", "가", "을", "를", "의", "로", "과", "와", "도", "만")
# 엔티티 링킹 풀텍스트에서 떼어낼 불용어 = 재무 지표어(랭킹 차원) + 회사/기업류 일반명사.
# 둘 다 회사명이 아닌데 FULLTEXT cjk 가 Product 노드(매출채권)·실재 노드(기업은행)에 매칭해
# 시드 슬롯을 잠식한다. 단어집은 config(relations.METRIC_TERMS·entities.GENERIC_ORG_TERMS).
_LINK_STOPWORDS = frozenset(METRIC_TERMS) | frozenset(GENERIC_ORG_TERMS)


def _strip_link_stopwords(query: str) -> str:
    """엔티티 링킹(풀텍스트)용으로 불용어 *독립 토큰*만 제거한다.

    "매출"(랭킹 차원)·"기업"(보통명사)은 회사명이 아닌데 entity_fulltext 에 매출채권 Product·
    기업은행 같은 노드가 있어 가짜 시드가 잡힌다. 토큰 단위로만 떼므로 '삼성자산운용'·'기업은행'
    같은 회사명 토큰은 보존된다(부분문자열 치환이 아님). 원 질의는 그대로 두고 풀텍스트에만 쓴다.
    """
    kept: list[str] = []
    for tok in (query or "").split():
        bare = tok
        for josa in _LINK_JOSA:
            if bare.endswith(josa) and len(bare) > len(josa):
                bare = bare[: -len(josa)]
                break
        if bare in _LINK_STOPWORDS:
            continue
        kept.append(tok)
    return " ".join(kept)


# ── 과매칭 억제 (튠 파라미터는 config.graphrag, 정규화는 config.entities) ──────────
# cjk analyzer 는 "에스케이하이닉스" 를 bigram 으로 쪼개 SK 계열 수십 곳에 매칭한다.
# 상위 N개를 그대로 시드로 쓰면 ego 그래프가 계열 합집합(헤어볼)이 된다. 그래서
# 결과를 (1) 일반어 제외 (2) 정규화 정확매칭 우선 (3) top score 밴드컷 (4) 시드 상한 으로 좁힌다.


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
        if is_generic_org(r.get("name") or ""):
            continue  # 특수관계자 등 일반어 placeholder 는 시드에서 제외
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
    # 풀텍스트 검색어에서만 불용어 토큰을 제거(원 query 는 _select 의 strong 판정에 그대로 씀).
    q = escape_lucene(_strip_link_stopwords(query).strip())
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


def _merge_seeds(*groups: list[Seed], max_seeds: int = MAX_SEEDS) -> list[Seed]:
    seen: set[str] = set()
    merged: list[Seed] = []
    for group in groups:
        for seed in group:
            if not seed.get("id") or seed["id"] in seen:
                continue
            seen.add(seed["id"])
            merged.append(seed)
            if len(merged) >= max_seeds:
                return merged
    return merged


def _exact_alias_codes(query: str) -> list[str]:
    qn = _norm(query)
    found: list[str] = []
    for alias, code in sorted(_EXACT_CORP_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if _norm(alias) in qn and code not in found:
            found.append(code)
    return found


def match_exact_corp_mentions(query: str, *, cap: int = MAX_SEEDS) -> list[Seed]:
    """RDB 회사명 해소로 명시적 회사 seed를 보강한다.

    FULLTEXT CJK가 'SK하이닉스'를 'SK(주)'로 과매칭하는 케이스를 막기 위한
    보조 경로다. RDB/Neo4j 중 하나라도 실패하면 빈 리스트로 degrade한다.
    """
    if not query:
        return []
    alias_codes = _exact_alias_codes(query)
    try:
        from tool.rdb_client import resolve_corp_codes_from_text  # noqa: PLC0415

        codes = [*alias_codes, *resolve_corp_codes_from_text(query, cap=cap)]
    except Exception as exc:
        print(f"⚠️ [GraphRAG matcher] exact corp resolve failed: {exc!r}")
        codes = alias_codes
    codes = [c for c in dict.fromkeys(codes) if c][:cap]
    try:
        return match_upstream(codes)
    except Exception as exc:
        print(f"⚠️ [GraphRAG matcher] exact corp seed fetch failed: {exc!r}")
        return []


def match(query: str, upstream_seeds: Iterable[str] | None = None,
          *, threshold: float = FULLTEXT_THRESHOLD, limit: int = FULLTEXT_POOL) -> list[Seed]:
    """Public entry — upstream 우선, 없으면 FULLTEXT."""
    seeds = match_upstream(upstream_seeds or [])
    if seeds:
        return seeds
    exact = match_exact_corp_mentions(query, cap=MAX_SEEDS)
    fulltext = match_fulltext(query, threshold=threshold, limit=limit)
    return _merge_seeds(exact, fulltext, max_seeds=MAX_SEEDS)
