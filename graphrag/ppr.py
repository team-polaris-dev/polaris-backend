"""Personalized PageRank 기반 시드 관련성 랭킹.

하이퍼 연결 그래프(허브 차수 수천)에서 시드에 *진짜* 가까운 노드를 뽑는다.
순수 파이썬 power iteration — GDS·networkx 등 외부 의존성 없음.

흐름:
1. seeds → elementId
2. APOC subgraphNodes 로 시드 depth-2 도메인 이웃(Org/Person/Product/Tech) 추출
3. 이웃 내부 도메인 엣지 수집 → 가중 무방향 인접
4. 시드 personalization 으로 PPR 반복 → 노드 관련성 점수
5. 상위 N 노드를 our-id/label/name/score 로 반환
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, TypedDict

from config.graphrag import (
    PPR_ALPHA,
    PPR_ITERS,
    PPR_NEIGHBORHOOD_LIMIT,
    PPR_TOP_NODES,
)
# 관계 가중치·도메인 관계 목록은 SSOT(config.relations)에서. HAS_METRIC 등 비도메인 관계는 제외.
from config.relations import REL_WEIGHT as _REL_WEIGHT, DOMAIN_RELS as _DOMAIN_RELS
from config.entities import LABEL_MAP as _LABEL_MAP
from tool.graph_client import neo4j_driver
from graphrag.schema import Seed


_REL_FILTER = "|".join(_DOMAIN_RELS)
_NOISE_NAMES = {"계", "합계", "소계", "-", "주", ""}

_OUR_ID_CYPHER = (
    "CASE "
    "WHEN node:Organization THEN coalesce(node.corp_code, 'org:' + node.er_name) "
    "WHEN node:Person THEN node.person_id "
    "WHEN node:Product THEN node.product_id "
    "WHEN node:Technology THEN node.tech_id END"
)


class PPRNode(TypedDict):
    element_id: str
    our_id: str
    label: str
    name: str
    score: float


def _seed_element_ids(session, seeds: list[Seed]) -> list[str]:
    """seed key_type/key_value → Organization/Person/... elementId."""
    eids: list[str] = []
    for sd in seeds:
        kt, kv = sd.get("key_type"), sd.get("key_value")
        if not kt or not kv:
            continue
        clause = {
            "corp_code": "MATCH (n:Organization {corp_code:$v})",
            "er_name": "MATCH (n:Organization {er_name:$v})",
            "person_id": "MATCH (n:Person {person_id:$v})",
            "product_id": "MATCH (n:Product {product_id:$v})",
            "tech_id": "MATCH (n:Technology {tech_id:$v})",
        }.get(kt)
        if not clause:
            continue
        row = session.run(f"{clause} RETURN elementId(n) AS eid LIMIT 1", v=kv).single()
        if row and row["eid"]:
            eids.append(row["eid"])
    return eids


def _neighborhood(session, seed_eids: list[str]) -> dict[str, dict[str, Any]]:
    """시드 depth-2 도메인 이웃 노드 메타 (elementId → {our_id,label,name})."""
    rows = session.run(
        f"""
        MATCH (s) WHERE elementId(s) IN $seed_eids
        WITH collect(s) AS starts
        CALL apoc.path.subgraphNodes(starts, {{
            maxLevel: 2, bfs: true, limit: $lim,
            relationshipFilter: '{_REL_FILTER}',
            labelFilter: '+Organization|+Person|+Product|+Technology'
        }}) YIELD node
        RETURN elementId(node) AS eid,
               {_OUR_ID_CYPHER.strip()} AS our_id,
               labels(node)[0] AS lab,
               coalesce(node.name, node.corp_code, node.er_name, '?') AS name
        """,
        seed_eids=seed_eids, lim=PPR_NEIGHBORHOOD_LIMIT,
    ).data()
    meta: dict[str, dict[str, Any]] = {}
    for r in rows:
        nm = r.get("name") or ""
        if nm in _NOISE_NAMES or not r.get("our_id"):
            continue
        meta[r["eid"]] = {
            "our_id": r["our_id"],
            "label": _LABEL_MAP.get(r["lab"], (r["lab"] or "").lower()),
            "name": nm,
        }
    return meta


def _domain_edges(session, eids: list[str]) -> list[tuple[str, str, str]]:
    """이웃 노드 집합 내부의 도메인 엣지 (a_eid, b_eid, rel_type). 시점·qc 필터."""
    rows = session.run(
        """
        MATCH (a)-[r]-(b)
        WHERE elementId(a) IN $eids AND elementId(b) IN $eids
          AND elementId(a) < elementId(b)
          AND type(r) IN $rels
          AND coalesce(r.valid_to,'') = '' AND r.qc_disabled_at IS NULL
        RETURN DISTINCT elementId(a) AS a, elementId(b) AS b, type(r) AS t
        """,
        eids=eids, rels=_DOMAIN_RELS,
    ).data()
    return [(r["a"], r["b"], r["t"]) for r in rows]


def _power_iteration(
    adj: dict[str, list[tuple[str, float]]], seed_eids: list[str]
) -> dict[str, float]:
    """가중 무방향 인접에서 시드 personalization PPR. r = (1-α)s + α·Wr."""
    nodes = list(adj.keys())
    if not nodes:
        return {}
    seeds_in = [e for e in seed_eids if e in adj] or nodes
    s = {n: (1.0 / len(seeds_in) if n in seeds_in else 0.0) for n in nodes}
    outw = {n: sum(w for _, w in adj[n]) or 1.0 for n in nodes}
    r = dict(s)
    for _ in range(PPR_ITERS):
        nr = {n: (1.0 - PPR_ALPHA) * s[n] for n in nodes}
        for n in nodes:
            rn = r[n]
            if not rn:
                continue
            share = PPR_ALPHA * rn / outw[n]
            for m, w in adj[n]:
                nr[m] += share * w
        r = nr
    return r


def ppr_rank(seeds: list[Seed]) -> list[PPRNode]:
    """seeds 기준 PPR 상위 노드 리스트(시드 포함, score 내림차순)."""
    if not seeds:
        return []
    with neo4j_driver.session() as session:
        seed_eids = _seed_element_ids(session, seeds)
        if not seed_eids:
            return []
        meta = _neighborhood(session, seed_eids)
        if not meta:
            return []
        eids = list(meta.keys())
        edges = _domain_edges(session, eids)

    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for e in eids:
        adj.setdefault(e, [])
    for a, b, t in edges:
        if a not in meta or b not in meta:
            continue
        w = _REL_WEIGHT.get(t, 1.0)
        adj[a].append((b, w))
        adj[b].append((a, w))

    scores = _power_iteration(adj, seed_eids)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])

    top = ranked[:PPR_TOP_NODES]
    # 시드는 항상 포함 (top-N 밖이어도)
    seed_set = set(seed_eids)
    have = {e for e, _ in top}
    for e in seed_eids:
        if e not in have and e in scores:
            top.append((e, scores[e]))

    out: list[PPRNode] = []
    mx = top[0][1] if top else 1.0
    for eid, sc in top:
        m = meta.get(eid)
        if not m:
            continue
        out.append(PPRNode(
            element_id=eid,
            our_id=m["our_id"],
            label=m["label"],
            name=m["name"],
            score=round(sc / mx, 4) if mx else 0.0,  # 시드=1.0 기준 정규화
        ))
    return out
