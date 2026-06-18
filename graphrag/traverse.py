"""Cypher 패턴 실행 + GraphHit 평탄화.

cypher/traverse.cypher의 패턴을 읽어 라벨로 인덱싱.
seed.label/key_type에 따라 적정 패턴 호출, 결과를 GraphHit 리스트로 변환.
정적 hit 0개인 경우 호출자(search.py)가 fallback 패턴 선택.
"""
from __future__ import annotations

import re
from itertools import zip_longest
from pathlib import Path
from typing import Any

from config.graphrag import (
    CROSS_EDGE_MIN_RELEVANCE,
    FIN_KEY_ACCOUNTS,
    INDUCED_EDGES,
    INDUCED_MAX_NODES,
    MAX_EDGES,
    MAX_INDUCED_EDGES,
    NODE_HIT_SCORE,
    REL_HIT_SCORE,
    SEED_SPOKE_CAP,
)
from tool.graph_client import neo4j_driver
from graphrag.ppr import (
    ppr_rank,
    _seed_element_ids as _ppr_seed_eids,
    _DOMAIN_RELS as _PPR_DOMAIN_RELS,
)
from graphrag.schema import GraphHit, Seed


_CYPHER_FILE = Path(__file__).resolve().parent / "cypher" / "traverse.cypher"


def _load_patterns() -> dict[str, str]:
    text = _CYPHER_FILE.read_text(encoding="utf-8")
    blocks = re.split(r"^--\s*@name\s+(\w+)\s*$", text, flags=re.MULTILINE)
    # blocks[0]은 헤더(주석 등), 이후 (name, body) 쌍 반복
    patterns: dict[str, str] = {}
    for i in range(1, len(blocks) - 1, 2):
        name = blocks[i].strip()
        body = blocks[i + 1].strip()
        # 끝의 다음 -- @name 또는 EOF까지가 본문
        patterns[name] = body
    return patterns


PATTERNS = _load_patterns()


def _org_match(var: str, key_type: str, key_value_param: str = "key_value") -> str:
    """ORG_MATCH 치환 문자열 생성.

    var: Cypher 변수명 (o, root 등)
    key_type: 'corp_code' | 'er_name'
    key_value_param: Cypher 파라미터 이름 ($로 시작 없이)
    """
    if key_type == "corp_code":
        return f"MATCH ({var}:Organization {{corp_code: ${key_value_param}}})"
    elif key_type == "er_name":
        return f"MATCH ({var}:Organization {{er_name: ${key_value_param}}})"
    else:
        raise ValueError(f"unsupported org key_type: {key_type}")


# ─────────────────────────────────────────────────────────────
# Hit 생성 헬퍼
# ─────────────────────────────────────────────────────────────

def _rel_hit(rel_type: str, from_id: str, from_name: str,
             to_id: str, to_name: str, attrs: dict[str, Any],
             source: str | None = None, chunk: str | None = None,
             score: float = REL_HIT_SCORE) -> GraphHit:
    full_attrs: dict[str, Any] = {
        "rel_type": rel_type,
        "from_id": from_id,
        "from_name": from_name,
        "to_id": to_id,
        "to_name": to_name,
        **attrs,
    }
    # 청크 출처(chunk_id)는 추출 엣지에만 존재. attrs 에 실어 adapt_to_legacy 가 읽는다.
    if chunk:
        full_attrs["chunk_id"] = chunk
    hit: GraphHit = {
        "id": f"rel:{rel_type}:{from_id}:{to_id}",
        "label": "relationship",
        "name": f"{from_name} → {to_name}",
        "attrs": full_attrs,
        "score": score,
        "seed_origin": "traversal",
    }
    if source:
        hit["source"] = source
    return hit


def _node_hit(label: str, id_: str, name: str, attrs: dict[str, Any] | None = None,
              source: str | None = None, score: float = NODE_HIT_SCORE) -> GraphHit:
    hit: GraphHit = {
        "id": id_,
        "label": label,  # type: ignore[typeddict-item]
        "name": name,
        "attrs": attrs or {},
        "score": score,
        "seed_origin": "traversal",
    }
    if source:
        hit["source"] = source
    return hit


# ─────────────────────────────────────────────────────────────
# 패턴 실행 + 결과 → GraphHit 변환
# ─────────────────────────────────────────────────────────────

def _run_company_immediate(seed: Seed) -> list[GraphHit]:
    pattern = PATTERNS["pattern_company_immediate"]
    cypher = pattern.replace("{{ORG_MATCH}}", _org_match("o", seed["key_type"]))

    with neo4j_driver.session() as s:
        row = s.run(
            cypher, key_value=seed["key_value"], fin_accounts=FIN_KEY_ACCOUNTS
        ).single()
    if not row:
        return []

    root_id = row["root_id"]
    root_name = row["root_name"]
    hits: list[GraphHit] = []

    # FinMetric → fin_metric hit
    for m in row.get("metrics") or []:
        if not m or m.get("metric_id") is None:
            continue
        hits.append(_node_hit(
            "fin_metric",
            m["metric_id"],
            m.get("account_id") or m["metric_id"],
            attrs={k: v for k, v in m.items() if v is not None},
            source=m.get("source"),
        ))

    # Executive → person hit + relationship hit
    for e in row.get("execs") or []:
        if not e or e.get("person_id") is None:
            continue
        person_id = e["person_id"]
        person_name = e.get("name") or person_id
        hits.append(_node_hit(
            "person", person_id, person_name,
            attrs={"pos": e.get("pos")},
            source=e.get("source"),
        ))
        hits.append(_rel_hit(
            "EXECUTIVE_OF",
            from_id=person_id, from_name=person_name,
            to_id=root_id, to_name=root_name,
            attrs={"pos": e.get("pos")},
            source=e.get("source"),
        ))

    # Shareholder → relationship hit
    for h in row.get("holders") or []:
        if not h or h.get("holder_id") is None:
            continue
        hits.append(_rel_hit(
            "IS_MAJOR_SHAREHOLDER_OF",
            from_id=h["holder_id"], from_name=h.get("name") or "",
            to_id=root_id, to_name=root_name,
            attrs={"qota_rt": h.get("qota_rt")},
            source=h.get("source"),
        ))

    # Investee
    for inv in row.get("invs") or []:
        if not inv or inv.get("investee_id") is None:
            continue
        hits.append(_rel_hit(
            "INVESTS_IN",
            from_id=root_id, from_name=root_name,
            to_id=inv["investee_id"], to_name=inv.get("name") or "",
            attrs={"qota_rt": inv.get("qota_rt")},
            source=inv.get("source"),
        ))

    # Related party
    for rp in row.get("related") or []:
        if not rp or rp.get("counterpart_id") is None:
            continue
        hits.append(_rel_hit(
            "RELATED_PARTY",
            from_id=root_id, from_name=root_name,
            to_id=rp["counterpart_id"], to_name=rp.get("name") or "",
            attrs={},
            source=rp.get("source"),
            chunk=rp.get("chunk"),
        ))

    # Interlocking
    for idn in row.get("interlocking") or []:
        if not idn or idn.get("counterpart_id") is None:
            continue
        hits.append(_rel_hit(
            "INTERLOCKING_DIRECTORATE",
            from_id=root_id, from_name=root_name,
            to_id=idn["counterpart_id"], to_name=idn.get("name") or "",
            attrs={},
            source=idn.get("source"),
            chunk=idn.get("chunk"),
        ))

    return hits


def _run_subsidiary_tree(seed: Seed) -> list[GraphHit]:
    pattern = PATTERNS["pattern_subsidiary_tree"]
    cypher = pattern.replace("{{ORG_MATCH_root}}", _org_match("root", seed["key_type"]))

    with neo4j_driver.session() as s:
        row = s.run(cypher, key_value=seed["key_value"]).single()
    if not row:
        return []

    root_id = row["root_id"]
    root_name = row["root_name"]
    hits: list[GraphHit] = []
    for sub in row.get("subs") or []:
        if not sub or sub.get("id") is None:
            continue
        hits.append(_rel_hit(
            "IS_SUBSIDIARY_OF",
            from_id=sub["id"], from_name=sub.get("name") or "",
            to_id=root_id, to_name=root_name,
            attrs={"depth": sub.get("depth")},
            source=sub.get("source"),
        ))
    return hits


def _run_supply_chain(seed: Seed) -> list[GraphHit]:
    pattern = PATTERNS["pattern_supply_chain"]
    cypher = pattern.replace("{{ORG_MATCH}}", _org_match("o", seed["key_type"]))

    with neo4j_driver.session() as s:
        row = s.run(cypher, key_value=seed["key_value"]).single()
    if not row:
        return []

    root_id = row["root_id"]
    root_name = row["root_name"]

    supplier_hits: list[GraphHit] = []
    for sp in row.get("suppliers") or []:
        if not sp or sp.get("id") is None:
            continue
        supplier_hits.append(_rel_hit(
            "SUPPLIES_TO",
            from_id=sp["id"], from_name=sp.get("name") or "",
            to_id=root_id, to_name=root_name,
            attrs={"role": "supplier", "tier": sp.get("tier")},
            source=sp.get("source"),
            chunk=sp.get("chunk"),
        ))
    buyer_hits: list[GraphHit] = []
    for b in row.get("buyers") or []:
        if not b or b.get("id") is None:
            continue
        buyer_hits.append(_rel_hit(
            "SUPPLIES_TO",
            from_id=root_id, from_name=root_name,
            to_id=b["id"], to_name=b.get("name") or "",
            attrs={"role": "buyer", "tier": b.get("tier")},
            source=b.get("source"),
            chunk=b.get("chunk"),
        ))

    # 인바운드(공급사)·아웃바운드(납품처)를 교차 배치 — 엣지 cap 라운드로빈이 한쪽으로
    # 쏠리지 않게. 공급사를 먼저 둬 '공급 리스크' 류 질문에 인바운드가 우선 보존되게 한다.
    hits: list[GraphHit] = []
    for sp, b in zip_longest(supplier_hits, buyer_hits):
        if sp is not None:
            hits.append(sp)
        if b is not None:
            hits.append(b)
    return hits


def _run_product_links(seed: Seed) -> list[GraphHit]:
    pattern = PATTERNS["pattern_product_links"]
    cypher = pattern.replace("{{ORG_MATCH}}", _org_match("o", seed["key_type"]))

    with neo4j_driver.session() as s:
        row = s.run(cypher, key_value=seed["key_value"]).single()
    if not row:
        return []

    root_id = row["root_id"]
    root_name = row["root_name"]
    hits: list[GraphHit] = []

    for p in row.get("products") or []:
        if not p or p.get("id") is None:
            continue
        hits.append(_rel_hit(
            "PRODUCES",
            from_id=root_id, from_name=root_name,
            to_id=p["id"], to_name=p.get("name") or "",
            attrs={},
        ))
    for t in row.get("techs") or []:
        if not t or t.get("id") is None:
            continue
        hits.append(_rel_hit(
            "USES_TECH",
            from_id=root_id, from_name=root_name,
            to_id=t["id"], to_name=t.get("name") or "",
            attrs={},
        ))
    return hits


def _run_product_seed_reverse(seed: Seed) -> list[GraphHit]:
    cypher = PATTERNS["pattern_product_seed_reverse"]
    with neo4j_driver.session() as s:
        row = s.run(cypher, key_value=seed["key_value"]).single()
    if not row:
        return []

    product_id = row["root_id"]
    product_name = row["root_name"]
    hits: list[GraphHit] = []
    for org in row.get("producers") or []:
        if not org or org.get("id") is None:
            continue
        hits.append(_rel_hit(
            "PRODUCES",
            from_id=org["id"], from_name=org.get("name") or "",
            to_id=product_id, to_name=product_name,
            attrs={},
        ))
    return hits


def _run_tech_seed_reverse(seed: Seed) -> list[GraphHit]:
    cypher = PATTERNS["pattern_tech_seed_reverse"]
    with neo4j_driver.session() as s:
        row = s.run(cypher, key_value=seed["key_value"]).single()
    if not row:
        return []

    tech_id = row["root_id"]
    tech_name = row["root_name"]
    hits: list[GraphHit] = []
    for org in row.get("users") or []:
        if not org or org.get("id") is None:
            continue
        hits.append(_rel_hit(
            "USES_TECH",
            from_id=org["id"], from_name=org.get("name") or "",
            to_id=tech_id, to_name=tech_name,
            attrs={},
        ))
    return hits


def _run_person_affiliations(seed: Seed) -> list[GraphHit]:
    cypher = PATTERNS["pattern_person_affiliations"]
    with neo4j_driver.session() as s:
        row = s.run(cypher, key_value=seed["key_value"]).single()
    if not row:
        return []

    person_id = row["root_id"]
    person_name = row["root_name"]
    hits: list[GraphHit] = []
    for a in row.get("affiliations") or []:
        if not a or a.get("id") is None:
            continue
        hits.append(_rel_hit(
            "EXECUTIVE_OF",
            from_id=person_id, from_name=person_name,
            to_id=a["id"], to_name=a.get("name") or "",
            attrs={"pos": a.get("pos")},
            source=a.get("source"),
        ))
    return hits


def _run_2hop_bridge(seed_a: Seed, seed_b: Seed) -> list[GraphHit]:
    cypher = PATTERNS["pattern_2hop_bridge"]
    with neo4j_driver.session() as s:
        rows = s.run(
            cypher,
            a_key_type=seed_a["key_type"], a_key_value=seed_a["key_value"],
            b_key_type=seed_b["key_type"], b_key_value=seed_b["key_value"],
        ).data()
    hits: list[GraphHit] = []
    for r in rows:
        nodes = r.get("nodes") or []
        rels = r.get("rels") or []
        # 경로를 1차로 relationship hit으로 한 줄 추가 (요약)
        if len(nodes) >= 2 and rels:
            from_name = nodes[0]
            to_name = nodes[-1]
            hits.append(_rel_hit(
                "BRIDGE",
                from_id=seed_a["id"], from_name=from_name or "",
                to_id=seed_b["id"], to_name=to_name or "",
                attrs={"path_nodes": nodes, "path_rels": rels},
                score=0.6,
            ))
    return hits


def _run_fallback(seed: Seed, use_apoc: bool = True) -> list[GraphHit]:
    """seed 주변 부분그래프. 정적 패턴 모두 hit 0일 때 호출."""
    name = "pattern_fallback_subgraph_apoc" if use_apoc else "pattern_fallback_subgraph_plain"
    cypher = PATTERNS[name]

    # seed 노드 internal id 조회
    with neo4j_driver.session() as s:
        # seed key 매칭으로 internal id 찾기
        match_label_clause = _seed_match_for_internal_id(seed)
        if match_label_clause is None:
            return []
        start_row = s.run(
            f"{match_label_clause} RETURN id(n) AS iid LIMIT 1",
            key_value=seed["key_value"],
        ).single()
        if not start_row:
            return []
        iid = start_row["iid"]

        try:
            rows = s.run(cypher, start_internal_id=iid).single()
        except Exception:
            # APOC 없는 환경 — plain으로 재시도
            if use_apoc:
                return _run_fallback(seed, use_apoc=False)
            return []

    if not rows:
        return []

    hits: list[GraphHit] = []
    nodes = rows.get("nodes") or rows.get("neighbors") or []
    rels = rows.get("relationships") or rows.get("rels") or []

    for n in nodes:
        if n is None:
            continue
        labels = list(n.labels) if hasattr(n, "labels") else []
        primary = next((_label_lower(l) for l in labels), "organization")
        nid = (n.get("corp_code") or n.get("person_id") or
               n.get("product_id") or n.get("tech_id") or
               (f"org:{n.get('er_name')}" if n.get("er_name") else None) or
               str(n.element_id))
        hits.append(_node_hit(primary, nid, n.get("name") or "", attrs={}, score=0.5))

    # rels는 list[list[rel]] 형태일 수 있음 (가변 hop path)
    flat_rels = _flatten_rels(rels)
    for r in flat_rels:
        if r is None:
            continue
        try:
            rtype = r.type
            start_node = r.start_node
            end_node = r.end_node
            from_id = (start_node.get("corp_code") or start_node.get("person_id") or
                       (f"org:{start_node.get('er_name')}" if start_node.get("er_name") else str(start_node.element_id)))
            to_id = (end_node.get("corp_code") or end_node.get("person_id") or
                     (f"org:{end_node.get('er_name')}" if end_node.get("er_name") else str(end_node.element_id)))
            hits.append(_rel_hit(
                rtype,
                from_id=from_id, from_name=start_node.get("name") or "",
                to_id=to_id, to_name=end_node.get("name") or "",
                attrs={},
                score=0.5,
            ))
        except Exception:
            continue

    return hits


def _flatten_rels(rels: Any) -> list:
    out: list = []
    for r in rels:
        if isinstance(r, list):
            out.extend(_flatten_rels(r))
        else:
            out.append(r)
    return out


def _label_lower(label: str) -> str:
    return {
        "Organization": "organization",
        "Person": "person",
        "Product": "product",
        "Technology": "technology",
        "FinMetric": "fin_metric",
    }.get(label, label.lower())


def _seed_match_for_internal_id(seed: Seed) -> str | None:
    kt = seed["key_type"]
    if kt == "corp_code":
        return "MATCH (n:Organization {corp_code: $key_value})"
    elif kt == "er_name":
        return "MATCH (n:Organization {er_name: $key_value})"
    elif kt == "person_id":
        return "MATCH (n:Person {person_id: $key_value})"
    elif kt == "product_id":
        return "MATCH (n:Product {product_id: $key_value})"
    elif kt == "tech_id":
        return "MATCH (n:Technology {tech_id: $key_value})"
    return None


# ─────────────────────────────────────────────────────────────
# 후처리 — 중복 제거(#3) + 엣지 cap(#4)
# ─────────────────────────────────────────────────────────────
# 금융기관(은행·펀드) 노이즈는 여기서 키워드로 거르지 않는다 — 그건 데이터/QC
# 레이어의 책임이다. SUPPLIES_TO 의 비회사 끝점(은행·국가 등)은 어드민 QC 가
# LLM 으로 판정해 qc_disabled_at 로 소프트삭제하고, traverse(cypher)의 SUPPLIES_TO
# 패턴이 그 플래그를 존중(qc_disabled_at IS NULL)한다. 쿼리타임 키워드 매칭 폐기.

# 라운드로빈 배분 시 타이브레이크용 관계 종류 우선순위(낮을수록 매 라운드 먼저).
# 단, 한 종류가 cap 을 독식하지 않게 라운드로빈으로 모든 종류에 슬롯을 고루 준다 —
# 예전엔 엄격 우선순위 정렬이라 IS_SUBSIDIARY_OF(해외 자회사 수십) 가 cap 을 꽉 채워
# 공급망 질문인데 SUPPLIES_TO 가 0개로 잘리는 역설이 있었다.
_REL_PRIORITY = {
    "IS_MAJOR_SHAREHOLDER_OF": 0,
    "EXECUTIVE_OF": 1,
    "SUPPLIES_TO": 2,
    "IS_SUBSIDIARY_OF": 3,
    "INVESTS_IN": 4,
    "PRODUCES": 5,
    "USES_TECH": 6,
    "RELATED_PARTY": 7,
    "INTERLOCKING_DIRECTORATE": 8,
    "BRIDGE": 9,
}


def _cap_relations(rel_hits: list[GraphHit], max_edges: int) -> list[GraphHit]:
    """관계 종류별 라운드로빈으로 max_edges 까지 추린다.

    종류별로 score 내림차순 정렬 후, 우선순위 순서로 한 바퀴씩 한 개꼴로 뽑아
    모든 관계 유형이 고르게 대표되게 한다. 적게 가진 종류는 일찍 소진되고 그 슬롯은
    많이 가진 종류(공급사·제품 등)로 흘러간다. 헤어볼은 막되 다양성은 보존.
    """
    if len(rel_hits) <= max_edges:
        return rel_hits
    by_type: dict[str, list[GraphHit]] = {}
    for h in rel_hits:
        rt = (h.get("attrs") or {}).get("rel_type") or ""
        by_type.setdefault(rt, []).append(h)
    for lst in by_type.values():
        lst.sort(key=lambda h: -float(h.get("score") or 0.0))
    types = sorted(by_type, key=lambda t: _REL_PRIORITY.get(t, 5))

    out: list[GraphHit] = []
    while len(out) < max_edges and any(by_type[t] for t in types):
        for t in types:
            bucket = by_type[t]
            if bucket:
                out.append(bucket.pop(0))
                if len(out) >= max_edges:
                    break
    return out


def _cap_relations_web(
    rel_hits: list[GraphHit],
    max_edges: int,
    seed_ids: set[str],
    seed_spoke_cap: int,
    cross_min_relevance: float = 0.0,
) -> list[GraphHit]:
    """별→웹 cap. 시드 직접 엣지(스포크)를 seed_spoke_cap 으로 제한하고 남는 예산을
    이웃끼리 관계(교차엣지=관계망)에 우선 배정한다. 각 파티션은 종류별 라운드로빈.

    시드 ego는 본질적으로 별이라(시드 차수 수백), 스포크를 그대로 두면 관계망(교차엣지)이
    cap 에 밀려 안 보인다. 스포크를 줄여 '속성 나열'을 '관계망'으로 전환.

    cross_min_relevance>0 이면 교차엣지(비시드↔비시드) 중 PPR 관련성(score)이 바닥값
    미만인 것을 버린다. cap 예산이 엣지 수보다 커서 cap 이 안 무는 경우에도, 시드가 아닌
    허브 하나를 공유한다는 이유만으로 끌려온 저관련 스포크를 떨군다. 시드 직결 엣지는
    직접 사실이라 floor 면제.
    """
    seed_inc: list[GraphHit] = []
    cross: list[GraphHit] = []
    for h in rel_hits:
        attrs = h.get("attrs") or {}
        if attrs.get("from_id") in seed_ids or attrs.get("to_id") in seed_ids:
            seed_inc.append(h)
        else:
            cross.append(h)
    if cross_min_relevance > 0.0:
        cross = [h for h in cross if float(h.get("score") or 0.0) >= cross_min_relevance]
    seed_kept = _cap_relations(seed_inc, min(seed_spoke_cap, max_edges))
    cross_kept = _cap_relations(cross, max_edges - len(seed_kept))
    return seed_kept + cross_kept


def _postprocess(hits: list[GraphHit]) -> list[GraphHit]:
    """#3 중복 제거 + #4 엣지 cap(라운드로빈, config.graphrag.MAX_EDGES).

    중복은 GraphHit.id(rel:type:from:to 또는 노드 id) 기준 1회만. 엣지가 너무 많으면
    관계 종류별 라운드로빈으로 잘라 노드 폭주(헤어볼)를 막되 다양성은 보존한다.
    """
    node_hits: list[GraphHit] = []
    rel_hits: list[GraphHit] = []
    seen: set[str] = set()

    for h in hits:
        hid = h.get("id")
        if not hid or hid in seen:
            continue
        seen.add(hid)
        (rel_hits if h.get("label") == "relationship" else node_hits).append(h)

    rel_hits = _cap_relations(rel_hits, MAX_EDGES)
    return node_hits + rel_hits


# ─────────────────────────────────────────────────────────────
# induced 엣지 — 별→망 (이웃끼리 잇는 내부 엣지)
# ─────────────────────────────────────────────────────────────

def _rel_key(rel_type: str, a: str, b: str) -> tuple[str, tuple[str, str]]:
    """방향 무관 dedup 키. 대칭 관계·역방향 induced 중복을 한 키로 묶는다."""
    return (rel_type, tuple(sorted((a, b))))  # type: ignore[return-value]


def _collect_node_ids(hits: list[GraphHit]) -> tuple[list[str], list[str]]:
    """assembled hit 에서 induced 멤버십용 id 집합 추출.

    Returns: (er_names, bare) — 'org:' 접두는 er_name 으로, 그 외 bare PK 후보로.
    """
    er_names: set[str] = set()
    bare: set[str] = set()

    def add(raw: str | None) -> None:
        if not raw or raw.startswith("rel:"):
            return
        if raw.startswith("org:"):
            er = raw[4:]
            if er:
                er_names.add(er)
        else:
            bare.add(raw)

    for h in hits:
        if h.get("label") == "relationship":
            attrs = h.get("attrs") or {}
            add(attrs.get("from_id"))
            add(attrs.get("to_id"))
        elif h.get("label") != "fin_metric":
            add(h.get("id"))

    er_list = list(er_names)[:INDUCED_MAX_NODES]
    bare_list = list(bare)[:INDUCED_MAX_NODES]
    return er_list, bare_list


def _induced_edges(assembled: list[GraphHit]) -> list[GraphHit]:
    """assembled 노드 집합 *내부*의 엣지를 조회해 induced rel hit 으로 반환.

    직접(seed) 엣지와 방향 무관 dedup 한다. 직접 엣지보다 후순위로 cap.
    """
    er_names, bare = _collect_node_ids(assembled)
    if not er_names and not bare:
        return []

    # 직접 엣지의 방향무관 키 — induced 가 같은 엣지를 재발견하면 버린다.
    seen: set[tuple[str, tuple[str, str]]] = set()
    for h in assembled:
        if h.get("label") != "relationship":
            continue
        attrs = h.get("attrs") or {}
        ft, to = attrs.get("from_id"), attrs.get("to_id")
        rt = attrs.get("rel_type")
        if ft and to and rt:
            seen.add(_rel_key(rt, ft, to))

    cap = MAX_INDUCED_EDGES * 4  # Cypher 단계 여유분; Python 에서 우선순위로 최종 cap
    with neo4j_driver.session() as s:
        rows = s.run(
            PATTERNS["pattern_induced_edges"],
            er_names=er_names, bare=bare, cap=cap,
        ).data()

    induced: list[GraphHit] = []
    for r in rows:
        rel_type = r.get("rel_type")
        a_id, b_id = r.get("a_id"), r.get("b_id")
        if not rel_type or not a_id or not b_id or a_id == b_id:
            continue
        if r.get("a_is_start"):
            from_id, from_name = a_id, r.get("a_name") or ""
            to_id, to_name = b_id, r.get("b_name") or ""
        else:
            from_id, from_name = b_id, r.get("b_name") or ""
            to_id, to_name = a_id, r.get("a_name") or ""

        key = _rel_key(rel_type, from_id, to_id)
        if key in seen:
            continue
        seen.add(key)

        attrs: dict[str, Any] = {}
        if r.get("qota_rt") is not None:
            attrs["qota_rt"] = r["qota_rt"]
        hit = _rel_hit(
            rel_type,
            from_id=from_id, from_name=from_name,
            to_id=to_id, to_name=to_name,
            attrs=attrs, source=r.get("source"), chunk=r.get("chunk"),
            score=0.55,
        )
        hit["seed_origin"] = "induced"
        induced.append(hit)

    return _cap_relations(induced, MAX_INDUCED_EDGES)


# ─────────────────────────────────────────────────────────────
# Public dispatch
# ─────────────────────────────────────────────────────────────

def expand(seeds: list[Seed]) -> tuple[list[GraphHit], list[str]]:
    """seeds에 적절한 정적 패턴을 모두 호출. fallback은 호출자 책임.

    Returns: (hits, patterns_run)
    """
    hits: list[GraphHit] = []
    patterns_run: list[str] = []

    org_seeds = [s for s in seeds if s.get("label") == "organization"]
    person_seeds = [s for s in seeds if s.get("label") == "person"]
    product_seeds = [s for s in seeds if s.get("label") == "product"]
    tech_seeds = [s for s in seeds if s.get("label") == "technology"]

    for s in org_seeds:
        hits.extend(_run_company_immediate(s))
        patterns_run.append(f"company_immediate({s['id']})")
        hits.extend(_run_subsidiary_tree(s))
        patterns_run.append(f"subsidiary_tree({s['id']})")
        hits.extend(_run_supply_chain(s))
        patterns_run.append(f"supply_chain({s['id']})")
        hits.extend(_run_product_links(s))
        patterns_run.append(f"product_links({s['id']})")

    for s in person_seeds:
        hits.extend(_run_person_affiliations(s))
        patterns_run.append(f"person_affiliations({s['id']})")

    for s in product_seeds:
        hits.extend(_run_product_seed_reverse(s))
        patterns_run.append(f"product_seed_reverse({s['id']})")

    for s in tech_seeds:
        hits.extend(_run_tech_seed_reverse(s))
        patterns_run.append(f"tech_seed_reverse({s['id']})")

    # 2-hop bridge: Org seed ≥2개
    if len(org_seeds) >= 2:
        hits.extend(_run_2hop_bridge(org_seeds[0], org_seeds[1]))
        patterns_run.append(f"2hop_bridge({org_seeds[0]['id']},{org_seeds[1]['id']})")

    assembled = _postprocess(hits)

    # induced 엣지(별→망): 모인 노드 집합 내부의 이웃끼리 엣지를 추가
    if INDUCED_EDGES:
        induced = _induced_edges(assembled)
        if induced:
            assembled = assembled + induced
            patterns_run.append(f"induced_edges(+{len(induced)})")

    return assembled, patterns_run


def _run_seed_financial(seed: Seed) -> list[GraphHit]:
    """시드 회사의 핵심 재무(HAS_METRIC) hit. PPR 경로는 관계는 PPR로 모으고 재무만 따로."""
    cypher = PATTERNS["pattern_seed_financial"].replace(
        "{{ORG_MATCH}}", _org_match("o", seed["key_type"])
    )
    with neo4j_driver.session() as s:
        row = s.run(
            cypher, key_value=seed["key_value"], fin_accounts=FIN_KEY_ACCOUNTS
        ).single()
    if not row:
        return []
    hits: list[GraphHit] = []
    for m in row.get("metrics") or []:
        if not m or m.get("metric_id") is None:
            continue
        hits.append(_node_hit(
            "fin_metric",
            m["metric_id"],
            m.get("account_id") or m["metric_id"],
            attrs={k: v for k, v in m.items() if v is not None},
            source=m.get("source"),
        ))
    return hits


def _run_shared_connections(seeds: list[Seed]) -> list[GraphHit]:
    """멀티 기업 숨은 연결 — 시드 2곳 이상에 동시 연결된 노드(공통 공급사·공통 주주 등).

    문서 하나로는 안 보이는 그래프 전용 인사이트. 엣지 cap 에 묻히지 않게 최우선 score 로
    태그(seed_origin='shared')해 호출자가 보존한다.
    """
    org_seeds = [s for s in seeds if s.get("label") == "organization"]
    if len(org_seeds) < 2:
        return []
    with neo4j_driver.session() as s:
        seed_eids = _ppr_seed_eids(s, org_seeds)
        if len(seed_eids) < 2:
            return []
        rows = s.run(
            PATTERNS["pattern_shared_connections"],
            seed_eids=seed_eids, rels=_PPR_DOMAIN_RELS,
        ).data()

    hits: list[GraphHit] = []
    for r in rows:
        shared_id = r.get("shared_id")
        shared_name = r.get("shared_name") or ""
        nseeds = r.get("nseeds") or 0
        if not shared_id:
            continue
        for link in r.get("links") or []:
            seed_id = link.get("seed_id")
            seed_name = link.get("seed_name") or ""
            rel = link.get("rel")
            if not seed_id or not rel:
                continue
            if link.get("seed_is_start"):
                from_id, from_name, to_id, to_name = seed_id, seed_name, shared_id, shared_name
            else:
                from_id, from_name, to_id, to_name = shared_id, shared_name, seed_id, seed_name
            attrs: dict[str, Any] = {"shared_with": nseeds}
            if link.get("qota_rt") is not None:
                attrs["qota_rt"] = link["qota_rt"]
            if rel == "SUPPLIES_TO":
                attrs["role"] = "supplier" if to_id == seed_id else "buyer"
            hit = _rel_hit(
                rel, from_id=from_id, from_name=from_name,
                to_id=to_id, to_name=to_name, attrs=attrs,
                source=link.get("source"), chunk=link.get("chunk"),
            )
            hit["seed_origin"] = "shared"
            hit["score"] = 0.97  # 공통연결은 최우선 — cap 에서 항상 생존
            hits.append(hit)
    return hits


def _ppr_typed_edges(eids: list[str], seed_our_ids: set[str]) -> list[GraphHit]:
    """PPR 선별 노드($eids) 내부의 타입 엣지(지분율·직책·출처). 2-hop 포함."""
    if not eids:
        return []
    with neo4j_driver.session() as s:
        rows = s.run(
            PATTERNS["pattern_typed_edges_among"],
            eids=eids, rels=_PPR_DOMAIN_RELS, cap=MAX_EDGES * 4,
        ).data()
    out: list[GraphHit] = []
    for r in rows:
        rel_type = r.get("rel_type")
        a_id, b_id = r.get("a_id"), r.get("b_id")
        if not rel_type or not a_id or not b_id or a_id == b_id:
            continue
        if r.get("a_is_start"):
            from_id, from_name = a_id, r.get("a_name") or ""
            to_id, to_name = b_id, r.get("b_name") or ""
        else:
            from_id, from_name = b_id, r.get("b_name") or ""
            to_id, to_name = a_id, r.get("a_name") or ""
        attrs: dict[str, Any] = {}
        if r.get("qota_rt") is not None:
            attrs["qota_rt"] = r["qota_rt"]
        if r.get("ofcps"):
            attrs["pos"] = r["ofcps"]
        if rel_type == "SUPPLIES_TO":
            if to_id in seed_our_ids:
                attrs["role"] = "supplier"
            elif from_id in seed_our_ids:
                attrs["role"] = "buyer"
        hit = _rel_hit(
            rel_type,
            from_id=from_id, from_name=from_name,
            to_id=to_id, to_name=to_name,
            attrs=attrs, source=r.get("source"), chunk=r.get("chunk"),
        )
        hit["seed_origin"] = "ppr"
        out.append(hit)
    return out


def expand_ppr(seeds: list[Seed]) -> tuple[list[GraphHit], list[str]]:
    """하이브리드 확장 — 패턴 확장(백본) + PPR 관련성 오버레이.

    백본(expand): 시드 1-hop 타입 엣지(임원·주주·공급역할·재무)·induced. 직접 사실 보장.
    PPR: 시드 관련성으로 노드 랭킹 + 멀티홉(2-hop) 관련 노드/엣지 보강. 헤어볼은 round-robin
    cap 으로 억제. 관련성 점수는 노드 hit 의 score·attrs.relevance 로 노출.
    """
    base_hits, patterns_run = expand(seeds)

    ranked = ppr_rank(seeds)
    if not ranked:
        return base_hits, patterns_run

    patterns_run = patterns_run + [f"ppr_rank(n={len(ranked)})"]
    relevance = {n["our_id"]: n["score"] for n in ranked}
    seed_our_ids = {s.get("id") for s in seeds}

    ppr_node_hits = [
        _node_hit(n["label"], n["our_id"], n["name"],
                  attrs={"relevance": n["score"]}, score=n["score"])
        for n in ranked
    ]
    ppr_edge_hits = _ppr_typed_edges([n["element_id"] for n in ranked], seed_our_ids)
    patterns_run.append(f"ppr_typed_edges(raw={len(ppr_edge_hits)})")

    # 병합 + dedup: 노드 id 1회(백본 우선, 관련성 주입), 엣지 round-robin cap
    node_hits: list[GraphHit] = []
    rel_hits: list[GraphHit] = []
    seen_nodes: set[str] = set()
    for h in base_hits + ppr_node_hits:
        if h.get("label") == "relationship":
            rel_hits.append(h)
            continue
        hid = h.get("id")
        if not hid or hid in seen_nodes:
            continue
        seen_nodes.add(hid)
        if "relevance" not in (h.get("attrs") or {}) and hid in relevance:
            h.setdefault("attrs", {})["relevance"] = relevance[hid]
        node_hits.append(h)

    def _edge_relevance(attrs: dict[str, Any]) -> float:
        """엣지 score = 양 끝(비-시드) 노드 PPR 관련성의 *최솟값*(병목/약한고리).
        엣지는 가장 먼(시드에서 약한) 끝만큼만 관련 있다 — 한쪽 끝이 허브(예: 비시드
        대기업)라 점수가 높아도, 반대쪽 끝이 시드와 무관하면 그 엣지는 무관하다.
        max 를 쓰면 허브에 닿은 모든 엣지가 허브 점수를 물려받아 허브 ego 가 답을
        점령한다(hub bias). min 으로 시드 직결 엣지와 허브 스포크를 변별한다.
        시드(1.0)는 제외해 스포크끼리도 상대 노드 관련성으로 변별."""
        a, b = attrs.get("from_id"), attrs.get("to_id")
        vals = [relevance.get(nid, 0.0) for nid in (a, b) if nid not in seed_our_ids]
        if vals:
            return min(vals)
        return max(relevance.get(a, 0.0), relevance.get(b, 0.0))

    # 멀티 기업 숨은 연결(공통 공급사·공통 주주 등) — cap 에 묻히지 않게 항상 보존
    shared_hits = _run_shared_connections(seeds)
    if shared_hits:
        patterns_run.append(f"shared_connections({len(shared_hits)})")

    seen_rel: set[str] = set()
    shared_kept: list[GraphHit] = []
    other_rel: list[GraphHit] = []
    for h in shared_hits + rel_hits + ppr_edge_hits:
        hid = h.get("id")
        if not hid or hid in seen_rel:
            continue
        seen_rel.add(hid)
        attrs = h.get("attrs") or {}
        # 시드 인접 공급엣지(induced 경유로 role 미설정분 포함)에 방향 라벨 보정
        if attrs.get("rel_type") == "SUPPLIES_TO" and not attrs.get("role"):
            if attrs.get("to_id") in seed_our_ids:
                attrs["role"] = "supplier"
            elif attrs.get("from_id") in seed_our_ids:
                attrs["role"] = "buyer"
        if h.get("seed_origin") == "shared":
            shared_kept.append(h)
        else:
            h["score"] = round(_edge_relevance(attrs), 4)
            other_rel.append(h)

    # 공통연결은 항상 보존, 나머지는 별→웹 cap(시드 스포크 제한 + 교차엣지 우선)
    budget = max(0, (MAX_EDGES + MAX_INDUCED_EDGES) - len(shared_kept))
    rel_capped = shared_kept + _cap_relations_web(
        other_rel, budget, seed_our_ids, SEED_SPOKE_CAP, CROSS_EDGE_MIN_RELEVANCE
    )

    return node_hits + rel_capped, patterns_run


def fallback_for(seed: Seed) -> list[GraphHit]:
    """정적 패턴이 비었을 때 호출자가 명시적으로 부르는 fallback."""
    return _postprocess(_run_fallback(seed, use_apoc=True))
