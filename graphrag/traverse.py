"""Cypher 패턴 실행 + GraphHit 평탄화.

cypher/traverse.cypher의 패턴을 읽어 라벨로 인덱싱.
seed.label/key_type에 따라 적정 패턴 호출, 결과를 GraphHit 리스트로 변환.
정적 hit 0개인 경우 호출자(search.py)가 fallback 패턴 선택.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tool.graph_client import neo4j_driver
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
             source: str | None = None, score: float = 0.8) -> GraphHit:
    full_attrs: dict[str, Any] = {
        "rel_type": rel_type,
        "from_id": from_id,
        "from_name": from_name,
        "to_id": to_id,
        "to_name": to_name,
        **attrs,
    }
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
              source: str | None = None, score: float = 1.0) -> GraphHit:
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
        row = s.run(cypher, key_value=seed["key_value"]).single()
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
    hits: list[GraphHit] = []

    for b in row.get("buyers") or []:
        if not b or b.get("id") is None:
            continue
        hits.append(_rel_hit(
            "SUPPLIES_TO",
            from_id=root_id, from_name=root_name,
            to_id=b["id"], to_name=b.get("name") or "",
            attrs={"role": "buyer", "tier": b.get("tier")},
        ))
    for sp in row.get("suppliers") or []:
        if not sp or sp.get("id") is None:
            continue
        hits.append(_rel_hit(
            "SUPPLIES_TO",
            from_id=sp["id"], from_name=sp.get("name") or "",
            to_id=root_id, to_name=root_name,
            attrs={"role": "supplier", "tier": sp.get("tier")},
        ))
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

    return hits, patterns_run


def fallback_for(seed: Seed) -> list[GraphHit]:
    """정적 패턴이 비었을 때 호출자가 명시적으로 부르는 fallback."""
    return _run_fallback(seed, use_apoc=True)
