"""랜딩 페이지 그래프 — Neo4j 실데이터를 프론트(GraphExplorer) 형태로 내려준다.

프론트 GraphExplorer 의 목업(buildMockUniverse)을 그대로 대체할 수 있도록
{ nodes:[{id,name,category,val}], links:[{source,target,kind}] } 형태로 반환한다.

카테고리 5종: 기업(company)·인물(person)·제품(product)·기술(technology)·재무(finance).

선별 기준(의미 있고 카테고리가 고른 그래프):
  - 허브 랭킹: 총차수가 아니라 회사↔회사 관계의 REL_WEIGHT 가중합(지배·지분 우선) — 지주사·
    대주주가 앞에 온다. 제품 수에 부풀려지던 총차수 편향 제거.
  - 균등 분배: 카테고리마다 node_limit/5 쿼터를 둬, 데이터가 회사관계 위주(기업 ~3600개 중
    위성 보유 기업은 ~95개뿐)여도 기업이 화면을 독식하지 않게 한다. 위성이 적은
    카테고리(인물·기술·재무)는 데이터 한계까지만 차고 총 노드가 node_limit 에 못 미칠 수 있다.
  - 엣지 유효성: ppr/structured_executor 와 동일하게 valid_to(만료)·qc_disabled_at(QC 오답) 제외.
  - 특수관계(RELATED_PARTY)는 거대 노이즈 허브라 망에서 제외.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from tool.graph_client import neo4j_driver
from config.relations import NETWORK_REL_TYPES, REL_LABELS, REL_WEIGHT

# Neo4j 라벨 → 프론트 카테고리.
_CATEGORY: dict[str, str] = {
    "Organization": "company",
    "Person": "person",
    "Product": "product",
    "Technology": "technology",
}
# 노드 크기(val) — 시각 비중.
_VAL: dict[str, int] = {
    "company": 16, "person": 5, "product": 7, "technology": 8, "finance": 6,
}

# 기업의 속성 위성(임원·제품·기술) 관계.
_ATTR_RELS = ["EXECUTIVE_OF", "PRODUCES", "USES_TECH"]

# 재무(FinMetric) 위성으로 보여줄 핵심 지표. IFRS account_id → 한글 라벨.
# FinMetric 은 account_id 가 영문 IFRS 태그(수백 종)라, 일반 사용자가 알아볼 핵심만 선별한다.
# 같은 지표가 연도별로 여러 노드라, 쿼리에서 기업·지표별 최신 연도 1개만 고른다.
_FIN_METRICS: dict[str, str] = {
    "ifrs-full_Assets": "자산총계",
    "ifrs-full_Liabilities": "부채총계",
    "ifrs-full_Equity": "자본총계",
    "ifrs-full_ProfitLoss": "당기순이익",
    "ifrs-full_ProfitLossBeforeTax": "세전이익",
    "ifrs-full_CashAndCashEquivalents": "현금성자산",
}

# 랜딩 망에서 쓸 회사↔회사 관계. 특수관계(RELATED_PARTY)는 한 기업에 수백 개가 달리는
# 거대 노이즈 허브라(entities.py 경고 참고) 지배구조 그림을 흐려서 제외한다.
_NET_RELS = [t for t in NETWORK_REL_TYPES if t != "RELATED_PARTY"]

# 엣지 유효성 필터 — 기존 검색(ppr.py/structured_executor.py)과 동일 규칙.
_VALID_EDGE = "coalesce(r.valid_to, '') = '' AND r.qc_disabled_at IS NULL"

# 기업당 위성 상한(카테고리별) — 소수 대기업이 한 카테고리를 독식하지 않게 고루 퍼뜨린다.
_PER_ORG_CAP: dict[str, int] = {"person": 4, "product": 3, "technology": 3, "finance": 4}


def build_landing_graph(node_limit: int = 1000) -> dict[str, Any]:
    """카테고리별 쿼터(node_limit/5)로 기업·인물·제품·기술·재무를 고르게 모은다."""
    # 카테고리별 상한 — 5종을 균등하게. 기업도 이 쿼터를 넘지 않아 독식이 막힌다.
    quota = max(20, node_limit // 5)
    # 위성을 붙일 허브 후보 수(쿼리 IN-리스트 크기 제한). 위성 보유 기업은 상위 랭킹에
    # 몰려 있어 넉넉히 600개면 전부 덮는다.
    hub_limit = max(200, node_limit // 2)

    nodes: dict[str, dict[str, Any]] = {}
    links: list[dict[str, str]] = []
    seen_links: set[tuple[str, str]] = set()
    cat_count: defaultdict[str, int] = defaultdict(int)

    def add_node(nid: str, name: str, category: str) -> bool:
        """추가 성공 시 True. 이미 있으면 True. 카테고리 쿼터/총량 초과면 False."""
        if nid in nodes:
            return True
        if cat_count[category] >= quota or len(nodes) >= node_limit:
            return False
        nodes[nid] = {"id": nid, "name": name or "?", "category": category, "val": _VAL[category]}
        cat_count[category] += 1
        return True

    def add_link(src: str, dst: str, kind: str) -> None:
        key = (src, dst) if src < dst else (dst, src)
        if key in seen_links:
            return
        seen_links.add(key)
        links.append({"source": src, "target": dst, "kind": kind})

    with neo4j_driver.session() as session:
        # 1) 후보 기업을 "가중 네트워크 차수"(REL_WEIGHT 가중합) 내림차순으로 받는다.
        #    지주사·대주주처럼 구조적으로 중요한 허브가 앞에 온다. 동점은 elementId 안정 정렬.
        org_rows = session.run(
            f"""
            MATCH (o:Organization)-[r]-(b:Organization)
            WHERE type(r) IN $net AND {_VALID_EDGE}
            WITH o, sum($weights[type(r)]) AS wdeg
            WHERE wdeg > 0
            ORDER BY wdeg DESC, elementId(o)
            LIMIT $cand_limit
            RETURN elementId(o) AS id,
                   coalesce(o.name, o.corp_code, o.er_name, '?') AS name
            """,
            net=_NET_RELS,
            weights=REL_WEIGHT,
            cand_limit=node_limit,
        )
        companies_ordered = [(row["id"], row["name"]) for row in org_rows]
        hub_ids = [cid for cid, _ in companies_ordered[:hub_limit]]

        # 2) 허브 기업의 위성을 모아 기업별로 버킷에 담는다(임원·제품·기술 + 재무).
        #    엣지엔 유효성 필터 적용(QC 오답/만료 제외).
        sat_by_org: defaultdict[str, list[tuple[str, str, str, str]]] = defaultdict(list)

        attr_rows = session.run(
            f"""
            MATCH (o:Organization)-[r]-(m)
            WHERE elementId(o) IN $ids
              AND (m:Person OR m:Product OR m:Technology) AND type(r) IN $attr_rels
              AND {_VALID_EDGE}
            RETURN elementId(o) AS o_id, elementId(m) AS m_id,
                   coalesce(m.name, '?') AS m_name, head(labels(m)) AS m_label, type(r) AS rel
            """,
            ids=hub_ids,
            attr_rels=_ATTR_RELS,
        )
        for row in attr_rows:
            cat = _CATEGORY.get(row["m_label"])
            if cat:
                sat_by_org[row["o_id"]].append(
                    (row["m_id"], row["m_name"], cat, REL_LABELS.get(row["rel"], row["rel"]))
                )

        # 재무: 기업·지표별 최신 연도 1개만(연도별 중복 방지). 핵심 지표만(_FIN_METRICS).
        fin_rows = session.run(
            """
            MATCH (o:Organization)-[:HAS_METRIC]->(m:FinMetric)
            WHERE elementId(o) IN $ids AND m.account_id IN $accounts
            WITH o, m.account_id AS acc, m ORDER BY m.bsns_year DESC
            WITH o, acc, head(collect(m)) AS m
            RETURN elementId(o) AS o_id, elementId(m) AS m_id, $labels[acc] AS m_name
            """,
            ids=hub_ids,
            accounts=list(_FIN_METRICS.keys()),
            labels=_FIN_METRICS,
        )
        for row in fin_rows:
            sat_by_org[row["o_id"]].append((row["m_id"], row["m_name"], "finance", "재무"))

        # 3) 랭킹 순으로 위성 보유 기업부터 적재 — 기업 + 그 위성(기업당·카테고리당 상한).
        #    카테고리 쿼터(add_node)가 균등 분배를 보장한다.
        for cid, cname in companies_ordered:
            if cid not in sat_by_org:
                continue
            if not add_node(cid, cname, "company"):
                continue  # 기업 쿼터가 찼으면 그 위성도 생략
            percat: defaultdict[str, int] = defaultdict(int)
            for m_id, m_name, cat, kind in sat_by_org[cid]:
                if percat[cat] >= _PER_ORG_CAP.get(cat, 3):
                    continue
                if add_node(m_id, m_name, cat):
                    percat[cat] += 1
                    add_link(cid, m_id, kind)

        # 4) 기업 쿼터가 남으면 다음 순위 기업으로 채운다(지배구조 망을 두텁게).
        for cid, cname in companies_ordered:
            if cat_count["company"] >= quota:
                break
            add_node(cid, cname, "company")

        # 5) 채택된 기업들 사이의 지분·계열·공급 등 망 엣지.
        #    유효 망 엣지 전체(~1.3만 행)를 받아 양끝이 채택 기업집합에 드는 것만 파이썬에서
        #    거른다(큰 IN-리스트 회피 → 노드 수와 무관하게 빠름).
        org_id_set = {nid for nid, n in nodes.items() if n["category"] == "company"}
        if org_id_set:
            net_rows = session.run(
                f"""
                MATCH (a:Organization)-[r]-(b:Organization)
                WHERE elementId(a) < elementId(b) AND type(r) IN $net AND {_VALID_EDGE}
                RETURN elementId(a) AS a_id, elementId(b) AS b_id, type(r) AS rel
                """,
                net=_NET_RELS,
            )
            for row in net_rows:
                if row["a_id"] in org_id_set and row["b_id"] in org_id_set:
                    add_link(row["a_id"], row["b_id"], REL_LABELS.get(row["rel"], row["rel"]))

    return {"nodes": list(nodes.values()), "links": links}
