// graphrag traversal 패턴 모음.
// traverse.py가 라벨로 split해서 읽음. 라벨 형식: -- @name <pattern_name>
// {{ORG_MATCH}}는 traverse.py가 corp_code/er_name 분기로 치환.

-- @name pattern_company_immediate
{{ORG_MATCH}}
CALL (o) {
  OPTIONAL MATCH (o)-[r:HAS_METRIC]->(m:FinMetric)
    WHERE m.reprt_code = '11011' AND m.fs_div = 'CFS' AND m.bsns_year >= 2024
      AND m.account_id IN $fin_accounts
  RETURN collect(DISTINCT m {.metric_id, .account_id, .value, .unit, .bsns_year, source: r.rcept_no})[..40] AS metrics
}
CALL (o) {
  OPTIONAL MATCH (p:Person)-[r:EXECUTIVE_OF]->(o) WHERE r.valid_to IS NULL
  RETURN collect(DISTINCT {person_id: p.person_id, name: p.name, pos: r.ofcps, source: r.rcept_no})[..30] AS execs
}
CALL (o) {
  OPTIONAL MATCH (h)-[r:IS_MAJOR_SHAREHOLDER_OF]->(o)
    WHERE r.valid_to IS NULL
      AND NOT coalesce(h.name,'') IN ['계','소계','합계','-','']
  RETURN collect(DISTINCT {
    holder_id: coalesce(h.corp_code, h.person_id, 'org:' + coalesce(h.er_name, h.name)),
    name: h.name,
    qota_rt: r.qota_rt,
    source: r.rcept_no
  })[..30] AS holders
}
CALL (o) {
  OPTIONAL MATCH (o)-[r:INVESTS_IN]->(inv) WHERE r.valid_to IS NULL
  RETURN collect(DISTINCT {
    investee_id: coalesce(inv.corp_code, 'org:' + coalesce(inv.er_name, inv.name)),
    name: inv.name,
    qota_rt: r.qota_rt,
    source: r.rcept_no
  })[..50] AS invs
}
CALL (o) {
  OPTIONAL MATCH (o)-[r:RELATED_PARTY]-(rp)
  RETURN collect(DISTINCT {
    counterpart_id: coalesce(rp.corp_code, 'org:' + coalesce(rp.er_name, rp.name)),
    name: rp.name,
    source: r.rcept_no
  })[..30] AS related
}
CALL (o) {
  OPTIONAL MATCH (o)-[r:INTERLOCKING_DIRECTORATE]-(idn)
  RETURN collect(DISTINCT {
    counterpart_id: coalesce(idn.corp_code, 'org:' + coalesce(idn.er_name, idn.name)),
    name: idn.name
  })[..30] AS interlocking
}
RETURN
  coalesce(o.corp_code, 'org:' + coalesce(o.er_name, o.name)) AS root_id,
  o.name AS root_name,
  metrics, execs, holders, invs, related, interlocking;

-- @name pattern_subsidiary_tree
{{ORG_MATCH_root}}
OPTIONAL MATCH path = (sub:Organization)-[:IS_SUBSIDIARY_OF*1..5]->(root)
WITH root, sub, length(path) AS depth
RETURN
  coalesce(root.corp_code, 'org:' + coalesce(root.er_name, root.name)) AS root_id,
  root.name AS root_name,
  collect(DISTINCT {
    id: coalesce(sub.corp_code, 'org:' + coalesce(sub.er_name, sub.name)),
    name: sub.name,
    depth: depth
  })[..100] AS subs;

-- @name pattern_supply_chain
{{ORG_MATCH}}
CALL (o) {
  OPTIONAL MATCH downpath = (o)-[:SUPPLIES_TO*1..3]->(buyer:Organization)
  WHERE ALL(rr IN relationships(downpath) WHERE rr.qc_disabled_at IS NULL)
  RETURN collect(DISTINCT {
    id: coalesce(buyer.corp_code, 'org:' + coalesce(buyer.er_name, buyer.name)),
    name: buyer.name,
    tier: length(downpath),
    role: 'buyer'
  })[..100] AS buyers
}
CALL (o) {
  OPTIONAL MATCH uppath = (supplier:Organization)-[:SUPPLIES_TO*1..3]->(o)
  WHERE ALL(rr IN relationships(uppath) WHERE rr.qc_disabled_at IS NULL)
  RETURN collect(DISTINCT {
    id: coalesce(supplier.corp_code, 'org:' + coalesce(supplier.er_name, supplier.name)),
    name: supplier.name,
    tier: length(uppath),
    role: 'supplier'
  })[..100] AS suppliers
}
RETURN
  coalesce(o.corp_code, 'org:' + coalesce(o.er_name, o.name)) AS root_id,
  o.name AS root_name,
  buyers, suppliers;

-- @name pattern_product_links
{{ORG_MATCH}}
CALL (o) {
  OPTIONAL MATCH (o)-[:PRODUCES]->(p:Product)
  RETURN collect(DISTINCT {id: p.product_id, name: p.name})[..50] AS products
}
CALL (o) {
  OPTIONAL MATCH (o)-[:USES_TECH]->(t:Technology)
  RETURN collect(DISTINCT {id: t.tech_id, name: t.name})[..50] AS techs
}
RETURN
  coalesce(o.corp_code, 'org:' + coalesce(o.er_name, o.name)) AS root_id,
  o.name AS root_name,
  products, techs;

-- @name pattern_product_seed_reverse
MATCH (p:Product {product_id: $key_value})
OPTIONAL MATCH (org:Organization)-[:PRODUCES]->(p)
RETURN
  p.product_id AS root_id,
  p.name AS root_name,
  collect(DISTINCT {
    id: coalesce(org.corp_code, 'org:' + coalesce(org.er_name, org.name)),
    name: org.name
  })[..100] AS producers;

-- @name pattern_tech_seed_reverse
MATCH (t:Technology {tech_id: $key_value})
OPTIONAL MATCH (org:Organization)-[:USES_TECH]->(t)
RETURN
  t.tech_id AS root_id,
  t.name AS root_name,
  collect(DISTINCT {
    id: coalesce(org.corp_code, 'org:' + coalesce(org.er_name, org.name)),
    name: org.name
  })[..100] AS users;

-- @name pattern_person_affiliations
MATCH (p:Person {person_id: $key_value})-[r:EXECUTIVE_OF]->(o:Organization)
WHERE r.valid_to IS NULL
RETURN
  p.person_id AS root_id,
  p.name AS root_name,
  collect(DISTINCT {
    id: coalesce(o.corp_code, 'org:' + coalesce(o.er_name, o.name)),
    name: o.name,
    pos: r.ofcps,
    source: r.rcept_no
  })[..50] AS affiliations;

-- @name pattern_2hop_bridge
MATCH (a:Organization), (b:Organization)
WHERE
  (CASE $a_key_type WHEN 'corp_code' THEN a.corp_code = $a_key_value
                    ELSE a.er_name = $a_key_value END)
  AND
  (CASE $b_key_type WHEN 'corp_code' THEN b.corp_code = $b_key_value
                    ELSE b.er_name = $b_key_value END)
MATCH path = shortestPath((a)-[*..3]-(b))
RETURN
  [n IN nodes(path) | coalesce(n.name, n.corp_code, n.er_name)] AS nodes,
  [r IN relationships(path) | type(r)] AS rels
LIMIT 5;

-- @name pattern_typed_edges_among
// PPR 선별 노드 집합($eids = elementId 리스트) 내부의 타입 엣지 + 속성(지분율·직책·출처).
// 시점·qc 필터. 방향은 startNode 로 복원.
MATCH (a)-[r]-(b)
WHERE elementId(a) IN $eids AND elementId(b) IN $eids
  AND elementId(a) < elementId(b)
  AND type(r) IN $rels
  AND coalesce(r.valid_to,'') = '' AND r.qc_disabled_at IS NULL
WITH DISTINCT r, a, b
RETURN
  type(r) AS rel_type,
  startNode(r) = a AS a_is_start,
  CASE WHEN a:Organization THEN coalesce(a.corp_code, 'org:' + a.er_name)
       WHEN a:Person THEN a.person_id
       WHEN a:Product THEN a.product_id
       WHEN a:Technology THEN a.tech_id END AS a_id,
  a.name AS a_name,
  CASE WHEN b:Organization THEN coalesce(b.corp_code, 'org:' + b.er_name)
       WHEN b:Person THEN b.person_id
       WHEN b:Product THEN b.product_id
       WHEN b:Technology THEN b.tech_id END AS b_id,
  b.name AS b_name,
  r.qota_rt AS qota_rt, r.ofcps AS ofcps, r.rcept_no AS source
LIMIT $cap;

-- @name pattern_seed_financial
{{ORG_MATCH}}
OPTIONAL MATCH (o)-[r:HAS_METRIC]->(m:FinMetric)
  WHERE m.reprt_code = '11011' AND m.fs_div = 'CFS' AND m.bsns_year >= 2024
    AND m.account_id IN $fin_accounts
RETURN
  coalesce(o.corp_code, 'org:' + coalesce(o.er_name, o.name)) AS root_id,
  o.name AS root_name,
  collect(DISTINCT m {.metric_id, .account_id, .value, .unit, .bsns_year, source: r.rcept_no})[..40] AS metrics;

-- @name pattern_induced_edges
// 확장으로 모인 노드 집합 내부의 엣지(induced subgraph). 별→망 변환.
// 파라미터: er_names(list), bare(list) — bare 는 corp_code/person/product/tech PK 후보를
// 라벨 구분 없이 한 리스트로 던진다(PK 전역 유일 가정). FinMetric 은 집합에서 제외.
MATCH (a)
WHERE (a:Organization AND (a.corp_code IN $bare OR a.er_name IN $er_names))
   OR (a:Person AND a.person_id IN $bare)
   OR (a:Product AND a.product_id IN $bare)
   OR (a:Technology AND a.tech_id IN $bare)
WITH collect(DISTINCT a) AS ns
UNWIND ns AS a
MATCH (a)-[r]-(b)
WHERE b IN ns AND elementId(a) < elementId(b)
  AND coalesce(r.valid_to, '') = ''
  AND r.qc_disabled_at IS NULL
WITH DISTINCT r, a, b
RETURN
  type(r) AS rel_type,
  startNode(r) = a AS a_is_start,
  CASE WHEN a:Organization THEN coalesce(a.corp_code, 'org:' + a.er_name)
       WHEN a:Person THEN a.person_id
       WHEN a:Product THEN a.product_id
       WHEN a:Technology THEN a.tech_id END AS a_id,
  a.name AS a_name,
  CASE WHEN b:Organization THEN coalesce(b.corp_code, 'org:' + b.er_name)
       WHEN b:Person THEN b.person_id
       WHEN b:Product THEN b.product_id
       WHEN b:Technology THEN b.tech_id END AS b_id,
  b.name AS b_name,
  r.qota_rt AS qota_rt
LIMIT $cap;

-- @name pattern_fallback_subgraph_apoc
MATCH (start) WHERE id(start) = $start_internal_id
CALL apoc.path.subgraphAll(start, {maxLevel: 2, limit: 80, bfs: true})
YIELD nodes, relationships
RETURN nodes, relationships;

-- @name pattern_fallback_subgraph_plain
MATCH (start)-[r*1..2]-(neighbor)
WHERE id(start) = $start_internal_id
WITH start, collect(DISTINCT neighbor)[..80] AS neighbors, collect(DISTINCT r)[..80] AS rels
RETURN start, neighbors, rels;
