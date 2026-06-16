// graphrag traversal 패턴 모음.
// traverse.py가 라벨로 split해서 읽음. 라벨 형식: -- @name <pattern_name>
// {{ORG_MATCH}}는 traverse.py가 corp_code/er_name 분기로 치환.

-- @name pattern_company_immediate
{{ORG_MATCH}}
CALL (o) {
  OPTIONAL MATCH (o)-[r:HAS_METRIC]->(m:FinMetric)
    WHERE m.reprt_code = '11011' AND m.fs_div = 'CFS' AND m.bsns_year >= 2024
  RETURN collect(DISTINCT m {.metric_id, .account_id, .value, .unit, .bsns_year, source: r.rcept_no})[..20] AS metrics
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
  RETURN collect(DISTINCT {
    id: coalesce(buyer.corp_code, 'org:' + coalesce(buyer.er_name, buyer.name)),
    name: buyer.name,
    tier: length(downpath),
    role: 'buyer'
  })[..100] AS buyers
}
CALL (o) {
  OPTIONAL MATCH uppath = (supplier:Organization)-[:SUPPLIES_TO*1..3]->(o)
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
