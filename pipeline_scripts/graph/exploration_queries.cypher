// POLARIS — 실측 탐색 쿼리 모음
// 2026-06-09 대화 세션에서 실행한 Neo4j 검증 쿼리.
// 실행: Neo4j Browser (bolt://localhost:7687) 또는 uv run python -c "..."

// ── 기본 통계 ────────────────────────────────────────────────────

// 전체 노드 수
MATCH (o:Organization) RETURN count(o) AS organizations;
// 결과: 3539

MATCH (m:FinMetric) RETURN count(m) AS fin_metrics;
// 결과: 37,679

// 재무 보유 회사 수
MATCH (o:Organization)-[:HAS_METRIC]->(:FinMetric)
RETURN count(DISTINCT o.corp_code) AS orgs_with_fin;
// 결과: 34

// 관계 수
MATCH ()-[r:IS_SUBSIDIARY_OF]->() RETURN count(r) AS subsidiary;
// 결과: 2142

MATCH ()-[r:IS_MAJOR_SHAREHOLDER_OF]->() RETURN count(r) AS shareholder;
// 결과: 337

MATCH ()-[r:INVESTS_IN]->() RETURN count(r) AS invests_in;
// 결과: 1222

// ── 지분율 50%↑ 분포 (전이지배 가능성 탐색) ─────────────────────

MATCH ()-[r:IS_MAJOR_SHAREHOLDER_OF]->()
WHERE r.qota_rt IS NOT NULL AND toFloat(r.qota_rt) >= 50
RETURN count(r) AS qota_50_plus;
// 결과: 5 → 전이지배 데이터 없음

MATCH path = (a:Organization)-[:IS_MAJOR_SHAREHOLDER_OF*2..4]->(c:Organization)
WHERE all(r IN relationships(path)
          WHERE r.qota_rt IS NOT NULL AND toFloat(r.qota_rt) >= 50)
  AND a.corp_code <> c.corp_code
RETURN count(DISTINCT [a.corp_code, c.corp_code]) AS transitive_control_candidates;
// 결과: 0 → CONTROLS_INDIRECTLY 비활성 근거

// ── 2단계 — 최대주주의 대표이사 (shareholder_ceo) ─────────────────

// 삼성전자 기준 테스트
MATCH (h:Organization)-[r:IS_MAJOR_SHAREHOLDER_OF]->(o:Organization {corp_code: '00126380'})
WITH h, r ORDER BY toFloat(r.qota_rt) DESC LIMIT 1
MATCH (p:Person)-[er:EXECUTIVE_OF]->(h)
WHERE er.ofcps CONTAINS '대표이사'
RETURN h.name AS holder, p.name AS ceo, er.ofcps AS pos;
// 결과: 3사 커버 (삼성물산 계열 정도)

// ── 3단계 — 자회사경로 재무 교차 (affiliates_fin_subsidiary) ──────

// 삼성전자 자회사 재무 (매출)
MATCH (sub:Organization)-[:IS_SUBSIDIARY_OF]->(o:Organization {corp_code: '00126380'})
MATCH (sub)-[:HAS_METRIC]->(m:FinMetric)
WHERE m.account_id = 'ifrs-full_Revenue'
  AND m.reprt_code = '11011' AND m.fs_div = 'CFS'
RETURN sub.name AS sub_name, m.bsns_year AS year, m.value AS value
ORDER BY year DESC, value DESC
LIMIT 10;
// 결과: 0건 — 자회사 310개이나 재무 미적재

// ── 3단계 — 지분경로 재무 교차 (affiliates_fin_ownership) ─────────

// 삼성전자 지분/출자 관계사 매출 (GraphRAG 핵심, 실측 작동)
MATCH (root:Organization {corp_code: '00126380'})
MATCH path = (root)-[:IS_MAJOR_SHAREHOLDER_OF|INVESTS_IN*1..2]->(target:Organization)
MATCH (target)-[:HAS_METRIC]->(m:FinMetric)
WHERE m.account_id = 'ifrs-full_Revenue'
  AND m.reprt_code = '11011' AND m.fs_div = 'CFS'
RETURN DISTINCT target.corp_code AS code, target.name AS org,
       m.bsns_year AS year, m.value AS value
ORDER BY year DESC, value DESC
LIMIT 100;
// 결과: 31행 DISTINCT (16사) — GraphRAG 핵심 가치 경로

// 전체 지분경로 커버리지 (몇 개사가 이 경로로 재무 응답 가능한가)
MATCH (root:Organization)
MATCH path = (root)-[:IS_MAJOR_SHAREHOLDER_OF|INVESTS_IN*1..2]->(target:Organization)
MATCH (target)-[:HAS_METRIC]->(:FinMetric)
RETURN count(DISTINCT root.corp_code) AS root_coverage;
// 결과: ~33사

// ── 4단계 — 공통임원 (interlocking_directorate) ───────────────────

// 공통임원 후보 수 (build_derived_edges.py --dry-run 동일)
MATCH (p1:Person)-[:EXECUTIVE_OF]->(a:Organization),
      (p2:Person)-[:EXECUTIVE_OF]->(b:Organization)
WHERE p1.name = p2.name
  AND a.corp_code IS NOT NULL AND b.corp_code IS NOT NULL
  AND a.corp_code <> b.corp_code
  AND elementId(a) < elementId(b)
  AND size(p1.name) >= 2
RETURN count(*) AS common_director_candidates;
// 결과: 66

// INTERLOCKING_DIRECTORATE 생성 (--rebuild, 멱등)
// → build_derived_edges.py --rebuild 으로 실행

// 생성 후 확인
MATCH ()-[d:INTERLOCKING_DIRECTORATE]->() RETURN count(d) AS edges;
// 결과: 66

// SK(주) 공통임원 연결 회사 수
MATCH (o:Organization {corp_code: '00126380'})-[d:INTERLOCKING_DIRECTORATE]-(c)
RETURN count(DISTINCT c.corp_code) AS linked_orgs;
// 주: SK(주) corp_code 는 실제 값으로 교체 필요. 실측 21개사 연결.

// ── 4단계 — 지분 회계분류 (ownership_class) ──────────────────────

MATCH (o:Organization {corp_code: '00126380'})-[r:IS_MAJOR_SHAREHOLDER_OF]->(t:Organization)
WHERE r.qota_rt IS NOT NULL
RETURN t.name AS target, r.qota_rt AS qota_rt,
       CASE WHEN toFloat(r.qota_rt) >= 50 THEN '지배(종속회사)'
            WHEN toFloat(r.qota_rt) >= 20 THEN '유의적영향(관계기업)'
            ELSE '단순투자' END AS accounting_class
ORDER BY toFloat(r.qota_rt) DESC
LIMIT 20;
