// POLARIS — 2~4단계 실증 쿼리 (실측 검증 완료)
// Neo4j Browser에서 블록 단위로 복붙해서 실행하세요.
// 검증일: 2026-06-09 / 주요 corp_code: 삼성전자=00126380, SK(주)=00181712
//
// [진짜 멀티홉 판정 기준]
//   진짜 멀티홉 = 중간 노드를 반드시 경유해야만 목적지 도달 가능
//   아닌 것     = 동일한 결과를 단일 엣지/JOIN 하나로 얻을 수 있는 경우


// ════════════════════════════════════════════════════════
// [2단계-Q1] "삼성이오스(주)의 최대주주인 회사의 대표이사는?"
// ════════════════════════════════════════════════════════
// 경로: (삼성이오스) <-IS_MAJOR_SHAREHOLDER_OF- (삼성전자) <-EXECUTIVE_OF- (Person)
// 홉 구성: 엣지1=IS_MAJOR_SHAREHOLDER_OF, 엣지2=EXECUTIVE_OF, 노드3종
// 진짜 멀티홉 이유: '삼성전자'라는 중간 Organization 없이 Person에 도달 불가.
//                  삼성이오스→대표이사 직접엣지 없음.
// 실측: 삼성전자가 삼성이오스 43.06% 보유, 삼성전자 대표이사 다수 존재 → 결과 나옴.

MATCH (h:Organization)-[r:IS_MAJOR_SHAREHOLDER_OF]->(o:Organization {corp_code: '00877059'})
WITH h, r ORDER BY toFloat(r.qota_rt) DESC LIMIT 1
MATCH (p:Person)-[er:EXECUTIVE_OF]->(h)
WHERE er.ofcps CONTAINS '대표이사'
RETURN h.name AS 최대주주법인, r.qota_rt AS 지분율,
       p.name AS 대표이사, er.ofcps AS 직책
LIMIT 10;
// 예상 결과: 삼성전자(주) / 43.06 / 대표이사 이름들

// [비교용 1홉 — 같은 결과가 나오는지 확인. 안 나오면 진짜 멀티홉 증명]
MATCH (p:Person)-[er:EXECUTIVE_OF]->(o:Organization {corp_code: '00877059'})
WHERE er.ofcps CONTAINS '대표이사'
RETURN p.name AS 대표이사, er.ofcps AS 직책 LIMIT 5;
// 예상 결과: 0건 (삼성이오스 자체 대표이사 없거나 다른 인물)


// ════════════════════════════════════════════════════════
// [2단계-Q2] "삼성전자가 출자한 회사들의 임원 목록"
// ════════════════════════════════════════════════════════
// 경로: (삼성전자) -INVESTS_IN-> (출자회사X) <-EXECUTIVE_OF- (Person)
// 홉 구성: 엣지1=INVESTS_IN, 엣지2=EXECUTIVE_OF, 노드3종
// 진짜 멀티홉 이유: '출자회사X' 없이 임원에 도달 불가.
//                  삼성전자→임원 직접엣지 없음.
// 실측: 15건 이상 반환 확인.

MATCH (o:Organization {corp_code: '00126380'})-[:INVESTS_IN]->(t:Organization)
MATCH (p:Person)-[er:EXECUTIVE_OF]->(t)
RETURN t.name AS 출자회사, p.name AS 임원명, er.ofcps AS 직책
ORDER BY t.name
LIMIT 30;


// ════════════════════════════════════════════════════════
// [3단계-Q1] "삼성전자 지분/출자 관계사들의 매출 (최신연도)"
// ════════════════════════════════════════════════════════
// 경로: (삼성전자) -[IS_MAJOR_SHAREHOLDER_OF|INVESTS_IN]*1..2-> (관계사) -HAS_METRIC-> (FinMetric)
// 홉 구성: 엣지1=지분관계(1~2홉), 엣지2=HAS_METRIC, 노드3종 + 경로변수
// 진짜 멀티홉 이유: ① '관계사'라는 중간 노드 없이 FinMetric 도달 불가
//                  ② RDB 단독으로는 '삼성전자와 지분관계인 회사'를 필터링 불가
//                  → Graph(관계망) + FinMetric(재무) 이종 DB 교차가 핵심
// 실측: 2025년 기준 14사 반환, hops 분포 {1:14} (2홉은 순환경로 제거 후)
// 주의: bsns_year는 정수(2023 아님 — 2024/2025 데이터 적재됨)

MATCH (root:Organization {corp_code: '00126380'})
MATCH path = (root)-[:IS_MAJOR_SHAREHOLDER_OF|INVESTS_IN*1..2]->(target:Organization)
WHERE root.corp_code <> target.corp_code
MATCH (target)-[:HAS_METRIC]->(m:FinMetric)
WHERE m.account_id = 'ifrs-full_Revenue'
  AND m.reprt_code = '11011' AND m.fs_div = 'CFS'
WITH target, m, length(path) AS hops
ORDER BY m.bsns_year DESC
WITH target, collect(m)[0] AS latest_m, min(hops) AS hops
RETURN target.name AS 관계사, latest_m.bsns_year AS 연도,
       latest_m.value AS 매출, hops AS 홉수
ORDER BY hops, 매출 DESC
LIMIT 20;
// 예상: 삼성에스디에스·삼성SDI·삼성물산 등 14사, 모두 hops=1

// [비교용 — RDB만으로는 불가능한 이유 확인]
// Q: "삼성전자와 지분관계인 회사" 목록을 SQL로 얻으려면?
// A: fin_metric 테이블에는 corp_code만 있음. '지분관계'라는 개념이 없어서 불가.
//    그래프 없이는 이 필터를 걸 수 없다 → 진짜 3단계 교차 가치.


// ════════════════════════════════════════════════════════
// [3단계-Q2] "SK(주) 관계사들의 자산총계 비교 (최신연도)"
// ════════════════════════════════════════════════════════

MATCH (root:Organization {corp_code: '00181712'})
MATCH path = (root)-[:IS_MAJOR_SHAREHOLDER_OF|INVESTS_IN*1..2]->(target:Organization)
WHERE root.corp_code <> target.corp_code
MATCH (target)-[:HAS_METRIC]->(m:FinMetric)
WHERE m.account_id = 'ifrs-full_Assets'
  AND m.reprt_code = '11011' AND m.fs_div = 'CFS'
WITH target, m, length(path) AS hops
ORDER BY m.bsns_year DESC
WITH target, collect(m)[0] AS latest_m, min(hops) AS hops
RETURN target.name AS 관계사, latest_m.bsns_year AS 연도,
       latest_m.value AS 자산총계, hops AS 홉수
ORDER BY 자산총계 DESC
LIMIT 10;


// ════════════════════════════════════════════════════════
// [4단계-Q1] "삼성전자와 임원이 겹치는 회사" (파생 온톨로지)
// ════════════════════════════════════════════════════════
// 경로: (삼성전자) -INTERLOCKING_DIRECTORATE- (연결회사)
// 파생 근거: Person.name 동일 + 두 회사 각각 EXECUTIVE_OF
//            → 배치에서 materialize (build_derived_edges.py --rebuild)
// 온톨로지 추론 이유: "임원 겹침"이라는 개념 자체를 규칙으로 정의해 엣지화
//                  런타임 1홉처럼 보이지만 배치에서 2홉 추론을 사전 체결한 것
// 실측: 삼성전자 기준 1건, SK(주) 기준 21개사

MATCH (o:Organization {corp_code: '00126380'})-[d:INTERLOCKING_DIRECTORATE]-(c:Organization)
RETURN c.name AS 연결회사, d.via AS 공통임원, d.confidence AS 신뢰도
ORDER BY c.name;

// 파생 근거 검증: 해당 임원이 실제로 두 회사에 EXECUTIVE_OF 있는지
MATCH (p:Person)-[:EXECUTIVE_OF]->(o:Organization)
WHERE p.name = '이재석'  // 위 결과의 via_person 값으로 교체
RETURN o.name AS 소속회사, p.name AS 임원;

// SK(주) 기준 (더 풍부)
MATCH (o:Organization {corp_code: '00181712'})-[d:INTERLOCKING_DIRECTORATE]-(c:Organization)
RETURN c.name AS 연결회사, d.via AS 공통임원, d.confidence AS 신뢰도
ORDER BY c.name
LIMIT 25;


// ════════════════════════════════════════════════════════
// [4단계-Q2] "SK(주) 지분 보유 회사 회계분류"
// ════════════════════════════════════════════════════════
// 온톨로지 즉석 추론: 지분율 50%/20% 임계값 → 회계기준 분류
// 규칙 기반 추론 이유: IFRS 기준(IAS28/IFRS10)을 CASE 문으로 직접 구현
//                     LLM 없이 DB에서 결정론적으로 분류
// 실측: SK트레이딩(51% 지배), SKC(40.64% 관계기업), 이노베이션(32.14% 관계기업)

MATCH (o:Organization {corp_code: '00181712'})-[r:IS_MAJOR_SHAREHOLDER_OF]->(t:Organization)
WHERE r.qota_rt IS NOT NULL
RETURN t.name AS 투자대상, r.qota_rt AS 지분율,
       CASE WHEN toFloat(r.qota_rt) >= 50 THEN '지배(종속회사)'
            WHEN toFloat(r.qota_rt) >= 20 THEN '유의적영향(관계기업)'
            ELSE '단순투자' END AS 회계분류
ORDER BY toFloat(r.qota_rt) DESC
LIMIT 10;
