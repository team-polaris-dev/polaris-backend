"""Neo4j → data/vocab.json 덤프.

대상: Organization / Person / Product / Technology (주어 노드만)
정렬: degree 내림차순 (Lost-in-the-Middle 완화)
ID 정책: Organization은 corp_code 있으면 그것, 없으면 "org:" + er_name.
        Person/Product/Technology는 각자의 PK.

실행: python -m pipeline_scripts.graph.dump_vocab [--out data/vocab.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tool.graph_client import neo4j_driver


# ─────────────────────────────────────────────────────────────
# 정적 단어집 — DB에서 추출 불가, 사람이 관리. vocab.json에 함께 출력.
# (재무 계정 한글명·관계 술어·코드값은 DB에 한글이 없어 여기서 정의)
# ─────────────────────────────────────────────────────────────
SCHEMA_SUMMARY = """\
[Neo4j 노드]
- Organization(corp_code, name, er_name): 회사
- Person(person_id, name): 인물(임원·주주)
- Product(product_id, name): 제품
- Technology(tech_id, name): 기술
- FinMetric(metric_id, account_id, value, bsns_year, reprt_code, fs_div): 재무지표

[관계]
- (Person)-[EXECUTIVE_OF]->(Organization): 임원직
- (Org)-[IS_MAJOR_SHAREHOLDER_OF]->(Org): 주요주주 (지분율 qota_rt)
- (Org)-[IS_SUBSIDIARY_OF]->(Org): 종속회사
- (Org)-[INVESTS_IN]->(Org): 출자 (지분율 qota_rt)
- (Org)-[HAS_METRIC]->(FinMetric): 재무 (연간확정치는 reprt_code='11011', fs_div='CFS')
- (Org)-[SUPPLIES_TO]->(Org): 공급 (방향 있음)
- (Org)-[PRODUCES]->(Product): 제조
- (Org)-[USES_TECH]->(Technology): 기술 사용
- (Org)-[RELATED_PARTY]-(Org): 특수관계자
- (Org)-[INTERLOCKING_DIRECTORATE]-(Org): 임원 겸직 (파생)
"""

# 재무 계정 (B): 한글 구어 → FinMetric.account_id. 영업이익은 dart_ 접두사 주의.
FIN_ACCOUNTS = {
    "ifrs-full_Revenue": ["매출액", "매출", "수익", "영업수익", "매출고"],
    "ifrs-full_CostOfSales": ["매출원가"],
    "ifrs-full_GrossProfit": ["매출총이익", "매출총손익"],
    "dart_OperatingIncomeLoss": ["영업이익", "영업손익", "영업이익률"],
    "ifrs-full_ProfitLossBeforeTax": ["법인세차감전순이익", "세전이익", "법인세비용차감전순이익"],
    "ifrs-full_IncomeTaxExpenseContinuingOperations": ["법인세비용", "법인세"],
    "ifrs-full_ProfitLoss": ["당기순이익", "순이익", "당기순손익", "순손익"],
    "ifrs-full_ComprehensiveIncome": ["총포괄이익", "포괄손익"],
    "ifrs-full_OtherComprehensiveIncome": ["기타포괄손익"],
    "ifrs-full_FinanceIncome": ["금융수익"],
    "ifrs-full_FinanceCosts": ["금융원가", "금융비용", "이자비용"],
    "ifrs-full_Assets": ["자산총계", "총자산", "자산"],
    "ifrs-full_CurrentAssets": ["유동자산"],
    "ifrs-full_NoncurrentAssets": ["비유동자산"],
    "ifrs-full_CashAndCashEquivalents": ["현금및현금성자산", "현금"],
    "ifrs-full_Inventories": ["재고자산", "재고"],
    "ifrs-full_PropertyPlantAndEquipment": ["유형자산"],
    "ifrs-full_OtherCurrentAssets": ["기타유동자산"],
    "ifrs-full_Liabilities": ["부채총계", "총부채", "부채"],
    "ifrs-full_CurrentLiabilities": ["유동부채"],
    "ifrs-full_NoncurrentLiabilities": ["비유동부채"],
    "ifrs-full_OtherCurrentLiabilities": ["기타유동부채"],
    "ifrs-full_Equity": ["자본총계", "자기자본", "자본"],
    "ifrs-full_EquityAndLiabilities": ["부채와자본총계", "부채및자본"],
    "ifrs-full_IssuedCapital": ["자본금"],
    "ifrs-full_RetainedEarnings": ["이익잉여금"],
    "ifrs-full_CashFlowsFromUsedInOperatingActivities": ["영업활동현금흐름", "영업활동으로인한현금흐름"],
    "ifrs-full_CashFlowsFromUsedInInvestingActivities": ["투자활동현금흐름", "투자활동으로인한현금흐름"],
    "ifrs-full_CashFlowsFromUsedInFinancingActivities": ["재무활동현금흐름", "재무활동으로인한현금흐름"],
    "ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities": ["유형자산취득", "capex", "설비투자"],
}

# 관계 술어 (C): 한글 구어 → 관계 타입
# 주의: 그룹 범위어(계열사·관계사 등 config.relations.GROUP_SCOPE_TERMS)는 여기 넣지 않는다.
# 특정 관계(IS_SUBSIDIARY_OF/RELATED_PARTY)로 치환되면 재구성 LLM 이 "삼성 계열사"를
# "삼성 종속회사"로 좁혀 그룹 군집 랭킹(community_member_rank)을 무너뜨린다.
RELATION_PREDICATES = {
    "EXECUTIVE_OF": ["임원", "등기임원", "대표이사", "사내이사", "사외이사", "경영진", "ceo", "대표"],
    "IS_MAJOR_SHAREHOLDER_OF": ["주주", "주요주주", "대주주", "최대주주", "지분보유", "주식보유", "지분"],
    "IS_SUBSIDIARY_OF": ["자회사", "종속회사", "종속기업", "모회사", "지배회사"],
    "INVESTS_IN": ["출자", "투자", "지분투자", "관계기업", "피투자"],
    "HAS_METRIC": ["재무", "재무제표", "실적", "재무지표"],
    "SUPPLIES_TO": ["공급", "납품", "벤더", "협력사", "공급망", "거래처", "매입처", "매출처"],
    "PRODUCES": ["제조", "생산", "제품", "생산제품"],
    "USES_TECH": ["기술사용", "보유기술", "핵심기술", "기술"],
    "RELATED_PARTY": ["특수관계자", "특수관계"],
    "INTERLOCKING_DIRECTORATE": ["임원겸직", "겸직", "겸임"],
}

# 보고서 코드 (D): 한글 구어 → reprt_code
REPRT_CODES = {
    "11011": ["사업보고서", "연간", "연간보고서", "연간실적", "연결연간", "결산"],
    "11012": ["반기보고서", "반기", "반기실적", "상반기"],
    "11013": ["1분기보고서", "1분기", "일분기", "q1", "1q"],
    "11014": ["3분기보고서", "3분기", "삼분기", "q3", "3q"],
}

# 재무제표 구분 (E): 한글 구어 → fs_div
FS_DIV = {
    "CFS": ["연결", "연결재무제표", "연결기준", "연결실적"],
    "OFS": ["별도", "별도재무제표", "별도기준", "개별", "개별재무제표"],
}


ORG_CYPHER = """
MATCH (o:Organization)
WHERE o.name IS NOT NULL AND NOT trim(o.name) IN $noise
OPTIONAL MATCH (o)-[r]-()
WITH o, count(r) AS deg
RETURN
  CASE WHEN o.corp_code IS NOT NULL THEN o.corp_code
       ELSE 'org:' + coalesce(o.er_name, o.name) END AS id,
  o.name AS name,
  coalesce(o.er_name, o.name) AS canonical
ORDER BY deg DESC, o.name
"""

_NOISE_NAMES = ['-', '–', '—', '계', '소계', '합계', 'N/A', '해당없음', '']

PERSON_CYPHER = """
MATCH (p:Person)
WHERE p.name IS NOT NULL AND NOT trim(p.name) IN $noise
OPTIONAL MATCH (p)-[r]-()
WITH p, count(r) AS deg
RETURN p.person_id AS id, p.name AS name
ORDER BY deg DESC, p.name
"""

PRODUCT_CYPHER = """
MATCH (pr:Product)
WHERE pr.name IS NOT NULL AND NOT trim(pr.name) IN $noise
OPTIONAL MATCH (pr)<-[r]-()
WITH pr, count(r) AS deg
RETURN pr.product_id AS id, pr.name AS name
ORDER BY deg DESC, pr.name
"""

TECH_CYPHER = """
MATCH (t:Technology)
WHERE t.name IS NOT NULL AND NOT trim(t.name) IN $noise
OPTIONAL MATCH (t)<-[r]-()
WITH t, count(r) AS deg
RETURN t.tech_id AS id, t.name AS name
ORDER BY deg DESC, t.name
"""


def _records(s, cypher: str, **params) -> list[dict]:
    return [dict(r) for r in s.run(cypher, **params)]


def dump() -> dict:
    with neo4j_driver.session() as s:
        organization = _records(s, ORG_CYPHER, noise=_NOISE_NAMES)
        person = _records(s, PERSON_CYPHER, noise=_NOISE_NAMES)
        product = _records(s, PRODUCT_CYPHER, noise=_NOISE_NAMES)
        technology = _records(s, TECH_CYPHER, noise=_NOISE_NAMES)

    return {
        "version": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stats": {
            "organization": len(organization),
            "person": len(person),
            "product": len(product),
            "technology": len(technology),
            "fin_accounts": len(FIN_ACCOUNTS),
            "relation_predicates": len(RELATION_PREDICATES),
        },
        # 정적 단어집 (사람 관리)
        "schema_summary": SCHEMA_SUMMARY,
        "fin_accounts": FIN_ACCOUNTS,
        "relation_predicates": RELATION_PREDICATES,
        "reprt_codes": REPRT_CODES,
        "fs_div": FS_DIV,
        # 동적 단어집 (DB 추출)
        "organization": organization,
        "person": person,
        "product": product,
        "technology": technology,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/vocab.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parents[2] / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vocab = dump()
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote {out_path}")
    print(f"      stats={vocab['stats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
