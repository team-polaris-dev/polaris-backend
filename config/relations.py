"""그래프 관계 타입 단일 진실원천(SSOT).

이전엔 관계 9종의 가중치·한글라벨·legacy매핑·망여부가 ppr.py/schema.py/serialize.py/
search.py/extract_helpers.py 6곳에 흩어져 부분집합·불일치·stale(INVESTS/ACQUIRES)이
생겼다. 여기서 한 번 정의하고 소비처는 파생 맵만 import 한다.

필드:
  type        Neo4j 관계 타입(엣지)
  weight      PPR 관련성 전파 가중치 (ppr.py)
  ko_label    패널/답변용 한글 라벨 (serialize.py)
  legacy_type adapt_to_legacy 의 UnifiedResult.type (schema.py)
  is_network  회사↔회사 사업관계 = 패널 망 엣지로 그림 (schema.py). 속성성 관계(임원·
              제품·기술)는 False → 망에서 제외, facts/텍스트에만.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Relation:
    type: str
    weight: float
    ko_label: str
    legacy_type: str
    is_network: bool


# 가중치 근거(ppr): 지배·지분(3.0) > 투자·공급(2.5) > 임원·제품·기술(1.5) > 특수관계(1.0)
# > 겸직(0.5). is_network: 지분·지배·공급·투자·특수관계·겸직 = 회사망. 임원·제품·기술 = 속성.
# 순서는 기존 ppr._REL_WEIGHT 와 동일하게 유지(DOMAIN_RELS 바이트 동일 보장).
RELATIONS: list[Relation] = [
    Relation("IS_MAJOR_SHAREHOLDER_OF", 3.0, "대주주",   "shareholder",              True),
    Relation("IS_SUBSIDIARY_OF",        3.0, "자회사",   "subsidiary",               True),
    Relation("INVESTS_IN",              2.5, "투자",     "investment",               True),
    Relation("SUPPLIES_TO",             2.5, "공급",     "supply",                   True),
    Relation("EXECUTIVE_OF",            1.5, "임원",     "executive",                False),
    Relation("PRODUCES",                1.5, "생산",     "produces",                 False),
    Relation("USES_TECH",               1.5, "기술",     "uses_tech",                False),
    Relation("RELATED_PARTY",           1.0, "특수관계", "related_party",            True),
    Relation("INTERLOCKING_DIRECTORATE",0.5, "겸직",     "interlocking_directorate", True),
]

# ── 파생 맵 (소비처는 이걸 import) ─────────────────────────────────
REL_WEIGHT: dict[str, float] = {r.type: r.weight for r in RELATIONS}
REL_LABELS: dict[str, str] = {r.type: r.ko_label for r in RELATIONS}
REL_TO_LEGACY_TYPE: dict[str, str] = {r.type: r.legacy_type for r in RELATIONS}
NETWORK_REL_TYPES: frozenset[str] = frozenset(r.type for r in RELATIONS if r.is_network)
DOMAIN_RELS: list[str] = [r.type for r in RELATIONS]  # PPR/관련성 전파 대상(전체 도메인 관계)

# 비정형 추출(extract_helpers)이 만드는 엣지 타입. 정형 관계(지분·지배·투자·임원·겸직)는
# 구조화 로더가 따로 적재하므로 추출 허용 목록에는 빠진다. chunk→object 출처 엣지 hasObject 포함.
# (= 기존 extract_helpers.EDGE_TYPES 와 동일)
INGEST_EDGE_TYPES: frozenset[str] = frozenset({
    "PRODUCES", "USES_TECH", "SUPPLIES_TO", "RELATED_PARTY", "hasObject",
})

# 랭킹 의도 키워드 SSOT. 예전엔 planner._RANK 와 llm_planner._RANK_TERMS 에 같은 8단어가
# 중복 정의돼 한쪽만 고치면 어긋났다. 여기서 한 번 정의하고 양쪽이 import 한다.
RANK_TERMS: tuple[str, ...] = ("가장", "최고", "1위", "상위", "제일", "많은", "높은", "잘나가")

# 질문 키워드 → 앞세울 관계 유형(검색 focus). search._relation_focus 가 사용한다.
# "주주" 질문이면 지분망만, "공급망"이면 공급망만 보여주도록 관계 hit 을 스코프한다.
FOCUS_KEYWORD_GROUPS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("주주", "지분", "대주주", "주식", "소유", "오너", "지배구조", "지배", "자회사",
      "계열", "모회사", "종속"),
     ("IS_MAJOR_SHAREHOLDER_OF", "IS_SUBSIDIARY_OF", "INVESTS_IN")),
    (("공급", "납품", "매입", "매출처", "고객", "공급망", "협력사", "벤더", "거래처",
      "수혜", "이득", "낙수", "납품처"),  # "수혜주" = 거래 상대(공급사)가 수혜
     ("SUPPLIES_TO",)),
    (("투자",), ("INVESTS_IN",)),
    (("특수관계", "계열거래"), ("RELATED_PARTY",)),
    (("제품", "생산", "만드", "품목"), ("PRODUCES",)),
    (("기술", "공정"), ("USES_TECH",)),
    (("임원", "대표", "경영진", "이사", "CEO"), ("EXECUTIVE_OF",)),
]
