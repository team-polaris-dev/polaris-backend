"""GraphRAG 검색 튜닝 파라미터.

하드코딩 상수를 한곳으로 모은다. 전부 환경변수로 override 가능(.env).
값의 의미·기본값 근거는 주석 참고. 코드(matcher/traverse)는 여기서만 읽는다.
"""
from __future__ import annotations

import os

# ── 시드 매칭 (matcher.py) ──────────────────────────────────────
# FULLTEXT cjk 는 "에스케이하이닉스"를 bigram 으로 쪼개 SK 계열 수십 곳에 매칭한다.
# 상위 N개를 그대로 시드로 쓰면 ego 그래프가 계열 합집합(헤어볼)이 되므로,
# (1) 정규화 정확매칭 우선 (2) top score 대비 밴드컷 (3) 시드 상한 으로 좁힌다.
SEED_SCORE_BAND = float(os.environ.get("GRAPHRAG_SEED_SCORE_BAND", "0.8"))  # top score 의 이 비율 이상만 유지
MAX_SEEDS = int(os.environ.get("GRAPHRAG_MAX_SEEDS", "6"))                  # 최종 시드 상한
FULLTEXT_THRESHOLD = float(os.environ.get("GRAPHRAG_FULLTEXT_THRESHOLD", "0.5"))  # FULLTEXT 절대 score 하한
FULLTEXT_POOL = int(os.environ.get("GRAPHRAG_FULLTEXT_POOL", "25"))        # FULLTEXT 후보 풀 크기

# ── 확장 (traverse.py) ──────────────────────────────────────────
# 그래프 노드 폭주(헤어볼) 방지용 엣지 상한. 정보가치 높은 관계부터 남긴다.
MAX_EDGES = int(os.environ.get("GRAPHRAG_MAX_EDGES", "90"))

# 회사 재무(HAS_METRIC)를 가져올 때 핵심 계정만 선별한다. 예전엔 collect[..20]이
# 아무 계정 20개나 집어 매출·영업이익이 빠지는 일이 있었다(IFRS 표준 계정 코드).
# 손익(매출~순이익) + 재무상태(자산·부채·자본·현금) 핵심만.
FIN_KEY_ACCOUNTS = [
    "ifrs-full_Revenue",                 # 매출액
    "ifrs-full_CostOfSales",             # 매출원가
    "ifrs-full_GrossProfit",             # 매출총이익
    "dart_OperatingIncomeLoss",          # 영업이익
    "ifrs-full_ProfitLossBeforeTax",     # 법인세차감전순이익
    "ifrs-full_ProfitLoss",              # 당기순이익
    "ifrs-full_Assets",                  # 자산총계
    "ifrs-full_CurrentAssets",           # 유동자산
    "ifrs-full_Liabilities",             # 부채총계
    "ifrs-full_CurrentLiabilities",      # 유동부채
    "ifrs-full_Equity",                  # 자본총계
    "ifrs-full_CashAndCashEquivalents",  # 현금및현금성자산
]
