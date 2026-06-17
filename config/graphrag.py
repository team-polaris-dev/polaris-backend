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

# induced 엣지: seed 중심 확장은 "별 모양"(이웃이 root에만 연결)이라 관계망이 안 보인다.
# 확장으로 모인 노드 집합 *내부*의 엣지를 한 번 더 조회해 이웃끼리도 이어 "망"으로 만든다.
# 직접(seed) 엣지보다 후순위로 cap 한다 — 헤어볼 재발 방지.
INDUCED_EDGES = os.environ.get("GRAPHRAG_INDUCED_EDGES", "1") not in ("0", "false", "False")
MAX_INDUCED_EDGES = int(os.environ.get("GRAPHRAG_MAX_INDUCED_EDGES", "60"))
INDUCED_MAX_NODES = int(os.environ.get("GRAPHRAG_INDUCED_MAX_NODES", "120"))

# ── Personalized PageRank (ppr.py) ──────────────────────────────
# 하이퍼 연결 그래프(허브 차수 수천)에서 시드 관련성 기반 멀티홉 추출. 시드에서 PPR 을
# 돌려 거리·허브통과로 점수가 감쇠하게 해, 시드에 *진짜* 가까운 서브그래프만 남기고
# 허브 우회 노이즈를 억제한다(HippoRAG 방식). 순수 파이썬 power iteration — GDS 불필요.
PPR_ENABLED = os.environ.get("GRAPHRAG_PPR", "1") not in ("0", "false", "False")
PPR_ALPHA = float(os.environ.get("GRAPHRAG_PPR_ALPHA", "0.85"))           # 댐핑(restart=1-alpha)
PPR_ITERS = int(os.environ.get("GRAPHRAG_PPR_ITERS", "30"))              # power iteration 횟수
PPR_NEIGHBORHOOD_LIMIT = int(os.environ.get("GRAPHRAG_PPR_NBR_LIMIT", "1500"))  # depth-2 이웃 상한
PPR_TOP_NODES = int(os.environ.get("GRAPHRAG_PPR_TOP_NODES", "50"))      # PPR 상위 N 노드 선별

# 시드 ego는 본질적으로 별(시드 차수만 수백). 시드에 직접 붙은 스포크를 이 수로 제한하고
# 남는 예산을 이웃끼리 관계(교차엣지=관계망)에 우선 배정해 '속성 나열' → '관계망'으로 바꾼다.
SEED_SPOKE_CAP = int(os.environ.get("GRAPHRAG_SEED_SPOKE_CAP", "35"))

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
