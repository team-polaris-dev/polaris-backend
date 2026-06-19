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

# 교차엣지(비시드↔비시드) PPR 관련성 바닥값. PPR score 는 시드=1.0 정규화이므로 depth-2 허브
# 스포크(예: 비시드 대기업 한 곳에만 붙은 공급사)는 매우 낮게 깔린다. 이 값 미만 교차엣지는
# 버려, 시드가 아닌 허브 하나를 공유한다는 이유만으로 그 허브의 ego 가 답을 점령하는 것을 막는다.
# 시드 직결 엣지(한쪽이 시드)는 직접 사실이라 floor 면제. 0 이면 비활성(기존 동작).
CROSS_EDGE_MIN_RELEVANCE = float(os.environ.get("GRAPHRAG_CROSS_EDGE_MIN_RELEVANCE", "0.12"))

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

# ── 패널 그래프 큐레이션 (core/serialize.py build_graph) ────────────
# 답이 언급한 회사들의 부분그래프만 남겨 헤어볼을 막는다. 엣지가 적으면(빈 패널 방지)
# 큐레이션을 건너뛴다.
PANEL_CURATION_MIN_EDGES = int(os.environ.get("GRAPHRAG_PANEL_MIN_EDGES", "6"))   # 이 수 초과면 큐레이션 시도
PANEL_CURATION_KEEP_MIN = int(os.environ.get("GRAPHRAG_PANEL_KEEP_MIN", "3"))     # 큐레이션 후 최소 엣지(미만이면 큐레이션 포기)
PANEL_MENTION_MIN_LEN = int(os.environ.get("GRAPHRAG_PANEL_MENTION_MIN_LEN", "2"))  # 답변 언급 매칭 최소 정규화 회사명 길이

# ── 구조화 GraphRAG 근거 게이트 (graphrag/structured_executor.py) ─────────
# 회계/투자/특수관계처럼 공시 주석 근거가 핵심인 관계는 높은 플로어를 적용하고,
# 공급 관계는 추출 텍스트가 거래 양쪽을 직접 언급하지 못하는 경우가 있어 중간 플로어를 둔다.
STRUCTURED_MIN_EVIDENCE = float(os.environ.get("GRAPHRAG_STRUCTURED_MIN_EVIDENCE", "0.8"))
STRUCTURED_MIN_EVIDENCE_OPERATING = float(os.environ.get("GRAPHRAG_STRUCTURED_MIN_EVIDENCE_OPERATING", "0.55"))
PANEL_MIN_EVIDENCE = float(os.environ.get("GRAPHRAG_PANEL_MIN_EVIDENCE", "0.55"))

# 매출(metric)로 순위를 매길 수 없는 '근거 확인된' 관계(해외·비상장 특수관계자 등)도
# 그래프에는 보여준다. branch별로 이 상한까지 confirmed 엣지를 렌더해 헤어볼은 막되
# 질문이 요구한 관계망 자체가 통째로 비는 것을 방지한다.
STRUCTURED_CONFIRMED_RENDER_CAP = int(os.environ.get("GRAPHRAG_STRUCTURED_CONFIRMED_RENDER_CAP", "8"))

# ── GraphRAG Global Search map-reduce (graphrag/global_search.py) ──────────
# MS GraphRAG global search 방식: 후보 군집마다 LLM이 질문 맞춤 부분답+점수를 만들고(map),
# gen 노드가 종합(reduce). 인덱스 시점 정적요약을 문자열매칭으로 고르던 기존 동작 대비
# 질문-맞춤 정확도가 오르지만 질의 시점 LLM 호출이 늘어난다(비용·지연).
# GLOBAL_MAP_REDUCE=0 이면 기존 정적-요약 읽기 경로로 폴백(견고성 스위치).
GLOBAL_MAP_REDUCE = os.environ.get("GRAPHRAG_GLOBAL_MAP_REDUCE", "1") not in ("0", "false", "False")
# select/map 대상 군집 상한 — map LLM 호출 수를 bound 한다(군집 size 상위 N).
GLOBAL_MAP_MAX_COMMUNITIES = int(os.environ.get("GRAPHRAG_GLOBAL_MAP_MAX_COMMUNITIES", "5"))
# 부분답 관련성 점수(0~100) 하한. 미만이면 폐기 — 좁은 질문에 무관 군집 노이즈 차단.
GLOBAL_MAP_MIN_SCORE = int(os.environ.get("GRAPHRAG_GLOBAL_MAP_MIN_SCORE", "1"))

# ── traverse hit 기본 점수 (graphrag/traverse.py) ──────────────────
REL_HIT_SCORE = float(os.environ.get("GRAPHRAG_REL_HIT_SCORE", "0.8"))   # 관계 hit 기본 score
NODE_HIT_SCORE = float(os.environ.get("GRAPHRAG_NODE_HIT_SCORE", "1.0"))  # 노드 hit 기본 score
