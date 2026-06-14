"""의도×fact_type 가중치 — 그래프 노드 내부 점수 통합 관리.

v2 (2026-06-11): 임의 단일값(FACT_BASE_SCORE 직감 테이블)을 4축 분해로 재구성.
Bloomberg KG 의 source-hierarchy 방식 흉내 — 점수의 "왜"를 축으로 설명 가능하게:

  score = source_authority × extraction_method   (fact_type 별 정적 base)
        × confidence × time_decay × consistency  (fact 별 동적 보정)
        × intent_boost                           (의도 정합 가산)

축 정의:
  source_authority — 출처 권위. 정형공시(DART 사실)=1.0 / 비정형(본문 LLM 추출·룰 파생·
    LLM 생성쿼리)=0.85. 같은 트리플이라도 출처가 낮으면 사실보다 아래.
  extraction_method — 산출 방식 오차. 결정론 적재=0.95 / 멀티홉 결합=0.90 /
    LLM 추출=0.90 / 룰 배치=0.85 / LLM Cypher=0.82.
  confidence — fact 에 confidence 필드가 있으면(claude 추출 엣지) 그대로 곱함 [0.5, 1.0].
  time_decay — fact 에 year 필드가 있으면 exp(-Δ년/16), 하한 0.85. 연차 패널티가
    아니라 동률 후보의 최신순 정렬용(완만). 슬롯이 특정 연도를 요구하면 호출측에서
    해당 연도만 통과시키므로 여기선 무시해도 안전.
  consistency — fact 에 conflict=True 가 있으면 ×0.5 (detect_conflicts.py 검토 큐와
    연동 — 양방향/원장충돌 엣지). 기본 1.0.

다른 출처(dense/sparse)와의 RRF 가중치는 Synthesizer 책임 → 이 모듈은
그래프 노드 자체 ranking 의 일관성만 보장.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any


# ── 정적 base: fact_type → (source_authority, extraction_method) ──
_AUTH_FILING = 1.00     # DART 정형 공시 (extracted_by IS NULL)
_AUTH_SOFT = 0.85       # 비정형: LLM 본문추출 / 룰 파생 / LLM 생성쿼리

_METHOD_DETERMINISTIC = 0.95  # 결정론 적재·미러 조회
_METHOD_MULTIHOP = 0.90       # 멀티홉 결합 (경로 결합 리스크)
_METHOD_LLM_EXTRACT = 0.90    # LLM 본문 추출 (anchor 검증 통과분)
_METHOD_RULE = 0.85           # 룰 배치 파생 (추론)
_METHOD_LLM_CYPHER = 0.82     # LLM 생성 Cypher (가드 통과분)

_AXES: dict[str, tuple[float, float]] = {
    # 정형 (DART 사실) — 권위 동일, 방식 동일. 구분은 intent_boost 가 담당.
    "fin_metric":  (_AUTH_FILING, _METHOD_DETERMINISTIC),
    "subsidiary":  (_AUTH_FILING, _METHOD_DETERMINISTIC),
    "shareholder": (_AUTH_FILING, _METHOD_DETERMINISTIC),
    "executive":   (_AUTH_FILING, _METHOD_DETERMINISTIC),
    "investment":  (_AUTH_FILING, _METHOD_DETERMINISTIC),
    "agg_count":   (_AUTH_FILING, _METHOD_DETERMINISTIC),
    "multihop":    (_AUTH_FILING, _METHOD_MULTIHOP),
    # 비정형 (본문 추출)
    "supplies_to": (_AUTH_SOFT, _METHOD_LLM_EXTRACT),
    "produces":    (_AUTH_SOFT, _METHOD_LLM_EXTRACT),
    "technology":  (_AUTH_SOFT, _METHOD_LLM_EXTRACT),
    "related_party": (_AUTH_SOFT, _METHOD_LLM_EXTRACT),
    # 파생 (룰 배치) / LLM 생성쿼리
    "derived":        (_AUTH_SOFT, _METHOD_RULE),
    "text_to_cypher": (_AUTH_SOFT, _METHOD_LLM_CYPHER),
}

# 호환용 — base = authority × method (v1 의 임의값을 축 곱으로 대체)
FACT_BASE_SCORE: dict[str, float] = {
    k: round(a * m, 4) for k, (a, m) in _AXES.items()
}

# intent = 04_graphrag.md §4 의 12의도. 해당 fact_type 에 가산.
INTENT_BOOST: dict[str, dict[str, float]] = {
    "ownership_in":  {"shareholder": 1.10, "multihop": 1.10},
    "ownership_out": {"investment": 1.10, "multihop": 1.10},
    "executives":    {"executive": 1.10, "multihop": 1.05},
    "subsidiaries":  {"subsidiary": 1.10},
    "affiliates_fin": {"fin_metric": 1.15, "multihop": 1.05},
    "supply_chain":  {"supplies_to": 1.10, "multihop": 1.05},
    "products":      {"produces": 1.10, "technology": 1.10},
    "related_party": {"derived": 1.15, "supplies_to": 1.05},
}

DEFAULT_SCORE = 0.70

_DECAY_FLOOR = 0.85
_DECAY_TAU_YEARS = 16.0


def _time_decay(year: Any) -> float:
    try:
        dy = max(0, date.today().year - int(year))
    except (TypeError, ValueError):
        return 1.0
    return max(_DECAY_FLOOR, math.exp(-dy / _DECAY_TAU_YEARS))


def _per_fact_factor(f: dict[str, Any]) -> float:
    """fact 별 동적 보정: confidence × time_decay × consistency."""
    factor = 1.0
    conf = f.get("confidence")
    if conf is not None:
        try:
            factor *= min(1.0, max(0.5, float(conf)))
        except (TypeError, ValueError):
            pass
    if f.get("year") is not None:
        factor *= _time_decay(f["year"])
    if f.get("conflict"):
        factor *= 0.5
    return factor


def apply_score_weights(
    facts: list[dict[str, Any]],
    intent: str | None,
) -> list[dict[str, Any]]:
    """fact dict 리스트에 score 부여 + 내림차순 정렬.

    각 fact 는 최소 'type' 키를 가져야 한다. score 키는 in-place 갱신.
    """
    boost = INTENT_BOOST.get(intent or "factoid", {})
    for f in facts:
        ftype = f.get("type", "")
        base = FACT_BASE_SCORE.get(ftype, DEFAULT_SCORE)
        mult = boost.get(ftype, 1.0)
        f["score"] = round(base * mult * _per_fact_factor(f), 4)
    return sorted(facts, key=lambda x: -x.get("score", 0.0))
