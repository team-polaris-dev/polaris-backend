"""의도분류 노드 — LLM 12의도 분류 + 라우팅(route) + 단계(stage).

04_graphrag.md §2 흐름의 [1] 의도분류. 업계표준: LLM 이 질문 → {intent, route,
stage, slots} JSON. graph 비대상(fin_value/fin_trend→rdb, disclosure→vec,
provenance)은 route 로 reject 신호를 준다.

LLM 호출은 apimaker(=Claude CLI 세션 영속 래퍼) 를 통해 나간다.
apimaker 서버 미가동 / 분류 실패 → 규칙 fallback 으로 degrade.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...apimaker_client import ApimakerClient, ApimakerError, NoApimakerAvailable
from ...query_shape import classify as classify_shape
from .text_to_cypher import _parse_json_block  # JSON 추출 재사용


# 12 의도 → (route, default_stage)
INTENT_ROUTE: dict[str, tuple[str, int]] = {
    "fin_value":     ("rdb", 1),
    "fin_trend":     ("rdb", 1),
    "ownership_in":  ("graph", 1),
    "ownership_out": ("graph", 1),
    "executives":    ("graph", 1),
    "subsidiaries":  ("graph", 1),
    "affiliates_fin": ("graph", 3),
    "supply_chain":  ("graph", 2),
    "products":      ("graph", 1),
    "related_party": ("graph", 2),
    "disclosure":    ("vec", 1),
    "provenance":    ("provenance", 1),
}

GRAPH_INTENTS = {k for k, (r, _) in INTENT_ROUTE.items() if r == "graph"}

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "intent_classify.md"
_prompt_cache: str | None = None


def _load_prompt() -> str:
    global _prompt_cache
    if _prompt_cache is None:
        _prompt_cache = _PROMPT_PATH.read_text(encoding="utf-8")
    return _prompt_cache


# ── 규칙 fallback (LLM 없을 때) ────────────────────────────────

_RULE_KEYWORDS: list[tuple[str, str]] = [
    # (정규식, intent) — 위에서부터 우선
    (r"간접.*지배|전이.*지배|겹치는|공통\s*임원|인적\s*연결", "related_party"),
    (r"자회사.*(매출|자산|재무|순이익)|종속회사.*(매출|재무)", "affiliates_fin"),
    (r"공급|매출처|고객사|납품", "supply_chain"),
    (r"최대주주|대주주|지분율|주주", "ownership_in"),
    (r"투자한|출자한|지분투자", "ownership_out"),
    (r"대표이사|임원|사내이사|사외이사|감사", "executives"),
    (r"자회사|종속회사|계열사", "subsidiaries"),
    (r"제품|생산|주력|기술|사용기술|활용기술", "products"),
    (r"특수관계|관계기업", "related_party"),
    (r"출처|근거\s*공시|어디서", "provenance"),
    (r"리스크|위험|사업\s*개요|사업\s*내용|전망", "disclosure"),
    (r"매출|자산|부채|자본|순이익|영업이익", "fin_value"),
]


def _rule_fallback(query: str) -> dict[str, Any]:
    intent = "disclosure"  # 기본
    for pat, name in _RULE_KEYWORDS:
        if re.search(pat, query):
            intent = name
            break
    # 시계열 키워드 있으면 fin_trend 로 승격
    if intent == "fin_value" and re.search(r"추이|연도별|변화|성장률|매년", query):
        intent = "fin_trend"
    route, stage = INTENT_ROUTE[intent]
    # 멀티홉 형태면 stage 상향
    if route == "graph" and classify_shape(query) == "multi_hop" and stage < 2:
        stage = 2
    return {
        "intent": intent,
        "route": route,
        "stage": stage,
        "slots": {},
        "relations": [],
        "rationale": "rule_fallback",
        "by": "rule",
    }


# ── LLM 분류기 (apimaker 경유) ─────────────────────────────────

class IntentClassifier:
    def __init__(self, model: str | None = None):
        self.model = model
        try:
            self.client: ApimakerClient | None = ApimakerClient(model=model)
        except NoApimakerAvailable:
            self.client = None

    def classify(self, query: str) -> dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return _rule_fallback("")
        if self.client is None:
            return _rule_fallback(query)

        try:
            text = self.client.chat(_load_prompt(), f"질문: {query}")
            obj = _parse_json_block(text)
        except (ApimakerError, Exception):
            return _rule_fallback(query)

        return _normalize(obj, query)


def _normalize(obj: dict[str, Any], query: str) -> dict[str, Any]:
    """LLM 출력 정합화 — 알 수 없는 intent/route 는 규칙으로 교정."""
    intent = obj.get("intent")
    if intent not in INTENT_ROUTE:
        return _rule_fallback(query)
    route, default_stage = INTENT_ROUTE[intent]
    # route 는 카탈로그가 SSOT (LLM 이 틀려도 교정)
    stage = obj.get("stage")
    if not isinstance(stage, int) or not (1 <= stage <= 5):
        stage = default_stage
    return {
        "intent": intent,
        "route": route,
        "stage": stage,
        "slots": obj.get("slots") or {},
        "relations": obj.get("relations") or [],
        "rationale": obj.get("rationale") or "",
        "by": "llm",
    }


def is_graph_target(result: dict[str, Any]) -> bool:
    return result.get("route") == "graph"
