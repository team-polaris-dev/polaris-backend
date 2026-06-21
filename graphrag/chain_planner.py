"""다중홉 랭킹 체인(cutline) 플래너 — 공식 text2cypher 위에 얹는 value-add 튜닝 레이어.

'A가 오르면 수혜 볼 기업 N개, 거기서 또 수혜 볼 기업 N개'처럼 영향이 단계적으로 전파되는
질문을, 각 홉에서 관계 후보를 노드 지표(매출 등)로 줄세워 상위 top_n 을 다음 홉 앵커로
넘기는 재귀 랭킹으로 만든다. 라이브러리(Text2CypherRetriever)는 단일 read-only Cypher 한 개만
생성하므로 이 재귀 줄세우기는 엔진이 아니라 오케스트레이션 결정 → 도메인 레이어로 남긴다.

LLM 은 제한된 hops 스키마(관계·방향·지표·top_n)만 채운다. 절대 Cypher/SQL/자유텍스트를
만들지 않으며 무효 출력은 폐기(None)돼 호출자가 폴백한다. 실행은 structured_executor.execute
(plan.kind=='multi_hop_chain' → _execute_multi_hop_chain)가 맡는다.
"""
from __future__ import annotations

import json
import os
from typing import Any

from graphrag.plan_schema import (
    HopStep,
    MetricId,
    MetricRankStep,
    RelationStep,
    StructuredPlan,
)

_FALSE = {"0", "false", "False", "no", "NO", "off", "OFF"}
_ENABLED = os.environ.get("GRAPHRAG_LLM_PLANNER", "1") not in _FALSE

_RELATION_TYPES = {
    "SUPPLIES_TO",
    "RELATED_PARTY",
    "IS_MAJOR_SHAREHOLDER_OF",
    "IS_SUBSIDIARY_OF",
    "INVESTS_IN",
}
_DIRECTIONS = {"incoming", "outgoing", "undirected", "auto"}
_METRICS = {
    "ifrs-full_Revenue",
    "dart_OperatingIncomeLoss",
    "ifrs-full_ProfitLoss",
    "ifrs-full_Assets",
}
_FIRST_POLICIES = {"default", "operating_counterparty"}

_RELATION_DEFAULTS: dict[tuple[str, str], tuple[str, str]] = {
    ("SUPPLIES_TO", "incoming"): ("suppliers", "supplier"),
    ("SUPPLIES_TO", "outgoing"): ("buyers", "buyer"),
    ("SUPPLIES_TO", "auto"): ("supply_counterparties", "supply_counterparty"),
    ("RELATED_PARTY", "undirected"): ("related_parties", "related_party"),
    ("IS_MAJOR_SHAREHOLDER_OF", "incoming"): ("shareholders", "shareholder"),
    ("IS_MAJOR_SHAREHOLDER_OF", "outgoing"): ("shareholdings", "shareholding"),
    ("IS_SUBSIDIARY_OF", "incoming"): ("subsidiaries", "subsidiary"),
    ("IS_SUBSIDIARY_OF", "outgoing"): ("parents", "parent"),
    ("INVESTS_IN", "outgoing"): ("investees", "investee"),
    ("INVESTS_IN", "incoming"): ("investors", "investor"),
    ("INVESTS_IN", "undirected"): ("investment_parties", "investment"),
}

# 다중홉 체인(cutline) cue. "수혜의 수혜"처럼 영향이 단계적으로 퍼지는 질문 신호.
# 두 축을 모두 요구한다(_looks_chain): 영향이 *전파*되는 어휘(수혜/낙수…) + 그 전파가
# *반복*된다는 표지(또/거기서/그다음…). 전파어만 있으면 단일 홉 '수혜주'질문이므로 체인 아님.
_CHAIN_PROPAGATION_TERMS = (
    "수혜", "낙수", "파급", "연쇄", "타고", "한 다리", "한다리", "건너",
)
_CHAIN_RECURSION_TERMS = (
    "또", "거기서", "거기에", "그다음", "그 다음", "이어서", "다시", "연달아",
    "단계적", "재차", "그곳",
)


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(t in text for t in terms)


def _looks_chain(query: str) -> bool:
    """다중홉 랭킹 체인(cutline)인가. 전파 cue + 반복 표지를 모두 요구한다.

    "SK하이닉스가 오르면 수혜 볼 기업, 거기서 또 수혜 볼 기업 N개" 류. 전파어만 있으면
    단일 홉 '수혜주' 질문이라 체인이 아니다 → 반복 표지(또/거기서…)까지 있어야 LLM 에 위임.
    """
    q = " ".join((query or "").split())
    if not q:
        return False
    return _has_any(q, _CHAIN_PROPAGATION_TERMS) and _has_any(q, _CHAIN_RECURSION_TERMS)


def _as_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    content = getattr(raw, "content", None)
    if isinstance(content, str):
        return json.loads(content)
    raise ValueError("chain planner response is not JSON")


def _relation_step(data: dict[str, Any] | None) -> RelationStep | None:
    if not isinstance(data, dict):
        return None
    rel = str(data.get("rel_type") or data.get("relation") or "").strip()
    direction = str(data.get("direction") or "").strip()
    if rel not in _RELATION_TYPES or direction not in _DIRECTIONS:
        return None
    default_alias, default_role = _RELATION_DEFAULTS.get((rel, direction), (rel.lower(), rel.lower()))
    alias = str(data.get("alias") or default_alias)
    role = str(data.get("role") or default_role)
    return RelationStep(rel, direction, alias, role)  # type: ignore[arg-type]


def _metric_id(data: dict[str, Any], query: str) -> MetricId | None:
    metric = str(data.get("rank_metric") or data.get("metric_id") or "").strip()
    if metric in _METRICS:
        return metric  # type: ignore[return-value]
    if "영업이익" in query:
        return "dart_OperatingIncomeLoss"
    if "순이익" in query or "당기순이익" in query:
        return "ifrs-full_ProfitLoss"
    if "자산" in query or "규모" in query:
        return "ifrs-full_Assets"
    if "매출" in query or "수익" in query or "잘나가" in query:
        return "ifrs-full_Revenue"
    return None


def _hop_step(data: dict[str, Any], query: str) -> HopStep | None:
    """체인 한 홉 JSON → HopStep. 관계/지표 미해소면 None(체인 전체 폐기)."""
    if not isinstance(data, dict):
        return None
    relation = _relation_step(data.get("relation") or data)
    if relation is None:
        return None
    metric_id = _metric_id(data, query)
    if not metric_id:
        return None
    try:
        top_n = int(data.get("top_n") or 3)
    except (TypeError, ValueError):
        top_n = 3
    top_n = max(1, min(top_n, 10))
    policy = str(data.get("policy") or data.get("first_candidate_policy") or "default")
    if policy not in _FIRST_POLICIES:
        policy = "default"
    return HopStep(
        relation=relation,
        rank=MetricRankStep(metric_id, alias="hop_" + relation.role),
        top_n=top_n,
        policy=policy,  # type: ignore[arg-type]
    )


def _coerce_chain_plan(data: dict[str, Any], query: str) -> StructuredPlan | None:
    """multi_hop_chain JSON → StructuredPlan. 홉이 2개 미만이거나 하나라도 무효면 None."""
    raw_hops = data.get("hops") or []
    if not isinstance(raw_hops, list) or not raw_hops:
        return None
    hops: list[HopStep] = []
    for raw_hop in raw_hops:
        hop = _hop_step(raw_hop, query) if isinstance(raw_hop, dict) else None
        if hop is None:
            return None
        hops.append(hop)
    if len(hops) < 2:
        return None  # 체인은 최소 2홉(수혜의 수혜) — 단일 홉은 일반 랭킹이 처리.
    return StructuredPlan(
        kind="multi_hop_chain",
        first_relation=hops[0].relation,
        first_rank=hops[0].rank,
        hops=hops,
        planner="llm",
        raw_reason=str(data.get("reason") or "LLM multi-hop chain"),
        steps=[
            {
                "op": "chain_hop",
                "relation": h.relation.rel_type,
                "direction": h.relation.direction,
                "metric": h.rank.metric_id,
                "top_n": h.top_n,
            }
            for h in hops
        ],
    )


def coerce_plan(data: dict[str, Any], query: str) -> StructuredPlan | None:
    """체인 LLM JSON 을 실행 가능한 multi_hop_chain StructuredPlan 으로 검증·변환.

    supported:false·kind 불일치·홉 무효는 None(호출자 폴백). 이 모듈은 체인만 다룬다.
    """
    if not data or data.get("supported") is False:
        return None
    if str(data.get("kind") or "").strip() != "multi_hop_chain":
        return None
    return _coerce_chain_plan(data, query)


def build_chain_plan(data: dict[str, Any], query: str) -> StructuredPlan | None:
    """통합 라우터용: hops 필드만으로 multi_hop_chain plan 빌드(kind 키 불요).

    router 의 통합 LLM 응답은 question_type='chain' + hops 형태라 coerce_plan 의 kind 검사
    대신 홉 검증만 적용한다. 홉이 없거나 무효면 None(라우터가 비-체인으로 폴백).
    """
    if not isinstance(data, dict):
        return None
    return _coerce_chain_plan(data, query)


def _invoke_llm(query: str) -> dict[str, Any]:
    from config.llm import json_llm  # noqa: PLC0415
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    raw = json_llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_USER_TEMPLATE.format(query=query)),
    ])
    return _as_dict(raw)


def plan(query: str) -> StructuredPlan | None:
    """체인 cue 가 있는 질의만 LLM 1회로 multi_hop_chain plan 추출. 그 외 None."""
    if not _ENABLED or not _looks_chain(query):
        return None
    data = _invoke_llm(query)
    return coerce_plan(data, query)


_SYSTEM_PROMPT = """너는 GraphRAG 다중홉 랭킹 체인 질의를 제한된 hops 계획으로 바꾸는 semantic parser다.
너는 Cypher, SQL, 자유 텍스트 계획을 절대 만들지 않는다. 아래 허용값 중에서만 고른다.

허용 relation:
- SUPPLIES_TO: 공급/납품 관계. incoming = 후보가 기준 회사에 공급, outgoing = 기준 회사가 후보에 공급.
- RELATED_PARTY: 특수관계자/관련 회사. direction은 보통 undirected.
- IS_MAJOR_SHAREHOLDER_OF: 주주/지분 관계.
- IS_SUBSIDIARY_OF: 자회사/종속회사 관계.
- INVESTS_IN: 투자 관계.

허용 rank_metric:
- ifrs-full_Revenue: 매출액, 수익, 잘나가는 회사의 기본 기준.
- dart_OperatingIncomeLoss: 영업이익.
- ifrs-full_ProfitLoss: 순이익/당기순이익.
- ifrs-full_Assets: 자산/규모.

출력은 JSON 객체 하나만 한다."""

_USER_TEMPLATE = """질문을 아래 JSON 스키마로만 변환하라.

{{
  "supported": true 또는 false,
  "kind": "multi_hop_chain",
  "hops": [
    {{
      "relation": {{
        "rel_type": "SUPPLIES_TO|RELATED_PARTY|IS_MAJOR_SHAREHOLDER_OF|IS_SUBSIDIARY_OF|INVESTS_IN",
        "direction": "incoming|outgoing|undirected|auto"
      }},
      "rank_metric": "ifrs-full_Revenue|dart_OperatingIncomeLoss|ifrs-full_ProfitLoss|ifrs-full_Assets",
      "top_n": 3,
      "policy": "default|operating_counterparty"
    }}
  ],
  "reason": "짧은 한국어 근거"
}}

규칙:
- 한 회사의 영향이 단계적으로 퍼져 '그 회사가 오르면/수혜를 보면, 거기서 또 수혜 보는 회사 N개'처럼 2단계 이상 체인을 각 단계 상위 N개로 묻는 질문만 supported=true, kind=multi_hop_chain.
- hops 배열에 단계별 {{relation, direction, rank_metric, top_n}}을 순서대로 넣는다(최소 2홉).
- 각 hop 의 top_n 은 질문이 N개를 명시하면 그 수, 아니면 3.
- '수혜/낙수/거래'는 보통 SUPPLIES_TO(고객·매출처=outgoing, 공급사=incoming, 애매하면 auto)이고 후보는 매출(ifrs-full_Revenue)로 줄세운다.
- 단일 홉(수혜주 한 단계)이거나 체인이 아니면 supported=false.

질문: {query}"""
