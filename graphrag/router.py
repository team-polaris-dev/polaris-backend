"""질문종류 통합 분류기 — 흩어진 키워드 게이트를 LLM 1회 판정으로 모은다.

기존엔 질문종류 판정이 네 곳의 키워드 게이트에 흩어져 있었다(chain_planner._looks_chain·
graph_mode._prefilter_mode/_is_silent·planner.plan). 키워드가 안 맞으면 LLM 이 *호출조차*
안 돼서 표현을 조금만 바꿔도 의도가 안 잡혔다(예: 재귀어 없는 체인 질문이 단일 홉으로 새는
Phase 1 버그). 이 모듈은 LLM 을 1차 판정자로 올려, 한 번의 호출로 8종 질문종류를 분류하고
체인이면 홉까지 함께 받는다. 키워드 로직은 LLM off·다운·무효일 때의 폴백으로 강등된다.

LLM 이 하지 않는 결정적 레이어는 튜닝으로 보존한다:
- 결정적 프리필터(순수 속성 silent·교집합 정확성이 핵심인 보존 kind)는 LLM 미호출 fast-path.
- 지표 해소(config.relations)·교집합 빌드(planner)·홉 검증(chain_planner)은 그대로 재사용.

반환은 Route 하나. search 는 route.plan(있으면 구조화 실행), node 는 route.type(모드 분기)을
본다 — 분류는 한 곳, 소비는 두 곳.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from config.relations import render_chain_relation_vocab
from graphrag import chain_planner, graph_mode, planner
from graphrag.plan_schema import StructuredPlan

_FALSE = {"0", "false", "False", "no", "NO", "off", "OFF"}
_ENABLED = os.environ.get("GRAPHRAG_LLM_PLANNER", "1") not in _FALSE

# route.type 값. 구조화 kind(plan 동반) + 모드(node 분기용).
_STRUCTURED_TYPES = {"chain", "community_member_rank", "multi_anchor_rank"}
_MODE_TYPES = {"relation_rank", "relation_explore", "relation_only", "macro", "silent"}
_ALL_TYPES = _STRUCTURED_TYPES | _MODE_TYPES
# 프리필터/폴백이 결정적으로 빌드하는 보존 kind(교집합·군집 정확성).
_PRESERVED = {"community_member_rank", "multi_anchor_rank"}


@dataclass(frozen=True)
class Route:
    """질문종류 판정 결과.

    type: 8종 중 하나. plan: 구조화 실행 plan(chain/community/multi_anchor 일 때만, 그 외 None).
    source: 판정 출처(prefilter=결정적 fast-path / llm=1차 판정 / fallback=LLM 불가 키워드).
    """

    type: str
    plan: StructuredPlan | None = None
    source: str = "llm"
    reason: str = ""


def _as_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    content = getattr(raw, "content", None)
    if isinstance(content, str):
        return json.loads(content)
    raise ValueError("router response is not JSON")


def _prefilter(query: str) -> Route | None:
    """LLM 미호출 결정적 fast-path. 명백·정확성요구 케이스만 확정, 나머지는 None(LLM 위임).

    - 순수 노드 속성(매출·자산 등만) → silent: 값싸고 신뢰도 높은 침묵 판정.
    - 보존 kind(community_member_rank/multi_anchor_rank) → 교집합·군집 정확성이 핵심이라
      결정적 planner 가 직접 빌드. 키워드가 잡히는 한 LLM 보다 정확하다.
    체인·일반 모드는 여기서 확정하지 않는다 — 표현 다양성을 LLM 이 흡수하도록(이 모듈의 목적).
    """
    if graph_mode._is_silent(query):
        return Route("silent", source="prefilter", reason="순수 노드 속성 질문")
    p = planner.plan(query)
    if p is not None and p.kind in _PRESERVED:
        return Route(p.kind, plan=p, source="prefilter", reason=p.raw_reason)
    return None


def _coerce(data: dict[str, Any], query: str, *, has_metric: bool) -> Route | None:
    """LLM JSON → Route. 무효·미지원·빌드불가는 None(호출자 폴백).

    chain 은 같은 응답의 hops 로 plan 을 빌드(없거나 무효면 None → 폴백, 체인 아님).
    보존 kind 는 결정적 planner 로 빌드(키워드로 실현 불가하면 관계 탐색으로 강등).
    relation_rank 인데 노드 지표가 없으면 relation_only 로 강등(억지 1위 금지).
    """
    if not isinstance(data, dict) or data.get("supported") is False:
        return None
    qtype = str(data.get("question_type") or data.get("type") or "").strip()
    if qtype not in _ALL_TYPES:
        return None
    if qtype == "chain":
        plan = chain_planner.build_chain_plan(data, query)
        if plan is None:
            return None
        return Route("chain", plan=plan, source="llm", reason=plan.raw_reason)
    if qtype in _PRESERVED:
        p = planner.plan(query)
        if p is not None and p.kind == qtype:
            return Route(qtype, plan=p, source="llm", reason=p.raw_reason)
        return Route(
            "relation_rank" if has_metric else "relation_explore",
            source="llm",
            reason=f"{qtype} 결정적 빌드 불가 → 관계 탐색 강등",
        )
    if qtype == "relation_rank" and not has_metric:
        return Route("relation_only", source="llm", reason="노드 지표 없음 → 억지 1위 금지")
    return Route(qtype, source="llm", reason=str(data.get("reason") or ""))


def _fallback(query: str, *, has_metric: bool) -> Route:
    """LLM off·다운·무효 시 결정적 백스톱. 흩어져 있던 키워드 게이트를 우선순위로 모은다.

    보존 kind(planner) → 모드(graph_mode 프리필터, 애매하면 _fallback_mode). 체인은 홉
    추출에 LLM 이 필요하므로 폴백에서 만들지 않는다 — 관계 탐색/랭킹으로 degrade 된다.
    """
    p = planner.plan(query)
    if p is not None and p.kind in _PRESERVED:
        return Route(p.kind, plan=p, source="fallback", reason=p.raw_reason)
    mode = graph_mode._prefilter_mode(query, has_anchor=False, has_metric=has_metric)
    if mode is None:
        mode = graph_mode._fallback_mode(query, has_anchor=False, has_metric=has_metric)
    return Route(mode.value, source="fallback", reason="키워드 폴백")


def _invoke_llm(query: str) -> dict[str, Any]:
    from config.llm import json_llm  # noqa: PLC0415
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    raw = json_llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_USER_TEMPLATE.format(query=query)),
    ])
    return _as_dict(raw)


def classify(query: str, *, has_metric: bool) -> Route:
    """질문 → Route. 결정적 프리필터 fast-path → LLM 1차 판정 → 키워드 폴백.

    has_metric: 질의에서 노드 지표(매출·영업이익 등)가 해소되는가(planner._metric_id).
    relation_rank 를 relation_only 로 강등할지 판단에 쓴다(억지 1위 금지).
    """
    q = " ".join((query or "").split())
    if not q:
        return Route("relation_explore", source="prefilter", reason="빈 질의")
    pre = _prefilter(q)
    if pre is not None:
        return pre
    if _ENABLED:
        try:
            data = _invoke_llm(q)
        except Exception:
            data = None
        if data is not None:
            route = _coerce(data, q, has_metric=has_metric)
            if route is not None:
                return route
    return _fallback(q, has_metric=has_metric)


_SYSTEM_PROMPT = """너는 한국어 회사 질의를 받아 '어떤 종류의 질문인가'를 고르는 분류기다.
너는 Cypher, SQL, 자유 텍스트를 절대 만들지 않는다. 아래 question_type 중 하나만 고른다.

question_type (질문 종류):
- chain: 한 회사의 영향이 단계적으로 퍼지는 다중홉 질문. 'A가 오르면 수혜 볼 기업, 거기서 또
  수혜 볼 기업 N개'처럼 2단계 이상으로 전파되며 각 단계 상위 N개를 묻는다. 이때만 hops 를 채운다.
- community_member_rank: 한 회사가 아니라 그 회사가 속한 그룹/계열 전체를 노드 지표(매출 등)로
  줄세우는 질문. '삼성 계열사 중 매출 1위' 류.
- multi_anchor_rank: 둘 이상 회사가 *공통으로* 가진 거래상대 교집합을 지표로 1위만 뽑는 질문.
  '삼성전자와 SK하이닉스가 동시에 거래하는 소재사 중 매출 1위' 류.
- relation_rank: 한 회사의 관계 상대(협력사·주주 등)를 노드 지표로 줄세우는 단일 단계 랭킹.
- relation_explore: 한 회사(또는 특정 제품)의 관계망을 펼쳐 보여주는 질문(랭킹 아님). 협력사·
  주주·자회사뿐 아니라 임원·대표이사·경영진(인물 관계)과 '특정 제품의 제조사·생산기업'(예:
  'HBM 제조사', 'D램 만드는 회사')도 모두 relation_explore 다.
- relation_only: 관계를 묻지만 줄세울 노드 지표가 없는 질문(거래금액·점유율 최대 등 — 데이터에
  없음). 억지 1위를 만들면 안 된다.
- macro: 특정 회사·제품 앵커가 *전혀 없이* 업종/산업/시장 전반을 묻는 sensemaking 질문. 질문에
  특정 회사나 특정 제품이 명시되면 macro 가 아니다(그 앵커의 관계 질문 → relation_*).
- silent: 한 회사의 단일 *숫자/스칼라* 속성(매출·자산·주가·시총·영업이익·직원수·설립일·본사 등)
  만 묻는 질문. 관계가 필요 없다. 대표이사·임원·경영진 같은 인물은 속성이 아니라 관계이므로
  silent 가 아니다(relation_explore).

chain 일 때 hops 각 단계의 relation 은 아래 단어집의 관계 타입만 쓴다. 각 관계의 한글
관련어를 보고 질문 표현이 어느 관계(들)에 해당하는지 판단하라. 한 표현이 여러 관계에 걸리는
복합 신호(수혜 등)는 그 관계들을 한 relation 배열에 함께 담아 펼친다 — 하나로 좁히지 말 것.

""" + render_chain_relation_vocab() + """

허용 rank_metric: ifrs-full_Revenue(매출), dart_OperatingIncomeLoss(영업이익),
ifrs-full_ProfitLoss(순이익), ifrs-full_Assets(자산).

출력은 JSON 객체 하나만 한다."""

_USER_TEMPLATE = """질문을 아래 JSON 스키마로만 변환하라.

{{
  "supported": true 또는 false,
  "question_type": "chain|community_member_rank|multi_anchor_rank|relation_rank|relation_explore|relation_only|macro|silent",
  "hops": [
    {{
      "relation": [
        {{
          "rel_type": "SUPPLIES_TO|RELATED_PARTY|IS_MAJOR_SHAREHOLDER_OF|IS_SUBSIDIARY_OF|INVESTS_IN",
          "direction": "incoming|outgoing|undirected|auto"
        }}
      ],
      "rank_metric": "ifrs-full_Revenue|dart_OperatingIncomeLoss|ifrs-full_ProfitLoss|ifrs-full_Assets",
      "top_n": 3,
      "policy": "default|operating_counterparty"
    }}
  ],
  "reason": "짧은 한국어 근거"
}}

규칙:
- question_type 은 항상 하나 고른다.
- hops 는 question_type 이 chain 일 때만 채운다(최소 2홉, 단계 순서대로). 그 외엔 빈 배열 [].
- 각 hop 의 relation 은 관계 객체의 '배열'이다. 한 관계만 맞으면 길이 1, '복합' 경로면 여러
  관계를 한 배열에 담는다 — 그 홉에서 모든 관계의 후보를 합쳐 한 지표로 함께 줄세운다.
- chain 의 '수혜/낙수/거래'는 복합 신호다. 한 회사가 오르면 이득은 공급(SUPPLIES_TO: 공급사
  =incoming, 고객=outgoing, 애매하면 auto)뿐 아니라 지분/지배(IS_MAJOR_SHAREHOLDER_OF,
  IS_SUBSIDIARY_OF), 투자(INVESTS_IN), 특수관계(RELATED_PARTY)로도 흐른다. 질문이 한 경로로
  못박지 않으면 relation 배열에 해당 관계들을 함께 담아라 — 공급 하나로 좁히지 말 것. 후보는
  기본 매출(ifrs-full_Revenue)로 줄세운다. top_n 은 질문이 N개를 명시하면 그 수, 아니면 3.
- 단일 단계 '수혜주' 질문은 chain 이 아니다(전파가 한 번뿐). 관계 랭킹이면 relation_rank.
- 거래금액·점유율처럼 노드 지표가 아닌 크기로 줄세우라는 질문은 relation_only(억지 1위 금지).
- 대표이사·임원·경영진 등 인물을 묻는 질문은 silent 가 아니라 relation_explore.
- 특정 제품(HBM·D램 등)의 제조사·생산기업을 묻는 질문은 macro 가 아니라 relation_explore.

질문: {query}"""
