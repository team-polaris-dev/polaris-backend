"""LLM-assisted planner for structured GraphRAG questions.

The LLM only chooses among the constrained logical-plan fields defined in
plan_schema.py. It never writes Cypher or SQL; invalid output is discarded and
the deterministic planner/fallback search path continues.
"""
from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any

from config.relations import MACRO_TERMS, RANK_TERMS, has_rank_intent
from graphrag.plan_schema import (
    BranchKind,
    BranchRankStep,
    Direction,
    FirstCandidatePolicy,
    MetricId,
    MetricRankStep,
    RelationStep,
    RelationType,
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
_BRANCH_KINDS = {"supplier", "major_customer", "related_party", "investment"}

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

_RANK_TERMS = RANK_TERMS  # config.relations SSOT (구 중복 정의 제거)
_RELATION_TERMS = (
    "거래", "공급", "납품", "협력", "벤더", "매출처", "고객", "관련",
    "특수관계", "관계자", "계열", "주주", "지분", "자회사", "종속", "투자",
)
# 순수 노드 속성(스칼라) 단어. 관계어·랭크어가 같이 없으면 그래프가 침묵할 신호.
# 관계 유형 단어(대표·임원 등 EXECUTIVE_OF 류)는 일부러 제외 — silent 가 관계질문을 삼키지 않도록.
_ATTRIBUTE_ONLY_TERMS = (
    "매출", "영업이익", "순이익", "당기순이익", "자산", "부채", "자본",
    "주가", "시가총액", "시총", "직원수", "종업원", "임직원", "영업이익률",
    "설립", "본사", "소재지", "배당", "주식수",
)
# 매크로/업계 질문 cue (config.relations SSOT). 업종어가 회사명에 퍼지매칭돼 has_anchor 가
# 켜지는 경우에도 MACRO 가 has_anchor 보다 먼저 발화하도록 _prefilter_mode 가 순서를 잡는다.
_MACRO_TERMS = MACRO_TERMS


class GraphMode(str, Enum):
    """그래프 노드가 질문마다 고르는 실행 모드(역할 계약).

    macro=커뮤니티 sensemaking, relation_rank=노드지표 랭킹(구조화),
    relation_explore=시드 관계망 펼치기, relation_only=랭킹 불가 → 시드 관계망으로 degrade,
    silent=순수 속성질문 → 그래프 침묵(rdb/vec 가 답).
    """

    MACRO = "macro"
    RELATION_RANK = "relation_rank"
    RELATION_EXPLORE = "relation_explore"
    RELATION_ONLY = "relation_only"
    SILENT = "silent"


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(t in text for t in terms)


def _looks_structured(query: str) -> bool:
    q = " ".join((query or "").split())
    return bool(q and has_rank_intent(q) and _has_any(q, _RELATION_TERMS))


def _is_silent(query: str) -> bool:
    """순수 노드 속성 질문인가. 속성어 present AND 관계어·랭크어 모두 없음.

    앵커 유무와 무관(앵커 있어도 '삼성전자 매출은?'은 침묵). 애매하면 침묵 금지(False) —
    관계어/랭크어가 하나라도 있으면 그래프가 기여할 여지가 있다고 보고 False.
    """
    q = " ".join((query or "").split())
    if not q or not _has_any(q, _ATTRIBUTE_ONLY_TERMS):
        return False
    return not (_has_any(q, _RELATION_TERMS) or _has_any(q, _RANK_TERMS))


def _as_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    content = getattr(raw, "content", None)
    if isinstance(content, str):
        return json.loads(content)
    raise ValueError("LLM planner response is not JSON")


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


def _make_steps(
    kind: str,
    first: RelationStep,
    metric_id: str,
    second: RelationStep | None,
    branches: list[BranchRankStep] | None = None,
) -> list[dict]:
    if kind == "single_anchor_branch_rank":
        steps: list[dict] = []
        for branch in branches or []:
            steps.extend([
                {
                    "op": "branch_traverse",
                    "branch": branch.kind,
                    "from": "anchor",
                    "relation": branch.relation.rel_type,
                    "direction": branch.relation.direction,
                    "as": branch.relation.alias,
                },
                {"op": "join_metric", "metric": metric_id, "target": branch.relation.alias},
                {"op": "argmax", "by": metric_id, "as": branch.rank.alias},
                {"op": "score_evidence", "target": branch.rank.alias},
            ])
        return steps

    steps: list[dict] = [
        {
            "op": "intersect_anchors"
            if kind in {"multi_anchor_branch_rank", "multi_anchor_rank"}
            else "traverse",
            "relation": first.rel_type,
            "direction": first.direction,
            "as": first.alias,
        },
        {"op": "join_metric", "metric": metric_id, "target": first.alias},
        {"op": "argmax", "by": metric_id, "as": "top_" + first.role},
    ]
    if kind == "two_hop_rank" and second is not None:
        steps.extend([
            {"op": "traverse", "from": "top_" + first.role, "relation": second.rel_type, "direction": second.direction, "as": second.alias},
            {"op": "join_metric", "metric": metric_id, "target": second.alias},
            {"op": "argmax", "by": metric_id, "as": "top_" + second.role},
        ])
    if kind == "multi_anchor_branch_rank":
        for branch in branches or []:
            steps.extend([
                {
                    "op": "branch_traverse",
                    "branch": branch.kind,
                    "from": "top_" + first.role,
                    "relation": branch.relation.rel_type,
                    "direction": branch.relation.direction,
                    "as": branch.relation.alias,
                },
                {"op": "join_metric", "metric": metric_id, "target": branch.relation.alias},
                {"op": "argmax", "by": metric_id, "as": branch.rank.alias},
                {"op": "score_evidence", "target": branch.rank.alias},
            ])
    return steps


def _branch_rank(data: dict[str, Any], metric_id: MetricId) -> BranchRankStep | None:
    kind = str(data.get("kind") or data.get("branch") or "").strip()
    if kind not in _BRANCH_KINDS:
        return None
    relation = _relation_step(data.get("relation") or data.get("branch_relation"))
    if relation is None:
        return None
    return BranchRankStep(
        kind=kind,  # type: ignore[arg-type]
        relation=relation,
        rank=MetricRankStep(metric_id, alias="top_" + kind),
    )


def coerce_plan(data: dict[str, Any], query: str) -> StructuredPlan | None:
    """Validate LLM JSON and convert it into an executable StructuredPlan."""
    if not data or data.get("supported") is False:
        return None

    kind = str(data.get("kind") or "").strip()
    if kind not in {"single_hop_rank", "two_hop_rank", "multi_anchor_rank", "multi_anchor_branch_rank", "single_anchor_branch_rank"}:
        return None

    metric_id = _metric_id(data, query)
    if not metric_id:
        return None

    branches: list[BranchRankStep] = []
    if kind in {"multi_anchor_branch_rank", "single_anchor_branch_rank"}:
        raw_branches = data.get("branch_relations") or data.get("branches") or []
        if not isinstance(raw_branches, list):
            return None
        for raw_branch in raw_branches:
            if isinstance(raw_branch, dict):
                branch = _branch_rank(raw_branch, metric_id)
                if branch:
                    branches.append(branch)
        allowed = {"major_customer", "related_party", "investment"}
        if kind == "single_anchor_branch_rank":
            allowed = {"supplier", "major_customer", "related_party", "investment"}
        branch_kinds = {b.kind for b in branches}
        if not branch_kinds or not branch_kinds <= allowed:
            return None

    first = _relation_step(data.get("first_relation"))
    if first is None and kind == "single_anchor_branch_rank" and branches:
        first = next((b.relation for b in branches if b.kind == "supplier"), branches[0].relation)
    if first is None:
        return None

    second = _relation_step(data.get("second_relation")) if kind == "two_hop_rank" else None
    if kind == "two_hop_rank" and second is None:
        return None

    first_policy = str(data.get("first_candidate_policy") or "default")
    if first_policy not in _FIRST_POLICIES:
        first_policy = "default"
    # 공통 공급사 교집합 단일 1위는 결정적 multi_anchor_rank 와 동일하게 운영 거래 게이트를 쓴다
    # (SUPPLIES_TO incoming → operating_counterparty). 실행기 _rank_candidates 가 같은 근거
    # 바닥값을 적용해야 LLM·결정적 경로가 같은 후보를 통과시킨다.
    if kind == "multi_anchor_rank" and first.rel_type == "SUPPLIES_TO" and first.direction == "incoming":
        first_policy = "operating_counterparty"

    exclude_anchor = data.get("exclude_original_anchor_from_second")
    if exclude_anchor is None:
        exclude_anchor = kind == "two_hop_rank"

    steps = _make_steps(kind, first, metric_id, second, branches)
    return StructuredPlan(
        kind=kind,  # type: ignore[arg-type]
        first_relation=first,
        first_rank=MetricRankStep(metric_id, alias="top_" + first.role),
        second_relation=second,
        second_rank=MetricRankStep(metric_id, alias="top_" + second.role) if second else None,
        branch_ranks=branches,
        common_anchor_min=int(data.get("common_anchor_min") or (2 if kind in {"multi_anchor_branch_rank", "multi_anchor_rank"} else 1)),
        first_candidate_policy=first_policy,  # type: ignore[arg-type]
        exclude_original_anchor_from_second=bool(exclude_anchor),
        planner="llm",
        raw_reason=str(data.get("reason") or "LLM constrained planner"),
        steps=steps,
    )


def _prefilter_mode(query: str, *, has_anchor: bool, has_metric: bool) -> GraphMode | None:
    """명백한 질문은 결정적으로 모드 확정. 애매하면 None(=LLM 에 위임).

    순서: SILENT(속성전용) → 구조화(랭크+관계) → MACRO(매크로cue) → EXPLORE(앵커) → 위임/EXPLORE.
    구조화면 has_metric 으로 RANK(노드지표 랭킹·가능) vs ONLY(금액랭킹류·degrade) 분리.

    MACRO 를 has_anchor 보다 먼저 보는 이유: "반도체 업종 공급망 전반" 의 '반도체'가 회사명에
    퍼지매칭돼 has_anchor 가 켜지면 EXPLORE 로 새 버려 global_search 가 안 탔다(MACRO 사문화).
    명시 매크로 cue 가 있으면 퍼지 앵커보다 우선한다. 사용자가 회사를 *명시 호명*했을 때의
    DRIFT 보존은 노드가 `explicit_company` 로 최종 판정한다(여기선 cue 우선만 결정).
    """
    q = " ".join((query or "").split())
    if not q:
        return None
    if _is_silent(q):
        return GraphMode.SILENT
    if _looks_structured(q):
        return GraphMode.RELATION_RANK if has_metric else GraphMode.RELATION_ONLY
    if _has_any(q, _MACRO_TERMS):
        return GraphMode.MACRO
    # 앵커가 해소된 질문은 그 회사의 관계망을 보여주는 게 안전한 기본값. 랭킹·속성 신호가
    # 없으면 explore 로 확정 — LLM 은 엔티티가 안 잡힌 관계질문에만 부른다.
    if has_anchor:
        return GraphMode.RELATION_EXPLORE
    if _has_any(q, _RELATION_TERMS):
        return None  # 관계를 묻지만 엔티티 미해소 → 모드를 LLM 에 위임
    return GraphMode.RELATION_EXPLORE  # 신호 없음 → 노드가 빈 로컬로 degrade(시드 0)


def _fallback_mode(query: str, *, has_anchor: bool, has_metric: bool) -> GraphMode:
    """결정적 최종 백스톱. LLM off·다운·invalid 전부 여기로 = graceful degrade.

    순수속성→SILENT, 앵커없는 매크로 cue→MACRO, 그 외→EXPLORE. 앵커 미해소를 무조건
    MACRO 로 보내면 global_search 가 강제돼 n_seeds=0 결과를 덮어쓰는 회귀가 나므로,
    매크로 cue 가 있을 때만 MACRO 로 보낸다. 앵커 없고 cue 도 없으면 EXPLORE 가 빈 로컬을 내며
    안전 degrade(노드 search 가 시드 0 → assemble_local 빈 결과).
    """
    if _is_silent(query):
        return GraphMode.SILENT
    if not has_anchor and _has_any(query, _MACRO_TERMS):
        return GraphMode.MACRO
    return GraphMode.RELATION_EXPLORE


def _coerce_mode(data: dict[str, Any] | None, query: str, *, has_anchor: bool, has_metric: bool) -> GraphMode:
    """LLM 의 mode 필드 검증. 누락·미지원 enum·supported:false → 결정적 fallback.

    relation_rank 인데 노드 지표가 없으면 ONLY 로 강등(억지 1위 금지).
    """
    if not data or data.get("supported") is False:
        return _fallback_mode(query, has_anchor=has_anchor, has_metric=has_metric)
    raw = str(data.get("mode") or "").strip()
    try:
        mode = GraphMode(raw)
    except ValueError:
        return _fallback_mode(query, has_anchor=has_anchor, has_metric=has_metric)
    if mode is GraphMode.RELATION_RANK and not has_metric:
        return GraphMode.RELATION_ONLY
    return mode


def _invoke_llm(query: str) -> dict[str, Any]:
    from config.llm import json_llm  # noqa: PLC0415
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    raw = json_llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_USER_TEMPLATE.format(query=query)),
    ])
    return _as_dict(raw)


def plan_with_mode(
    query: str, *, has_anchor: bool, has_metric: bool
) -> tuple[GraphMode, StructuredPlan | None]:
    """질문 → (모드, plan). 결정적 프리필터 우선, 애매하면 LLM 1회로 모드+plan 동시 추출.

    plan 은 relation_rank 일 때만 채워질 수 있다(search 가 자체 빌드하므로 노드는 보통 무시).
    프리필터 hit·LLM off·다운·invalid 는 전부 plan None + degrade 모드.
    """
    pre = _prefilter_mode(query, has_anchor=has_anchor, has_metric=has_metric)
    if pre is not None:
        return pre, None
    if not _ENABLED:
        return _fallback_mode(query, has_anchor=has_anchor, has_metric=has_metric), None
    try:
        data = _invoke_llm(query)
    except Exception:
        return _fallback_mode(query, has_anchor=has_anchor, has_metric=has_metric), None
    mode = _coerce_mode(data, query, has_anchor=has_anchor, has_metric=has_metric)
    plan_obj: StructuredPlan | None = None
    if mode is GraphMode.RELATION_RANK:
        plan_obj = coerce_plan(data, query)
        if plan_obj is None:
            mode = GraphMode.RELATION_ONLY  # 쓸 plan 없음 → degrade, 억지 1위 금지
    return mode, plan_obj


_SYSTEM_PROMPT = """너는 GraphRAG 질의를 제한된 실행 계획으로 바꾸는 semantic parser다.
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

first_candidate_policy:
- default: 질문이 그룹/특수관계/지배 관계까지 포함해도 되는 경우.
- operating_counterparty: 거래처, 협력사, 공급사처럼 운영상 거래 상대를 묻는 경우.

mode (그래프 역할 — 질문마다 무엇을 할지 고른다):
- macro: 특정 회사 앵커 없이 업계/산업/시장 전반을 묻는 sensemaking 질문.
- relation_rank: 회사들을 노드 지표(매출·영업이익·순이익·자산)로 줄세우는 랭킹 질문. 이때만 위 plan 필드를 채운다.
- relation_explore: 특정 회사의 관계망(협력사·주주·자회사 등)을 펼쳐 보여주는 질문(랭킹 아님).
- relation_only: 관계를 묻지만 줄세울 노드 지표가 없는 질문(예: 거래금액 최대 공급처 — 금액은 데이터에 없음). 억지 1위를 만들지 말 것.
- silent: 한 회사의 단일 속성(매출·자산·주가 등)만 묻는 질문. 관계가 필요 없다.

출력은 JSON 객체 하나만 한다."""

_USER_TEMPLATE = """질문을 아래 JSON 스키마로만 변환하라.

{{
  "supported": true 또는 false,
  "mode": "macro|relation_rank|relation_explore|relation_only|silent",
  "kind": "single_hop_rank" 또는 "two_hop_rank" 또는 "multi_anchor_rank" 또는 "multi_anchor_branch_rank" 또는 "single_anchor_branch_rank",
  "first_relation": {{
    "rel_type": "SUPPLIES_TO|RELATED_PARTY|IS_MAJOR_SHAREHOLDER_OF|IS_SUBSIDIARY_OF|INVESTS_IN",
    "direction": "incoming|outgoing|undirected|auto"
  }},
  "rank_metric": "ifrs-full_Revenue|dart_OperatingIncomeLoss|ifrs-full_ProfitLoss|ifrs-full_Assets",
  "second_relation": {{
    "rel_type": "SUPPLIES_TO|RELATED_PARTY|IS_MAJOR_SHAREHOLDER_OF|IS_SUBSIDIARY_OF|INVESTS_IN",
    "direction": "incoming|outgoing|undirected"
  }} 또는 null,
  "branch_relations": [
    {{
      "kind": "supplier|major_customer|related_party|investment",
      "relation": {{
        "rel_type": "SUPPLIES_TO|RELATED_PARTY|INVESTS_IN",
        "direction": "incoming|outgoing|undirected|auto"
      }}
    }}
  ],
  "common_anchor_min": 2,
  "exclude_original_anchor_from_second": true,
  "first_candidate_policy": "default|operating_counterparty",
  "reason": "짧은 한국어 근거"
}}

규칙:
- mode 는 항상 채운다. macro/relation_rank/relation_explore/relation_only/silent 중 하나.
- mode=relation_rank 이고 줄세울 노드 지표(매출·영업이익·순이익·자산)가 있을 때만 supported=true 로 두고 plan 필드(kind, first_relation 등)를 채운다. 그 외 mode 는 supported=false.
- 거래금액·점유율처럼 노드 지표가 아닌 크기로 줄세우라는 질문은 데이터에 없으므로 mode=relation_only(억지 1위 금지).
- 랭킹/최댓값/1위 질문이 아니면 supported=false.
- 두 번째로 '그 회사/해당 기업의 관련 회사 중'처럼 이어지면 two_hop_rank.
- 기준 회사가 둘 이상이고(예: 'A와 B에 공급하는', 'A와 B가 거래하는', 'A와 B 공통/둘 다/동시에'), 그 둘이 공유하는 거래 상대(공급사/협력사 등) 중 단일 지표 1위만 물으면 multi_anchor_rank. 관계 유형별 분기 비교가 없으면 여기다. 핵심은 표현이 아니라 의미: 기업이 둘 명시되고 그 교집합 거래상대를 줄세우면 multi_anchor_rank다(키워드가 없어도). 예: '삼성전자와 에스케이하이닉스에 제품을 공급하는 회사 중 매출액이 가장 높은 회사' → kind=multi_anchor_rank, first_relation=SUPPLIES_TO incoming, common_anchor_min=2.
- 반대로 기준 회사가 하나뿐이면(예: 'SK하이닉스에 공급하는 회사 중 매출 1위') multi_anchor_rank가 아니라 single_hop_rank다.
- 'A와 B 둘 다/공통 협력사'를 먼저 찾고, 그 회사의 매출처/특수관계자/투자 관계를 각각 비교하라고 하면 multi_anchor_branch_rank.
- 한 기준 회사에 대해 공급/특수관계/투자 등 관계 유형별 1위와 근거를 비교하라고 하면 single_anchor_branch_rank.
- single_anchor_branch_rank의 branch_relations에는 질문에 나온 관계만 넣는다. 질문에 없는 investment/related_party/supplier를 보충하지 않는다.
- 공급 방향이 고객/매출처/납품처이면 SUPPLIES_TO outgoing, 협력사/공급사/벤더/매입이면 incoming, 문맥상 애매하면 auto.
- '잘나가는'은 별도 지표가 없으면 매출액(ifrs-full_Revenue)으로 둔다.
- 2-hop에서는 원래 기준 회사가 다시 답으로 돌아오지 않도록 exclude_original_anchor_from_second=true.
- 협력사/공급사/거래처 후보를 매출액으로 고르는 질문은 first_candidate_policy=operating_counterparty. 단 공통 협력사 교집합 질문은 아래 multi_anchor 규칙을 따른다.
- multi_anchor_branch_rank의 first_relation은 보통 SUPPLIES_TO incoming이며, first_candidate_policy는 보통 default다. 공통 공급사 교집합 자체가 운영 거래 필터다.
- multi_anchor_branch_rank의 branch_relations는 major_customer=SUPPLIES_TO outgoing, related_party=RELATED_PARTY undirected, investment=INVESTS_IN undirected 세 개를 모두 포함한다.
- single_anchor_branch_rank의 branch_relations는 supplier=SUPPLIES_TO incoming 또는 auto, major_customer=SUPPLIES_TO outgoing, related_party=RELATED_PARTY undirected, investment=INVESTS_IN undirected 중 질문에 나온 subset만 포함한다.

질문: {query}"""


def plan(query: str) -> StructuredPlan | None:
    """Return an LLM-derived constrained plan, or None when unsupported/invalid."""
    if not _ENABLED or not _looks_structured(query):
        return None
    data = _invoke_llm(query)
    return coerce_plan(data, query)
