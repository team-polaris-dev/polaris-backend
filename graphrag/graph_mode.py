"""그래프 모드(역할) 키워드 분류기 — 통합 라우터(graphrag.router)의 결정적 폴백 소스.

질문마다 "그래프가 무엇을 할지/말지"를 고른다: macro(커뮤니티 sensemaking)/relation_rank
(노드지표 랭킹)/relation_explore(관계망 펼치기)/relation_only(랭킹 불가 degrade)/silent
(순수 속성 → 침묵). LLM 1차 판정은 router 가 맡고, 이 모듈은 router 가 프리필터(_is_silent)·
LLM off·다운·무효일 때 호출하는 결정적 키워드 백스톱만 제공한다(자체 LLM 호출 없음).
이 라우팅은 라이브러리(Text2CypherRetriever)가 하지 않는 오케스트레이션 결정이라 도메인
레이어로 남긴다(어느 서브시스템이 답하는가: 그래프 vs 커뮤니티 vs 침묵).
"""
from __future__ import annotations

from enum import Enum

from config.relations import MACRO_TERMS, RANK_TERMS, has_rank_intent

_RANK_TERMS = RANK_TERMS  # config.relations SSOT
_RELATION_TERMS = (
    "거래", "공급", "납품", "협력", "벤더", "매출처", "고객", "관련",
    "특수관계", "관계자", "계열", "주주", "지분", "자회사", "종속", "투자",
)
# 순수 노드 속성(스칼라) 단어. 관계어·랭크어가 같이 없으면 그래프가 침묵할 신호.
_ATTRIBUTE_ONLY_TERMS = (
    "매출", "영업이익", "순이익", "당기순이익", "자산", "부채", "자본",
    "주가", "시가총액", "시총", "직원수", "종업원", "임직원", "영업이익률",
    "설립", "본사", "소재지", "배당", "주식수",
)
# 매크로/업계 질문 cue (config.relations SSOT).
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
