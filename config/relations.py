"""그래프 관계 타입 + 질의 어휘 단어집 로더(SSOT).

관계 9종의 가중치·한글라벨·legacy매핑·망여부, 그리고 랭킹/매크로/지표/포커스 어휘는
사람이 읽고 고치는 단어집 config/relations.json 한 곳에 정의한다. 이 모듈은 그 JSON 을
읽어 파생 맵·함수로 제공하는 얇은 로더다 — 코드에 단어를 하드코딩하지 않는다.

이전엔 관계어가 ppr.py/schema.py/serialize.py/search.py/extract_helpers.py 등에 흩어져
부분집합·불일치·stale 가 생겼고, LLM 이 코드 분기를 볼 수 없었다. 이제 relations.json 이
단일 출처고, 프롬프트는 render_chain_relation_vocab() 로 이 단어집을 LLM 에 주입한다.

JSON relations[] 필드:
  type        Neo4j 관계 타입(엣지)
  weight      PPR 관련성 전파 가중치 (ppr.py)
  ko_label    패널/답변용 한글 라벨 (serialize.py)
  legacy_type adapt_to_legacy 의 UnifiedResult.type (schema.py)
  is_network  회사↔회사 사업관계 = 패널 망 엣지(schema.py). 속성성 관계(임원·제품·기술)는 False.
  aliases     이 관계를 가리키는 한글 관련어(LLM 참고용 단어집).
  desc        방향/의미 설명(프롬프트 주입용).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_VOCAB_PATH = Path(__file__).with_name("relations.json")
with _VOCAB_PATH.open("r", encoding="utf-8") as _f:
    _VOCAB = json.load(_f)


@dataclass(frozen=True)
class Relation:
    type: str
    weight: float
    ko_label: str
    legacy_type: str
    is_network: bool
    aliases: tuple[str, ...] = ()
    desc: str = ""


# 가중치 근거(ppr): 지배·지분(3.0) > 투자·공급(2.5) > 임원·제품·기술(1.5) > 특수관계(1.0)
# > 겸직(0.5). relations.json 배열 순서 = 기존 ppr._REL_WEIGHT 순서(DOMAIN_RELS 바이트 동일 보장).
RELATIONS: list[Relation] = [
    Relation(
        type=r["type"], weight=r["weight"], ko_label=r["ko_label"],
        legacy_type=r["legacy_type"], is_network=r["is_network"],
        aliases=tuple(r.get("aliases", ())), desc=r.get("desc", ""),
    )
    for r in _VOCAB["relations"]
]

# ── 파생 맵 (소비처는 이걸 import) ─────────────────────────────────
REL_WEIGHT: dict[str, float] = {r.type: r.weight for r in RELATIONS}
REL_LABELS: dict[str, str] = {r.type: r.ko_label for r in RELATIONS}
REL_TO_LEGACY_TYPE: dict[str, str] = {r.type: r.legacy_type for r in RELATIONS}
NETWORK_REL_TYPES: frozenset[str] = frozenset(r.type for r in RELATIONS if r.is_network)
DOMAIN_RELS: list[str] = [r.type for r in RELATIONS]  # PPR/관련성 전파 대상(전체 도메인 관계)

# 비정형 추출(extract_helpers)이 만드는 엣지 타입. 정형 관계(지분·지배·투자·임원·겸직)는
# 구조화 로더가 따로 적재하므로 추출 허용 목록에는 빠진다. chunk→object 출처 엣지 hasObject 포함.
INGEST_EDGE_TYPES: frozenset[str] = frozenset(_VOCAB["ingest_edge_types"])

# ── 체인/복합 관계 단어집 (LLM 프롬프트 주입용) ────────────────────
# 한 회사가 오르면 이익이 전파되는 경로(체인) 후보 관계. COMPOSITE_CONCEPTS 는 '수혜'처럼
# 단일 관계로 못박을 수 없는 질의어를 여러 관계의 합집합으로 펼치도록 LLM 에 알려준다.
CHAIN_RELATION_TYPES: tuple[str, ...] = tuple(_VOCAB["chain_relation_types"])
COMPOSITE_CONCEPTS: dict[str, dict] = _VOCAB["composite_concepts"]


def render_chain_relation_vocab() -> str:
    """체인/라우터 LLM 프롬프트용 관계 단어집 텍스트. 관계 어휘의 단일 출처(relations.json).

    예전엔 관계어가 router._SYSTEM_PROMPT 하드코딩·search.FOCUS_KEYWORD_GROUPS·planner 의
    키워드 분기에 흩어져 LLM 이 볼 수 없었다. 이 함수가 단어집을 사람이 읽는 텍스트로 렌더해
    프롬프트에 주입한다.
    """
    by_type = {r.type: r for r in RELATIONS}
    lines = ["허용 relation (각 관계의 한글 관련어 = 단어집):"]
    for rtype in CHAIN_RELATION_TYPES:
        r = by_type[rtype]
        desc = r.desc or r.ko_label
        lines.append(f"- {r.type}: {desc} (관련어: {'/'.join(r.aliases)})")
    for term, spec in COMPOSITE_CONCEPTS.items():
        lines.append("")
        lines.append(f"'{term}'(관련어: {'/'.join(spec['aliases'])})은(는) "
                     f"단일 관계가 아니라 복합 신호다. {spec['note']}")
    return "\n".join(lines)


# ── 랭킹 의도 어휘 단어집 ─────────────────────────────────────────
# 예전엔 같은 단어가 여러 플래너에 중복 정의돼 한쪽만 고치면 어긋났다. 단일 출처(JSON).
RANK_TERMS: tuple[str, ...] = tuple(_VOCAB["rank_terms"])
# "최대주주/대주주" 의 '최대'·'대'는 랭킹이 아니라 지분구조 관계어다. RANK_TERMS 에 "최대"가
# 있어 "삼성전자 최대주주는?" 이 랭킹으로 오분류되지 않게, 판정 전 이 복합어를 먼저 제거한다.
_RANK_NONINTENT_COMPOUNDS: tuple[str, ...] = tuple(_VOCAB["rank_nonintent_compounds"])


def has_rank_intent(query: str) -> bool:
    """질의에 랭킹 의도(가장/최대/1위 …)가 있는가. '최대주주' 같은 복합어는 랭킹이 아님.

    planner.plan·graph_mode._looks_structured·cypher_executor SQL 랭킹이 공유한다. 단순
    `term in query` 가 아니라 이 함수를 거쳐야 '최대주주'의 '최대'가 랭킹으로 새지 않는다.
    """
    q = " ".join((query or "").split())
    if not q:
        return False
    for compound in sorted(_RANK_NONINTENT_COMPOUNDS, key=len, reverse=True):
        q = q.replace(compound, " ")
    return any(t in q for t in RANK_TERMS)


# ── 랭킹 지표 단어집. 질의 어휘 → fin_metric account_id. 순서가 우선순위(영업이익이 매출보다 먼저).
_METRIC_BY_TERM: tuple[tuple[tuple[str, ...], str], ...] = tuple(
    (tuple(terms), metric_id) for terms, metric_id in _VOCAB["metric_by_term"]
)


def metric_for_query(query: str) -> str | None:
    """질의 어휘에서 랭킹 지표(account_id) 해소. 없으면 None.

    has_rank_intent 가 '줄세울 의도'를, 이 함수가 '무엇으로 줄세울지'를 답한다. text2cypher
    SQL 랭킹 후처리(cypher_executor.rank_results)와 결정적 planner 가 같은 어휘를 공유한다.
    """
    q = query or ""
    for terms, metric_id in _METRIC_BY_TERM:
        if any(t in q for t in terms):
            return metric_id
    return None


# 지표 account_id → 한글 라벨. planner 의 랭킹 근거 문구("매출액 기준 랭킹" 등)에 쓴다.
_METRIC_LABELS: dict[str, str] = dict(_VOCAB.get("metric_labels", {}))


def metric_label_for(metric_id: str) -> str:
    """지표 account_id 의 한글 라벨. 미정의면 account_id 그대로."""
    return _METRIC_LABELS.get(metric_id, metric_id)


# ── 그룹/계열 범위 단어집. "삼성 계열사 중 매출 1위"처럼 그룹 군집 전체를 줄세우는 질문 식별.
GROUP_SCOPE_TERMS: tuple[str, ...] = tuple(_VOCAB["group_scope_terms"])

# ── 매크로/업계 sensemaking cue 단어집. graph_mode._prefilter_mode 가 has_anchor 보다 먼저
# 봐서 "반도체 업종 공급망 전반" 의 '반도체'가 회사명에 매칭돼 EXPLORE 로 새는 것을 막는다.
MACRO_TERMS: tuple[str, ...] = tuple(_VOCAB["macro_terms"])

# ── 재무 지표어 단어집. 엔티티 링킹에서 이 단어들은 '회사명'이 아니라 '랭킹 차원'이다.
# 매처가 풀텍스트 질의에서 이 토큰(독립 토큰)만 제거해 "매출"이 Product 노드로 잡히는 오염을 막는다.
METRIC_TERMS: tuple[str, ...] = tuple(_VOCAB["metric_terms"])

# ── SUPPLIES_TO 랭킹 노이즈 단어집. 적재 시 공급망에 섞이는 금융기관(대주·인수단)은 운영
# 거래상대가 아니므로 랭킹 후보에서만 비파괴적으로 뺀다(structured_executor._passes_noise_gate).
SUPPLY_NOISE_NAME_TERMS: tuple[str, ...] = tuple(_VOCAB.get("supply_noise_name_terms", ()))

# ── 회사명 별칭 단어집. 근거 청크 본문이 노드명과 표기가 달라도(에스케이↔SK 음역, 하이닉스
# 약칭 등) 같은 회사로 매칭되게 한다(structured_executor._name_variants). 코드 하드코딩 대신 JSON.
_CORP_ALIASES: dict = _VOCAB.get("corp_name_aliases", {})
_PREFIX_TRANSLITERATIONS: tuple[tuple[str, str], ...] = tuple(
    (a, b) for a, b in _CORP_ALIASES.get("prefix_transliterations", ())
)
_SUBSTRING_SHORTCUTS: tuple[str, ...] = tuple(_CORP_ALIASES.get("substring_shortcuts", ()))


def corp_name_variants(base: str) -> set[str]:
    """정규화된 회사명 base 의 별칭 변형 집합(2자 미만 변형은 버린다).

    prefix_transliterations: 접두 음역 쌍(에스케이↔sk)을 양방향으로 치환해 변형을 추가.
    substring_shortcuts: 약칭(하이닉스)이 포함되면 그 약칭 자체도 변형으로 추가.
    단어집(config/relations.json)을 SSOT 로 삼아 회사 추가 시 코드 수정이 필요 없게 한다.
    """
    variants = {base} if base else set()
    for a, b in _PREFIX_TRANSLITERATIONS:
        if base.startswith(a):
            variants.add(b + base[len(a):])
        if base.startswith(b):
            variants.add(a + base[len(b):])
    for s in _SUBSTRING_SHORTCUTS:
        if s in base:
            variants.add(s)
    return {v for v in variants if len(v) >= 2}
