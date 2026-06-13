"""polaris-backend 그래프 RAG 노드 (LangGraph node).

polaris-backend 의 `nodes/rag.py` 가 그대로 import 하는 공개 함수 한 개:

    def graph_search_node(state: AgentState) -> dict[str, Any]

I/O 계약 (`polaris-backend/core/state.py:AgentState` 기준 — 변경 금지):

  입력  state["reconstructed_query"]: str
  출력  graph_facts:      List[UnifiedResult]   # {type, code, name, value, extra, source}
        graph_paths:      List[List[str]]
        graph_provenance: List[str]              # rcept_no

내부적으로 `GraphAgentRunner.run({"query": ...})` 를 호출하고, 런너의 풍부한
출력(13키)을 polaris-backend 가 받는 3키로 축약·정규화한다.

⚠ 런너가 산출하지만 polaris-backend AgentState 에 슬롯이 없어서 **버리는** 키:
   graph_chunk_ids · template_used · cypher_executed · graph_self_check
   · graph_latency_ms · graph_errors · actions_taken · intent · route · stage
   · graph_rejected
   필요해지면 AgentState 에 필드 추가 후 이 모듈도 함께 갱신.
"""
from __future__ import annotations

from typing import Any

from ...neo4j_client import neo4j_driver
from ...query_shape import classify as classify_shape
from ..graph.runner import GraphAgentRunner


_runner: GraphAgentRunner | None = None


def build_default_runner() -> GraphAgentRunner:
    """싱글톤 러너 (드라이버·LLM·분류기 1회 초기화)."""
    global _runner
    if _runner is None:
        _runner = GraphAgentRunner()
    return _runner


# ── facts(runner 내부 shape) → UnifiedResult(polaris-backend 6키) 정규화 ─────
_CODE_KEYS = ("corp_code", "code", "sub_code", "s_code", "b_code")
_NAME_KEYS = (
    "name", "org", "sub_name", "supplier", "buyer", "person", "holder",
    "investee", "investor", "parent",
)
_USED_TOP = {"type", "rcept_no"} | set(_CODE_KEYS) | set(_NAME_KEYS)

# (시작점 키, 끝점 키, 관계 라벨) — 템플릿 RETURN 별칭과 1:1 대응
_PAIR_RULES = (
    ("supplier", "buyer", "공급(SUPPLIES_TO)"),
    ("investor", "investee", "출자(INVESTS_IN)"),
    ("holder", "org", "주요주주(지분율%)"),
    ("sub_name", "parent", "종속회사(IS_SUBSIDIARY_OF)"),
    ("person", "org", "임원(EXECUTIVE_OF)"),
)


def _normalize_fact(fact: dict[str, Any]) -> dict[str, Any]:
    """런너 fact dict → UnifiedResult(6키).

    fact 키가 의도/템플릿마다 다른 형태(template_used 따라 RETURN 키 상이) 이므로
    best-effort 매핑. 매칭 안 된 키는 모두 `extra` 로 보존.
    """
    fact_type = fact.get("type") or "graph_fact"

    # community 카드는 cluster_id/anchors 가 자연 키 — 우선 매핑.
    if fact_type == "community":
        anchors = fact.get("anchors") or []
        return {
            "type": "community",
            "code": f"cluster_{fact.get('cluster_id', '')}",
            "name": (anchors[0] if anchors else "") or "",
            "value": fact.get("size"),
            "extra": {k: v for k, v in fact.items() if k != "type"},
            "source": "",
        }

    code = next((str(fact[k]) for k in _CODE_KEYS if fact.get(k)), "")

    # 쌍(pair) 관계는 'A → B' 로 양끝을 보존한다 — 한쪽 이름만 남기면 syn 단계
    # (name/value 만 렌더링)에서 상대방·방향·관계종류가 통째로 소실된다.
    name = ""
    rel_label: str | None = None
    for a, b, label in _PAIR_RULES:
        if fact.get(a) and fact.get(b):
            name = f"{fact[a]} → {fact[b]}"
            rel_label = label
            break
    if not name:
        name = next((str(fact[k]) for k in _NAME_KEYS if fact.get(k)), "")
    if not name:
        # LLM Cypher 가 별칭 규칙을 어기고 자유형 키(company_name 등)로 반환한
        # 경우의 폴백 — 이름일 가능성이 높은 첫 문자열 값을 채택.
        deny = {"type", "rcept_no", "relation", "unit", "rationale"}
        name = next(
            (str(v) for k, v in fact.items()
             if k not in deny and isinstance(v, str) and v.strip()),
            "",
        )

    if "value" in fact:
        value: Any = fact["value"]
    else:
        # 내용 있는 필드만 값으로 쓴다. score 는 랭킹 가중치라 값으로 쓰면
        # LLM 이 '연관성 0.84' 같은 의미를 창작하므로 절대 사용 금지.
        detail = next(
            (fact[k] for k in ("qota_rt", "qota", "pos", "relation_type",
                               "relation", "hops", "year")
             if fact.get(k) not in (None, "")),
            None,
        )
        if rel_label and detail is not None:
            value = f"{rel_label} {detail}"
        elif detail is not None:
            value = detail
        else:
            value = rel_label or ""
    if isinstance(value, str):
        value = " ".join(value.split())  # 직책 등의 개행이 syn 라인 포맷을 깨지 않게
    source = str(fact.get("rcept_no") or "")
    extra = {k: v for k, v in fact.items() if k not in _USED_TOP and k != "value"}
    return {
        "type": str(fact_type),
        "code": code,
        "name": name,
        "value": value,
        "extra": extra,
        "source": source,
    }


# ── chunk_ids → rcept_no provenance 해석 ────────────────────────────────────
def _resolve_provenance(
    facts: list[dict[str, Any]],
    chunk_ids: list[str],
) -> list[str]:
    """1) fact 자체에 rcept_no 있으면 채택  2) chunk_id → (:Chunk).rcept_no 조회.

    Neo4j 미가용 시 1)만 사용 (degrade).
    """
    prov: list[str] = []
    seen: set[str] = set()
    for f in facts:
        rn = f.get("rcept_no")
        if rn and rn not in seen:
            seen.add(rn)
            prov.append(str(rn))
    if chunk_ids:
        try:
            with neo4j_driver().session() as s:
                rows = s.run(
                    "MATCH (c:Chunk) WHERE c.chunk_id IN $ids "
                    "RETURN DISTINCT c.rcept_no AS rn",
                    ids=list(chunk_ids),
                ).data()
            for r in rows:
                rn = r.get("rn")
                if rn and rn not in seen:
                    seen.add(rn)
                    prov.append(str(rn))
        except Exception:
            pass
    return prov


# ── 쌍(pair) fact → graph_paths 경로 합성 ────────────────────────────────────
# build_graph(core/serialize.py)는 그래프 패널을 graph_paths 의 홀수 인덱스를
# 관계 토큰으로 보고 REL_LABELS 로 한글화하므로, serialize.REL_LABELS 와 같은
# 정규 토큰을 내보낸다.
_PATH_PAIRS = (
    ("supplier", "buyer", "SUPPLIES_TO"),
    ("investor", "investee", "INVESTS"),
    ("holder", "org", "IS_MAJOR_SHAREHOLDER_OF"),
    ("sub_name", "parent", "IS_SUBSIDIARY_OF"),
    ("person", "org", "EXECUTIVE_OF"),
)


def _pair_path(fact: dict[str, Any]) -> list[str]:
    """쌍 관계 fact → [시작노드, 관계토큰, 끝노드]. 쌍이 아니면 []."""
    for a, b, rel in _PATH_PAIRS:
        if fact.get(a) and fact.get(b):
            return [str(fact[a]), rel, str(fact[b])]
    return []


def graph_search_node(state: dict[str, Any]) -> dict[str, Any]:
    """polaris-backend LangGraph 그래프 노드 — 공개 진입점.

    입력 state 키: reconstructed_query (str)
    출력 state delta: graph_facts / graph_paths / graph_provenance
    """
    query = (state.get("reconstructed_query") or "").strip()
    empty = {"graph_facts": [], "graph_paths": [], "graph_provenance": []}
    if not query:
        return empty

    print(f"🛠️ [GraphRAG]  검색 시뮬레이션 중: {query}")
    try:
        runner = build_default_runner()
        out = runner.run({"query": query})
    except Exception as e:
        print(f"⚠️ [GraphRAG] 실패 → 빈 결과로 degrade: {type(e).__name__}: {e}")
        return empty

    print(
        f"   -> intent={out.get('intent')} route={out.get('route')}"
        f" stage={out.get('stage')} rejected={out.get('rejected')}"
        f" template={out.get('template_used')}"
        f" facts={len(out.get('graph_facts') or [])}"
        f" chunks={len(out.get('graph_chunk_ids') or [])}"
        f" errors={out.get('errors') or []}"
    )

    raw_facts: list[dict[str, Any]] = list(out.get("graph_facts") or [])
    paths: list[list[str]] = list(out.get("graph_paths") or [])
    chunk_ids: list[str] = list(out.get("graph_chunk_ids") or [])

    # global 질의(클러스터/커뮤니티/생태계) — community_cards 결정론 카드 선두 배치.
    # Microsoft GraphRAG global search 의 축소판. 데이터 없으면 자동 비활성화.
    try:
        if classify_shape(query) == "global":
            for h in runner.r.community_cards(query):
                if h.fact_card:
                    raw_facts.insert(0, dict(h.fact_card))
                if h.path:
                    paths.insert(0, h.path)
    except Exception:
        pass

    facts = [_normalize_fact(f) for f in raw_facts]
    provenance = _resolve_provenance(raw_facts, chunk_ids)

    # runner 는 anchor_chunks(stage>=2 / 무facts)에서만 graph_paths 를 채운다.
    # stage-1 에서 템플릿/LLM 으로 쌍 관계 fact 를 찾으면 paths 가 비어, 그래프
    # 패널(graph_paths 만 읽음)이 비게 된다. paths 가 비면 쌍 fact 에서 경로를
    # 합성한다(raw_facts 와 1:1 정렬 → build_graph 의 rcept_no 매칭 유지).
    if not paths:
        paths = [_pair_path(f) for f in raw_facts]

    return {
        "graph_facts": facts,
        "graph_paths": paths,
        "graph_provenance": provenance,
    }
