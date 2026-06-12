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
    name = next((str(fact[k]) for k in _NAME_KEYS if fact.get(k)), "")
    if "value" in fact:
        value: Any = fact["value"]
    else:
        value = fact.get("qota") or fact.get("year") or fact.get("score")
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


def graph_search_node(state: dict[str, Any]) -> dict[str, Any]:
    """polaris-backend LangGraph 그래프 노드 — 공개 진입점.

    입력 state 키: reconstructed_query (str)
    출력 state delta: graph_facts / graph_paths / graph_provenance
    """
    query = (state.get("reconstructed_query") or "").strip()
    empty = {"graph_facts": [], "graph_paths": [], "graph_provenance": []}
    if not query:
        return empty

    runner = build_default_runner()
    try:
        out = runner.run({"query": query})
    except Exception:
        return empty

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

    return {
        "graph_facts": facts,
        "graph_paths": paths,
        "graph_provenance": provenance,
    }
