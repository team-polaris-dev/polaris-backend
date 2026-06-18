"""LangGraph 진입점. AgentState → search → 신·구 키 동시 emit (셰임).

Issue #17 후속에서 nodes/rag.py의 합성기가 제거되고 gen이 graph_facts를
직접 포매팅하므로, legacy 키(graph_facts/paths/provenance) 셰임 유지가 필수.
"""
from __future__ import annotations

import logging

from tool.graph_client import neo4j_driver
from graphrag.schema import adapt_to_legacy
from graphrag.search import search
from graphrag.global_search import global_search

log = logging.getLogger(__name__)

_preflight_done = False


def _last_human_text(state: dict) -> str:
    """state.messages 에서 마지막 사용자(human) 메시지 본문을 반환. 없으면 ''."""
    for msg in reversed(state.get("messages") or []):
        if getattr(msg, "type", "") == "human":
            return str(msg.content)
    return ""


def _preflight() -> None:
    """1회성 DB 사전 점검. entity_fulltext 인덱스 없으면 경고만(검색은 빈 결과로 degrade)."""
    global _preflight_done
    if _preflight_done:
        return
    _preflight_done = True
    try:
        with neo4j_driver.session() as s:
            row = s.run(
                "SHOW INDEXES YIELD name WHERE name = 'entity_fulltext' "
                "RETURN count(*) AS c"
            ).single()
        if not row or row["c"] == 0:
            log.warning(
                "graphrag: entity_fulltext index missing — "
                "run `python -m pipeline_scripts.graph.setup_fulltext_index`"
            )
    except Exception as e:
        log.warning("graphrag preflight skipped (Neo4j unreachable?): %s", e)


def graph_search_node(state: dict) -> dict:
    """GraphRAG 단일 진입점. intent 로 Local/Global 을 분기한다.

    - global(매크로/업계): 인덱스 시점에 만들어둔 Community 요약 map-reduce
      (Global Search) → community_results.
    - 그 외: 시드 매칭→멀티홉/PPR 순회(Local Search) → graph_facts 등.
    둘 다 같은 GraphRAG 노드가 소유하므로 별도 플로우 노드를 두지 않는다.
    """
    # 글로벌은 Cypher 순회가 아니라 커뮤니티 요약을 읽어 종합하므로 분기.
    if state.get("intent") == "global":
        # global 은 ctx 를 건너뛰므로(core.graph) reconstructed_query 가 비어 있다.
        # 이때 원문 질문으로 폴백해 커뮤니티 멤버명 매칭을 살린다.
        query = state.get("reconstructed_query") or _last_human_text(state)
        results = global_search(query)
        print(f"🌐 [GraphRAG/Global] 커뮤니티 {len(results)}개 선택")
        return {"community_results": results}

    _preflight()
    query = state.get("reconstructed_query") or ""
    upstream = state.get("reconstructed_seeds") or []

    out = search(query, upstream_seeds=upstream)
    legacy = adapt_to_legacy(out["graph_hits"])

    return {
        # 신규
        "graph_hits": out["graph_hits"],
        "graph_seeds": out["graph_seeds"],
        "graph_meta": out["graph_meta"],
        # 셰임 (Issue #17 result_check/gen이 이 키를 직접 봄)
        "graph_facts": legacy["facts"],
        "graph_paths": legacy["paths"],
        "graph_provenance": legacy["provenance"],
    }
