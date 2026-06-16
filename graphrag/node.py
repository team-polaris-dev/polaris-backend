"""LangGraph 진입점. AgentState → search → 신·구 키 동시 emit (셰임).

Issue #17 후속에서 nodes/rag.py의 합성기가 제거되고 gen이 graph_facts를
직접 포매팅하므로, legacy 키(graph_facts/paths/provenance) 셰임 유지가 필수.
"""
from __future__ import annotations

import logging

from tool.graph_client import neo4j_driver
from graphrag.schema import adapt_to_legacy
from graphrag.search import search

log = logging.getLogger(__name__)

_preflight_done = False


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
