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


def _anchor_corp_codes(graph_seeds: list[dict]) -> list[str]:
    """로컬 시드에서 corp_code 앵커만 추출(순서·중복제거). DRIFT 군집 선택용.

    Seed.key_type=='corp_code' 인 시드의 key_value 가 실제 corp_code(schema.Seed).
    """
    codes: list[str] = []
    seen: set[str] = set()
    for sd in graph_seeds or []:
        if sd.get("key_type") != "corp_code":
            continue
        code = sd.get("key_value")
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _assemble_local(out: dict) -> dict:
    """search() 출력 → Local Search 결과 dict(신규 키 + legacy 셰임 키).

    Issue #17 이후 result_check/gen 이 legacy 키(graph_facts/paths/provenance)를 직접
    보므로 셰임을 함께 emit 한다. ctx 경로와 global-앵커(DRIFT) 경로가 공유한다.
    """
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
        # 패널 엣지별 출처 — graph_paths 와 행 정렬(serialize.build_graph 가 i 로 읽음).
        "graph_path_sources": legacy["path_sources"],
        "graph_path_chunks": legacy["path_chunks"],
    }


def _attach_communities(result: dict, query: str, out: dict) -> None:
    """DRIFT: 로컬 앵커가 속한 군집의 map-reduce 부분답을 result 에 결합(in-place).

    out["graph_seeds"] 의 corp_code 앵커가 속한 군집만 map-reduce 한다. 앵커가 없으면
    (엔티티 미해소) 아무것도 붙이지 않는다 — 노이즈 0. ctx·global-앵커 경로 공유.
    """
    anchors = _anchor_corp_codes(out["graph_seeds"])
    if not anchors:
        return
    community_results = global_search(query, anchor_corp_codes=anchors)
    if community_results:
        print(f"🌐 [GraphRAG/DRIFT] 앵커 군집 {len(community_results)}개 결합")
        result["community_results"] = community_results


def graph_search_node(state: dict) -> dict:
    """GraphRAG 단일 진입점. intent 로 Local/Global 을 분기하고 DRIFT 로 결합한다.

    - global(매크로/업계): 먼저 로컬 search() 로 corp_code 앵커 해소를 시도한다.
      · 앵커가 잡히면(라우터는 global 로 봤지만 실은 구체 엔티티 질문) → ctx 와 동일한
        DRIFT 융합(로컬 graph_facts + 앵커 군집 community_results)을 반환(대칭 DRIFT).
      · 앵커가 없으면(순수 매크로) → 커뮤니티 요약 map-reduce 만(community_results).
    - 그 외(ctx/local): 시드 매칭→멀티홉/PPR 순회(Local Search) → graph_facts 등 +
      앵커 군집 DRIFT 결합.
    둘 다 같은 GraphRAG 노드가 소유하므로 별도 플로우 노드를 두지 않는다.
    """
    if state.get("intent") == "global":
        # global 은 ctx 를 건너뛰므로(core.graph) reconstructed_query 가 비어 있다.
        # 이때 원문 질문으로 폴백해 시드 매칭·커뮤니티 멤버명 매칭을 살린다.
        query = state.get("reconstructed_query") or _last_human_text(state)
        _preflight()
        # ctx 를 건너뛴 경로라 upstream_seeds 가 없다 — 질의 텍스트로만 시드 해소.
        out = search(query)
        anchors = _anchor_corp_codes(out["graph_seeds"])
        if anchors:
            # 대칭 DRIFT: 엔티티가 해소되면 로컬 사실 + 앵커 군집을 함께 반환.
            result = _assemble_local(out)
            _attach_communities(result, query, out)
            print(f"🌐 [GraphRAG/DRIFT-global] 앵커 {len(anchors)}개 + 로컬 사실 결합")
            return result
        # 순수 매크로: 군집 요약 map-reduce 만.
        results = global_search(query)
        print(f"🌐 [GraphRAG/Global] 커뮤니티 {len(results)}개 선택")
        return {"community_results": results}

    _preflight()
    query = state.get("reconstructed_query") or ""
    upstream = state.get("reconstructed_seeds") or []

    out = search(query, upstream_seeds=upstream)
    result = _assemble_local(out)
    _attach_communities(result, query, out)
    return result
