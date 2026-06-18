"""GraphRAG Global Search — 커뮤니티 요약 map-reduce.

매크로/업계/주제형 질문(intent="global")의 검색측. Cypher 생성이 아니라,
인덱스 시점에 미리 만들어둔 Community 노드(군집별 LLM 요약)를 읽어 관련 군집을
고른다. graphrag.node.graph_search_node 가 intent="global" 일 때 global_search()
를 호출해 community_results 로 넘기고, gen 노드가 이를 종합한다.

선택 규칙: 재구성 질의에 멤버사 이름이 등장하는 군집만 고른다. 하나도 안 걸리면
(폭넓은 "업계 전체" 류 질문) 전체 군집을 포함한다(군집 ~5개라 부담 없음).
Neo4j 가 끊겨 있으면 빈 리스트로 degrade(파이프라인 보호).
"""
from __future__ import annotations

import json
import logging

from tool.graph_client import neo4j_driver

log = logging.getLogger(__name__)


def _load_communities() -> list[dict]:
    """Neo4j 의 모든 Community 노드를 dict 리스트로 로드. 실패 시 []."""
    try:
        with neo4j_driver.session() as s:
            rows = s.run(
                "MATCH (c:Community) "
                "RETURN c.cluster_id AS cluster_id, c.summary AS summary, "
                "       c.size AS size, c.member_names AS member_names, "
                "       c.anchor_names AS anchor_names, c.edge_dist AS edge_dist "
                "ORDER BY c.size DESC"
            ).data()
        return rows or []
    except Exception as e:
        log.warning("global_search: Community 로드 실패(Neo4j 불가?): %s", e)
        return []


def _matches(query: str, community: dict) -> bool:
    """재구성 질의에 이 군집의 멤버명/대표명이 하나라도 등장하면 True."""
    members = (community.get("member_names") or []) + (community.get("anchor_names") or [])
    for nm in members:
        if not nm:
            continue
        # '삼성전자(주)' 같은 정식명·접미사 흔들림 흡수를 위해 양방향 포함 검사.
        if nm in query or query.find(nm.replace("(주)", "").replace("주식회사", "").strip()) != -1:
            return True
    return False


def _to_unified(community: dict) -> dict:
    """Community dict → UnifiedResult(type='community')."""
    anchors = community.get("anchor_names") or community.get("member_names") or []
    cid = community.get("cluster_id")
    name = anchors[0] if anchors else f"군집 {cid}"
    edge_dist = community.get("edge_dist")
    if isinstance(edge_dist, str):
        try:
            edge_dist = json.loads(edge_dist)
        except Exception:
            edge_dist = {}
    return {
        "type": "community",
        "code": str(cid),
        "name": str(name),
        "value": community.get("summary") or "",
        "extra": {
            "size": community.get("size"),
            "edge_dist": edge_dist,
            "member_names": community.get("member_names") or [],
        },
        "source": f"community:{cid}",
    }


def global_search(query: str) -> list[dict]:
    """질의 → 관련 Community 요약 UnifiedResult 리스트.

    매칭 군집이 없으면 전체 군집을 반환(폭넓은 매크로 질문). Neo4j 불가 시 [].
    """
    communities = _load_communities()
    if not communities:
        return []

    selected = [c for c in communities if _matches(query or "", c)]
    if not selected:
        # 특정 군집이 안 걸리는 광범위 질문 → 전체(군집 수가 적어 OK).
        selected = communities

    return [_to_unified(c) for c in selected]
