"""GraphRAG 하이브리드 검색 패키지.

진입점:
  graph_search_node(state)  → GraphRAG 단일 진입점. intent 로 분기:
                              local  = graph_facts / paths / provenance
                              global = community_results (커뮤니티 요약)
"""
from graphrag.node import effective_query, graph_search_node

__all__ = ["effective_query", "graph_search_node"]
