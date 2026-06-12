"""POLARIS Graph RAG 패키지.

polaris-backend `nodes/rag.py` 가 import 하는 공개 진입점은
`graphrag.langgraph_app.nodes.graph_agent.graph_search_node` 한 개.

설계 SSOT: docs/DBdocs/03_neo4j.md · 04_graphrag.md.
"""
from .langgraph_app.nodes.graph_agent import graph_search_node

__all__ = ["graph_search_node"]
