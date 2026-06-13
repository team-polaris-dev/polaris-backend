"""Neo4j 접속 유틸 — 그래프 RAG 노드 전용.

POLARIS 3-DB 중 그래프 노드가 쓰는 Neo4j 만 남김. Mariadb/Qdrant/Ollama 관련
함수·import 는 본 패키지가 그래프 RAG 전용으로 추려진 후 제거됨.

이식 후 polaris-backend 단일 출처(`tool/graph_client.py` 또는 pydantic-settings)
로 통합 권장 — 접속정보 이중 관리 금지.
"""
from __future__ import annotations

import os

from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "polaris_dev_only")


def neo4j_driver():
    d = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    d.verify_connectivity()
    return d
