import os
from typing import List, Dict, Any
from neo4j import GraphDatabase

class GraphQueryError(Exception):
    """LLM이 생성한 쿼리가 무효하거나, 그래프DB 실행에서 실패"""
    pass

# 환경 변수 로드 (Docker Compose에서 주입됨)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "polaris_dev_only")

# Neo4j 드라이버 초기화
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def execute_cypher_query(query: str) -> List[Dict[str, Any]]:
    """실제 Neo4j 데이터베이스에 Cypher 쿼리를 실행합니다."""
    print(f"🛠️ [Graph DB] 쿼리 실행 시뮬레이션 중: {query}")
    if not query or "MATCH" not in query.upper():
        raise GraphQueryError(f"invalid cypher: {query!r}")
    

    
    try:
        with neo4j_driver.session() as session:
            result = session.run(query)
            # 결과를 List[Dict] 형태로 변환하여 반환
            return [record.data() for record in result]
    except Exception as e:
        raise GraphQueryError(f"그래프DB 실행 실패: {str(e)}")