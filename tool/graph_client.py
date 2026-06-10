# tools/graph_client.py
from typing import List, Dict, Any


class GraphQueryError(Exception):
    """LLM이 생성한 쿼리가 무효하거나, 그래프DB 실행에서 실패"""
    pass


def execute_cypher_query(query: str) -> List[Dict[str, Any]]:
    """
    실제 구현 전 mock. 
    - 입력: Cypher (str)
    - 출력: row 리스트. row는 노드/관계/스칼라가 섞인 dict.
      예) {"path": [{"corp_code":"...","name":"..."}, {"rel":"IS_MAJOR_SHAREHOLDER_OF","qota_rt":42.5}, {...}], "rcept_no": "..."}
    
    빈 쿼리 / 키워드 없는 쿼리 → GraphQueryError.
    """
    if not query or "MATCH" not in query.upper():
        raise GraphQueryError(f"invalid cypher: {query!r}")
    print(f"[Tool: graph_client] CYPHER = {query}")
    # mock row 한두 개
    return [{
        "path": [
            {"corp_code": "00126380", "name": "삼성전자"},
            {"rel": "IS_SUBSIDIARY_OF", "qota_rt": 42.5},
            {"corp_code": "00164742", "name": "삼성디스플레이"},
        ],
        "rcept_no": "20250515000001",
    }]
    
    