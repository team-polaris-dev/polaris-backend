# tools/rdb_client.py

def execute_sql_query(query: str) -> str:
    """
    RDB에 SQL을 실행하여 정형 데이터를 가져오는 목업 함수
    """
    print(f"\n[🔧 Tool: RDB] SQL 실행 중... 쿼리: {query}")
    
    # 임시 목업 데이터 반환
    return """
    임시목업 데이터
    """