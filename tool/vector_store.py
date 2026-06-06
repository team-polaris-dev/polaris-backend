# tools/vector_store.py

def search_vector_db(query: str, top_k: int = 3) -> str:
    """
    Vector DB에 임베딩 기반 유사도 검색을 수행하는 목업 함수
    실전: ChromaDB, FAISS, Pinecone 등 연동
    """
    print(f"\n[🔧 Tool: Vector] 문서 검색 중... 질문: '{query}' (top_k={top_k})")
    
    # 임시 목업 문서 반환
    return """
    [Vector 유사 문서 결과]
    """