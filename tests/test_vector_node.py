import sys
from types import ModuleType


lc_messages = ModuleType("langchain_core.messages")
lc_messages.BaseMessage = object
sys.modules.setdefault("langchain_core", ModuleType("langchain_core"))
sys.modules.setdefault("langchain_core.messages", lc_messages)

lg_message = ModuleType("langgraph.graph.message")
lg_message.add_messages = lambda *args, **kwargs: None
sys.modules.setdefault("langgraph", ModuleType("langgraph"))
sys.modules.setdefault("langgraph.graph", ModuleType("langgraph.graph"))
sys.modules.setdefault("langgraph.graph.message", lg_message)

rdb_client = ModuleType("tool.rdb_client")
rdb_client.execute_sql_query = lambda sql: {"ok": True, "rows": [], "error": "", "sql": sql}
sys.modules.setdefault("tool.rdb_client", rdb_client)

text_to_sql = ModuleType("tool.text_to_sql")
text_to_sql.generate_sql = lambda *args, **kwargs: "SELECT 1"
sys.modules.setdefault("tool.text_to_sql", text_to_sql)

import nodes.rag as rag


def test_vector_node_formats_rows(monkeypatch):
    def fake_search(query, top_k=10):
        return [
            {
                "chunk_id": "abc123",
                "text": "삼성전자의 환경 리스크 관련 본문",
                "corp_name": "삼성전자",
                "year": 2024,
                "title": "사업보고서",
                "score": 0.12345,
            }
        ]

    monkeypatch.setattr(rag, "search_vector_db", fake_search)
    out = rag.vector_search_node({"reconstructed_query": "삼성전자 환경 리스크", "messages": []})
    text = out["search_results"][0]
    assert "[Vector 검색]" in text
    assert "삼성전자" in text
    assert "abc123" in text
    assert "0.1235" in text


def test_vector_node_graceful_on_error(monkeypatch):
    def boom(query, top_k=10):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(rag, "search_vector_db", boom)
    out = rag.vector_search_node({"reconstructed_query": "테스트", "messages": []})
    assert "오류" in out["search_results"][0]
    assert "qdrant down" in out["search_results"][0]


def test_vector_node_empty_result(monkeypatch):
    monkeypatch.setattr(rag, "search_vector_db", lambda query, top_k=10: [])
    out = rag.vector_search_node({"reconstructed_query": "테스트", "messages": []})
    assert "해당 데이터 없음" in out["search_results"][0]
