import nodes.rag as rag


def test_node_graceful_when_llm_raises(monkeypatch):
    """Ollama 등 LLM 호출이 예외를 던져도 노드가 죽지 않고 결과를 반환한다."""
    def boom(*a, **k):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(rag, "generate_sql", boom)
    out = rag.rdb_search_node({"reconstructed_query": "테스트", "messages": []})
    assert "search_results" in out
    assert "오류" in out["search_results"][0]


def test_node_retries_once_on_failure(monkeypatch):
    """실행 실패 시 generate_sql/execute 가 정확히 2번(최초+재시도) 호출된다."""
    calls = {"gen": 0, "exec": 0}

    def fake_gen(question, read_run_id=None, error_feedback=None):
        calls["gen"] += 1
        return "SELECT 1"

    def fake_exec(sql, max_rows=50):
        calls["exec"] += 1
        return {"ok": False, "rows": [], "error": "boom", "sql": sql}

    monkeypatch.setattr(rag, "generate_sql", fake_gen)
    monkeypatch.setattr(rag, "execute_sql_query", fake_exec)
    rag.rdb_search_node({"reconstructed_query": "테스트", "messages": []})
    assert calls["gen"] == 2
    assert calls["exec"] == 2
