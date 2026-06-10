from types import SimpleNamespace

import tool.vector_store as vector_store


def test_search_vector_db_normalizes_dataclass_like_rows(monkeypatch):
    class FakeChunk:
        def to_dict(self):
            return {"chunk_id": "c1", "text": "body"}

    def fake_import(name, *args, **kwargs):
        if name == "polaris.retrieve":
            return SimpleNamespace(hybrid_search=lambda query, top_k=10: [FakeChunk()])
        return real_import(name, *args, **kwargs)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)
    assert vector_store.search_vector_db("질문", top_k=1) == [{"chunk_id": "c1", "text": "body"}]


def test_search_vector_db_empty_query():
    assert vector_store.search_vector_db("   ") == []


def test_search_vector_db_loads_backend_env_before_import(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE", raising=False)

    def fake_import(name, *args, **kwargs):
        if name == "polaris.retrieve":
            import os

            assert os.environ["OLLAMA_BASE"] == "http://220.80.16.79:11434"
            return SimpleNamespace(hybrid_search=lambda query, top_k=10: [])
        return real_import(name, *args, **kwargs)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)
    assert vector_store.search_vector_db("질문", top_k=1) == []
