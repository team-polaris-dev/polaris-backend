"""LangGraph 그래프 RAG 서브에이전트 — polaris-backend 그래프 노드 본체.

공개 진입점: `nodes.graph_agent.graph_search_node` (polaris-backend
`nodes/rag.py` 가 그대로 import 해서 LangGraph 노드로 등록하는 함수).

I/O 계약 = polaris-backend `core/state.py:AgentState`:
  입력  state["reconstructed_query"]: str
  출력  graph_facts: List[UnifiedResult]  ({type, code, name, value, extra, source})
        graph_paths: List[List[str]]
        graph_provenance: List[str]      (rcept_no)
"""
