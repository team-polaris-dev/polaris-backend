# core/state.py
from typing import TypedDict, Annotated, List,Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class UnifiedResult(TypedDict):
    type: str       # subsidiary / executive / fin_metric / rdb_row / vec_chunk ...
    code: str       # corp_code
    name: str
    value: Any      # 값의 형태가 다양할 수 있으므로 Any 사용
    extra: dict
    source: str     # rcept_no 또는 chunk_id

# [제안 1 반영] State 키는 용도별로 분리 유지
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_preferences: dict
    intent: str
    reconstructed_query: str
    search_plan: List[str]
    
    # 분리된 State 키들에 통일된 UnifiedResult 규격 적용
    rdb_results: List[UnifiedResult]
    vec_results: List[UnifiedResult]
    graph_facts: List[UnifiedResult]
    
    # 기타 그래프 관련 키 분리 유지
    graph_paths: List[List[str]]
    graph_provenance: List[str]  # 근거 rcept_no
    
    synthesized_info: str
    final_draft: str