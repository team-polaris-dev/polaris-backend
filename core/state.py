# core/state.py
from typing import TypedDict, Annotated, List, Any
try:
    from typing import NotRequired  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import NotRequired  # type: ignore[no-redef]

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def merge_timings(a: dict, b: dict) -> dict:
    """node_timings 리듀서. rdb/vec/graph 가 병렬로 써도 충돌 없이 합치도록 dict 병합."""
    return {**(a or {}), **(b or {})}


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

    final_draft: str

    # GraphRAG rewrite (graphrag/) — Sync Node로 넘기는 신규 키. 옵셔널.
    graph_hits: NotRequired[List[dict]]
    graph_seeds: NotRequired[List[dict]]
    graph_meta: NotRequired[dict]

    # 앞단(Gemini)이 단어집 보고 동봉한 엔티티 식별자 (corp_code/org:er_name/...)
    reconstructed_seeds: NotRequired[List[str]]

    # 계측용 — 파이프라인 시작 시각(perf_counter)과 노드별 소요시간(초). ResultCheck 덤프에서 사용.
    pipeline_started_at: NotRequired[float]
    node_timings: NotRequired[Annotated[dict, merge_timings]]
