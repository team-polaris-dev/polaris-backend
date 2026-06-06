# core/state.py
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_preferences: dict
    intent: str
    reconstructed_query: str
    search_plan: List[str]
    search_results: List[str]
    synthesized_info: str
    final_draft: str