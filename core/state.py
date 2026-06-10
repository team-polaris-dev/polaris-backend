# core/state.py
import operator
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_preferences: dict
    intent: str
    reconstructed_query: str
    search_plan: List[str]
    search_results: Annotated[List[str], operator.add]
    synthesized_info: str
    is_sufficient: bool
    final_draft: str
