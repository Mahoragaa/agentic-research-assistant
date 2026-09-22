"""
Agent State Definition

Defines the typed state schema for the LangGraph ReAct agent.
Uses the `add_messages` reducer for proper message history management.
"""

from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State schema for the Agentic Research Assistant.

    Attributes:
        messages:           Conversation history with the add_messages reducer
                            that correctly appends new interactions.
        active_collection:  Name of the ChromaDB collection currently being queried.
                            Set when the user selects a topic / ingested collection.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    active_collection: str
