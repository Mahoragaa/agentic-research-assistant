"""
LangGraph ReAct Agent

Builds a StateGraph-based ReAct agent that dynamically routes queries
to one of three tools: retrieve, web_search, or clarify.

The agent uses Google Gemini 1.5 Flash for reasoning and tool calling,
with streaming support for real-time UI updates.
"""

import logging
from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from app.config import settings
from app.agent.state import AgentState
from app.agent.tools.retrieve import retrieve
from app.agent.tools.web_search import web_search
from app.agent.tools.clarify import clarify

logger = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Research Assistant specialized in analyzing academic papers from arXiv. You help researchers understand, compare, and explore scientific literature.

You have access to three tools. Choose the RIGHT tool for each query:

## Tool Selection Guide

### 1. `retrieve` — Search Ingested Papers
Use when the user asks about:
- Specific methods, algorithms, or techniques described in papers
- Experimental results, benchmarks, or comparisons from papers
- Definitions, formulas, or theoretical concepts from papers
- "What does paper X say about..." or "How does the method work..."
IMPORTANT: Always pass the `collection_name` parameter from the active collection.

### 2. `web_search` — Search the Web
Use when the user asks about:
- Recent news or developments NOT in the ingested papers
- Author profiles, affiliations, h-index, or other publications
- Comparisons with work outside the ingested collection
- Current state-of-the-art, leaderboards, or benchmarks
- Background context, tutorials, or explanations of general concepts

### 3. `clarify` — Ask for Clarification
Use when:
- The query is too vague (e.g., "tell me about stuff")
- It is ambiguous which tool to use
- The user hasn't specified what papers/topic they are asking about
- More context would significantly improve the answer quality

## Response Guidelines
- Always cite your sources. For paper results, include the paper title, authors, and arXiv ID.
- For web results, mention the source.
- Be concise but thorough. Prioritize accuracy over verbosity.
- If retrieval returns low-similarity results (< 0.3), acknowledge uncertainty.
- You may call multiple tools in sequence if needed (e.g., retrieve then web_search for comparison).
"""

# ── Tool definitions ──────────────────────────────────────────────────

TOOLS = [retrieve, web_search, clarify]

# ── LLM setup ─────────────────────────────────────────────────────────


def _get_llm():
    """Create the Gemini LLM with tools bound."""
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.1,
        max_retries=2,
    )
    return llm.bind_tools(TOOLS)


# ── Graph Nodes ───────────────────────────────────────────────────────


def agent_node(state: AgentState) -> dict:
    """
    The reasoning node: calls the LLM with the current message history.
    The LLM decides whether to call a tool or respond directly.
    """
    llm = _get_llm()
    messages = state["messages"]

    # Inject system prompt and active collection context
    active_collection = state.get("active_collection", "(no collection selected)")
    system_msg = SystemMessage(
        content=SYSTEM_PROMPT + f"\n\nCurrently active collection: `{active_collection}`"
    )

    response = llm.invoke([system_msg] + list(messages))
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Conditional edge: route to tools if the LLM made tool calls,
    otherwise end the conversation turn.
    """
    last_message = state["messages"][-1]

    # Check if the LLM wants to call tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# ── Graph Construction ────────────────────────────────────────────────

# Maximum iterations to prevent infinite loops
MAX_ITERATIONS = 10


def build_graph():
    """
    Build and compile the LangGraph ReAct agent.

    Returns:
        A compiled StateGraph ready for invocation.
    """
    # Create tool node
    tool_node = ToolNode(TOOLS)

    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Set entry point
    graph.set_entry_point("agent")

    # Add conditional edge from agent
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # Tools always route back to agent (ReAct loop)
    graph.add_edge("tools", "agent")

    # Compile with memory checkpointer
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# ── Convenience functions ─────────────────────────────────────────────

# Module-level compiled graph (lazy initialization)
_graph = None


def get_graph():
    """Get or create the compiled agent graph (singleton)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def invoke_agent(
    query: str,
    collection_name: str,
    thread_id: str = "default",
) -> str:
    """
    Invoke the agent with a user query.

    Args:
        query:            The user's question.
        collection_name:  Active ChromaDB collection name.
        thread_id:        Conversation thread ID for memory persistence.

    Returns:
        The agent's final text response.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        {
            "messages": [("user", query)],
            "active_collection": collection_name,
        },
        config=config,
    )

    # Extract the final response
    final_message = result["messages"][-1]
    return final_message.content


async def stream_agent(
    query: str,
    collection_name: str,
    thread_id: str = "default",
):
    """
    Stream the agent's response for real-time UI updates.

    Yields dicts with keys:
        - type: "tool_call" | "tool_result" | "text" | "end"
        - content: The content of the event

    Args:
        query:            The user's question.
        collection_name:  Active ChromaDB collection name.
        thread_id:        Conversation thread ID for memory persistence.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    async for event in graph.astream_events(
        {
            "messages": [("user", query)],
            "active_collection": collection_name,
        },
        config=config,
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            # Streaming text tokens
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield {"type": "text", "content": chunk.content}

        elif kind == "on_tool_start":
            # Tool invocation started
            tool_name = event.get("name", "unknown")
            tool_input = event.get("data", {}).get("input", {})
            yield {
                "type": "tool_call",
                "content": f"🔧 Calling **{tool_name}**...",
                "tool_name": tool_name,
                "tool_input": tool_input,
            }

        elif kind == "on_tool_end":
            # Tool finished
            tool_name = event.get("name", "unknown")
            yield {
                "type": "tool_result",
                "content": f"✅ **{tool_name}** completed.",
                "tool_name": tool_name,
            }

    yield {"type": "end", "content": ""}
