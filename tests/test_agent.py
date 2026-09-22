"""Tests for the full agent graph (LangGraph ReAct agent)."""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.agent.state import AgentState
from app.agent.graph import (
    SYSTEM_PROMPT,
    TOOLS,
    build_graph,
    get_graph,
    should_continue,
    agent_node,
)


# ═══════════════════════════════════════════════════════════════════════
# Agent State Tests
# ═══════════════════════════════════════════════════════════════════════


class TestAgentState:
    """Tests for the AgentState TypedDict."""

    def test_agent_state_keys(self):
        """Test that AgentState has the expected keys."""
        # AgentState is a TypedDict, check annotations
        annotations = AgentState.__annotations__
        assert "messages" in annotations
        assert "active_collection" in annotations


# ═══════════════════════════════════════════════════════════════════════
# System Prompt Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSystemPrompt:
    """Tests for the system prompt configuration."""

    def test_system_prompt_exists(self):
        """Test that the system prompt is defined and has content."""
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_mentions_tools(self):
        """Test that the system prompt describes all three tools."""
        assert "retrieve" in SYSTEM_PROMPT.lower()
        assert "web_search" in SYSTEM_PROMPT.lower()
        assert "clarify" in SYSTEM_PROMPT.lower()

    def test_system_prompt_has_guidelines(self):
        """Test that the system prompt includes response guidelines."""
        assert "cite" in SYSTEM_PROMPT.lower()
        assert "source" in SYSTEM_PROMPT.lower()


# ═══════════════════════════════════════════════════════════════════════
# Tool Registration Tests
# ═══════════════════════════════════════════════════════════════════════


class TestToolRegistration:
    """Tests for tool registration."""

    def test_three_tools_registered(self):
        """Test that exactly three tools are registered."""
        assert len(TOOLS) == 3

    def test_tool_names(self):
        """Test that the tools have correct names."""
        tool_names = [t.name for t in TOOLS]
        assert "retrieve" in tool_names
        assert "web_search" in tool_names
        assert "clarify" in tool_names


# ═══════════════════════════════════════════════════════════════════════
# Graph Structure Tests
# ═══════════════════════════════════════════════════════════════════════


class TestGraphStructure:
    """Tests for the LangGraph agent structure."""

    def test_build_graph_returns_compiled_graph(self):
        """Test that build_graph returns a compiled graph object."""
        graph = build_graph()
        assert graph is not None

    def test_get_graph_singleton(self):
        """Test that get_graph returns the same instance."""
        # Reset the singleton
        import app.agent.graph as graph_module
        graph_module._graph = None

        graph1 = get_graph()
        graph2 = get_graph()
        assert graph1 is graph2

        # Reset after test
        graph_module._graph = None


# ═══════════════════════════════════════════════════════════════════════
# Conditional Edge Tests
# ═══════════════════════════════════════════════════════════════════════


class TestShouldContinue:
    """Tests for the should_continue conditional edge."""

    def test_routes_to_tools_when_tool_calls_present(self):
        """Test routing to tools when LLM makes tool calls."""
        mock_msg = MagicMock()
        mock_msg.tool_calls = [{"name": "retrieve", "args": {"query": "test"}}]

        state = {"messages": [mock_msg]}
        assert should_continue(state) == "tools"

    def test_routes_to_end_when_no_tool_calls(self):
        """Test routing to END when LLM responds directly."""
        mock_msg = MagicMock()
        mock_msg.tool_calls = []

        state = {"messages": [mock_msg]}
        assert should_continue(state) == "end"

    def test_routes_to_end_when_no_tool_calls_attr(self):
        """Test routing to END when message has no tool_calls attribute."""
        mock_msg = MagicMock(spec=[])  # No attributes

        state = {"messages": [mock_msg]}
        assert should_continue(state) == "end"


# ═══════════════════════════════════════════════════════════════════════
# Agent Node Tests
# ═══════════════════════════════════════════════════════════════════════


class TestAgentNode:
    """Tests for the agent_node function."""

    @patch("app.agent.graph._get_llm")
    def test_agent_node_returns_messages(self, mock_get_llm):
        """Test that agent_node returns a dict with messages."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [("user", "What is attention?")],
            "active_collection": "test-collection",
        }

        result = agent_node(state)
        assert "messages" in result
        assert len(result["messages"]) == 1

    @patch("app.agent.graph._get_llm")
    def test_agent_node_includes_collection_context(self, mock_get_llm):
        """Test that the system prompt includes the active collection."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        state = {
            "messages": [("user", "test")],
            "active_collection": "my-research-topic",
        }

        agent_node(state)

        # Verify the LLM was called with messages that include the collection
        call_args = mock_llm.invoke.call_args[0][0]
        system_msg = call_args[0]
        assert "my-research-topic" in system_msg.content
