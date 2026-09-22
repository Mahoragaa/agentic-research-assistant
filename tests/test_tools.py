"""Tests for the agent tools: retrieve, web_search, clarify."""

from unittest.mock import MagicMock, patch

import pytest

from app.agent.tools.clarify import clarify
from app.agent.tools.web_search import web_search
from app.agent.tools.retrieve import retrieve


# ═══════════════════════════════════════════════════════════════════════
# Clarify Tool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestClarifyTool:
    """Tests for the clarify tool."""

    def test_clarify_returns_formatted_response(self):
        """Test that clarify returns a well-formatted clarification request."""
        result = clarify.invoke({
            "original_query": "tell me about stuff",
            "reason": "The query is too vague to determine which papers to search.",
            "suggested_queries": [
                "What are the key contributions of the attention mechanism?",
                "Compare transformer and RNN architectures",
            ],
        })

        assert "tell me about stuff" in result
        assert "too vague" in result
        assert "attention mechanism" in result
        assert "Compare transformer" in result

    def test_clarify_includes_all_suggestions(self):
        """Test that all suggested queries appear in the output."""
        suggestions = [
            "Suggestion one",
            "Suggestion two",
            "Suggestion three",
        ]
        result = clarify.invoke({
            "original_query": "test query",
            "reason": "test reason",
            "suggested_queries": suggestions,
        })

        for suggestion in suggestions:
            assert suggestion in result

    def test_clarify_includes_reason(self):
        """Test that the reason is included in the output."""
        result = clarify.invoke({
            "original_query": "what?",
            "reason": "The query lacks specificity about the research domain.",
            "suggested_queries": ["Be more specific"],
        })

        assert "lacks specificity" in result


# ═══════════════════════════════════════════════════════════════════════
# Web Search Tool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestWebSearchTool:
    """Tests for the web_search tool (mocked DuckDuckGo)."""

    @patch("app.agent.tools.web_search._ddg_search")
    def test_web_search_returns_results(self, mock_ddg):
        """Test successful web search."""
        mock_ddg.invoke.return_value = "Search result: Transformers are state-of-the-art..."

        result = web_search.invoke({"query": "latest transformer models"})

        assert "latest transformer models" in result
        assert "Transformers are state-of-the-art" in result

    @patch("app.agent.tools.web_search._ddg_search")
    def test_web_search_empty_results(self, mock_ddg):
        """Test web search with no results."""
        mock_ddg.invoke.return_value = ""

        result = web_search.invoke({"query": "completely obscure topic"})

        assert "No web search results" in result

    @patch("app.agent.tools.web_search._ddg_search")
    def test_web_search_handles_error(self, mock_ddg):
        """Test web search error handling."""
        mock_ddg.invoke.side_effect = Exception("Network error")

        result = web_search.invoke({"query": "test"})

        assert "error" in result.lower()


# ═══════════════════════════════════════════════════════════════════════
# Retrieve Tool Tests
# ═══════════════════════════════════════════════════════════════════════


class TestRetrieveTool:
    """Tests for the retrieve tool (mocked ChromaDB + Embeddings)."""

    @patch("app.agent.tools.retrieve._get_embeddings")
    @patch("app.ingestion.indexer._get_client")
    def test_retrieve_returns_formatted_results(self, mock_get_client, mock_get_embeddings):
        """Test retrieval with mocked ChromaDB and embeddings."""
        # Mock embedding model
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_get_embeddings.return_value = mock_embed

        # Mock ChromaDB collection
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["This is a test document about attention mechanisms."]],
            "metadatas": [[{
                "title": "Attention Paper",
                "authors": "Author A",
                "page_number": 1,
                "arxiv_id": "2401.12345",
                "source_url": "https://arxiv.org/abs/2401.12345",
            }]],
            "distances": [[0.2]],
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        result = retrieve.invoke({
            "query": "attention mechanism",
            "collection_name": "test-collection",
        })

        assert "Attention Paper" in result
        assert "Author A" in result
        assert "2401.12345" in result

    @patch("app.agent.tools.retrieve._get_embeddings")
    @patch("app.ingestion.indexer._get_client")
    def test_retrieve_collection_not_found(self, mock_get_client, mock_get_embeddings):
        """Test retrieval when collection doesn't exist."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_get_embeddings.return_value = mock_embed

        mock_client = MagicMock()
        mock_client.get_collection.side_effect = ValueError("Collection not found")
        mock_get_client.return_value = mock_client

        result = retrieve.invoke({
            "query": "test",
            "collection_name": "nonexistent",
        })

        assert "not found" in result.lower() or "error" in result.lower()

    @patch("app.agent.tools.retrieve._get_embeddings")
    @patch("app.ingestion.indexer._get_client")
    def test_retrieve_no_results(self, mock_get_client, mock_get_embeddings):
        """Test retrieval when no documents match."""
        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [0.1] * 768
        mock_get_embeddings.return_value = mock_embed

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        result = retrieve.invoke({
            "query": "nonexistent topic",
            "collection_name": "test-collection",
        })

        assert "No relevant results" in result
