"""Tests for the ingestion pipeline components."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from app.ingestion.fetcher import PaperMetadata, fetch_papers
from app.ingestion.parser import ParsedPage, parse_pdf, parse_pdf_safe
from app.ingestion.chunker import TextChunk, chunk_pages
from app.ingestion.indexer import (
    topic_to_collection_name,
    get_or_create_collection,
    list_collections,
    delete_collection,
    CollectionInfo,
)
from app.ingestion.pipeline import ingest_papers, IngestionResult


# ═══════════════════════════════════════════════════════════════════════
# Fetcher Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPaperMetadata:
    """Tests for the PaperMetadata dataclass."""

    def test_paper_metadata_creation(self, sample_paper_metadata):
        assert sample_paper_metadata.arxiv_id == "2401.12345"
        assert sample_paper_metadata.title == "Attention Is All You Need"
        assert len(sample_paper_metadata.authors) == 3

    def test_authors_str(self, sample_paper_metadata):
        assert sample_paper_metadata.authors_str == "Ashish Vaswani, Noam Shazeer, Niki Parmar"

    def test_paper_metadata_defaults(self):
        paper = PaperMetadata(
            arxiv_id="test",
            title="Test Paper",
            authors=["Author One"],
            abstract="Abstract text",
            pdf_url="https://example.com/pdf",
            source_url="https://example.com/abs",
            published="2024-01-01",
        )
        assert paper.local_pdf_path is None
        assert paper.categories == []


class TestFetchPapers:
    """Tests for the fetch_papers function."""

    @patch("app.ingestion.fetcher.arxiv.Client")
    @patch("app.ingestion.fetcher.arxiv.Search")
    def test_fetch_papers_basic(self, mock_search_cls, mock_client_cls):
        """Test basic paper fetching with mocked arXiv API."""
        # Create a mock result
        mock_result = MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/2401.12345"
        mock_result.title = "  Test Paper Title  "
        mock_result.authors = [MagicMock(name="Author A")]
        mock_result.authors[0].name = "Author A"
        mock_result.summary = "  Test abstract  "
        mock_result.pdf_url = "https://arxiv.org/pdf/2401.12345"
        mock_result.published = MagicMock()
        mock_result.published.isoformat.return_value = "2024-01-15T00:00:00+00:00"
        mock_result.categories = ["cs.AI"]

        # Make client.results() return our mock result
        mock_client = MagicMock()
        mock_client.results.return_value = [mock_result]
        mock_client_cls.return_value = mock_client

        papers = fetch_papers("test topic", max_results=1, download_pdfs=False)

        assert len(papers) == 1
        assert papers[0].arxiv_id == "2401.12345"
        assert papers[0].title == "Test Paper Title"
        assert papers[0].abstract == "Test abstract"

    @patch("app.ingestion.fetcher.arxiv.Client")
    @patch("app.ingestion.fetcher.arxiv.Search")
    def test_fetch_papers_empty_results(self, mock_search_cls, mock_client_cls):
        """Test fetching when no papers are found."""
        mock_client = MagicMock()
        mock_client.results.return_value = []
        mock_client_cls.return_value = mock_client

        papers = fetch_papers("nonexistent topic", max_results=5, download_pdfs=False)
        assert len(papers) == 0

    @patch("app.ingestion.fetcher.arxiv.Client")
    @patch("app.ingestion.fetcher.arxiv.Search")
    def test_fetch_papers_with_progress_callback(self, mock_search_cls, mock_client_cls):
        """Test that progress callback is called."""
        mock_result = MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/2401.99999"
        mock_result.title = "Progress Test Paper"
        mock_result.authors = []
        mock_result.summary = "Summary"
        mock_result.pdf_url = "https://arxiv.org/pdf/2401.99999"
        mock_result.published = MagicMock()
        mock_result.published.isoformat.return_value = "2024-01-01"
        mock_result.categories = []

        mock_client = MagicMock()
        mock_client.results.return_value = [mock_result]
        mock_client_cls.return_value = mock_client

        callback = MagicMock()
        fetch_papers("test", max_results=1, download_pdfs=False, progress_callback=callback)
        callback.assert_called_once_with(1, 1, "Progress Test Paper")


# ═══════════════════════════════════════════════════════════════════════
# Parser Tests
# ═══════════════════════════════════════════════════════════════════════


class TestParser:
    """Tests for the PDF parser."""

    def test_parse_pdf_nonexistent_file(self):
        """Test parsing a file that doesn't exist."""
        result = parse_pdf("/nonexistent/path/to/file.pdf")
        assert result == []

    def test_parse_pdf_safe_handles_exception(self):
        """Test that parse_pdf_safe catches exceptions."""
        with patch("app.ingestion.parser.parse_pdf", side_effect=RuntimeError("test error")):
            result = parse_pdf_safe("dummy.pdf")
            assert result == []

    @patch("app.ingestion.parser.pymupdf4llm.to_markdown")
    def test_parse_pdf_page_chunks(self, mock_to_markdown):
        """Test parsing with page chunks enabled."""
        mock_to_markdown.return_value = [
            {"text": "This is page one content with enough text to pass the minimum threshold for filtering.", "metadata": {"page": 1}},
            {"text": "This is page two content with enough text to pass the minimum threshold for filtering.", "metadata": {"page": 2}},
        ]

        with patch("app.ingestion.parser.Path.exists", return_value=True):
            result = parse_pdf("test.pdf", page_chunks=True)

        assert len(result) == 2
        assert result[0].page_number == 1
        assert result[1].page_number == 2
        assert isinstance(result[0], ParsedPage)

    @patch("app.ingestion.parser.pymupdf4llm.to_markdown")
    def test_parse_pdf_skips_short_pages(self, mock_to_markdown):
        """Test that pages with very little content are skipped."""
        mock_to_markdown.return_value = [
            {"text": "Short", "metadata": {"page": 1}},  # < 50 chars
            {"text": "This page has enough content to pass the minimum character threshold for filtering in the parser.", "metadata": {"page": 2}},
        ]

        with patch("app.ingestion.parser.Path.exists", return_value=True):
            result = parse_pdf("test.pdf", page_chunks=True)

        assert len(result) == 1
        assert result[0].page_number == 2


# ═══════════════════════════════════════════════════════════════════════
# Chunker Tests
# ═══════════════════════════════════════════════════════════════════════


class TestChunker:
    """Tests for the text chunker."""

    def test_chunk_pages_basic(self, sample_parsed_pages):
        """Test basic chunking of parsed pages."""
        chunks = chunk_pages(
            pages=sample_parsed_pages,
            arxiv_id="2401.12345",
            title="Test Paper",
            authors="Author A, Author B",
            source_url="https://arxiv.org/abs/2401.12345",
            chunk_size=500,
            chunk_overlap=100,
        )

        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_chunk_metadata_preservation(self, sample_parsed_pages):
        """Test that metadata is preserved through chunking."""
        chunks = chunk_pages(
            pages=sample_parsed_pages,
            arxiv_id="2401.12345",
            title="Test Paper",
            authors="Author A",
            source_url="https://arxiv.org/abs/2401.12345",
            chunk_size=500,
            chunk_overlap=100,
        )

        for chunk in chunks:
            assert chunk.metadata["arxiv_id"] == "2401.12345"
            assert chunk.metadata["title"] == "Test Paper"
            assert chunk.metadata["authors"] == "Author A"
            assert chunk.metadata["source_url"] == "https://arxiv.org/abs/2401.12345"
            assert "page_number" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_chunk_id_format(self, sample_parsed_pages):
        """Test that chunk IDs follow the expected format."""
        chunks = chunk_pages(
            pages=sample_parsed_pages,
            arxiv_id="2401.12345",
            title="Test Paper",
            authors="Author A",
            source_url="https://example.com",
            chunk_size=500,
            chunk_overlap=100,
        )

        for chunk in chunks:
            assert chunk.chunk_id.startswith("2401.12345::chunk_")

    def test_chunk_sequential_indices(self, sample_parsed_pages):
        """Test that chunk indices are sequential."""
        chunks = chunk_pages(
            pages=sample_parsed_pages,
            arxiv_id="test",
            title="Test",
            authors="A",
            source_url="url",
            chunk_size=500,
            chunk_overlap=100,
        )

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_empty_pages(self):
        """Test chunking with empty pages."""
        empty_pages = [
            ParsedPage(text="", page_number=1, metadata={}),
            ParsedPage(text="   ", page_number=2, metadata={}),
        ]

        chunks = chunk_pages(
            pages=empty_pages,
            arxiv_id="test",
            title="Test",
            authors="A",
            source_url="url",
            chunk_size=500,
            chunk_overlap=100,
        )

        assert len(chunks) == 0


# ═══════════════════════════════════════════════════════════════════════
# Indexer Tests
# ═══════════════════════════════════════════════════════════════════════


class TestIndexer:
    """Tests for the ChromaDB indexer."""

    def test_topic_to_collection_name(self):
        """Test topic string slugification."""
        assert topic_to_collection_name("transformer architectures") == "transformer-architectures"
        assert topic_to_collection_name("NLP") == "nlp"  # will be padded
        assert len(topic_to_collection_name("ab")) >= 3  # minimum length

    def test_topic_to_collection_name_special_chars(self):
        """Test slugification with special characters."""
        name = topic_to_collection_name("multi-agent LLM systems (2024)")
        assert name.isascii()
        assert " " not in name
        assert len(name) >= 3
        assert len(name) <= 63

    def test_get_or_create_collection(self):
        """Test collection creation with ChromaDB."""
        collection = get_or_create_collection("test topic")
        assert collection is not None
        assert collection.name == topic_to_collection_name("test topic")

    def test_list_collections(self):
        """Test listing collections."""
        # Create a couple of collections
        get_or_create_collection("test collection alpha")
        get_or_create_collection("test collection beta")

        collections = list_collections()
        names = [c.name for c in collections]
        assert topic_to_collection_name("test collection alpha") in names
        assert topic_to_collection_name("test collection beta") in names

    def test_delete_collection(self):
        """Test deleting a collection."""
        get_or_create_collection("topic to delete")
        assert delete_collection("topic to delete") is True

        # Second delete should return False (doesn't exist)
        # Note: ChromaDB may raise ValueError — our function handles that
        result = delete_collection("topic to delete")
        assert result is False


# ═══════════════════════════════════════════════════════════════════════
# Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════


class TestIngestionResult:
    """Tests for the IngestionResult dataclass."""

    def test_success_property(self):
        result = IngestionResult(
            topic="test",
            papers_fetched=5,
            papers_parsed=3,
            total_chunks=30,
            chunks_stored=30,
            duplicates_skipped=0,
            errors=0,
        )
        assert result.success is True

    def test_failure_when_no_papers_parsed(self):
        result = IngestionResult(
            topic="test",
            papers_fetched=0,
            papers_parsed=0,
            total_chunks=0,
            chunks_stored=0,
            duplicates_skipped=0,
            errors=0,
        )
        assert result.success is False

    def test_summary_output(self):
        result = IngestionResult(
            topic="test",
            papers_fetched=5,
            papers_parsed=3,
            total_chunks=30,
            chunks_stored=28,
            duplicates_skipped=2,
            errors=0,
        )
        summary = result.summary()
        assert "test" in summary
        assert "5" in summary
        assert "28" in summary
