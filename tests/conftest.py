"""Shared pytest fixtures for the Agentic Research Assistant tests."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Ensure the app module is importable ─────────────────────────────────
# Add the project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Mock settings to avoid requiring GOOGLE_API_KEY in tests ────────────

@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """
    Provide mock settings for all tests so they don't need real API keys.
    Uses monkeypatch to set env vars before any settings are loaded.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key-for-testing")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", tempfile.mkdtemp(prefix="test_chroma_"))
    monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-flash")
    monkeypatch.setenv("EMBEDDING_MODEL", "models/text-embedding-004")
    monkeypatch.setenv("DEFAULT_MAX_PAPERS", "5")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")

    # Clear the cached settings so each test gets fresh settings
    from app.config import get_settings
    get_settings.cache_clear()

    yield

    # Clean up cache after test
    get_settings.cache_clear()


# ── Sample data fixtures ────────────────────────────────────────────────

@pytest.fixture
def sample_paper_metadata():
    """Create a sample PaperMetadata for testing."""
    from app.ingestion.fetcher import PaperMetadata

    return PaperMetadata(
        arxiv_id="2401.12345",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        abstract="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        pdf_url="https://arxiv.org/pdf/2401.12345",
        source_url="https://arxiv.org/abs/2401.12345",
        published="2024-01-15T00:00:00+00:00",
        categories=["cs.CL", "cs.AI"],
        local_pdf_path=None,
    )


@pytest.fixture
def sample_parsed_pages():
    """Create sample parsed pages for testing."""
    from app.ingestion.parser import ParsedPage

    return [
        ParsedPage(
            text="# Introduction\n\nThe Transformer architecture has revolutionized natural language processing. "
                 "It relies entirely on self-attention mechanisms, dispensing with recurrence and convolutions entirely. "
                 "This paper explores the key innovations that make transformers effective for sequence modeling tasks.",
            page_number=1,
            metadata={"page": 1, "file_path": "test.pdf"},
        ),
        ParsedPage(
            text="## Methodology\n\nWe propose a novel self-attention mechanism that computes attention weights "
                 "using scaled dot-product attention. The attention function maps a query and a set of key-value "
                 "pairs to an output, where all are vectors. The output is a weighted sum of the values.",
            page_number=2,
            metadata={"page": 2, "file_path": "test.pdf"},
        ),
        ParsedPage(
            text="## Results\n\nOur model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, "
                 "improving over the existing best results by over 2 BLEU. The model is also significantly "
                 "faster to train than architectures based on recurrent or convolutional layers.",
            page_number=3,
            metadata={"page": 3, "file_path": "test.pdf"},
        ),
    ]


@pytest.fixture
def sample_text_chunks(sample_parsed_pages):
    """Create sample text chunks for testing."""
    from app.ingestion.chunker import chunk_pages

    return chunk_pages(
        pages=sample_parsed_pages,
        arxiv_id="2401.12345",
        title="Attention Is All You Need",
        authors="Ashish Vaswani, Noam Shazeer, Niki Parmar",
        source_url="https://arxiv.org/abs/2401.12345",
        chunk_size=500,
        chunk_overlap=100,
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    d = tempfile.mkdtemp(prefix="test_artifacts_")
    yield d
    # Cleanup is handled by OS temp directory management
