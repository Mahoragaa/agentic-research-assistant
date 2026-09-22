"""
Text Chunker

Splits parsed PDF pages into smaller chunks suitable for embedding and retrieval.
Uses LangChain's RecursiveCharacterTextSplitter which respects Markdown structure.

Preserves page-level metadata through the splitting process and attaches
enriched metadata (arxiv_id, title, authors, etc.) to each chunk.
"""

import logging
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.ingestion.parser import ParsedPage

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunk of text ready for embedding with full metadata."""

    text: str
    chunk_index: int
    metadata: dict

    @property
    def chunk_id(self) -> str:
        """Unique identifier for deduplication: arxiv_id::chunk_index."""
        arxiv_id = self.metadata.get("arxiv_id", "unknown")
        return f"{arxiv_id}::chunk_{self.chunk_index}"


def chunk_pages(
    pages: list[ParsedPage],
    arxiv_id: str,
    title: str,
    authors: str,
    source_url: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """
    Split parsed pages into smaller text chunks with enriched metadata.

    Args:
        pages:          List of ParsedPage objects from the parser.
        arxiv_id:       arXiv paper ID (e.g., "2401.12345").
        title:          Paper title.
        authors:        Comma-separated author names.
        source_url:     arXiv URL for citation.
        chunk_size:     Max characters per chunk. Defaults to settings.chunk_size.
        chunk_overlap:  Overlap between chunks. Defaults to settings.chunk_overlap.

    Returns:
        List of TextChunk objects with sequential chunk indices and full metadata.
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap

    # Configure the splitter to respect Markdown structure
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",      # H2 headers
            "\n### ",     # H3 headers
            "\n#### ",    # H4 headers
            "\n\n",       # Paragraph breaks
            "\n",         # Line breaks
            ". ",         # Sentence boundaries
            " ",          # Word boundaries
            "",           # Character level (last resort)
        ],
        length_function=len,
    )

    chunks: list[TextChunk] = []
    global_chunk_index = 0

    for page in pages:
        if not page.text.strip():
            continue

        # Split the page text
        page_chunks = splitter.split_text(page.text)

        for chunk_text in page_chunks:
            if not chunk_text.strip():
                continue

            chunk = TextChunk(
                text=chunk_text.strip(),
                chunk_index=global_chunk_index,
                metadata={
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": authors,
                    "page_number": page.page_number,
                    "chunk_index": global_chunk_index,
                    "source_url": source_url,
                },
            )
            chunks.append(chunk)
            global_chunk_index += 1

    logger.info(
        f"Chunked paper '{title[:50]}...' into {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks
