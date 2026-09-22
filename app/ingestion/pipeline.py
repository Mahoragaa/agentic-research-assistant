"""
End-to-End Ingestion Pipeline Orchestrator

Wires fetcher -> parser -> chunker -> indexer into a single
`ingest_papers(topic, max_results)` function with progress callbacks.
"""

import logging
import tempfile
from dataclasses import dataclass, field

from app.config import settings
from app.ingestion.fetcher import fetch_papers
from app.ingestion.parser import parse_pdf_safe
from app.ingestion.chunker import chunk_pages
from app.ingestion.indexer import embed_and_upsert, get_collection_stats

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Summary of a completed ingestion run."""

    topic: str
    papers_fetched: int
    papers_parsed: int
    total_chunks: int
    chunks_stored: int
    duplicates_skipped: int
    errors: int
    failed_papers: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.papers_parsed > 0 and self.chunks_stored > 0

    def summary(self) -> str:
        """Human-readable summary of the ingestion."""
        lines = [
            f"Ingestion complete for topic: '{self.topic}'",
            f"  Papers fetched:     {self.papers_fetched}",
            f"  Papers parsed:      {self.papers_parsed}",
            f"  Total chunks:       {self.total_chunks}",
            f"  Chunks stored:      {self.chunks_stored}",
            f"  Duplicates skipped: {self.duplicates_skipped}",
            f"  Errors:             {self.errors}",
        ]
        if self.failed_papers:
            lines.append(f"  Failed papers: {', '.join(self.failed_papers)}")
        return "\n".join(lines)


def ingest_papers(
    topic: str,
    max_results: int | None = None,
    progress_callback=None,
) -> IngestionResult:
    """
    Full ingestion pipeline: fetch -> parse -> chunk -> embed -> store.

    Args:
        topic:              Search topic for arXiv (e.g., "transformer architectures").
        max_results:        Max papers to fetch. Defaults to settings.default_max_papers.
        progress_callback:  Optional callable(stage: str, detail: str) for UI updates.

    Returns:
        IngestionResult with comprehensive statistics.
    """
    if max_results is None:
        max_results = settings.default_max_papers

    def notify(stage: str, detail: str):
        logger.info(f"[{stage}] {detail}")
        if progress_callback:
            progress_callback(stage, detail)

    result = IngestionResult(
        topic=topic,
        papers_fetched=0,
        papers_parsed=0,
        total_chunks=0,
        chunks_stored=0,
        duplicates_skipped=0,
        errors=0,
    )

    # ── Stage 1: Fetch papers from arXiv ──────────────────────────
    notify("fetch", f"Fetching up to {max_results} papers on '{topic}'...")

    download_dir = tempfile.mkdtemp(prefix="arxiv_ingest_")

    papers = fetch_papers(
        topic=topic,
        max_results=max_results,
        download_pdfs=True,
        download_dir=download_dir,
        progress_callback=lambda cur, tot, title: notify(
            "fetch", f"[{cur}/{tot}] {title[:60]}"
        ),
    )
    result.papers_fetched = len(papers)
    notify("fetch", f"Fetched {len(papers)} papers.")

    if not papers:
        notify("error", "No papers fetched. Check your topic query.")
        return result

    # ── Stage 2 & 3: Parse PDFs and chunk text ────────────────────
    all_chunks = []

    for i, paper in enumerate(papers):
        notify("parse", f"[{i+1}/{len(papers)}] Parsing: {paper.title[:60]}...")

        if not paper.local_pdf_path:
            logger.warning(f"No local PDF for {paper.arxiv_id}, skipping.")
            result.failed_papers.append(paper.arxiv_id)
            continue

        # Parse PDF
        pages = parse_pdf_safe(paper.local_pdf_path)

        if not pages:
            logger.warning(f"No content extracted from {paper.arxiv_id}")
            result.failed_papers.append(paper.arxiv_id)
            continue

        result.papers_parsed += 1

        # Chunk the parsed pages
        chunks = chunk_pages(
            pages=pages,
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors_str,
            source_url=paper.source_url,
        )

        all_chunks.extend(chunks)
        notify("chunk", f"  -> {len(chunks)} chunks from {len(pages)} pages")

    result.total_chunks = len(all_chunks)
    notify("chunk", f"Total chunks across all papers: {len(all_chunks)}")

    if not all_chunks:
        notify("error", "No chunks produced. Check PDF parsing.")
        return result

    # ── Stage 4: Embed and store in ChromaDB ──────────────────────
    notify("embed", f"Embedding and storing {len(all_chunks)} chunks...")

    upsert_stats = embed_and_upsert(
        chunks=all_chunks,
        topic=topic,
        progress_callback=lambda cur, tot: notify(
            "embed", f"Embedded {cur}/{tot} chunks"
        ),
    )

    result.chunks_stored = upsert_stats["chunks_stored"]
    result.duplicates_skipped = upsert_stats["duplicates_skipped"]
    result.errors = upsert_stats["errors"]

    # ── Final summary ─────────────────────────────────────────────
    stats = get_collection_stats(topic)
    notify("done", result.summary())
    notify(
        "done",
        f"Collection '{stats.get('name')}' now has {stats.get('document_count', 0)} documents.",
    )

    return result
