"""
arXiv Paper Fetcher

Fetches papers from the arXiv API based on a topic query.
Uses the arxiv.py library with rate-limit-aware settings.

Returns structured PaperMetadata objects with PDF download capability.
"""

import os
import tempfile
import logging
from dataclasses import dataclass, field
from pathlib import Path

import arxiv

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Structured metadata for a fetched arXiv paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    pdf_url: str
    source_url: str  # arXiv abstract page
    published: str
    categories: list[str] = field(default_factory=list)
    local_pdf_path: str | None = None

    @property
    def authors_str(self) -> str:
        """Comma-separated author names for metadata storage."""
        return ", ".join(self.authors)


def fetch_papers(
    topic: str,
    max_results: int | None = None,
    download_pdfs: bool = True,
    download_dir: str | None = None,
    progress_callback=None,
) -> list[PaperMetadata]:
    """
    Fetch papers from arXiv for the given topic.

    Args:
        topic:              Search query string (e.g., "transformer architectures").
        max_results:        Maximum number of papers to fetch. Defaults to settings.default_max_papers.
        download_pdfs:      If True, download PDFs to a local directory.
        download_dir:       Directory to save PDFs. If None, uses a temp directory.
        progress_callback:  Optional callable(current, total, paper_title) for progress updates.

    Returns:
        List of PaperMetadata objects with populated fields.
    """
    if max_results is None:
        max_results = settings.default_max_papers

    # Create arXiv client with rate limiting
    client = arxiv.Client(
        page_size=min(max_results, 50),
        delay_seconds=3.0,
        num_retries=3,
    )

    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    # Prepare download directory
    if download_pdfs:
        if download_dir is None:
            download_dir = tempfile.mkdtemp(prefix="arxiv_papers_")
        Path(download_dir).mkdir(parents=True, exist_ok=True)

    papers: list[PaperMetadata] = []

    logger.info(f"Fetching up to {max_results} papers for topic: '{topic}'")

    for i, result in enumerate(client.results(search)):
        try:
            # Extract clean arXiv ID
            arxiv_id = result.entry_id.split("/")[-1]

            paper = PaperMetadata(
                arxiv_id=arxiv_id,
                title=result.title.strip(),
                authors=[a.name for a in result.authors],
                abstract=result.summary.strip(),
                pdf_url=result.pdf_url or "",
                source_url=result.entry_id,
                published=result.published.isoformat() if result.published else "",
                categories=[c for c in (result.categories or [])],
            )

            # Download PDF
            if download_pdfs and download_dir:
                try:
                    safe_filename = f"{arxiv_id.replace('/', '_')}.pdf"
                    pdf_path = os.path.join(download_dir, safe_filename)
                    result.download_pdf(dirpath=download_dir, filename=safe_filename)
                    paper.local_pdf_path = pdf_path
                    logger.info(f"  [{i+1}/{max_results}] Downloaded: {paper.title[:60]}...")
                except Exception as e:
                    logger.warning(f"  [{i+1}/{max_results}] Failed to download PDF for {arxiv_id}: {e}")

            papers.append(paper)

            if progress_callback:
                progress_callback(i + 1, max_results, paper.title)

        except Exception as e:
            logger.error(f"Error processing result {i}: {e}")
            continue

    logger.info(f"Fetched {len(papers)} papers for topic: '{topic}'")
    return papers
