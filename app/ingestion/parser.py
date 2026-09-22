"""
PDF Parser

Converts arXiv PDFs into structured Markdown text using pymupdf4llm.
Supports page-level chunking with metadata preservation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pymupdf4llm

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    """A single parsed page from a PDF."""

    text: str
    page_number: int
    metadata: dict


def parse_pdf(
    pdf_path: str,
    page_chunks: bool = True,
) -> list[ParsedPage]:
    """
    Parse a PDF file into structured Markdown text.

    Args:
        pdf_path:     Path to the PDF file.
        page_chunks:  If True, returns one ParsedPage per page.
                      If False, returns the entire document as a single ParsedPage.

    Returns:
        List of ParsedPage objects with Markdown text and metadata.
    """
    pdf_path = str(Path(pdf_path).resolve())

    if not Path(pdf_path).exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return []

    try:
        if page_chunks:
            # Get page-level chunks with metadata
            pages_data = pymupdf4llm.to_markdown(
                pdf_path,
                page_chunks=True,
            )

            parsed_pages = []
            for i, page_data in enumerate(pages_data):
                # pymupdf4llm returns dicts with 'text' and 'metadata' keys
                if isinstance(page_data, dict):
                    text = page_data.get("text", "")
                    meta = page_data.get("metadata", {})
                else:
                    text = str(page_data)
                    meta = {}

                # Skip pages with very little content (likely blank or just headers)
                if len(text.strip()) < 50:
                    continue

                parsed_pages.append(
                    ParsedPage(
                        text=text.strip(),
                        page_number=meta.get("page", i + 1),
                        metadata=meta,
                    )
                )

            logger.info(f"Parsed {len(parsed_pages)} pages from {Path(pdf_path).name}")
            return parsed_pages
        else:
            # Single document mode
            full_text = pymupdf4llm.to_markdown(pdf_path)
            if not full_text or len(full_text.strip()) < 50:
                logger.warning(f"No meaningful content extracted from {pdf_path}")
                return []
            return [
                ParsedPage(
                    text=full_text.strip(),
                    page_number=0,
                    metadata={"mode": "full_document"},
                )
            ]

    except Exception as e:
        logger.error(f"Failed to parse PDF {pdf_path}: {e}")
        return []


def parse_pdf_safe(pdf_path: str) -> list[ParsedPage]:
    """
    Parse a PDF with graceful error handling.

    Wraps parse_pdf to catch all exceptions and return an empty list on failure.
    Useful for batch processing where one corrupt PDF shouldn't stop the pipeline.
    """
    try:
        return parse_pdf(pdf_path, page_chunks=True)
    except Exception as e:
        logger.error(f"Unrecoverable error parsing {pdf_path}: {e}")
        return []
