"""Ingestion pipeline — arXiv fetch, PDF parse, chunk, and index."""

from app.ingestion.pipeline import ingest_papers, IngestionResult
from app.ingestion.indexer import (
    list_collections,
    delete_collection,
    get_collection_stats,
    get_or_create_collection,
)

__all__ = [
    "ingest_papers",
    "IngestionResult",
    "list_collections",
    "delete_collection",
    "get_collection_stats",
    "get_or_create_collection",
]
