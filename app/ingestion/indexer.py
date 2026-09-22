"""
ChromaDB Collection Manager

Manages persistent ChromaDB collections with one collection per topic/session.
Collections are named using slugified topic strings (e.g., "transformer-architectures").

Metadata schema per document:
    - arxiv_id:     str   (e.g., "2401.12345")
    - title:        str   (paper title)
    - authors:      str   (comma-separated author names)
    - page_number:  int   (source page in the PDF)
    - chunk_index:  int   (sequential chunk index within the paper)
    - source_url:   str   (arXiv abstract / PDF URL)
"""

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from slugify import slugify

from app.config import settings


# ── Collection metadata schema documentation ───────────────────────────
METADATA_FIELDS = [
    "arxiv_id",
    "title",
    "authors",
    "page_number",
    "chunk_index",
    "source_url",
]


@dataclass
class CollectionInfo:
    """Summary of a ChromaDB collection."""

    name: str
    display_name: str
    document_count: int


def _get_client() -> ClientAPI:
    """Create a persistent ChromaDB client."""
    persist_dir = settings.chroma_path
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def topic_to_collection_name(topic: str) -> str:
    """
    Convert a human-readable topic string into a valid ChromaDB collection name.

    ChromaDB constraints:
        - 3–63 characters
        - Must start and end with alphanumeric
        - Only alphanumeric, underscores, hyphens allowed
        - No consecutive periods
    """
    slug = slugify(topic, max_length=63, word_boundary=True)
    # Ensure minimum length
    if len(slug) < 3:
        slug = slug.ljust(3, "x")
    return slug


def get_or_create_collection(
    topic: str,
    client: ClientAPI | None = None,
) -> Collection:
    """
    Get an existing collection or create a new one for the given topic.

    Args:
        topic:  Human-readable topic string (e.g., "transformer architectures").
        client: Optional pre-existing ChromaDB client. Creates one if None.

    Returns:
        A ChromaDB Collection ready for upserts and queries.
    """
    if client is None:
        client = _get_client()
    collection_name = topic_to_collection_name(topic)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"topic": topic, "hnsw:space": "cosine"},
    )


def list_collections(client: ClientAPI | None = None) -> list[CollectionInfo]:
    """
    List all existing collections with their document counts.

    Returns:
        A list of CollectionInfo objects sorted by name.
    """
    if client is None:
        client = _get_client()
    collections = client.list_collections()
    result = []
    for col_meta in collections:
        col = client.get_collection(col_meta.name)
        # Recover the display name from stored metadata, or fall back to name
        display_name = (col.metadata or {}).get("topic", col_meta.name)
        result.append(
            CollectionInfo(
                name=col_meta.name,
                display_name=display_name,
                document_count=col.count(),
            )
        )
    return sorted(result, key=lambda c: c.name)


def delete_collection(
    topic: str,
    client: ClientAPI | None = None,
) -> bool:
    """
    Delete a collection by topic name.

    Returns:
        True if the collection was deleted, False if it didn't exist.
    """
    if client is None:
        client = _get_client()
    collection_name = topic_to_collection_name(topic)
    try:
        client.delete_collection(collection_name)
        return True
    except (ValueError, Exception) as e:
        # ChromaDB may raise NotFoundError (subclass of Exception)
        # or ValueError depending on version
        if "NotFoundError" in type(e).__name__ or isinstance(e, ValueError):
            return False
        raise


def get_collection_stats(
    topic: str,
    client: ClientAPI | None = None,
) -> dict:
    """
    Get detailed statistics for a collection.

    Returns:
        Dict with keys: name, topic, document_count, metadata.
    """
    if client is None:
        client = _get_client()
    collection_name = topic_to_collection_name(topic)
    try:
        col = client.get_collection(collection_name)
        return {
            "name": col.name,
            "topic": (col.metadata or {}).get("topic", col.name),
            "document_count": col.count(),
            "metadata": col.metadata,
        }
    except ValueError:
        return {
            "name": collection_name,
            "topic": topic,
            "document_count": 0,
            "metadata": None,
            "error": "Collection does not exist",
        }


# ── Embedding & Upsert Logic ──────────────────────────────────────────


def embed_and_upsert(
    chunks: list,
    topic: str,
    client: ClientAPI | None = None,
    batch_size: int = 50,
    progress_callback=None,
) -> dict:
    """
    Embed text chunks using Google Generative AI and upsert into ChromaDB.

    Args:
        chunks:             List of TextChunk objects from the chunker.
        topic:              Topic string to determine the target collection.
        client:             Optional pre-existing ChromaDB client.
        batch_size:         Number of chunks to process per embedding batch.
        progress_callback:  Optional callable(current, total) for progress updates.

    Returns:
        Dict with keys: chunks_stored, duplicates_skipped, errors.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    # Initialize embedding model
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )

    # Get or create collection for this topic
    collection = get_or_create_collection(topic, client)

    stats = {"chunks_stored": 0, "duplicates_skipped": 0, "errors": 0}
    total_chunks = len(chunks)

    # Process in batches
    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        batch = chunks[batch_start:batch_end]

        try:
            # Prepare batch data
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [c.metadata for c in batch]

            # Check for existing IDs (deduplication)
            existing = collection.get(ids=ids)
            existing_ids = set(existing["ids"]) if existing["ids"] else set()

            # Filter out duplicates
            new_indices = [
                i for i, cid in enumerate(ids) if cid not in existing_ids
            ]

            if not new_indices:
                stats["duplicates_skipped"] += len(batch)
                continue

            new_texts = [texts[i] for i in new_indices]
            new_ids = [ids[i] for i in new_indices]
            new_metadatas = [metadatas[i] for i in new_indices]

            # Generate embeddings
            embeddings = embeddings_model.embed_documents(new_texts)

            # Upsert into ChromaDB
            collection.add(
                ids=new_ids,
                documents=new_texts,
                embeddings=embeddings,
                metadatas=new_metadatas,
            )

            stats["chunks_stored"] += len(new_indices)
            stats["duplicates_skipped"] += len(batch) - len(new_indices)

        except Exception as e:
            logger.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
            stats["errors"] += len(batch)

        if progress_callback:
            progress_callback(min(batch_end, total_chunks), total_chunks)

    return stats
