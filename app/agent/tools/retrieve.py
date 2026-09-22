"""
Retrieve Tool

Performs similarity search against the active ChromaDB collection.
Returns formatted results with source citations (title, page, arXiv link).
"""

from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings
from app.ingestion.indexer import get_or_create_collection


def _get_embeddings():
    """Get the configured embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )


@tool
def retrieve(query: str, collection_name: str, top_k: int = 5) -> str:
    """Search the ingested research papers for specific technical details.

    Use this tool when the user asks about specific concepts, methods,
    results, or technical details that would be found in the ingested
    arXiv papers. This searches the vector database of paper chunks.

    Args:
        query: The search query describing what information to find.
        collection_name: The name of the ChromaDB collection to search.
        top_k: Number of top results to return (default: 5).

    Returns:
        Formatted search results with source citations.
    """
    try:
        embeddings_model = _get_embeddings()

        # Embed the query
        query_embedding = embeddings_model.embed_query(query)

        # Get the collection
        from app.ingestion.indexer import _get_client
        client = _get_client()
        try:
            collection = client.get_collection(collection_name)
        except ValueError:
            return f"Error: Collection '{collection_name}' not found. Please ingest papers first."

        # Perform similarity search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return "No relevant results found in the ingested papers."

        # Format results with citations
        formatted_results = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            similarity = 1 - dist  # cosine distance to similarity
            title = meta.get("title", "Unknown")
            authors = meta.get("authors", "Unknown")
            page = meta.get("page_number", "?")
            arxiv_id = meta.get("arxiv_id", "")
            source_url = meta.get("source_url", "")

            formatted_results.append(
                f"### Result {i+1} (similarity: {similarity:.2f})\n"
                f"**Paper:** {title}\n"
                f"**Authors:** {authors}\n"
                f"**Page:** {page} | **arXiv:** {arxiv_id}\n"
                f"**Link:** {source_url}\n\n"
                f"{doc[:500]}...\n"
            )

        return "\n---\n".join(formatted_results)

    except Exception as e:
        return f"Error during retrieval: {str(e)}"
