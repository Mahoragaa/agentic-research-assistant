"""
Web Search Tool

Uses DuckDuckGo search (free, no API key required) to find
recent developments, author information, or context outside
the ingested papers.
"""

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


# Initialize DuckDuckGo search (no API key needed)
_ddg_search = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web for recent information not found in the ingested papers.

    Use this tool when the user asks about:
    - Recent developments or news in a research area
    - Author profiles, affiliations, or other publications
    - Context, background, or comparisons outside the ingested papers
    - Current state-of-the-art or benchmarks
    - Any information that would not be in the arXiv papers already ingested

    Args:
        query: The search query to look up on the web.

    Returns:
        Web search results with relevant snippets.
    """
    try:
        results = _ddg_search.invoke(query)
        if not results:
            return "No web search results found for this query."
        return f"**Web Search Results for:** {query}\n\n{results}"
    except Exception as e:
        return f"Web search error: {str(e)}. Try rephrasing your query."
