"""
Clarify Tool

Returns a structured clarification request when the user's query
is too ambiguous, broad, or lacking context to answer effectively.
"""

from langchain_core.tools import tool


@tool
def clarify(original_query: str, reason: str, suggested_queries: list[str]) -> str:
    """Request clarification from the user when their query is too ambiguous.

    Use this tool when:
    - The query is too vague or broad to give a useful answer
    - It is unclear whether the user wants information from the papers or the web
    - The query could refer to multiple different concepts
    - More context is needed to provide an accurate response

    Args:
        original_query: The user's original query that needs clarification.
        reason: Why the query needs clarification.
        suggested_queries: 2-4 more specific queries the user could ask instead.

    Returns:
        A formatted clarification request for the user.
    """
    suggestions = "\n".join(
        f"  {i+1}. {sq}" for i, sq in enumerate(suggested_queries)
    )

    return (
        f"I need a bit more context to give you the best answer.\n\n"
        f"**Your question:** {original_query}\n\n"
        f"**Why I need clarification:** {reason}\n\n"
        f"**Here are some more specific questions you could ask:**\n{suggestions}\n\n"
        f"Please rephrase your question or pick one of the suggestions above."
    )
