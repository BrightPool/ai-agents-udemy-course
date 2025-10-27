"""Marketing Blog Agent Tools.

Tools supporting a ReAct-style marketing blog writer workflow:
- Vector search over example marketing corpus (FAISS, with numpy fallback)
- Outline management (persist outline across tool calls)
- Section persistence and editing
- Final blog assembly
"""

from __future__ import annotations

import json
from typing import List

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from typing_extensions import Annotated

from .utils import _search


# --- Tools ---
@tool
def search_context(query: str, k: int = 4) -> str:
    """Perform semantic search over the marketing knowledge corpus.

    This tool searches through a curated collection of marketing content using
    vector similarity to find relevant information for blog writing. It supports
    both FAISS (fast) and NumPy (fallback) implementations for different environments.

    Args:
        query: Natural language search query (e.g., "pricing strategy", "customer personas")
        k: Number of top results to return (default: 4, max: 10)

    Returns:
        JSON string containing:
        - tool: "search_context"
        - results: List of search results with id, text, and similarity score

    The search results help ground blog content in company knowledge, ensuring
    accurate and consistent messaging across marketing materials.
    """
    hits = _search(query, k=max(1, min(int(k), 10)))
    return json.dumps({"tool": "search_context", "results": hits})


@tool
def change_outline(
    new_outline: List[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Update the blog post outline structure.

    This tool replaces the current outline with a new structure, allowing the
    blog writing process to evolve and adapt based on research findings or
    changing requirements. The outline defines the section structure that will
    guide the writing process.

    Args:
        new_outline: List of section titles in the desired order
        tool_call_id: Injected tool call ID for the message

    Returns:
        Command object that updates the graph state with the new outline

    The outline serves as the skeleton for the blog post, ensuring logical
    flow and comprehensive coverage of the topic.
    """
    return Command(
        update={
            "outline": list(new_outline),
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        {"tool": "change_outline", "outline": list(new_outline)}
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def write_section(
    section_title: str,
    draft: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Save a section draft to the blog state.

    This tool persists a newly written section draft, allowing the blog writing
    process to build content incrementally. Each section is stored with its title
    as a key, enabling easy retrieval and editing later in the process.

    Args:
        section_title: The heading/title of the section being written
        draft: The complete content for this section
        tool_call_id: Injected tool call ID for the message

    Returns:
        Command object that updates the graph state with the new section

    Note: The LLM should generate comprehensive, well-structured draft content
    that aligns with the overall blog topic and maintains consistent tone.
    """
    return Command(
        update={
            "sections": {section_title: draft},
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        {
                            "tool": "write_section",
                            "section_title": section_title,
                            "saved": True,
                        }
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def edit_section(
    section_title: str,
    new_draft: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Update an existing section with improved content.

    This tool allows for iterative refinement of blog sections by replacing
    existing content with an improved version. It's useful for making corrections,
    adding more detail, improving clarity, or adjusting tone and style.

    Args:
        section_title: The title of the section to be updated
        new_draft: The revised content to replace the existing section
        tool_call_id: Injected tool call ID for the message

    Returns:
        Command object that updates the graph state with the edited section

    This tool enables the writing process to be iterative and quality-focused,
    allowing for continuous improvement of individual sections.
    """
    return Command(
        update={
            "sections": {section_title: new_draft},
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        {
                            "tool": "edit_section",
                            "section_title": section_title,
                            "saved": True,
                        }
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def assemble_blog(
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Signal to assemble the blog - actual assembly happens in a custom node.

    This tool simply returns a marker message. The actual blog assembly
    will be handled by a separate node that has access to the full state.

    Args:
        tool_call_id: Injected tool call ID for the message

    Returns:
        JSON message indicating assembly should be triggered
    """
    return json.dumps({"tool": "assemble_blog", "status": "triggered"})


__all__ = [
    "search_context",
    "change_outline",
    "write_section",
    "edit_section",
    "assemble_blog",
]
