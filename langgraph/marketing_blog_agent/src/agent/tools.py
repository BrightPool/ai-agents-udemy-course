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

from langchain_core.tools import tool

from .utils import _BLOG_STATE, _search


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
def change_outline(new_outline: List[str]) -> str:
    """Update the blog post outline structure.

    This tool replaces the current outline with a new structure, allowing the
    blog writing process to evolve and adapt based on research findings or
    changing requirements. The outline defines the section structure that will
    guide the writing process.

    Args:
        new_outline: List of section titles in the desired order

    Returns:
        JSON string containing:
        - tool: "change_outline"
        - outline: The updated list of section titles

    The outline serves as the skeleton for the blog post, ensuring logical
    flow and comprehensive coverage of the topic.
    """
    _BLOG_STATE.outline = list(new_outline)
    return json.dumps({"tool": "change_outline", "outline": _BLOG_STATE.outline})


@tool
def write_section(section_title: str, draft: str) -> str:
    """Save a section draft to the blog state.

    This tool persists a newly written section draft, allowing the blog writing
    process to build content incrementally. Each section is stored with its title
    as a key, enabling easy retrieval and editing later in the process.

    Args:
        section_title: The heading/title of the section being written
        draft: The complete content for this section

    Returns:
        JSON string containing:
        - tool: "write_section"
        - section_title: The title of the saved section
        - saved: Boolean indicating successful save

    Note: The LLM should generate comprehensive, well-structured draft content
    that aligns with the overall blog topic and maintains consistent tone.
    """
    _BLOG_STATE.sections[section_title] = draft
    return json.dumps(
        {"tool": "write_section", "section_title": section_title, "saved": True}
    )


@tool
def edit_section(section_title: str, new_draft: str) -> str:
    """Update an existing section with improved content.

    This tool allows for iterative refinement of blog sections by replacing
    existing content with an improved version. It's useful for making corrections,
    adding more detail, improving clarity, or adjusting tone and style.

    Args:
        section_title: The title of the section to be updated
        new_draft: The revised content to replace the existing section

    Returns:
        JSON string containing:
        - tool: "edit_section"
        - section_title: The title of the updated section
        - saved: Boolean indicating successful update

    This tool enables the writing process to be iterative and quality-focused,
    allowing for continuous improvement of individual sections.
    """
    _BLOG_STATE.sections[section_title] = new_draft
    return json.dumps(
        {"tool": "edit_section", "section_title": section_title, "saved": True}
    )


@tool
def assemble_blog() -> str:
    """Compile all sections into the final blog post.

    This tool takes the current outline and all written sections to create
    the complete blog post. It combines sections in the order specified by
    the outline, formatting them with proper headers and spacing.

    Returns:
        JSON string containing:
        - tool: "assemble_blog"
        - final_blog: The complete assembled blog post as a formatted string

    The assembled blog includes:
    - Section headers formatted as markdown H1 (#)
    - Proper spacing between sections
    - Only sections that exist and have content
    - Clean formatting suitable for publishing

    This represents the final step in the blog writing process, producing
    the complete, ready-to-publish article.
    """
    parts: List[str] = []
    for title in _BLOG_STATE.outline:
        body = _BLOG_STATE.sections.get(title, "")
        parts.append(f"# {title}\n\n{body}".strip())
    final_blog = "\n\n".join([p for p in parts if p])
    return json.dumps({"tool": "assemble_blog", "final_blog": final_blog})


__all__ = [
    "search_context",
    "change_outline",
    "write_section",
    "edit_section",
    "assemble_blog",
]
