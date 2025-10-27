"""Pydantic models for the marketing blog agent.

This module defines data models used throughout the blog agent for type safety,
validation, and structured data handling in the blog writing workflow.
"""

from __future__ import annotations

from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


def merge_sections(left: Dict[str, str], right: Dict[str, str]) -> Dict[str, str]:
    """Merge section dictionaries, with right values overwriting left."""
    return {**left, **right}


# --- State Schema ---
class BlogAgentState(TypedDict, total=False):
    """Complete state schema for the marketing blog agent.

    This TypedDict defines the complete state structure used throughout the
    LangGraph blog writing workflow. It includes conversation history, blog
    content state, and metadata for tracking the writing process.

    The state is designed to support the full blog writing pipeline:
    1. Topic specification and research
    2. Outline creation and management
    3. Section-by-section writing
    4. Content editing and refinement
    5. Final blog assembly

    Attributes:
        messages: Conversation history with proper message handling via add_messages reducer
        topic: The main blog topic/theme specified by the user
        outline: List of section titles that form the blog structure
        sections: Dictionary mapping section titles to their content
        final_blog: The assembled complete blog post (populated at the end)
    """

    # Core conversation state
    messages: Annotated[List[AnyMessage], add_messages]

    # Blog content state
    topic: str
    outline: Annotated[List[str], lambda x, y: y]  # Replace with new outline
    sections: Annotated[
        Dict[str, str], merge_sections
    ]  # Merge sections dict (accumulate sections)
    final_blog: str  # Final assembled blog content


class SearchResult(BaseModel):
    """Result from a semantic search operation."""

    id: str = Field(..., description="Unique identifier for the result")
    text: str = Field(..., description="The content/text of the result")
    score: float = Field(..., description="Similarity score (higher is better)")
    metadata: Optional[dict] = Field(
        None, description="Additional metadata about the result"
    )
