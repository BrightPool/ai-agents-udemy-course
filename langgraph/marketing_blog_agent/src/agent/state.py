"""Marketing Blog LangGraph Agent.

A ReAct-style marketing blog writer that:
- Retrieves context from a small marketing corpus via vector search
- Manages an outline
- Writes and edits section drafts
- Assembles a final blog post
"""

from __future__ import annotations

from typing import Annotated, Dict, List, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class BlogAgentState(TypedDict):
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
        Dict[str, str], lambda x, y: y
    ]  # Replace with new sections dict
    final_blog: str  # Final assembled blog content
