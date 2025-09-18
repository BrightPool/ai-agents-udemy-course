"""Pydantic models for the marketing blog agent.

This module defines data models used throughout the blog agent for type safety,
validation, and structured data handling in the blog writing workflow.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class BlogMetadata(BaseModel):
    """Metadata for a blog post."""

    title: str = Field(..., description="The blog post title")
    topic: str = Field(..., description="The main topic or theme of the blog")
    author: Optional[str] = Field(None, description="Author name if specified")
    target_audience: Optional[str] = Field(
        None, description="Target audience for the blog"
    )
    tone: str = Field(
        "professional",
        description="Writing tone (professional, casual, technical, etc.)",
    )
    word_count_goal: Optional[int] = Field(
        None, description="Target word count for the blog"
    )


class BlogSection(BaseModel):
    """Represents a single section of a blog post."""

    title: str = Field(..., description="Section heading/title")
    content: str = Field(..., description="The actual content of the section")
    word_count: Optional[int] = Field(None, description="Word count for this section")
    is_complete: bool = Field(
        False, description="Whether this section has been fully written"
    )


class SearchResult(BaseModel):
    """Result from a semantic search operation."""

    id: str = Field(..., description="Unique identifier for the result")
    text: str = Field(..., description="The content/text of the result")
    score: float = Field(..., description="Similarity score (higher is better)")
    metadata: Optional[dict] = Field(
        None, description="Additional metadata about the result"
    )


class BlogOutline(BaseModel):
    """Structure representing the blog post outline."""

    sections: List[str] = Field(..., description="List of section titles in order")
    estimated_word_count: Optional[int] = Field(
        None, description="Estimated total word count"
    )
    is_finalized: bool = Field(False, description="Whether the outline is finalized")


class ToolResponse(BaseModel):
    """Standardized response format for tool operations."""

    tool_name: str = Field(..., description="Name of the tool that was executed")
    success: bool = Field(True, description="Whether the operation was successful")
    message: str = Field(..., description="Human-readable message about the result")
    data: Optional[dict] = Field(
        None, description="Structured data returned by the tool"
    )


class BlogConfiguration(BaseModel):
    """Configuration settings for blog generation."""

    max_sections: int = Field(8, description="Maximum number of sections allowed")
    min_section_words: int = Field(100, description="Minimum words per section")
    max_section_words: int = Field(500, description="Maximum words per section")
    search_results_limit: int = Field(
        4, description="Maximum search results to retrieve"
    )
    enable_research: bool = Field(
        True, description="Whether to perform research before writing"
    )
    style_guide: Optional[str] = Field(
        None, description="Custom style guide instructions"
    )
