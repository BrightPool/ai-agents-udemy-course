"""Pydantic models used by the competitor monitoring agent."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class CompetitorSource(BaseModel):
    """Represents a single row from the competitor tracking sheet."""

    company: str = Field(default="Unknown", description="Competitor display name")
    root_url: str = Field(
        ..., description="Canonical marketing site for the competitor"
    )
    blog_url: str = Field(..., description="Blog or updates URL to crawl for links")


class DiscoveredLink(BaseModel):
    """Normalized link discovered on a competitor blog."""

    normalized_url: str = Field(..., description="Absolute URL after normalization")
    source_site: str = Field(
        ..., description="Competitor root domain that emitted the link"
    )
    competitor: str = Field(..., description="Human readable competitor name")
    discovered_at: datetime = Field(default_factory=datetime.utcnow)


class SPRDocument(BaseModel):
    """Sparse Priming Representation generated from article content."""

    source_url: str
    title: str
    date: str
    spr: List[str]


class ArticleSummary(BaseModel):
    """Human-facing competitor summary derived from an SPR."""

    competitor: str
    source_url: str
    content: str


class SummaryText(BaseModel):
    """Structured LLM response for summary content only."""

    content: str


class DigestEmail(BaseModel):
    """Rendered digest email payload."""

    subject: str
    recipient: str
    body_html: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class PipelineReport(BaseModel):
    """Final structured output returned by the LangGraph run."""

    newly_discovered: List[DiscoveredLink] = Field(default_factory=list)
    spr_documents: List[SPRDocument] = Field(default_factory=list)
    summaries: List[ArticleSummary] = Field(default_factory=list)
    executive_summary: Optional[str] = None
    email: Optional[DigestEmail] = None
    warnings: List[str] = Field(default_factory=list)
