"""Pydantic models for customer service agent data structures.

This module defines the data models used throughout the customer service agent
for type safety and validation of documentation search requests and responses.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Documentation category enum as Literal for strict typing
DocumentationCategory = Literal[
    "shipping",
    "returns",
    "products",
    "account",
    "payment",
]


class DocumentationSearchRequest(BaseModel):
    """Schema for searching documentation with optional category classification."""

    query: str = Field(..., description="End-user query string")
    category: Optional[DocumentationCategory | Literal["auto"]] = Field(
        "auto",
        description=(
            "Target documentation category. When 'auto', the system will classify"
            " the query into one of the known categories."
        ),
    )


class DocumentationSearchResult(BaseModel):
    """Normalized search result payload returned by documentation search tool."""

    category: DocumentationCategory = Field(..., description="Resolved category")
    content: str = Field(..., description="Concatenated documentation response text")
