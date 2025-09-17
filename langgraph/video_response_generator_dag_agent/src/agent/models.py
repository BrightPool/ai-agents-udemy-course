"""Pydantic models for testimonial video generator DAG agent."""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class BuyingSituation(BaseModel):
    """Describe a single buying situation used to craft a testimonial scene."""
    situation: str
    location: str
    trigger: str
    evaluation: str
    conclusion: str


class BuyingSituationsOutput(BaseModel):
    """Pydantic model for LLM output containing three buying situations."""
    persona: str = Field(..., description="Name or identifier for the persona")
    buying_situations: Dict[str, BuyingSituation]


class SceneDescription(BaseModel):
    """Character and appearance details for a generated scene."""
    persona: str
    appearance: str


class PromptModel(BaseModel):
    """Model grouping scene description, quote, and final prompt string."""
    scene_description: SceneDescription
    quote: str
    prompt_string: str


class PromptOutput(BaseModel):
    """Wrapper for prompt model to match expected output schema."""
    prompt: PromptModel

