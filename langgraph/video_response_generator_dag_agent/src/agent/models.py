"""Pydantic models for testimonial video generator DAG agent."""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class BuyingSituation(BaseModel):
    situation: str
    location: str
    trigger: str
    evaluation: str
    conclusion: str


class BuyingSituationsOutput(BaseModel):
    persona: str = Field(..., description="Name or identifier for the persona")
    buying_situations: Dict[str, BuyingSituation]


class SceneDescription(BaseModel):
    persona: str
    appearance: str


class PromptModel(BaseModel):
    scene_description: SceneDescription
    quote: str
    prompt_string: str


class PromptOutput(BaseModel):
    prompt: PromptModel

