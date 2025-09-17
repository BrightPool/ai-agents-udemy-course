"""Data models for the joke generator agent."""

from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field


class Context(TypedDict):
    """Context parameters for the joke generator agent."""

    openai_api_key: str


class ComedianSignature(BaseModel):
    """Pydantic model for comedian joke generation."""
    
    topic: str = Field(description="The topic to create a joke about")
    joke: str = Field(description="A funny joke about the topic")


class AudienceSignature(BaseModel):
    """Pydantic model for audience evaluation."""
    
    joke: str = Field(description="A joke to evaluate")
    profiles: List[str] = Field(description="Profiles of the audience members")
    reactions: List[str] = Field(description="Short reaction from each audience member explaining their inner thought process when hearing the joke (one per profile, same order)")
    responses: List[Literal["hilarious", "funny", "meh", "not funny", "offensive"]] = Field(description="Rating from each audience member (one per profile, same order)")


class AudienceEvaluationResult(BaseModel):
    """Structured result from audience evaluation."""
    
    avg_score: float = Field(description="Average score out of 5")
    responses: List[Literal["hilarious", "funny", "meh", "not funny", "offensive"]] = Field(description="Individual ratings")
    reactions: List[str] = Field(description="Individual reactions")
    profiles: List[str] = Field(description="Audience profiles used")
    joke: str = Field(description="The joke that was evaluated")


class JokeGeneratorState(TypedDict, total=False):
    """State for the joke generator agent.

    Marked as partial (total=False) because LangGraph states are updated
    incrementally across nodes, so not all keys are present at all times.
    """

    # Core joke generation state
    topic: str
    joke: str
    audience_score: float
    audience_responses: List[Literal["hilarious", "funny", "meh", "not funny", "offensive"]]
    audience_reactions: List[str]
    lm_type: Literal["cheap", "smart"]
    # Optional: explicit audience profiles to use during evaluation
    audience_profiles: List[str]
    # If True and a 'joke' is provided, skip generation and only evaluate
    evaluate_only: bool