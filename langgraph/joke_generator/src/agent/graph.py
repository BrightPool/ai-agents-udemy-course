"""Joke Generator LangGraph Agent.

An AI-powered joke generator with audience evaluation capabilities using LangGraph.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Literal, cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils import convert_to_secret_str
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from src.agent.models import (
    AudienceSignature,
    ComedianSignature,
    Context,
    JokeGeneratorState,
)

# Load environment variables from .env file if it exists
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent  # Navigate to project root
_env_file = _project_root / ".env"

if _env_file.exists():
    load_dotenv(_env_file)
    print(f"✅ Loaded environment variables from {_env_file}")  # noqa: T201
else:
    print(f"ℹ️  No .env file found at {_env_file}")  # noqa: T201
    print(  # noqa: T201
        "   Environment variables will be loaded from system environment or runtime context"
    )


def get_llm(
    runtime: Runtime[Context], model_type: Literal["cheap", "smart"]
) -> ChatOpenAI:
    """Get the appropriate LLM based on model type."""
    ctx = getattr(runtime, "context", None)
    api_key_value = None
    if isinstance(ctx, dict):
        api_key_value = ctx.get("openai_api_key")  # type: ignore[assignment]
    if not api_key_value:
        api_key_value = os.getenv("OPENAI_API_KEY")

    if model_type == "cheap":
        return ChatOpenAI(
            model="gpt-5-mini",
            api_key=convert_to_secret_str(api_key_value or ""),
            temperature=1.0,
        )
    else:  # smart
        return ChatOpenAI(
            model="gpt-5",
            api_key=convert_to_secret_str(api_key_value or ""),
            temperature=1.0,
        )


def comedian_node(
    state: JokeGeneratorState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Generate a joke about the given topic using structured output."""
    llm = get_llm(runtime, state.get("lm_type", "cheap"))

    topic = state.get("topic", "programming")

    # Use structured output with Pydantic
    structured_llm = llm.with_structured_output(ComedianSignature)

    system_message = SystemMessage(
        content="""You are a professional comedian. Your job is to tell funny, clever jokes about any topic.
        Make sure your jokes are:
        - Appropriate for a general audience
        - Clever and witty
        - Original and creative
        - Not offensive or inappropriate

        Return a structured response with the topic and your joke."""
    )

    human_message = HumanMessage(content=f"Tell me a funny joke about {topic}")

    response = cast(
        ComedianSignature, structured_llm.invoke([system_message, human_message])
    )

    return {"joke": response.joke, "topic": response.topic}


def audience_evaluator_node(
    state: JokeGeneratorState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Evaluate the joke using different audience personas with structured output."""
    llm = get_llm(runtime, "cheap")  # Use cheap model for evaluation

    joke = state.get("joke", "")

    # Define audience personas (mirroring the DSPy example)
    profiles = [
        "35-year-old comedy club owner who has seen every major standup special and demands originality",
        "42-year-old comedy critic who writes for The New Yorker and analyzes joke structure and social commentary",
        "38-year-old professional comedian who performs nightly and is tired of hacky material",
        "45-year-old comedy festival curator who looks for unique voices and fresh perspectives",
        "40-year-old comedy writing professor who teaches advanced joke construction and timing",
    ]

    # Use structured output with Pydantic
    structured_llm = llm.with_structured_output(AudienceSignature)

    system_message = SystemMessage(
        content="""You are evaluating a joke from the perspective of different audience members.
        For each persona, provide:
        1. A short reaction explaining their inner thought process when hearing the joke
        2. A rating on this scale:
           - "hilarious" (5 points)
           - "funny" (4 points)
           - "meh" (3 points)
           - "not funny" (2 points)
           - "offensive" (1 point)

        Return structured responses with reactions and ratings for each persona."""
    )

    human_message = HumanMessage(content=f"Evaluate this joke: {joke}")

    response = cast(
        AudienceSignature, structured_llm.invoke([system_message, human_message])
    )

    # Calculate average score
    rating_scores = {
        "hilarious": 5,
        "funny": 4,
        "meh": 3,
        "not funny": 2,
        "offensive": 1,
    }

    total_score = sum(rating_scores.get(rating, 3) for rating in response.responses)
    avg_score = round(total_score / len(response.responses), 2)

    return {
        "audience_responses": response.responses,
        "audience_reactions": response.reactions,
        "audience_score": avg_score,
    }


def should_continue(
    state: JokeGeneratorState,
) -> Literal["audience_evaluator", "__end__"]:
    """Decide whether to continue to audience evaluation or end."""
    # Always continue to audience evaluation after joke generation
    return "audience_evaluator"


def should_end(state: JokeGeneratorState) -> Literal["__end__"]:
    """End the graph after audience evaluation."""
    return "__end__"


# Define the graph
graph = (
    StateGraph(JokeGeneratorState, context_schema=Context)
    .add_node("comedian", comedian_node)
    .add_node("audience_evaluator", audience_evaluator_node)
    # Add edges to connect nodes
    .add_edge(START, "comedian")
    .add_conditional_edges("comedian", should_continue, ["audience_evaluator"])
    .add_edge("audience_evaluator", END)
    .compile(name="Joke Generator Agent")
)
