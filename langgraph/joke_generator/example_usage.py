#!/usr/bin/env python3
# ruff: noqa: T201
"""Run example usage of the LangGraph-based joke generator.

This script demonstrates the functionality that mirrors the DSPy example provided.
"""

import os
from typing import Any, Dict, Literal, cast

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from src.agent.graph import graph
from src.agent.models import AudienceEvaluationResult, JokeGeneratorState

# Load environment variables
load_dotenv()


def create_audience_evaluator():
    """Create an audience evaluator class similar to the DSPy example."""

    class AudienceEvaluator:
        """LangGraph-based audience evaluator that defines five audience personas
        and queries them to produce an aggregate funniness score for a given joke.
        """

        def __init__(self):
            # Define audience personas inside the module (single source of truth)
            self.profiles = [
                "35-year-old comedy club owner who has seen every major standup special and demands originality",
                "42-year-old comedy critic who writes for The New Yorker and analyzes joke structure and social commentary",
                "38-year-old professional comedian who performs nightly and is tired of hacky material",
                "45-year-old comedy festival curator who looks for unique voices and fresh perspectives",
                "40-year-old comedy writing professor who teaches advanced joke construction and timing",
            ]

            # Map qualitative ratings to a numeric score
            self.rating_scores = {
                "hilarious": 5,
                "funny": 4,
                "meh": 3,
                "not funny": 2,
                "offensive": 1,
            }

        def evaluate_joke(self, joke: str) -> AudienceEvaluationResult:
            """Evaluate a joke using the LangGraph audience evaluator."""
            # Run the graph with just the audience evaluation
            initial_state: JokeGeneratorState = {
                "topic": "evaluation",
                "joke": joke,
                "lm_type": "cheap",
                "audience_profiles": self.profiles,
                "evaluate_only": True,
            }

            # Run the graph
            config_dict: Dict[str, Any] = {
                "context": {"openai_api_key": os.getenv("OPENAI_API_KEY")}
            }
            config = cast(RunnableConfig, config_dict)
            result = graph.invoke(initial_state, config=config)

            # Return structured result
            return AudienceEvaluationResult(
                avg_score=result["audience_score"],
                responses=result["audience_responses"],
                reactions=result["audience_reactions"],
                profiles=self.profiles,
                joke=joke,
            )

    return AudienceEvaluator()


def generate_joke_with_evaluation(topic: str, lm_type: Literal["cheap", "smart"] = "cheap"):
    """Generate a joke and evaluate it using the LangGraph system."""
    # Initial state for joke generation
    initial_state: JokeGeneratorState = {"topic": topic, "lm_type": lm_type}

    # Run the complete graph
    config_dict: Dict[str, Any] = {
        "context": {"openai_api_key": os.getenv("OPENAI_API_KEY")}
    }
    config = cast(RunnableConfig, config_dict)
    result = graph.invoke(initial_state, config=config)

    return result


def main():
    """Run the main demonstration."""
    print("🎭 LangGraph Joke Generator with Pydantic Structured Outputs")
    print("=" * 60)

    # Initialize audience evaluator
    audience = create_audience_evaluator()

    # Test topics
    test_topics = ["programming", "Meditation", "Social Media", "Coffee", "Remote Work"]

    print("\nComparing cheap vs smart model outputs:\n")

    for topic in test_topics:
        print(f"\nTopic: {topic}")
        print("-" * 50)

        # Generate with cheap model
        cheap_result = generate_joke_with_evaluation(topic, "cheap")
        print("Cheap LM response:")
        print(f"Joke: {cheap_result['joke']}")
        print(f"Average audience score: {cheap_result['audience_score']}/5")
        print()

        # Generate with smart model
        smart_result = generate_joke_with_evaluation(topic, "smart")
        print("Smart LM response:")
        print(f"Joke: {smart_result['joke']}")
        print(f"Average audience score: {smart_result['audience_score']}/5")
        print("-" * 50)

    # Demonstrate standalone audience evaluation
    print("\n\n🎯 Standalone Audience Evaluation Example:")
    print("=" * 50)

    sample_joke = "Why does a programmer prefer dark mode? Because light attracts bugs!"
    print(f"Evaluating joke: {sample_joke}")

    evaluation_result = audience.evaluate_joke(sample_joke)

    print(f"\nAverage Score: {evaluation_result.avg_score}/5")
    print("\nIndividual Reactions:")
    for i, (profile, reaction, rating) in enumerate(
        zip(
            evaluation_result.profiles,
            evaluation_result.reactions,
            evaluation_result.responses,
        )
    ):
        print(f"\n{i + 1}. {profile}")
        print(f"   Reaction: {reaction}")
        print(f"   Rating: {rating}")


if __name__ == "__main__":
    main()
