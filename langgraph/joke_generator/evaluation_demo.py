#!/usr/bin/env python3
# ruff: noqa: T201
"""Run the evaluation demonstration for the LangGraph joke generator.

This script demonstrates evaluation capabilities similar to the DSPy example,
including dataset evaluation and comparison between different models.
"""

import os
import random
from typing import Any, Dict, List, Literal, cast

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from professional_jokes_dataset import (
    PROFESSIONAL_JOKES,
)
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
                "lm_type": cast(Literal["cheap", "smart"], "cheap"),
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


def metric_with_feedback(
    example: Dict[str, Any],
    pred: Dict[str, Any],
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> Dict[str, Any]:
    """Compute a normalized audience score and feedback text.

    Uses the precomputed audience score from AudienceEvaluator and sets
    feedback to a concatenated string of "<profile>: <reaction>" for
    each persona (in order).
    """
    audience = create_audience_evaluator()
    result = audience.evaluate_joke(pred["joke"])
    avg_score = result.avg_score / 5
    reactions = result.reactions
    profiles = result.profiles
    feedback = " ".join(
        f"Persona: {p}\nReaction: {r}\n\n---\n" for p, r in zip(profiles, reactions)
    )

    # Always return structured feedback for consistency with callers
    return {
        "score": avg_score,
        "feedback": feedback,
        "responses": result.responses,
        "reactions": result.reactions,
    }


def evaluate_dataset(
    jokes: List[Dict[str, Any]], lm_type: Literal["cheap", "smart"] = "cheap"
) -> List[Dict[str, Any]]:
    """Evaluate a dataset of jokes using the LangGraph system."""
    results = []
    audience = create_audience_evaluator()

    print(f"Evaluating {len(jokes)} jokes with {lm_type} model...")

    for i, joke_data in enumerate(jokes):
        topic = joke_data["topic"]
        joke = joke_data["joke"]

        print(f"Evaluating joke {i + 1}/{len(jokes)}: {topic}")

        # Generate new joke for comparison
        initial_state: JokeGeneratorState = {
            "topic": topic,
            "lm_type": cast(Literal["cheap", "smart"], lm_type),
        }

        config_dict: Dict[str, Any] = {
            "context": {"openai_api_key": os.getenv("OPENAI_API_KEY")}
        }
        config = cast(RunnableConfig, config_dict)
        generated_result = graph.invoke(initial_state, config=config)

        # Evaluate both original and generated jokes
        original_eval = audience.evaluate_joke(joke)
        generated_eval = audience.evaluate_joke(generated_result["joke"])

        results.append(
            {
                "topic": topic,
                "original_joke": joke,
                "generated_joke": generated_result["joke"],
                "original_score": original_eval.avg_score,
                "generated_score": generated_eval.avg_score,
                "original_responses": original_eval.responses,
                "generated_responses": generated_eval.responses,
                "comedian": joke_data.get("comedian", "Unknown"),
            }
        )

    return results


def compare_models_on_topics(topics: List[str]) -> Dict[str, Any]:
    """Compare cheap vs smart models on specific topics."""
    print("Comparing original vs optimized model outputs:\n")
    results = {"cheap": [], "smart": []}

    for topic in topics:
        print(f"\nTopic: {topic}")
        print("-" * 50)

        # Test both models
        for lm_type_value in ["cheap", "smart"]:
            lm_type = cast(Literal["cheap", "smart"], lm_type_value)
            initial_state: JokeGeneratorState = {"topic": topic, "lm_type": lm_type}

            config_dict: Dict[str, Any] = {
                "context": {"openai_api_key": os.getenv("OPENAI_API_KEY")}
            }
            config = cast(RunnableConfig, config_dict)
            result = graph.invoke(initial_state, config=config)

            results[lm_type].append(
                {
                    "topic": topic,
                    "joke": result["joke"],
                    "audience_score": result["audience_score"],
                    "audience_responses": result["audience_responses"],
                    "audience_reactions": result["audience_reactions"],
                }
            )

            print(f"{lm_type.title()} model:")
            print(f"Joke: {result['joke']}")
            print(f"Audience score: {result['audience_score']}/5")
            print()

        print("-" * 50)

    return results


def main():
    """Run the main demonstration."""
    print("🎭 LangGraph Joke Generator Evaluation Demo")
    print("=" * 60)

    # Test topics
    test_topics = ["programming", "Meditation", "Social Media", "Coffee", "Remote Work"]

    # Compare models on test topics
    _model_comparison = compare_models_on_topics(test_topics)

    # Evaluate a sample of professional jokes
    print("\n\n📊 Professional Jokes Evaluation")
    print("=" * 50)

    # Sample some professional jokes for evaluation
    sample_jokes = random.sample(PROFESSIONAL_JOKES, min(10, len(PROFESSIONAL_JOKES)))

    print(f"Evaluating {len(sample_jokes)} professional jokes...")

    # Evaluate with cheap model
    cheap_results = evaluate_dataset(sample_jokes, "cheap")

    # Evaluate with smart model
    smart_results = evaluate_dataset(sample_jokes, "smart")

    # Calculate average scores
    cheap_avg = sum(r["generated_score"] for r in cheap_results) / len(cheap_results)
    smart_avg = sum(r["generated_score"] for r in smart_results) / len(smart_results)
    original_avg = sum(r["original_score"] for r in cheap_results) / len(cheap_results)

    print("\n📈 Evaluation Results:")
    print(f"Original professional jokes average score: {original_avg:.2f}/5")
    print(f"Generated jokes (cheap model) average score: {cheap_avg:.2f}/5")
    print(f"Generated jokes (smart model) average score: {smart_avg:.2f}/5")

    # Show detailed results for top performing jokes
    print("\n🏆 Top 3 Generated Jokes (Smart Model):")
    smart_results_sorted = sorted(
        smart_results, key=lambda x: x["generated_score"], reverse=True
    )

    for i, result in enumerate(smart_results_sorted[:3]):
        print(f"\n{i + 1}. Topic: {result['topic']}")
        print(f"   Joke: {result['generated_joke']}")
        print(f"   Score: {result['generated_score']}/5")
        print(f"   Ratings: {result['generated_responses']}")

    # Demonstrate metric with feedback
    print("\n🔍 Metric with Feedback Example:")
    print("=" * 50)

    example_joke = {
        "topic": "programming",
        "joke": "Why does a programmer prefer dark mode? Because light attracts bugs!",
    }

    generated_joke = {
        "topic": "programming",
        "joke": "What do you call a programmer who doesn't comment their code? A silent partner.",
    }

    feedback_result = metric_with_feedback(
        example_joke, generated_joke, trace="feedback"
    )

    print(f"Example joke: {example_joke['joke']}")
    print(f"Generated joke: {generated_joke['joke']}")
    print(f"Score: {feedback_result['score']:.2f}")
    print(f"Feedback preview: {feedback_result['feedback'][:200]}...")


if __name__ == "__main__":
    main()
