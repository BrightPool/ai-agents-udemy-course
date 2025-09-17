#!/usr/bin/env python3
"""Run the joke generator graph."""

import os
from typing import Any, Dict, cast

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from src.agent.graph import graph
from src.agent.models import JokeGeneratorState

# Load environment variables
load_dotenv()

def main() -> None:
    """Run the joke generator with example topics."""
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")  # noqa: T201
        print("Please set your OpenAI API key in a .env file or environment variable")  # noqa: T201
        return
    
    # Example topics to test
    topics = ["programming", "cats", "coffee", "space"]
    lm_types = ["cheap", "smart"]
    
    for lm_type in lm_types:
        print(f"\n{'='*50}")  # noqa: T201
        print(f"Testing with {lm_type.upper()} model")  # noqa: T201
        print(f"{'='*50}")  # noqa: T201
        
        for topic in topics:
            print(f"\n🎭 Topic: {topic}")  # noqa: T201
            print("-" * 30)  # noqa: T201
            
            # Run the graph
            state: JokeGeneratorState = {
                "topic": topic,
                "lm_type": lm_type,  # type: ignore[assignment]
                "joke": "",
                "audience_score": 0.0,
                "audience_responses": [],
            }
            
            config_dict: Dict[str, Any] = {
                "context": {
                    "openai_api_key": os.getenv("OPENAI_API_KEY"),
                    "max_iterations": 10
                }
            }
            
            config = cast(RunnableConfig, config_dict)
            result = graph.invoke(state, config=config)
            
            # Display results
            print(f"Joke: {result['joke']}")  # noqa: T201
            print(f"Audience Score: {result['audience_score']}/5")  # noqa: T201
            print(f"Audience Responses: {result['audience_responses']}")  # noqa: T201
            print()  # noqa: T201

if __name__ == "__main__":
    main()