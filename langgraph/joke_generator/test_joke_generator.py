#!/usr/bin/env python3
# ruff: noqa: T201
"""Run simple tests for the LangGraph joke generator."""

import os
from dotenv import load_dotenv
from typing import Any, Dict, Literal, cast
from langchain_core.runnables import RunnableConfig
from src.agent.graph import graph

# Load environment variables
load_dotenv()

def test_basic_functionality():
    """Test basic joke generation and evaluation."""
    
    print("🧪 Testing LangGraph Joke Generator")
    print("=" * 40)
    
    # Test with programming topic
    initial_state = {
        "topic": "programming",
        "lm_type": cast(Literal["cheap", "smart"], "cheap"),
    }
    
    print("Generating joke about 'programming'...")
    
    try:
        config_dict: Dict[str, Any] = {
            "context": {"openai_api_key": os.getenv("OPENAI_API_KEY")}
        }
        config = cast(RunnableConfig, config_dict)
        result = graph.invoke(initial_state, config=config)
        
        print(f"✅ Success!")
        print(f"Topic: {result['topic']}")
        print(f"Joke: {result['joke']}")
        print(f"Audience Score: {result['audience_score']}/5")
        print(f"Audience Responses: {result['audience_responses']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_structured_outputs():
    """Test that structured outputs are working correctly."""
    
    print("\n🔍 Testing Structured Outputs")
    print("=" * 40)
    
    try:
        from src.agent.models import (
            ComedianSignature,
            AudienceSignature,
            AudienceEvaluationResult,
        )
        
        # Test Pydantic models
        comedian_sig = ComedianSignature(topic="test", joke="Test joke")
        audience_sig = AudienceSignature(
            joke="Test joke",
            profiles=["test profile"],
            reactions=["test reaction"],
            responses=["funny"]
        )
        eval_result = AudienceEvaluationResult(
            avg_score=4.0,
            responses=["funny"],
            reactions=["test reaction"],
            profiles=["test profile"],
            joke="Test joke"
        )
        
        print("✅ Pydantic models working correctly")
        print(f"ComedianSignature: {comedian_sig.model_dump()}")
        print(f"AudienceSignature: {audience_sig.model_dump()}")
        print(f"AudienceEvaluationResult: {eval_result.model_dump()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with structured outputs: {e}")
        return False


def main():
    """Run all tests."""
    
    print("🚀 LangGraph Joke Generator Test Suite")
    print("=" * 50)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key in a .env file or environment variable")
        return
    
    # Run tests
    tests = [
        test_structured_outputs,
        test_basic_functionality,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The joke generator is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()