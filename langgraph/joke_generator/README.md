# LangGraph Joke Generator with Pydantic Structured Outputs

An AI-powered joke generator with audience evaluation capabilities using LangGraph and Pydantic structured outputs. This implementation mirrors the functionality of the DSPy example provided, using LangGraph's graph-based approach with structured data validation.

## Features

- **Structured Outputs**: Uses Pydantic models for type-safe, validated responses
- **Dual Model Support**: Supports both "cheap" (gpt-5-mini) and "smart" (gpt-5) models
- **Audience Evaluation**: Five professional comedy personas evaluate jokes with detailed feedback
- **Professional Dataset**: Includes 100+ jokes from famous comedians for evaluation
- **LangGraph Integration**: Graph-based workflow with state management

## Architecture

### Pydantic Models

- `ComedianSignature`: Structured output for joke generation
- `AudienceSignature`: Structured output for audience evaluation
- `AudienceEvaluationResult`: Comprehensive evaluation results

### Graph Nodes

- `comedian_node`: Generates jokes using structured output
- `audience_evaluator_node`: Evaluates jokes with five professional personas

### Audience Personas

1. **Comedy Club Owner** (35): Demands originality, seen every major standup special
2. **Comedy Critic** (42): Writes for The New Yorker, analyzes structure and social commentary
3. **Professional Comedian** (38): Performs nightly, tired of hacky material
4. **Festival Curator** (45): Looks for unique voices and fresh perspectives
5. **Comedy Professor** (40): Teaches advanced joke construction and timing

## Installation

```bash
# Install dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

## Usage

### Basic Usage

```python
from agent.graph import graph

# Generate a joke with evaluation
result = graph.invoke({
    "topic": "programming",
    "lm_type": "cheap",  # or "smart"
    "messages": []
}, config={"openai_api_key": "your-key"})

print(f"Joke: {result['joke']}")
print(f"Audience Score: {result['audience_score']}/5")
```

### Example Scripts

```bash
# Run basic functionality test
python test_joke_generator.py

# Run comprehensive example
python example_usage.py

# Run evaluation demo with dataset
python evaluation_demo.py
```

### Professional Dataset

The system includes 100+ professional comedian jokes for evaluation:

```python
from professional_jokes_dataset import PROFESSIONAL_JOKES, get_random_joke

# Get a random professional joke
joke = get_random_joke()
print(f"Topic: {joke['topic']}")
print(f"Joke: {joke['joke']}")
print(f"Comedian: {joke['comedian']}")
```

## Comparison with DSPy

This LangGraph implementation provides equivalent functionality to the DSPy example:

| Feature              | DSPy | LangGraph     |
| -------------------- | ---- | ------------- |
| Structured Outputs   | ✅   | ✅ (Pydantic) |
| Dual Models          | ✅   | ✅            |
| Audience Evaluation  | ✅   | ✅            |
| Professional Dataset | ✅   | ✅            |
| Graph Workflow       | ❌   | ✅            |
| State Management     | ❌   | ✅            |

## Key Differences

1. **Graph-Based**: Uses LangGraph's state graph for workflow management
2. **Pydantic Validation**: Type-safe structured outputs with validation
3. **State Management**: Comprehensive state tracking throughout the workflow
4. **Extensibility**: Easy to add new nodes and modify the workflow

## File Structure

```
├── src/agent/
│   ├── models.py          # Pydantic models and state definitions
│   └── graph.py           # LangGraph workflow implementation
├── example_usage.py       # Basic usage examples
├── evaluation_demo.py     # Comprehensive evaluation demo
├── test_joke_generator.py # Test suite
├── professional_jokes_dataset.py # Professional comedian jokes
└── README.md             # This file
```

## Environment Variables

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
