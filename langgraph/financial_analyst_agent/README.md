# Financial Analyst Agent with LangGraph

A market-focused LangGraph agent that blends Yahoo Finance data with OpenAI reasoning. The agent runs a ReAct loop: it inspects the latest user message, chooses one of the available tools, iterates until it has enough evidence, and then emits a concise answer that always references the numeric facts it used. For intensive calculations it can escalate to the native OpenAI Code Interpreter.

## Workflow Overview

```
[user_message]
    ↓
[LLM plans next action]
    ├─► yf_get_price        → latest price snapshot
    ├─► yf_get_history      → last ≤10 OHLCV rows
    ├─► run_finance_analysis → low-temp Responses API reasoning
    └─► code_interpreter_tool → OpenAI Python tool
        ↓
[Compose final answer → {action, tool_result, answer}]
```

Outputs always include:
- `action`: the last tool used (`yf_get_price`, `yf_get_history`, `run_finance_analysis`, `code_interpreter_tool`, or `answer_direct`).
- `tool_result`: the text returned by that tool (empty when answering directly).
- `answer`: the user-facing financial summary.

## Project Structure

```
financial_analyst_agent/
├── langgraph.json   # LangGraph wiring for the agent graph
├── pyproject.toml   # Dependencies (LangGraph, LangChain, OpenAI, yfinance)
├── .env.example     # Required environment variables
├── run_graph.py     # Streaming demo using langgraph-sdk
└── src/agent/
    ├── __init__.py  # Exports graph + TOOLS list
    ├── graph.py     # ReAct loop with tool execution + final reducer
    ├── tools.py     # Yahoo Finance + OpenAI Responses/Code Interpreter tools
    └── models.py    # Pydantic request schemas for the tools
```

> This package is a pure LangGraph + OpenAI port—no DSPy dependency is required.

## Setup

### 1. Install uv

```bash
brew install uv
# or:
# pipx install uv
```

### 2. Create and activate a virtual environment

```bash
cd financial_analyst_agent
uv venv
source .venv/bin/activate
# Tip: skip activation and prefix commands with `uv run` when preferred.
```

### 3. Install dependencies

```bash
uv sync
# Include dev tools (ruff, pytest, etc.):
# uv sync --group dev
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and supply your real values
OPENAI_API_KEY=sk-...
FINANCIAL_AGENT_MAX_ITERS=5
FINANCIAL_ANALYST_REASONING_MODEL=gpt-5-mini
FINANCIAL_ANALYST_CODE_MODEL=gpt-4.1
```

Environment variable reference:

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | Auth key for all OpenAI calls | _required_ |
| `FINANCIAL_AGENT_MAX_ITERS` | Max tool-use iterations per request | `5` |
| `FINANCIAL_ANALYST_REASONING_MODEL` | Model for `run_finance_analysis` | `gpt-5-mini` |
| `FINANCIAL_ANALYST_CODE_MODEL` | Model for `code_interpreter_tool` | `gpt-4.1` |

### 5. Launch the LangGraph server

```bash
uv run langgraph dev
```

Example output:

```
> Ready!
>
> - API: http://localhost:2024
> - Docs: http://localhost:2024/docs
> - LangGraph Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### 6. Stream a sample request

Open a new terminal (or stop the server afterwards) and run:

```bash
uv run python run_graph.py
```

The script calls the graph with the prompt _“Compare the 5-day percentage return for AAPL and MSFT...”_. You should see streaming events culminating in a JSON-like object containing `action`, `tool_result`, and `answer`.

## Tools Reference

| Tool | Description | Typical use |
| --- | --- | --- |
| `yf_get_price` | Fetches the most recent price + currency | Quick spot price checks |
| `yf_get_history` | Returns the last ≤10 OHLCV rows | Percent returns, volatility inspection |
| `run_finance_analysis` | Low-temp OpenAI reasoning (text only) | Deterministic comparisons, summarising numbers |
| `code_interpreter_tool` | OpenAI Code Interpreter (Python sandbox) | Complex calculations, ad-hoc modelling |

## Usage Examples

### Quick price inquiry

```
Input: "What is the latest price for NVDA?"
Output (`action= yf_get_price`): "NVDA price: 912.32 USD"
```

### Comparative return analysis

```
Input: "Compare AAPL and MSFT 5-day performance."
Tools: `yf_get_history` → `run_finance_analysis`
Answer: Percent returns with the outperformer highlighted.
```

### Code Interpreter scenario

```
Input: "Download TSLA daily prices for the past month and fit a linear trend."
Tools: `yf_get_history` → `code_interpreter_tool`
Answer: Numeric summary plus interpretation of the fitted slope.
```

## Troubleshooting

- Ensure `OPENAI_API_KEY` is valid and has Responses + Code Interpreter access.
- If Code Interpreter returns an error, the agent reports the failure and can fall back to `run_finance_analysis`.
- Trimmed conversation history keeps outputs focused—set `FINANCIAL_AGENT_MAX_ITERS` higher if your task needs more steps.

Happy analysing!
