"""Financial analyst LangGraph agent that mixes tool use with ReAct."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from agent.tools import (
    code_interpreter_tool,
    run_finance_analysis,
    yf_get_history,
    yf_get_price,
)
from agent.utils import _int_env, _trim_history

# ---------------------------------------------------------------------------
# Environment bootstrap
# ---------------------------------------------------------------------------

_CURRENT_FILE = Path(__file__)
_PROJECT_ROOT = _CURRENT_FILE.parent.parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

if _ENV_FILE.exists():  # pragma: no cover - side effect for interactive runs
    load_dotenv(_ENV_FILE)
    print(f"✅ Loaded environment variables from {_ENV_FILE}")  # noqa: T201
else:  # pragma: no cover - informative logging
    print(f"ℹ️  No .env file found at {_ENV_FILE}")  # noqa: T201
    print("   Environment variables will be read from the process context.")  # noqa: T201

# ---------------------------------------------------------------------------
# Context and state definitions
# ---------------------------------------------------------------------------


class Context(TypedDict, total=False):
    """Runtime context for configuring the agent."""

    openai_api_key: str
    max_iterations: int


class FinancialAnalystState(TypedDict, total=False):
    """Agent state carrying conversation history and reporting metadata."""

    messages: Annotated[List[AnyMessage], add_messages]
    iterations: int
    max_iterations: int


TOOLS = [
    yf_get_price,
    yf_get_history,
    run_finance_analysis,
    code_interpreter_tool,
]

TOOL_REGISTRY = {tool.name: tool for tool in TOOLS}


# ---------------------------------------------------------------------------
# Core LangGraph nodes
# ---------------------------------------------------------------------------


def llm_call(state: FinancialAnalystState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """LLM node that decides which tool to call next (ReAct style)."""
    ctx = getattr(runtime, "context", {})
    base_max_iters = _int_env("FINANCIAL_AGENT_MAX_ITERS", 5)
    max_iters = base_max_iters
    if isinstance(ctx, dict):
        ctx_iters = ctx.get("max_iterations")
        if ctx_iters is not None:
            try:
                max_iters = int(ctx_iters)
            except (TypeError, ValueError):
                max_iters = base_max_iters
        api_key = ctx.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY must be configured for the financial analyst agent."
        )

    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.0,
        timeout=60,
    )

    llm_with_tools = llm.bind_tools(TOOLS)

    system_message = SystemMessage(
        content=(
            "You are a financial analyst who answers market questions with precise numbers. "
            "Use the available tools to fetch Yahoo Finance data (prices/history) and run deterministic "
            "analysis (reasoning or the native OpenAI python tool). Always use the code interpreter tool "
            "for any mathematical calculations or computations rather than computing them manually. "
            "Combine tool outputs, then respond with a concise summary that highlights key figures and "
            "comparisons. Maintain a short memory of the conversation to stay on-topic."
        )
    )

    history = state.get("messages", []) or []
    trimmed_history = _trim_history(history)

    response = llm_with_tools.invoke([system_message] + trimmed_history)

    new_state: Dict[str, Any] = {
        "messages": [response],
        "iterations": state.get("iterations", 0) + 1,
        "max_iterations": state.get("max_iterations", max_iters),
    }

    return new_state


def tool_node(state: FinancialAnalystState) -> Dict[str, Any]:
    """Execute the tool requested by the previous LLM call and record the output."""
    messages = state.get("messages", []) or []
    if not messages:
        return {}

    last_ai = next(
        (
            msg
            for msg in reversed(messages)
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)
        ),
        None,
    )
    if last_ai is None:
        return {}

    observations: List[ToolMessage] = []

    for call in last_ai.tool_calls:
        tool_name = call.get("name")
        tool = TOOL_REGISTRY.get(tool_name)
        if tool is None:
            content = f"Unknown tool: {tool_name}"
        else:
            try:
                content = tool.invoke(call.get("args", {}))
            except Exception as exc:  # pragma: no cover - runtime safeguard
                content = f"Tool {tool_name} errored: {exc}"
        observations.append(ToolMessage(content=str(content), tool_call_id=call["id"]))

    return {"messages": observations}


def should_continue(state: FinancialAnalystState) -> str:
    messages = state.get("messages", []) or []
    if not messages:
        return END

    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        tool_calls = getattr(last_message, "tool_calls", None)
        if tool_calls and state.get("iterations", 0) < state.get("max_iterations", 5):
            return "tool_node"
    return END


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------


graph = (
    StateGraph(FinancialAnalystState, context_schema=Context)
    .add_node("llm_call", llm_call)
    .add_node("tool_node", tool_node)
    .add_edge(START, "llm_call")
    .add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    .add_edge("tool_node", "llm_call")
    .compile(name="Financial Analyst Agent")
)

__all__ = ["graph"]
