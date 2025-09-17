"""Marketing Blog LangGraph Agent.

A ReAct-style marketing blog writer that:
- Retrieves context from a small marketing corpus via vector search
- Manages an outline
- Writes and edits section drafts
- Assembles a final blog post
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.utils import convert_to_secret_str
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from agent.tools import (
    assemble_blog,
    change_outline,
    edit_section,
    search_context,
    write_section,
)

# Load environment variables from .env file if it exists
# This allows developers to run the graph directly with python-dotenv support
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent  # Navigate to project root
_env_file = _project_root / ".env"

if _env_file.exists():
    load_dotenv(_env_file)
    # Note: Using print here for initialization feedback - consider using logging in production
    print(f"✅ Loaded environment variables from {_env_file}")  # noqa: T201
else:
    print(f"ℹ️  No .env file found at {_env_file}")  # noqa: T201
    print(  # noqa: T201
        "   Environment variables will be loaded from system environment or runtime context"
    )


class Context(TypedDict):
    """Context parameters for the marketing blog agent."""

    anthropic_api_key: str
    max_iterations: int


class BlogAgentState(TypedDict):
    """State for the marketing blog agent."""

    # Core state
    messages: Annotated[List[AnyMessage], add_messages]

    # Required input
    topic: str


def is_relevant_query(query: str) -> bool:
    """Basic filter for marketing/blog requests to avoid off-topic usage."""
    q = query.lower()
    keywords = [
        "blog",
        "outline",
        "section",
        "marketing",
        "launch",
        "product",
        "write",
        "post",
        "draft",
        "copy",
    ]
    return any(k in q for k in keywords)


def llm_call(state: BlogAgentState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """LLM decides whether to call a tool or not for blog writing."""
    # Set up the LLM with Anthropic Claude
    ctx = getattr(runtime, "context", None)
    api_key_value = None
    if isinstance(ctx, dict):
        api_key_value = ctx.get("anthropic_api_key")  # type: ignore[assignment]
    if not api_key_value:
        api_key_value = os.getenv("ANTHROPIC_API_KEY")

    llm = ChatAnthropic(
        model_name="claude-3-5-sonnet-20241022",
        api_key=convert_to_secret_str(api_key_value or ""),
        temperature=0.1,  # Low temperature for consistent responses
        timeout=30,
        stop=None,
    )

    # Define tools
    tools = [
        search_context,
        change_outline,
        write_section,
        edit_section,
        assemble_blog,
    ]

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create system message with injected context
    topic = state.get("topic", "")
    context_info = f"TOPIC: {topic}\n" if topic else ""

    system_message = SystemMessage(
        content=f"""
You are a senior marketing blog writer who plans before writing and uses tools to manage state.
{context_info}
GOALS:
- Create a clear multi-level outline for the blog
- Retrieve relevant prior company snippets via search_context to ground claims
- Write each section concisely (3–6 paragraphs) in a practical, no-hype voice
- Optionally edit sections for continuity and style
- Assemble the final blog draft via assemble_blog

TOOLS:
- search_context(query: str, k: int=4) -> JSON with id/text/score
- change_outline(new_outline: List[str])
- write_section(section_title: str, draft: str)
- edit_section(section_title: str, new_draft: str)
- assemble_blog() -> JSON with final_blog

STYLE:
- Voice: practical, crisp verbs, short sentences; avoid exclamation marks
- Use US English, Oxford comma, sentence case for headings
"""
    )

    # Derive the user's textual request for relevance filtering
    user_request: str | None = None
    # Try to extract from the latest HumanMessage
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_request = msg.content if isinstance(msg.content, str) else None
            break

    # Check if query is relevant (only if we have text to evaluate)
    if user_request and not is_relevant_query(user_request):
        return {
            "messages": [
                AIMessage(
                    content="""I can help with marketing blog tasks: outlining, retrieving context, drafting, and editing sections. Please provide a blog topic or request."""
                )
            ],
        }

    # Invoke LLM with tools
    response = llm_with_tools.invoke([system_message] + state["messages"])

    return {"messages": [response]}


tool_node = ToolNode(
    tools=[
        search_context,
        change_outline,
        write_section,
        edit_section,
        assemble_blog,
    ]
)


def should_continue(state: BlogAgentState) -> Literal["tool_node", END]:  # type: ignore
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call."""
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM (AIMessage) makes a tool call, then perform an action
    if isinstance(last_message, AIMessage):
        tool_calls = getattr(last_message, "tool_calls", None)
        if tool_calls:
            return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


# Define the graph
graph = (
    StateGraph(BlogAgentState, context_schema=Context)
    .add_node("llm_call", llm_call)
    .add_node("tool_node", tool_node)
    # Add edges to connect nodes
    .add_edge(START, "llm_call")
    .add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    .add_edge("tool_node", "llm_call")
    .compile(name="Marketing Blog Agent")
)
