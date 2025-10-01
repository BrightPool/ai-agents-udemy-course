"""Marketing Blog LangGraph Agent.

A ReAct-style marketing blog writer that:
- Retrieves context from a small marketing corpus via vector search
- Manages an outline
- Writes and edits section drafts
- Assembles a final blog post
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.utils import convert_to_secret_str
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from agent.models import BlogAgentState
from agent.tools import (
    assemble_blog,
    change_outline,
    edit_section,
    search_context,
    write_section,
)


class Context(TypedDict):
    """Runtime context parameters for the marketing blog agent.

    This TypedDict defines the runtime configuration that can be passed
    to the LangGraph agent, including API keys and execution limits.

    Attributes:
        anthropic_api_key: API key for Anthropic Claude model access
        max_iterations: Maximum number of graph execution steps allowed
    """

    openai_api_key: str
    max_iterations: int


def is_relevant_query(query: str) -> bool:
    """Check if a query is relevant to marketing/blog writing tasks.

    Args:
        query: The user's query string to evaluate

    Returns:
        bool: True if the query contains marketing/blog-related keywords, False otherwise

    This function filters out off-topic requests by checking for relevant keywords
    like 'blog', 'outline', 'marketing', etc. to ensure the agent only responds
    to appropriate blog writing tasks.
    """
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
    """Orchestrate the blog writing workflow using the main LLM node.

    This function serves as the central decision-making node in the LangGraph.
    It processes the current conversation state and determines whether to:
    - Call a tool to perform blog writing actions (search, outline, write, edit)
    - Respond directly to the user
    - Continue the conversation flow

    Args:
        state: Current state of the blog agent including messages and topic
        runtime: Runtime context containing API keys and configuration

    Returns:
        Dict containing updated messages from the LLM response

    The function includes relevance filtering to ensure only blog-related queries
    are processed, and handles tool calling for the various blog writing operations.
    """
    # Set up the LLM with OpenAI GPT
    ctx = getattr(runtime, "context", None)
    api_key_value = None
    if isinstance(ctx, dict):
        api_key_value = ctx.get("openai_api_key")  # type: ignore[assignment]
    if not api_key_value:
        api_key_value = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(
        model="gpt-5-nano",
        api_key=convert_to_secret_str(api_key_value or ""),
        temperature=0.1,  # Low temperature for consistent responses
        timeout=30,
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


# Create the tool node using LangGraph's prebuilt ToolNode
# This automatically handles Command objects from tools
tool_node = ToolNode(
    tools=[
        search_context,
        change_outline,
        write_section,
        edit_section,
        assemble_blog,
    ]
)


def assemble_node(state: BlogAgentState) -> Dict[str, Any]:
    """Assemble the final blog from outline and sections.

    This node is triggered after the assemble_blog tool is called.
    It has access to the full state to compile the blog.
    """
    outline = state.get("outline", [])
    sections = state.get("sections", {})

    parts: list[str] = []
    for title in outline:
        body = sections.get(title, "")
        if body:
            parts.append(f"# {title}\n\n{body}".strip())

    final_blog = "\n\n".join([p for p in parts if p])

    return {"final_blog": final_blog}


def route_after_tools(state: BlogAgentState) -> Literal["assemble_node", "llm_call"]:
    """Route to assembly node if assemble_blog tool was called, otherwise back to LLM."""
    messages = state["messages"]
    last_message = messages[-1]

    # Check if the last message is a tool message from assemble_blog
    if hasattr(last_message, "content"):
        try:
            content = (
                json.loads(last_message.content)
                if isinstance(last_message.content, str)
                else last_message.content
            )
            if isinstance(content, dict) and content.get("tool") == "assemble_blog":
                return "assemble_node"
        except (json.JSONDecodeError, AttributeError):
            pass

    return "llm_call"


def should_continue(state: BlogAgentState) -> Literal["tool_node", END]:  # type: ignore
    """Conditional router that determines the next step in the graph execution.

    This function acts as a conditional edge in the LangGraph, examining the
    most recent message to decide whether:
    - To route to "tool_node" if the LLM made a tool call that needs execution
    - To route to END if the conversation should terminate

    Args:
        state: Current state containing the conversation messages

    Returns:
        Either "tool_node" to execute a tool or END to terminate the graph

    The decision is based on whether the last AIMessage contains tool_calls,
    which indicates the LLM wants to perform an action rather than respond directly.
    """
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
    .add_node("assemble_node", assemble_node)
    # Add edges to connect nodes
    .add_edge(START, "llm_call")
    .add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    .add_conditional_edges(
        "tool_node", route_after_tools, ["assemble_node", "llm_call"]
    )
    .add_edge("assemble_node", "llm_call")
    .compile(name="Marketing Blog Agent")
)
