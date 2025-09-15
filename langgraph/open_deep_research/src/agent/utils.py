"""Utility functions and helpers for the Deep Research agent (OpenAI-only)."""

import os
from datetime import datetime
from typing import Annotated, List, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
)
from langchain_core.tools import (
    tool as tool_decorator,
)

try:
    ExceptionGroup = __import__("builtins").ExceptionGroup  # Python 3.11+
except AttributeError:
    # Fallback for older Python versions - define a simple ExceptionGroup-like class
    class ExceptionGroup(Exception):
        """A collection of exceptions that occurred concurrently.

        This is a simplified version of ExceptionGroup for Python < 3.11 compatibility.
        """

        def __init__(self, message: str, exceptions: list[Exception]):
            """Initialize ExceptionGroup with a message and list of exceptions.

            Args:
                message: Description of the exception group
                exceptions: List of exceptions that occurred
            """
            super().__init__(message)
            self.exceptions = exceptions


from agent.configuration import Configuration
from agent.state import ResearchComplete

##########################
# OpenAI Search Tool Utils
##########################
OPENAI_SEARCH_DESCRIPTION = (
    "An AI-powered search tool that uses OpenAI's knowledge to provide comprehensive information. "
    "Useful for research questions, current events, and general knowledge queries. "
    "Note: This tool provides information based on the model's training data and may not include real-time web results."
)


@tool_decorator(description=OPENAI_SEARCH_DESCRIPTION)
async def openai_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news"], InjectedToolArg] = "general",
    config: Optional[RunnableConfig] = None,
) -> str:
    """Generate comprehensive research responses using OpenAI's knowledge base.

    Args:
        queries: List of research queries to answer
        max_results: Maximum number of key points/results to return (not directly used in OpenAI search)
        topic: Topic type for search focus (general or news)
        config: Runtime configuration for API keys and model settings

    Returns:
        Formatted string containing comprehensive research information
    """
    # Handle None config
    cfg: RunnableConfig = config if config is not None else {}

    # Get configuration
    configurable = Configuration.from_runnable_config(cfg)

    # Initialize OpenAI model for research
    model_api_key = configurable.get_openai_api_key()
    research_model = init_chat_model(
        model=configurable.summarization_model,  # Using summarization model for research
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"],
    )

    # Process each query
    all_results = []
    for query in queries:
        try:
            # Create research prompt based on topic
            if topic == "news":
                prompt = f"""Please provide a comprehensive research summary for the following current events query: "{query}"

Focus on recent developments and provide:
1. Key facts and developments
2. Timeline of important events
3. Current status and implications
4. Sources and references you know of

Be thorough and objective. Include dates where relevant."""
            else:
                prompt = f"""Please provide comprehensive information for the following research query: "{query}"

Structure your response to include:
1. Key facts and background information
2. Important details and context
3. Current understanding and implications
4. Any relevant examples or case studies

Be thorough and provide detailed, accurate information based on your knowledge."""

            # Get research response from OpenAI
            response = await research_model.ainvoke([HumanMessage(content=prompt)])

            result = {
                "query": query,
                "title": f"Research Results for: {query}",
                "content": response.content,
                "url": f"openai-research://{query.replace(' ', '-').lower()}",
            }
            all_results.append(result)

        except Exception as e:
            # Handle errors gracefully
            error_result = {
                "query": query,
                "title": f"Research Error for: {query}",
                "content": f"Unable to complete research for this query. Error: {str(e)}",
                "url": f"openai-research-error://{query.replace(' ', '-').lower()}",
            }
            all_results.append(error_result)

    # Format the final output
    if not all_results:
        return "No research results could be generated. Please try different queries."

    formatted_output = f"OpenAI Research Results ({topic} focus):\n\n"
    for i, result in enumerate(all_results, 1):
        formatted_output += f"\n\n--- RESEARCH RESULT {i}: {result['title']} ---\n"
        formatted_output += f"Query: {result['query']}\n"
        formatted_output += "Source: OpenAI Knowledge Base\n\n"
        formatted_output += f"CONTENT:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"

    return formatted_output


##########################
# Reflection Tool Utils
##########################


@tool_decorator(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


##########################
# MCP Utils - Removed for OpenAI-only setup
##########################


##########################
# Tool Utils
##########################


##########################
# Tool Utils
##########################


async def get_all_tools(config: RunnableConfig):
    """Assemble toolkit for research: OpenAI-only + OpenAI search + think_tool."""
    tools: list[BaseTool] = [tool_decorator(ResearchComplete), think_tool]
    # Always include OpenAI search tool
    tools.append(openai_search)
    return tools


def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """Extract notes from tool call messages."""
    return [
        tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")
    ]


##########################
# Model Provider Native Websearch Utils
##########################


def anthropic_websearch_called(response):
    """Detect if Anthropic's native web search was used in the response.

    Args:
        response: The response object from Anthropic's API

    Returns:
        True if web search was called, False otherwise
    """
    try:
        # Navigate through the response metadata structure
        usage = response.response_metadata.get("usage")
        if not usage:
            return False

        # Check for server-side tool usage information
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False

        # Look for web search request count
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False

        # Return True if any web search requests were made
        return web_search_requests > 0

    except (AttributeError, TypeError):
        # Handle cases where response structure is unexpected
        return False


def openai_websearch_called(response):
    """Detect if OpenAI's web search functionality was used in the response.

    Args:
        response: The response object from OpenAI's API

    Returns:
        True if web search was called, False otherwise
    """
    # Check for tool outputs in the response metadata
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if not tool_outputs:
        return False

    # Look for web search calls in the tool outputs
    for tool_output in tool_outputs:
        if tool_output.get("type") == "web_search_call":
            return True

    return False


##########################
# Token Limit Exceeded Utils
##########################


def is_token_limit_exceeded(
    exception: Exception, model_name: Optional[str] = None
) -> bool:
    """Determine if an exception indicates a token/context limit was exceeded.

    Args:
        exception: The exception to analyze
        model_name: Optional model name to optimize provider detection

    Returns:
        True if the exception indicates a token limit was exceeded, False otherwise
    """
    error_str = str(exception).lower()

    # Step 1: Determine provider from model name if available
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith("openai:"):
            provider = "openai"
        elif model_str.startswith("anthropic:"):
            provider = "anthropic"
        elif model_str.startswith("gemini:") or model_str.startswith("google:"):
            provider = "gemini"

    # Step 2: Check provider-specific token limit patterns
    if provider == "openai":
        return _check_openai_token_limit(exception, error_str)
    elif provider == "anthropic":
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == "gemini":
        return _check_gemini_token_limit(exception, error_str)

    # Step 3: If provider unknown, check all providers
    return (
        _check_openai_token_limit(exception, error_str)
        or _check_anthropic_token_limit(exception, error_str)
        or _check_gemini_token_limit(exception, error_str)
    )


def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates OpenAI token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, "__module__", "")

    # Check if this is an OpenAI exception
    is_openai_exception = (
        "openai" in exception_type.lower() or "openai" in module_name.lower()
    )

    # Check for typical OpenAI token limit error types
    is_request_error = class_name in ["BadRequestError", "InvalidRequestError"]

    if is_openai_exception and is_request_error:
        # Look for token-related keywords in error message
        token_keywords = ["token", "context", "length", "maximum context", "reduce"]
        if any(keyword in error_str for keyword in token_keywords):
            return True

    # Check for specific OpenAI error codes
    if hasattr(exception, "code") and hasattr(exception, "type"):
        error_code = getattr(exception, "code", "")
        error_type = getattr(exception, "type", "")

        if (
            error_code == "context_length_exceeded"
            or error_type == "invalid_request_error"
        ):
            return True

    return False


def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Anthropic token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, "__module__", "")

    # Check if this is an Anthropic exception
    is_anthropic_exception = (
        "anthropic" in exception_type.lower() or "anthropic" in module_name.lower()
    )

    # Check for Anthropic-specific error patterns
    is_bad_request = class_name == "BadRequestError"

    if is_anthropic_exception and is_bad_request:
        # Anthropic uses specific error messages for token limits
        if "prompt is too long" in error_str:
            return True

    return False


def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Google/Gemini token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, "__module__", "")

    # Check if this is a Google/Gemini exception
    is_google_exception = (
        "google" in exception_type.lower() or "google" in module_name.lower()
    )

    # Check for Google-specific resource exhaustion errors
    is_resource_exhausted = class_name in [
        "ResourceExhausted",
        "GoogleGenerativeAIFetchError",
    ]

    if is_google_exception and is_resource_exhausted:
        return True

    # Check for specific Google API resource exhaustion patterns
    if "google.api_core.exceptions.resourceexhausted" in exception_type.lower():
        return True

    return False


# NOTE: This may be out of date or not applicable to your models. Please update this as needed.
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
    "bedrock:us.amazon.nova-premier-v1:0": 1000000,
    "bedrock:us.amazon.nova-pro-v1:0": 300000,
    "bedrock:us.amazon.nova-lite-v1:0": 300000,
    "bedrock:us.amazon.nova-micro-v1:0": 128000,
    "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
    "bedrock:us.anthropic.claude-opus-4-20250514-v1:0": 200000,
    "anthropic.claude-opus-4-1-20250805-v1:0": 200000,
}


def get_model_token_limit(model_string):
    """Look up the token limit for a specific model.

    Args:
        model_string: The model identifier string to look up

    Returns:
        Token limit as integer if found, None if model not in lookup table
    """
    # Search through known model token limits
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit

    # Model not found in lookup table
    return None


def remove_up_to_last_ai_message(
    messages: list[MessageLikeRepresentation],
) -> list[MessageLikeRepresentation]:
    """Truncate message history by removing up to the last AI message.

    This is useful for handling token limit exceeded errors by removing recent context.

    Args:
        messages: List of message objects to truncate

    Returns:
        Truncated message list up to (but not including) the last AI message
    """
    # Search backwards through messages to find the last AI message
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # Return everything up to (but not including) the last AI message
            return messages[:i]

    # No AI messages found, return original list
    return messages


##########################
# Misc Utils
##########################


def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.

    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"


def get_config_value(value):
    """Extract value from configuration, handling enums and None values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value


def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """Get API key for a specific model from environment or config."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"):
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None
