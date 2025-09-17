"""Shared utilities for the Mem0 coaching agent.

This module centralizes environment loading, LLM construction, Mem0 client
configuration (Qdrant + Neo4j), and common message helpers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.utils import convert_to_secret_str
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime
from mem0 import Memory


def load_env_if_exists() -> None:
    """Load environment variables from a project-level .env file if present.

    Attempts to load a .env file from the project root directory. If the file
    exists, loads all environment variables defined in it. If not found,
    prints an informational message indicating that environment variables
    will be loaded from system environment or runtime context.

    This function is safe to call multiple times and provides user feedback
    about the loading status.
    """
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded environment variables from {env_file}")  # noqa: T201
    else:
        print(f"ℹ️  No .env file found at {env_file}")  # noqa: T201
        print(  # noqa: T201
            "   Environment variables will be loaded from system environment or runtime context"
        )


def get_openai_llm(runtime: Runtime[Any]) -> ChatOpenAI:
    """Create an OpenAI chat model using environment variables only.

    Initializes a ChatOpenAI instance configured for the coaching agent.
    The model uses gpt-5-mini with moderate temperature for balanced
    creativity and consistency in responses.

    Args:
        runtime: LangGraph runtime instance (currently unused but required
                for interface consistency).

    Returns:
        ChatOpenAI: Configured OpenAI chat model instance.

    Raises:
        This function does not explicitly raise exceptions but may propagate
        OpenAI client initialization errors if the API key is invalid.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    return ChatOpenAI(
        model="gpt-5-mini",
        api_key=convert_to_secret_str(api_key),
        temperature=0.3,
        timeout=30,
    )


def get_mem0_client(runtime: Runtime[Any]) -> Memory:
    """Build a Mem0 client configured for Qdrant (vector) and Neo4j (graph).

    Creates a Mem0 memory client with dual storage configuration:
    - Qdrant for vector similarity search
    - Neo4j for graph-based memory relationships

    All configuration values are read from environment variables to align
    with docker-compose setup and allow runtime configuration.

    Args:
        runtime: LangGraph runtime instance (currently unused but required
                for interface consistency).

    Returns:
        Memory: Configured Mem0 memory client instance.

    Environment Variables:
        QDRANT_HOST: Qdrant server hostname (default: localhost)
        QDRANT_PORT: Qdrant server port (default: 6333)
        NEO4J_URI: Neo4j connection URI (default: bolt://localhost:7687)
        NEO4J_USERNAME: Neo4j username (default: neo4j)
        NEO4J_PASSWORD: Neo4j password (default: mem0-graph)
    """
    q_host = os.getenv("QDRANT_HOST", "localhost")
    q_port_val = int(os.getenv("QDRANT_PORT", "6333"))
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "mem0-graph")

    config: dict[str, Any] = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": q_host,
                "port": q_port_val,
            },
        },
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": neo4j_uri,
                "username": neo4j_user,
                "password": neo4j_pass,
            },
        },
    }

    return Memory.from_config(config_dict=config)


def extract_latest_user_text(messages: List[AnyMessage]) -> str:
    """Return latest human message content from a list of messages.

    Iterates through the message list in reverse order to find the most
    recent human message with string content. This is useful for extracting
    user input from conversation history.

    Args:
        messages: List of message objects to search through.

    Returns:
        str: Content of the latest human message, or empty string if none found.
             Returns empty string for None input or when no valid human
             messages are present.
    """
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content
    return ""


def get_default_k(runtime: Runtime[Any], fallback: int = 3) -> int:
    """Return default 'k' for Mem0 search from MEM0_DEFAULT_K env variable.

    Retrieves the default number of memories to retrieve during Mem0 search
    operations. Falls back to a provided default value or 3 if the environment
    variable is not set or cannot be parsed as an integer.

    Args:
        runtime: LangGraph runtime instance (currently unused but required
                for interface consistency).
        fallback: Default value to return if environment variable is not set
                 or invalid (default: 3).

    Returns:
        int: Number of memories to retrieve, either from MEM0_DEFAULT_K env var
             or the fallback value.

    Environment Variables:
        MEM0_DEFAULT_K: Integer specifying default memory retrieval count.
    """
    try:
        return int(os.getenv("MEM0_DEFAULT_K", str(fallback)))
    except Exception:
        return fallback
