"""Utility functions for the financial analyst agent."""

import os
from typing import List

from langchain_core.messages import AnyMessage


def _int_env(name: str, default: int) -> int:
    """Get an integer value from environment variables with fallback."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _message_to_text(message: AnyMessage) -> str:
    """Extract textual content from LangChain message variants.

    Uses the built-in message content extraction when possible.
    """
    # For most LangChain messages, .content contains the text
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: List[str] = []
        for item in content:
            if isinstance(item, str):
                fragments.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                fragments.append(str(item.get("text", "")))
        return "\n".join(filter(None, fragments))
    return str(content)


def _trim_history(messages: List[AnyMessage], max_items: int = 10) -> List[AnyMessage]:
    """Trim message history to keep only the most recent messages."""
    if len(messages) <= max_items:
        return messages
    return messages[-max_items:]


__all__ = ["_int_env", "_message_to_text", "_trim_history"]
