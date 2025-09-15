"""Minimal configuration for OpenAI-only Deep Research (GPT-5-mini)."""

import os
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """OpenAI-only configuration for the Deep Research agent."""

    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
    )
    allow_clarification: bool = Field(
        default=True,
    )
    max_concurrent_research_units: int = Field(
        default=5,
    )
    max_researcher_iterations: int = Field(
        default=6,
    )
    max_react_tool_calls: int = Field(
        default=10,
    )
    # Model Configuration
    summarization_model: str = Field(default="openai:gpt-5-mini")
    summarization_model_max_tokens: int = Field(
        default=8192,
    )
    max_content_length: int = Field(
        default=50000,
    )
    research_model: str = Field(default="openai:gpt-5-mini")
    research_model_max_tokens: int = Field(
        default=10000,
    )
    compression_model: str = Field(default="openai:gpt-5-mini")
    compression_model_max_tokens: int = Field(
        default=8192,
    )
    final_report_model: str = Field(default="openai:gpt-5-mini")
    final_report_model_max_tokens: int = Field(
        default=10000,
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    def get_openai_api_key(self) -> str:
        """Get OpenAI API key from environment or config."""
        return os.environ.get("OPENAI_API_KEY", "")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
