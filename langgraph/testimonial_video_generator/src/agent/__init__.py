"""Video Response Generator DAG Agent package."""

from .graph import graph
from .tools import (
    analyze_brand_tone_tool,
    create_video_prompt_tool,
    generate_veo3_response_tool,
    generate_video_script_tool,
)

# Define tools list for easy access
tools = [
    generate_video_script_tool,
    analyze_brand_tone_tool,
    create_video_prompt_tool,
    generate_veo3_response_tool,
]

__all__ = [
    "graph",
    "tools",
    "generate_video_script_tool",
    "analyze_brand_tone_tool",
    "create_video_prompt_tool",
    "generate_veo3_response_tool",
]