"""Video Generation LangGraph Agent.

A sophisticated agent for creating videos with iterative quality improvement.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from .models import (
    VideoGenerationContext,
    VideoGenerationState,
)
from .tools import (
    cleanup_tmp_directories,
    initialize_tmp_directories,
)
from .tools import (
    analyze_video_quality as analyze_video_quality_tool,
)
from .tools import (
    create_video as create_video_tool,
)
from .tools import (
    elevenlabs_text_to_speech as elevenlabs_tts_tool,
)
from .tools import (
    execute_ffmpeg as execute_ffmpeg_tool,
)
from .tools import (
    generate_ass_file_tool as generate_ass_file_tool_tool,
)
from .tools import (
    list_recent_renders as list_recent_renders_tool,
)
from .tools import (
    search_media_library as search_media_library_tool,
)
from .tools import (
    search_unsplash_media as search_unsplash_media_tool,
)
from .tools import (
    transcribe_audio_openai as transcribe_audio_openai_tool,
)


async def initialize_agent(
    state: VideoGenerationState, runtime: Runtime[VideoGenerationContext]
) -> Dict[str, Any]:
    """Initialize the video generation agent."""
    # Initialize tmp directories
    initialize_tmp_directories()

    # Ensure ffmpeg is installed
    try:
        import shutil

        if not shutil.which("ffmpeg"):
            warning = SystemMessage(
                content=(
                    "FFmpeg is not installed or not on PATH. Video creation tools will fail. "
                    "Please install ffmpeg and ensure it is accessible."
                )
            )
            initial_messages: List[BaseMessage] = [warning]
        else:
            initial_messages = []
    except Exception:
        initial_messages = []

    # Create system message
    system_message = SystemMessage(
        content="""
You are a professional video generation agent with access to multiple tools:

1. elevenlabs_text_to_speech: Generate high-quality speech from text
2. unsplash_search_and_download: Search and download images from Unsplash
3. generate_ass_file_tool: Create subtitle files for videos
4. search_media_library: Search a hardcoded library of assets
5. create_video: Plan a single ffmpeg command string
6. execute_ffmpeg: Execute a provided ffmpeg command
7. transcribe_audio_openai: Transcribe audio with word timestamps (OpenAI)

Your goal is to create high-quality videos based on user requests. You should:
- Analyze the user's request carefully
- Plan the video creation process
- Use appropriate tools to gather assets
- Generate audio and subtitles as needed
- Create the video with proper quality settings
- Iteratively improve based on quality feedback

Always provide detailed feedback about the video quality and suggest improvements.
"""
    )

    return {
        "messages": initial_messages + [system_message],
        "current_iteration": 0,
        "quality_satisfied": False,
        "max_iterations_reached": False,
        "renders": [],
    }


async def finalize_video(
    state: VideoGenerationState, runtime: Runtime[VideoGenerationContext]
) -> Dict[str, Any]:
    """Finalize the video generation process."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=runtime.context.google_api_key,
        temperature=0.7,
    )

    finalization_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Provide a final summary of the video generation process:

1. Summarize what was created
2. Highlight the final quality achieved
3. List all assets and tools used
4. Provide recommendations for future improvements

Be comprehensive and professional.""",
            ),
            ("human", "Final video: {video_path}"),
        ]
    )

    try:
        messages = finalization_prompt.format_messages(
            video_path=state.final_video_path or ""
        )
        response = await llm.ainvoke(messages)
    except Exception:
        from langchain_core.messages import SystemMessage

        response = SystemMessage(content="Final summary unavailable due to an error.")

    # Try to infer final video path from tool outputs if not recorded in state
    inferred_final_path = state.final_video_path or ""
    if not inferred_final_path:
        try:
            for m in reversed(list(state.messages or [])):
                if isinstance(m, ToolMessage) and getattr(m, "name", "") == "execute_ffmpeg":
                    raw = getattr(m, "content", "")
                    if isinstance(raw, str) and raw:
                        try:
                            payload = json.loads(raw)
                            if isinstance(payload, dict):
                                outp = payload.get("output_path")
                                if isinstance(outp, str) and outp:
                                    inferred_final_path = outp
                                    break
                        except Exception:
                            pass
        except Exception:
            pass

    # Perform best-effort cleanup of temporary artifacts while preserving final output
    try:
        preserve: list[str] = []
        if inferred_final_path:
            preserve.append(inferred_final_path)
        cleanup_result = cleanup_tmp_directories(preserve_paths=preserve)
        from langchain_core.messages import SystemMessage

        cleanup_msg = SystemMessage(
            content=(
                "Temporary files cleaned. "
                f"Preserved: {', '.join(cleanup_result.get('preserved', [])) or 'none'}. "
                f"Removed: {len(cleanup_result.get('removed_files', []))}. "
                f"Errors: {len(cleanup_result.get('errors', []))}."
            )
        )
        return {"messages": [response, cleanup_msg], "final_video_path": inferred_final_path}
    except Exception:
        # Even if cleanup fails, return the summary response
        return {"messages": [response], "final_video_path": inferred_final_path}


# Flattened, agentic flow using ToolNode


async def agentic_step(
    state: VideoGenerationState, runtime: Runtime[VideoGenerationContext]
) -> Dict[str, Any]:
    """Single agent step: think and decide whether to call tools."""
    # Ensure tmp dirs exist; idempotent
    initialize_tmp_directories()

    tools = [
        search_media_library_tool,
        search_unsplash_media_tool,
        elevenlabs_tts_tool,
        generate_ass_file_tool_tool,
        transcribe_audio_openai_tool,
        create_video_tool,
        execute_ffmpeg_tool,
        analyze_video_quality_tool,
        list_recent_renders_tool,
    ]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=runtime.context.google_api_key,
        temperature=0.6,
    ).bind_tools(tools)

    # Continue the conversation based on state.messages
    prior = list(state.messages or [])
    ai_msg = await llm.ainvoke(prior)

    next_iter = (state.current_iteration or 0) + 1
    max_reached = next_iter >= (runtime.context.max_iterations or 1)

    return {
        "messages": [ai_msg],
        "current_iteration": next_iter,
        "max_iterations_reached": max_reached,
    }


def route_after_agent(state: VideoGenerationState) -> str:
    """Route to tools if AI requested tool calls, or finalize on stopping conditions."""
    if getattr(state, "max_iterations_reached", False):
        return "finalize"
    if not state.messages:
        return "finalize"
    last = state.messages[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    return "tools" if tool_calls else "finalize"


tool_node = ToolNode(
    [
        search_media_library_tool,
        search_unsplash_media_tool,
        elevenlabs_tts_tool,
        generate_ass_file_tool_tool,
        transcribe_audio_openai_tool,
        create_video_tool,
        execute_ffmpeg_tool,
        analyze_video_quality_tool,
        list_recent_renders_tool,
    ]
)


# Define the graph
graph = (
    StateGraph(VideoGenerationState, context_schema=VideoGenerationContext)
    .add_node("initialize", initialize_agent)
    .add_node("agent", agentic_step)
    .add_node("tools", tool_node)
    .add_node("finalize", finalize_video)
    .add_edge(START, "initialize")
    .add_edge("initialize", "agent")
    .add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "finalize": "finalize"}
    )
    .add_edge("tools", "agent")
    .add_edge("finalize", END)
    .compile(name="Video Generation Agent")
)
