"""Video Generation LangGraph Agent.

A sophisticated agent for creating videos with iterative quality improvement.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from agent.models import (
    VideoGenerationContext,
    VideoGenerationState,
)
from agent.tools import (
    analyze_video_quality as analyze_video_quality_tool,
)
from agent.tools import (
    concat_videos as concat_videos_tool,
)
from agent.tools import (
    create_story_board as create_story_board_tool,
)
from agent.tools import (
    generate_image as generate_image_tool,
)
from agent.tools import (
    run_ffmpeg_binary as run_ffmpeg_tool,
)
from agent.tools import (
    update_scene as update_scene_tool,
)
from agent.tools import (
    veo3_generate_video as veo3_tool,
)
from agent.tools import (
    veo3_generate_videos_batch as veo3_batch_tool,
)
from agent.utils import cleanup_tmp_directories, initialize_tmp_directories


async def initialize_agent(
    state: VideoGenerationState, runtime: Runtime[VideoGenerationContext]
) -> Dict[str, Any]:
    """Initialize the video generation agent."""
    # Initialize tmp directories
    await initialize_tmp_directories()

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
You are a product advertisement creative agent.

Format goal: Create a Product Advertisement creative concept → write storyboard (3 scenes) → edit storyboard → generate_image → ensure each scene has a hero image → generate video per scene (veo3 image-to-video) → wait for all clips to finish → concat_videos → analyze_video_quality. Produce a 3-scene video: beginning, middle, end.

Guidance: Only start generating videos when each scene has an image. Generate all scene videos asynchronously and wait for all to complete (use the batch tool) before stitching. Then call concat_videos with the ordered list of clip paths.

Audio: ALWAYS generate videos with audio (generate_audio: true) unless the user explicitly requests silent videos. Audio enhances the viewing experience and engagement.

Note: Video generation endpoints often have short max durations. For longer videos, generate multiple clips and then concatenate them.
"""
    )

    return {
        "messages": initial_messages + [system_message],
        "current_iteration": 0,
        "quality_satisfied": False,
        "max_iterations_reached": False,
        "renders": [],
    }


async def clean_up_files(
    state: VideoGenerationState, runtime: Runtime[VideoGenerationContext]
) -> Dict[str, Any]:
    """Clean up temporary files while preserving the final video output."""
    # Perform best-effort cleanup of temporary artifacts while preserving final output
    inferred_final_path = state.final_video_path
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
        return {
            "messages": [cleanup_msg],
            "final_video_path": inferred_final_path,
        }
    except Exception:
        cleanup_msg = AIMessage(
            content=(
                "Failed to clean up files. "
                f"Preserved: {', '.join(cleanup_result.get('preserved', [])) or 'none'}. "
                f"Removed: {len(cleanup_result.get('removed_files', []))}. "
                f"Errors: {len(cleanup_result.get('errors', []))}."
            )
        )
        return {"messages": [cleanup_msg], "final_video_path": inferred_final_path}


async def agentic_step(
    state: VideoGenerationState, runtime: Runtime[VideoGenerationContext]
) -> Dict[str, Any]:
    """Single agent step: think and decide whether to call tools."""
    # Ensure tmp dirs exist; idempotent
    await initialize_tmp_directories()

    tools = [
        create_story_board_tool,
        update_scene_tool,
        generate_image_tool,
        veo3_tool,
        veo3_batch_tool,
        concat_videos_tool,
        run_ffmpeg_tool,
        analyze_video_quality_tool,
    ]

    llm = (
        ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=runtime.context.google_api_key,
            temperature=0.6,
            max_retries=3,  # Built-in retry for API calls
        )
        .bind_tools(tools)
        .with_retry(
            retry_if_exception_type=(
                Exception,  # Retry on all exceptions (network, rate limit, etc.)
            ),
            wait_exponential_jitter=True,  # Add jitter to avoid thundering herd
            stop_after_attempt=3,  # Try up to 3 times total
            exponential_jitter_params={
                "initial": 1,  # Start with 1 second delay
                "max": 10,  # Max 10 seconds between retries
                "exp_base": 2,  # Exponential backoff base
            },
        )
    )

    # Continue the conversation based on state.messages
    prior = list(state.messages or [])
    if not prior:
        # Ensure at least one human message (Gemini requires contents)
        prior = [
            HumanMessage(
                content=(
                    "No prior input provided. Propose a 3-scene product ad storyboard "
                    "(beginning, middle, end). Ask for missing details if needed."
                )
            )
        ]
    else:
        # If there are no human messages yet, append a bootstrap human turn
        has_human = any(isinstance(m, HumanMessage) for m in prior)
        if not has_human:
            prior.append(
                HumanMessage(
                    content=(
                        "Continue the creative workflow. If context is insufficient, "
                        "ask clarifying questions to proceed."
                    )
                )
            )

    ai_msg = await llm.ainvoke(prior)

    # Increment iteration counter
    next_iter = (state.current_iteration or 0) + 1

    return {
        "messages": [ai_msg],
        "current_iteration": next_iter,
        "max_iterations_reached": False,  # Never stop iterations automatically
    }


def route_after_agent(state: VideoGenerationState) -> str:
    """Route to tools if AI requested tool calls, or end when agent has no more actions."""
    if not state.messages:
        return "tools"
    last = state.messages[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    return "tools" if tool_calls else "end"


tool_node = ToolNode(
    [
        create_story_board_tool,
        update_scene_tool,
        generate_image_tool,
        veo3_tool,
        veo3_batch_tool,
        concat_videos_tool,
        run_ffmpeg_tool,
        analyze_video_quality_tool,
    ]
)


# Define the graph
graph = (
    StateGraph(VideoGenerationState, context_schema=VideoGenerationContext)
    .add_node("initialize", initialize_agent)
    .add_node("agent", agentic_step)
    .add_node("tools", tool_node)
    .add_edge(START, "initialize")
    .add_edge("initialize", "agent")
    .add_conditional_edges("agent", route_after_agent, {"tools": "tools", "end": END})
    .add_edge("tools", "agent")
    .compile(name="Video Generation Agent")
)
