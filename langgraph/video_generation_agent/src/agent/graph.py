"""Video Generation LangGraph Agent.

A sophisticated agent for creating videos with iterative quality improvement.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from typing import Any, Dict, List, Optional, cast

from google import genai  # type: ignore[import-untyped]
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from agent.models import (
    VideoGenerationContext,
    VideoGenerationState,
)
from agent.tools import (
    cleanup_tmp_directories,
    initialize_tmp_directories,
)
from agent.tools import (
    create_story_board as create_story_board_tool,
)
from agent.tools import (
    generate_image as generate_image_tool,
)
from agent.tools import (
    kling_generate_video_from_image as kling_tool,
)
from agent.tools import (
    run_ffmpeg_binary as run_ffmpeg_tool,
)
from agent.tools import (
    score_video as score_video_tool,
)
from agent.tools import (
    update_story_board as update_story_board_tool,
)


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
Tools available:
1. create_story_board: Draft a 3-scene storyboard (beginning, middle, end)
2. update_story_board: Apply edits to the storyboard
3. generate_image: Generate a hero image (product + person) using Gemini
4. kling_generate_video_from_image: Generate a short video from the image (fal.ai Kling)
5. run_ffmpeg_binary: Execute ffmpeg to post-process if needed
6. score_video: Evaluate video quality (0-10) and feedback

Format goal: Create a Product Advertisement creative concept → write storyboard → edit storyboard → generate_image → generate_video_from_image → score_video. Produce a 3-scene video: beginning, middle, end.

The other important things that are important is that kling can only produce 10 seconds of video. So if you need to generate a longer video, you need to generate multiple videos and then concatenate them.
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
                if isinstance(m, ToolMessage) and getattr(m, "name", "") in (
                    "execute_ffmpeg",
                    "run_ffmpeg_binary",
                ):
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
        return {
            "messages": [response, cleanup_msg],
            "final_video_path": inferred_final_path,
        }
    except Exception:
        # Even if cleanup fails, return the summary response
        return {"messages": [response], "final_video_path": inferred_final_path}


async def agentic_step(
    state: VideoGenerationState, runtime: Runtime[VideoGenerationContext]
) -> Dict[str, Any]:
    """Single agent step: think and decide whether to call tools."""
    # Ensure tmp dirs exist; idempotent
    await initialize_tmp_directories()

    tools = [
        create_story_board_tool,
        update_story_board_tool,
        generate_image_tool,
        kling_tool,
        run_ffmpeg_tool,
        score_video_tool,
    ]

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=runtime.context.google_api_key,
        temperature=0.6,
    ).bind_tools(tools)

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


async def attach_video_renders_to_chat(
    state: VideoGenerationState, runtime: Runtime[VideoGenerationContext]
) -> Dict[str, Any]:
    """Attach recent video renders as base64 and optional provider file IDs to chat history.

    Looks at the latest ToolMessage, extracts fields like output_path/video_path,
    reads the file if present, encodes as base64, and appends a HumanMessage
    with multimodal content blocks. If Google GenAI client is available, also
    uploads file and appends a message with provider-managed file_id.
    """
    attachments: List[HumanMessage] = []

    # Find most recent tool payload with possible video paths
    tool_payload: Optional[Dict[str, Any]] = None
    last_tool_name: Optional[str] = None
    for m in reversed(list(state.messages or [])):
        if isinstance(m, ToolMessage):
            last_tool_name = getattr(m, "name", None)
            raw = getattr(m, "content", "")
            if isinstance(raw, str) and raw:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        tool_payload = parsed
                        break
                except Exception:
                    continue

    if not tool_payload:
        return {"messages": []}

    candidate_paths: List[str] = []
    for key in ("output_path", "video_path"):
        val = tool_payload.get(key)
        if isinstance(val, str):
            candidate_paths.append(val)
    for key in ("outputs", "videos"):
        val = tool_payload.get(key)
        if isinstance(val, list):
            candidate_paths.extend([v for v in val if isinstance(v, str)])

    if not candidate_paths:
        return {"messages": []}

    client: Optional[genai.Client] = None
    api_key = runtime.context.google_api_key
    try:
        if api_key:
            client = genai.Client(api_key=api_key)
    except Exception:
        client = None

    for path in candidate_paths:
        try:
            if not (isinstance(path, str) and os.path.exists(path)):
                continue
            mime_type, _ = mimetypes.guess_type(path)
            mime_type = mime_type or "video/mp4"
            with open(path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode("utf-8")

            content_blocks = [
                {
                    "type": "text",
                    "text": f"Rendered video from {last_tool_name or 'tool'}: {os.path.basename(path)}",
                },
                {"type": "video", "base64": b64, "mime_type": mime_type},
            ]
            attachments.append(HumanMessage(content=cast(List[Any], content_blocks)))

            if client is not None:
                try:
                    uploaded = client.files.upload(file=path)
                    file_id = getattr(uploaded, "name", None) or getattr(
                        uploaded, "id", None
                    )
                    if isinstance(file_id, str):
                        content_file_id = [
                            {
                                "type": "text",
                                "text": "Provider file reference for rendered video.",
                            },
                            {"type": "video", "file_id": file_id},
                        ]
                        attachments.append(
                            HumanMessage(content=cast(List[Any], content_file_id))
                        )
                except Exception:
                    pass
        except Exception:
            continue

    if not attachments:
        return {}
    return {"messages": state.messages + attachments}


tool_node = ToolNode(
    [
        create_story_board_tool,
        update_story_board_tool,
        # generate_image_tool,
        # kling_tool,
        # run_ffmpeg_tool,
        # score_video_tool,
    ]
)


# Define the graph
graph = (
    StateGraph(VideoGenerationState, context_schema=VideoGenerationContext)
    .add_node("initialize", initialize_agent)
    .add_node("agent", agentic_step)
    .add_node("tools", tool_node)
    .add_node("attach_media", attach_video_renders_to_chat)
    .add_node("finalize", finalize_video)
    .add_edge(START, "initialize")
    .add_edge("initialize", "agent")
    .add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "finalize": "finalize"}
    )
    .add_edge("tools", "attach_media")
    .add_edge("attach_media", "agent")
    .add_edge("finalize", END)
    .compile(name="Video Generation Agent")
)
