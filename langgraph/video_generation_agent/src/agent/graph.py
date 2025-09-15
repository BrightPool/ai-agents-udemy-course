"""Video Generation LangGraph Agent.

A sophisticated agent for creating videos with iterative quality improvement.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

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

from .models import (
    QualityBreakdown,
    RenderRecord,
    VideoGenerationContext,
    VideoGenerationState,
)
from .tools import (
    ToolExecutionError,
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


async def plan_video_creation(
    state: VideoGenerationState, runtime: Runtime[VideoGenerationContext]
) -> Dict[str, Any]:
    """Plan the video creation process based on user request."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=runtime.context.google_api_key,
        temperature=0.7,
    )

    planning_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a video production expert. Analyze the user's request and create a detailed plan for video creation.

Consider:
- What type of video is needed?
- What assets (images, audio, video clips) are required?
- What text-to-speech content is needed?
- What subtitles or captions are required?
- What should be the video duration and quality settings?

Provide a structured plan with specific steps.""",
            ),
            ("human", "User request: {user_request}"),
        ]
    )

    messages = planning_prompt.format_messages(user_request=state.user_request)
    response = await llm.ainvoke(messages)

    return {
        "messages": [HumanMessage(content=state.user_request), response],
        "current_iteration": (state.current_iteration or 0) + 1,
    }


async def gather_assets(
    state: VideoGenerationState, runtime: Runtime[VideoGenerationContext]
) -> Dict[str, Any]:
    """Gather required assets for video creation."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=runtime.context.google_api_key,
        temperature=0.4,
    )

    # Enable tool calling
    tools = [
        search_media_library_tool,
        search_unsplash_media_tool,
        elevenlabs_tts_tool,
        generate_ass_file_tool_tool,
    ]
    llm_with_tools = llm.bind_tools(tools)

    # Seed conversation with plan and instructions
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You can call tools to acquire assets and create them when needed.
Always prefer existing media from the media library before fetching from Unsplash.
If narration is present, generate TTS using ElevenLabs and then create an ASS subtitle file with highlighted karaoke captions.
Return a short confirmation when done; do not repeat tool results.
                """.strip(),
            ),
            ("human", "Video plan: {plan}"),
        ]
    )

    plan_content = state.messages[-1].content if state.messages else ""
    messages = prompt.format_messages(plan=plan_content)

    # Execute up to 4 rounds of tool calls
    gathered_images: List[str] = list(state.downloaded_images or [])
    generated_audio_path: str | None = state.generated_audio_path
    generated_subtitle_path: str | None = state.generated_subtitle_path
    assets_used: List[str] = list(state.assets_used or [])

    for _ in range(4):
        ai_msg = await llm_with_tools.ainvoke(messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            break

        for call in tool_calls:
            name = call.get("name")
            args = call.get("args", {})

            # Dispatch tool
            if name == "search_unsplash_media":
                result = search_unsplash_media_tool.invoke(args)
                # result is List[str]
                if isinstance(result, list):
                    gathered_images.extend([p for p in result if isinstance(p, str)])
                    assets_used.extend([p for p in result if isinstance(p, str)])
                messages.append(
                    ToolMessage(
                        content=json.dumps(result, default=str),
                        tool_call_id=call.get("id", ""),
                    )
                )
            elif name == "search_media_library":
                result = search_media_library_tool.invoke(args)
                # result is AssetSearchResult
                try:
                    assets = result.assets if hasattr(result, "assets") else []  # type: ignore[attr-defined]
                    paths = [getattr(a, "path", None) for a in assets]
                    for p in paths:
                        if isinstance(p, str):
                            assets_used.append(p)
                except Exception:
                    pass
                messages.append(
                    ToolMessage(
                        content=json.dumps(
                            result,
                            default=lambda o: getattr(
                                o, "model_dump", lambda: str(o)
                            )(),
                        ),
                        tool_call_id=call.get("id", ""),
                    )
                )
            elif name == "elevenlabs_text_to_speech":
                result = elevenlabs_tts_tool.invoke(args)
                if isinstance(result, str) and os.path.exists(result):
                    generated_audio_path = result
                    assets_used.append(result)
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=call.get("id", ""))
                )
            elif name == "generate_ass_file_tool":
                result = generate_ass_file_tool_tool.invoke(args)
                if isinstance(result, str) and os.path.exists(result):
                    generated_subtitle_path = result
                    assets_used.append(result)
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=call.get("id", ""))
                )
            else:
                messages.append(
                    ToolMessage(
                        content=f"Unknown tool: {name}", tool_call_id=call.get("id", "")
                    )
                )

    # Deduplicate assets
    def _dedupe(seq: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    gathered_images = _dedupe([p for p in gathered_images if isinstance(p, str)])
    assets_used = _dedupe([p for p in assets_used if isinstance(p, str)])

    # Compose response message for chat history
    final_ack = SystemMessage(
        content=(
            f"Assets ready. Images: {len(gathered_images)}, "
            f"Audio: {'yes' if generated_audio_path else 'no'}, "
            f"Subtitles: {'yes' if generated_subtitle_path else 'no'}."
        )
    )

    return {
        "messages": [final_ack],
        "downloaded_images": gathered_images or state.downloaded_images,
        "generated_audio_path": generated_audio_path or state.generated_audio_path,
        "generated_subtitle_path": generated_subtitle_path
        or state.generated_subtitle_path,
        "assets_used": assets_used or state.assets_used,
    }


async def create_video_content(
    state: VideoGenerationState, runtime: Runtime[VideoGenerationContext]
) -> Dict[str, Any]:
    """Create the actual video content using gathered assets."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=runtime.context.google_api_key,
        temperature=0.3,
    )

    # Generate video creation instructions
    creation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are orchestrating iterative video creation in a single thread.

You will be given:
- The two most recent renders (command, outputs, quality).
- Currently available assets (images, audio, subtitles).

Decide the next step:
- If assets are insufficient, call tools to gather or generate assets.
- Otherwise, produce a better ffmpeg command with create_video.

Be concise. Return a short rationale followed by the next action.""",
            ),
            ("human", "Recent renders: {recent}\nAssets available: {assets}"),
        ]
    )

    assets_info = (
        f"Audio: {state.generated_audio_path or ''}, "
        f"Images: {state.downloaded_images or []}, "
        f"Subtitles: {state.generated_subtitle_path or ''}"
    )
    # Fetch last two renders for context
    recent_renders = list(state.renders or [])
    # Convert to dicts for tool compatibility
    recent_payload = [r.model_dump() for r in recent_renders[-2:]]
    recent = list_recent_renders_tool.invoke({"renders": recent_payload, "limit": 2})

    messages = creation_prompt.format_messages(
        recent=json.dumps(recent), assets=assets_info
    )
    response = await llm.ainvoke(messages)

    # Plan the ffmpeg command, then execute it
    try:
        # Compose minimal request from state
        input_files = state.downloaded_images or []
        if state.generated_audio_path:
            input_files = input_files + [state.generated_audio_path or ""]

        plan_payload = {
            "input_files": input_files,
            "output_path": state.final_video_path or "/tmp/videos/output.mp4",
            "duration": 10.0,
            "resolution": "1920x1080",
            "fps": 30,
            "subtitle_path": state.generated_subtitle_path or None,
        }

        plan_result = create_video_tool.invoke(plan_payload)
        command_str = plan_result.get("ffmpeg_command", "")
        planned_output = plan_result.get("output_path", plan_payload["output_path"])

        exec_result = execute_ffmpeg_tool.invoke(
            {
                "command": command_str,
                "timeout_seconds": 180,
                "output_path": planned_output,
            }
        )

        final_path = exec_result.get("output_path", planned_output)
    except ToolExecutionError as e:
        final_path = ""
        command_str = getattr(e, "command", "") or ""

    # Append render record
    new_renders = list(state.renders or [])
    import time

    new_renders.append(
        RenderRecord(
            media_used=state.assets_used or [],
            ffmpeg_command=command_str,
            generated_video_file_path=final_path or None,
            created_at=time.time(),
            quality_score=None,
            quality_breakdown=None,
        )
    )

    return {
        "messages": [response],
        "number_of_renders": (state.number_of_renders or 0) + 1,
        "final_video_path": final_path,
        "ffmpeg_command": command_str,
        "renders": new_renders,
    }


async def assess_quality(
    state: VideoGenerationState, runtime: Runtime[VideoGenerationContext]
) -> Dict[str, Any]:
    """Assess the quality of the generated video."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=runtime.context.google_api_key,
        temperature=0.7,
    )

    # Simulate quality assessment (in practice, you might analyze the actual video)
    quality_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Assess the quality of the generated video and provide feedback:

1. Rate the overall quality (0-10 scale)
2. Identify areas for improvement
3. Suggest specific changes for the next iteration
4. Determine if quality target is met

Be specific and actionable in your feedback.""",
            ),
            ("human", "Video details: {video_info}"),
        ]
    )

    video_info = (
        f"Path: {state.final_video_path or ''}, "
        f"Renders: {state.number_of_renders or 0}, "
        f"Assets: {len(state.assets_used or [])}"
    )
    messages = quality_prompt.format_messages(video_info=video_info)
    response = await llm.ainvoke(messages)

    # Try real quality analysis via tool, fallback to heuristic
    # Provide recent context (last 3 renders) and target score to calibrate evaluation
    recent_payload = [r.model_dump() for r in (state.renders or [])][-3:]
    tool_quality = analyze_video_quality_tool.invoke(
        {
            "video_path": state.final_video_path or "",
            "recent_renders": recent_payload,
            "target_quality_score": runtime.context.target_quality_score,
        }
    )
    if isinstance(tool_quality, dict) and "quality_score" in tool_quality:
        quality_score = float(tool_quality["quality_score"])  # type: ignore[arg-type]
        feedback_text = str(tool_quality.get("feedback", ""))
        breakdown_dict = (
            tool_quality.get("breakdown", {}) if isinstance(tool_quality, dict) else {}
        )
    else:
        quality_score = 7.5
        feedback_text = response.content
        breakdown_dict = {}
    target_quality = runtime.context.target_quality_score
    max_iterations = runtime.context.max_iterations

    quality_satisfied = quality_score >= target_quality
    max_iterations_reached = (state.current_iteration or 0) >= max_iterations

    # Update the most recent render with the quality score if present
    updated_renders = list(state.renders or [])
    if updated_renders:
        breakdown_obj = None
        if breakdown_dict:
            try:
                breakdown_obj = QualityBreakdown(
                    visual_quality=int(breakdown_dict.get("visual_quality", 0)),
                    audio_quality=int(breakdown_dict.get("audio_quality", 0)),
                    narrative_coherence=int(
                        breakdown_dict.get("narrative_coherence", 0)
                    ),
                    total=int(breakdown_dict.get("total", 0)),
                )
            except Exception:
                breakdown_obj = None

        last = updated_renders[-1].model_copy(
            update={
                "quality_score": quality_score,
                "quality_breakdown": breakdown_obj,
            }
        )
        updated_renders[-1] = last

    return {
        "messages": [response],
        "video_quality": quality_score,
        "quality_satisfied": quality_satisfied,
        "max_iterations_reached": max_iterations_reached,
        "quality_feedback": feedback_text,
        "renders": updated_renders,
    }


def should_continue_iteration(
    state: VideoGenerationState,
) -> str:
    """Determine if we should continue iterating or finish."""
    if state.quality_satisfied:
        return "finish"
    elif state.max_iterations_reached:
        return "finish"
    else:
        return "iterate"


async def improve_video(
    state: VideoGenerationState, runtime: Runtime[VideoGenerationContext]
) -> Dict[str, Any]:
    """Improve the video based on quality feedback."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=runtime.context.google_api_key,
        temperature=0.7,
    )

    improvement_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Based on the quality feedback, improve the video:

1. Address specific quality issues mentioned
2. Enhance visual and audio elements
3. Adjust video parameters for better quality
4. Regenerate or modify assets as needed

Focus on the most impactful improvements.""",
            ),
            ("human", "Quality feedback: {feedback}"),
        ]
    )

    messages = improvement_prompt.format_messages(feedback=state.quality_feedback or "")
    response = await llm.ainvoke(messages)

    return {
        "messages": [response],
        "current_iteration": (state.current_iteration or 0) + 1,
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
