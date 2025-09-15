"""Video Generation LangGraph Agent.

A sophisticated agent for creating videos with iterative quality improvement.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from .models import (
    QualityBreakdown,
    RenderRecord,
    VideoGenerationContext,
    VideoGenerationState,
)
from .tools import (
    ToolExecutionError,
    initialize_tmp_directories,
)
from .tools import (
    analyze_video_quality as analyze_video_quality_tool,
)
from .tools import (
    create_video as create_video_tool,
)

# Context and State are now Pydantic models imported from .models


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
5. create_video: Create videos using ffmpeg

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
        model="gemini-2.0-flash-exp",
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
        model="gemini-2.0-flash-exp",
        google_api_key=runtime.context.google_api_key,
        temperature=0.7,
    )

    # Analyze what assets are needed
    asset_analysis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Based on the video plan, determine what assets are needed:

1. Search the media library for existing assets
2. Search Unsplash for additional images if needed
3. Generate text-to-speech audio if required
4. Create subtitle files if needed

Provide specific search queries and requirements.""",
            ),
            ("human", "Video plan: {plan}"),
        ]
    )

    plan_content = state.messages[-1].content if state.messages else ""
    messages = asset_analysis_prompt.format_messages(plan=plan_content)
    response = await llm.ainvoke(messages)

    # Extract asset requirements from the response
    # This is a simplified version - in practice, you'd parse the response more carefully

    return {"messages": [response]}


async def create_video_content(
    state: VideoGenerationState, runtime: Runtime[VideoGenerationContext]
) -> Dict[str, Any]:
    """Create the actual video content using gathered assets."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=runtime.context.google_api_key,
        temperature=0.7,
    )

    # Generate video creation instructions
    creation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Based on the gathered assets and plan, create the video:

1. Use the create_video tool with appropriate settings
2. Set proper resolution, duration, and quality parameters
3. Combine all assets effectively
4. Generate the final video file

Provide specific parameters for video creation.""",
            ),
            ("human", "Assets available: {assets}"),
        ]
    )

    assets_info = (
        f"Audio: {state.generated_audio_path or ''}, "
        f"Images: {state.downloaded_images or []}, "
        f"Subtitles: {state.generated_subtitle_path or ''}"
    )
    messages = creation_prompt.format_messages(assets=assets_info)
    response = await llm.ainvoke(messages)

    # Attempt to render video via tool to get path + command
    try:
        # Compose minimal request from state
        input_files = state.downloaded_images or []
        if state.generated_audio_path:
            input_files = input_files + [state.generated_audio_path or ""]

        result = create_video_tool.invoke(
            {
                "input_files": input_files,
                "output_path": state.final_video_path or "/tmp/videos/output.mp4",
                "duration": 10.0,
                "resolution": "1920x1080",
                "fps": 30,
            }
        )
        final_path = result.get("output_path", "")
        command_str = result.get("ffmpeg_command", "")
    except ToolExecutionError as e:
        final_path = ""
        command_str = getattr(e, "command", "") or ""

    # Append render record
    new_renders = list(state.renders or [])
    new_renders.append(
        RenderRecord(
            media_used=state.assets_used or [],
            ffmpeg_command=command_str,
            generated_video_file_path=final_path or None,
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
        model="gemini-2.0-flash-exp",
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
    tool_quality = analyze_video_quality_tool.invoke(state.final_video_path or "")
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
        model="gemini-2.0-flash-exp",
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
        model="gemini-2.0-flash-exp",
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

    messages = finalization_prompt.format_messages(
        video_path=state.final_video_path or ""
    )
    response = await llm.ainvoke(messages)

    return {"messages": [response]}


# Define the graph
graph = (
    StateGraph(VideoGenerationState, context_schema=VideoGenerationContext)
    .add_node("initialize", initialize_agent)
    .add_node("plan", plan_video_creation)
    .add_node("gather_assets", gather_assets)
    .add_node("create_content", create_video_content)
    .add_node("assess_quality", assess_quality)
    .add_node("improve", improve_video)
    .add_node("finalize", finalize_video)
    # Define the flow
    .add_edge(START, "initialize")
    .add_edge("initialize", "plan")
    .add_edge("plan", "gather_assets")
    .add_edge("gather_assets", "create_content")
    .add_edge("create_content", "assess_quality")
    # Conditional edges for iteration
    .add_conditional_edges(
        "assess_quality",
        should_continue_iteration,
        {"iterate": "improve", "finish": "finalize"},
    )
    .add_edge("improve", "gather_assets")
    .add_edge("finalize", END)
    .compile(name="Video Generation Agent")
)
