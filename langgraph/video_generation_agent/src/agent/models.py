"""Pydantic models for video generation agent data structures.

This module defines the data models used throughout the video generation agent
for type safety and validation of video generation workflows.
"""

from __future__ import annotations

import os
import uuid
from typing import Annotated, List, Literal, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class VideoGenerationState(BaseModel):
    """Minimal state for the video generation agent."""

    # Conversation and required input
    messages: Annotated[List[BaseMessage], add_messages] = []

    # Iteration control
    current_iteration: Optional[int] = Field(
        None, description="Current iteration count"
    )
    max_iterations_reached: Optional[bool] = Field(
        None, description="Whether max iterations reached"
    )

    # Final artifact
    final_video_path: Optional[str] = Field(None, description="Final video file path")
    storyboard: Optional[Storyboard] = Field(None, description="Current storyboard")


# Video resolution presets
VideoResolution = Literal[
    "480p", "720p", "1080p", "1440p", "2160p", "3840x2160", "1920x1080", "1280x720"
]


class VideoGenerationContext(BaseModel):
    """Context parameters for the video generation agent."""

    max_iterations: int = Field(
        3, description="Maximum quality improvement iterations", ge=1, le=10
    )
    target_quality_score: float = Field(
        8.0, description="Target quality score", ge=0, le=10
    )
    google_api_key: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""),
        description="Google Gemini API key",
    )
    fal_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("FAL_API_KEY"),
        description="fal.ai API key for video generation endpoints",
    )


class FFmpegExecuteRequest(BaseModel):
    """Schema for executing a prepared ffmpeg command."""

    command: str = Field(..., description="The ffmpeg command to run", min_length=1)
    timeout_seconds: int = Field(
        180, ge=10, le=7200, description="Max seconds to allow ffmpeg to run"
    )
    output_path: Optional[str] = Field(
        None,
        description=(
            "Optional expected output path to ensure parent directory exists and "
            "to echo back in results"
        ),
    )


"""
Additional models for storyboard and image/video generation tools.
"""


class StoryboardScene(BaseModel):
    """Single storyboard scene with prompt and duration."""

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique scene id"
    )
    title: Optional[str] = Field(None, description="Scene title")
    description: Optional[str] = Field(None, description="Scene description")
    prompt: str = Field(..., description="Visual prompt for image/video generation")
    duration_seconds: float = Field(6.0, gt=0, description="Scene duration in seconds")
    rendered_video_file_path: Optional[str] = Field(
        None, description="Path to rendered video file for this scene"
    )
    # Simple scoring attached to each scene (KISS)
    quality_score: Optional[float] = Field(
        None, ge=0, le=10, description="Quality score (0-10)"
    )
    feedback: Optional[str] = Field(None, description="Short qualitative feedback")


class Storyboard(BaseModel):
    """Array of storyboard scenes."""

    scenes: List[StoryboardScene] = Field(default_factory=list)


class StoryboardCreateRequest(BaseModel):
    """Request to create a storyboard for a product advertisement."""

    product_name: str
    brand: str
    target_audience: str
    key_message: str
    tone: str
    scenes_count: int = Field(3, ge=1, le=10)
    default_scene_duration_seconds: float = Field(6.0, gt=0)


class StoryboardUpdateRequest(BaseModel):
    """Request to update an existing storyboard given natural language instructions."""

    storyboard: Storyboard
    instructions: str = Field(
        ..., description="Editing instructions for the storyboard"
    )


class GenerateImageRequest(BaseModel):
    """Request to generate image(s) from text using Gemini Image API."""

    prompt: str = Field(..., min_length=1)
    output_basename: str = Field(
        "product_ad_image", description="Base filename for saved images (no ext)"
    )


class Veo3VideoRequest(BaseModel):
    """Request schema for generating video with fal.ai Veo 3 Fast."""

    prompt: str = Field(..., description="Video prompt")
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = Field(
        "16:9", description="Aspect ratio for the resulting video"
    )
    duration: Literal["4s", "6s", "8s"] = Field(
        "8s", description="Desired video duration"
    )
    negative_prompt: Optional[str] = Field(
        None, description="Negative prompt to guide the model away from elements"
    )
    enhance_prompt: bool = Field(
        True, description="Whether to let the model enhance the prompt"
    )
    auto_fix: bool = Field(
        True, description="Automatically attempt to fix policy violations"
    )
    resolution: Literal["720p", "1080p"] = Field(
        "720p", description="Target output resolution"
    )
    generate_audio: bool = Field(
        True, description="Whether to produce audio with the video"
    )
    seed: Optional[int] = Field(None, description="Seed for reproducibility")
    image_path: Optional[str] = Field(
        None, description="Optional local reference image path"
    )
    image_url: Optional[str] = Field(
        None, description="Optional hosted reference image URL"
    )
    image_base64: Optional[str] = Field(
        None, description="Optional raw base64 payload for reference image"
    )
    image_mime_type: Optional[str] = Field(
        "image/png", description="Mime type for the base64 reference image"
    )


class Veo3VideoResult(BaseModel):
    """Result metadata for Veo 3 Fast generation."""

    request_id: Optional[str] = None
    video_url: Optional[str] = None
    video_identifier: Optional[str] = None
    logs: Optional[List[str]] = None


class VideoConcatRequest(BaseModel):
    """Request to concatenate multiple video files using ffmpeg concat."""

    video_paths: List[str] = Field(
        ..., min_length=2, description="Ordered list of video file paths to concatenate"
    )
    output_path: Optional[str] = Field(
        None,
        description="Explicit path for the concatenated video. Defaults to /tmp/videos",
    )
    output_basename: str = Field(
        "concatenated_video", description="Base name used when auto-generating output"
    )


class VideoConcatResult(BaseModel):
    """Result payload for concatenated videos."""

    video_path: Optional[str] = Field(
        None, description="Path to the concatenated video file"
    )
    segments: Optional[List[str]] = Field(
        None, description="Resolved local paths for each segment"
    )
    logs: Optional[List[str]] = Field(
        None, description="ffmpeg logs captured during run"
    )


class VideoQualityAssessment(BaseModel):
    """Structured output for video quality assessment."""

    visual_quality: int = Field(ge=0, le=10, description="Visual quality score (0-10)")
    audio_quality: int = Field(ge=0, le=10, description="Audio quality score (0-10)")
    narrative_coherence: int = Field(
        ge=0, le=10, description="Narrative coherence score (0-10)"
    )
    feedback: str = Field(description="Detailed feedback and suggestions")
