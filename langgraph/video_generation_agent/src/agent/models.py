"""Pydantic models for video generation agent data structures.

This module defines the data models used throughout the video generation agent
for type safety and validation of video generation workflows.
"""

from __future__ import annotations

import os
from typing import List, Literal, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

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
        description="fal.ai API key for Kling endpoints",
    )
    # Legacy/unused keys kept optional for compatibility
    elevenlabs_api_key: Optional[str] = Field(None, description="ElevenLabs API key")
    unsplash_access_key: Optional[str] = Field(None, description="Unsplash API key")


class VideoGenerationState(BaseModel):
    """Minimal state for the video generation agent."""

    # Conversation and required input
    messages: List[BaseMessage] = Field(default_factory=list)

    # Iteration control
    current_iteration: Optional[int] = Field(
        None, description="Current iteration count"
    )
    max_iterations_reached: Optional[bool] = Field(
        None, description="Whether max iterations reached"
    )

    # Final artifact
    final_video_path: Optional[str] = Field(None, description="Final video file path")

    # User-provided visuals (first chat)
    user_image_paths: Optional[List[str]] = Field(None, description="Local image paths")
    user_image_urls: Optional[List[str]] = Field(None, description="Image URLs")


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

    prompt: str = Field(..., description="Visual prompt for image/video generation")
    duration_seconds: float = Field(6.0, gt=0, description="Scene duration in seconds")
    rendered_video_file_path: Optional[str] = Field(
        None, description="Path to rendered video file for this scene"
    )


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
    num_images: int = Field(1, ge=1, le=4)
    output_basename: str = Field(
        "product_ad_image", description="Base filename for saved images (no ext)"
    )
    input_image_paths: Optional[List[str]] = Field(
        None, description="Optional local image paths for edit/composition"
    )
    input_image_urls: Optional[List[str]] = Field(
        None, description="Optional image URLs for edit/composition"
    )


class KlingVideoRequest(BaseModel):
    """Request to generate a video from a still image using fal.ai Kling endpoint."""

    prompt: str = Field(..., description="Video prompt / scene description")
    image_path: Optional[str] = Field(
        None, description="Local path to input image; uploaded if provided"
    )
    image_url: Optional[str] = Field(
        None, description="Public URL to input image; used if provided"
    )
    duration_seconds: float = Field(6.0, gt=1.0, le=60.0)
    endpoint: str = Field(
        "fal-ai/kling/v1", description="fal endpoint id for Kling variant"
    )
    seed: Optional[int] = Field(None, description="Optional seed for reproducibility")


class KlingVideoResult(BaseModel):
    """Result metadata for Kling video generation."""

    request_id: Optional[str] = None
    video_url: Optional[str] = None
    video_path: Optional[str] = None
    logs: Optional[List[str]] = None


class VideoQualityAssessment(BaseModel):
    """Structured output for video quality assessment."""

    visual_quality: int = Field(ge=0, le=10, description="Visual quality score (0-10)")
    audio_quality: int = Field(ge=0, le=10, description="Audio quality score (0-10)")
    narrative_coherence: int = Field(
        ge=0, le=10, description="Narrative coherence score (0-10)"
    )
    feedback: str = Field(description="Detailed feedback and suggestions")
