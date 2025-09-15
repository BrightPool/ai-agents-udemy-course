"""Pydantic models for video generation agent data structures.

This module defines the data models used throughout the video generation agent
for type safety and validation of video creation requests, asset management,
and quality assessment responses.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# Asset type enum as Literal for strict typing
AssetType = Literal["image", "audio", "video"]

# Image quality options for Unsplash
ImageQuality = Literal["raw", "full", "regular", "small", "thumb"]

# Video resolution presets
VideoResolution = Literal[
    "480p", "720p", "1080p", "1440p", "2160p", "3840x2160", "1920x1080", "1280x720"
]


class MediaAsset(BaseModel):
    """Schema for media library assets."""

    id: str = Field(..., description="Unique asset identifier")
    type: AssetType = Field(..., description="Asset type (image, audio, video)")
    filename: str = Field(..., description="Original filename")
    description: str = Field(..., description="Human-readable description")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    path: str = Field(..., description="File system path")


class RenderRecord(BaseModel):
    """Record for a single render attempt/output."""

    media_used: List[str] = Field(
        default_factory=list, description="List of asset file paths used in render"
    )
    ffmpeg_command: str = Field(
        ..., description="FFmpeg command or sub-command used for this render"
    )
    generated_video_file_path: Optional[str] = Field(
        None, description="Path to the generated video file"
    )
    quality_score: Optional[float] = Field(
        None, ge=0, le=10, description="Quality score assessed for this render"
    )
    quality_breakdown: Optional[QualityBreakdown] = Field(
        None, description="Breakdown of component quality scores"
    )


class Veo3ShotSpec(BaseModel):
    """Minimal composition spec for Veo3-style generation."""

    prompt: str = Field(..., description="Shot prompt for Veo3")
    duration: float = Field(5.0, gt=0, description="Shot duration in seconds")
    assets: List[str] = Field(
        default_factory=list, description="Referenced asset paths"
    )
    camera: Optional[str] = Field(None, description="Optional camera instruction")
    transition: Optional[str] = Field(None, description="Transition from previous shot")


class VideoScript(BaseModel):
    """Structured video script with TTS text and Veo3 JSON composition."""

    script_text: str = Field(..., description="Narration script for TTS")
    veo3_composition: List[Veo3ShotSpec] = Field(
        default_factory=list, description="Veo3 composition shots"
    )


class QualityBreakdown(BaseModel):
    """Component scores for video quality analysis."""

    visual_quality: int = Field(..., ge=0, le=10)
    audio_quality: int = Field(..., ge=0, le=10)
    narrative_coherence: int = Field(..., ge=0, le=10)
    total: int = Field(..., ge=0, le=30)


class VideoGenerationContext(BaseModel):
    """Context parameters for the video generation agent."""

    max_iterations: int = Field(
        3, description="Maximum quality improvement iterations", ge=1, le=10
    )
    target_quality_score: float = Field(
        8.0, description="Target quality score", ge=0, le=10
    )
    google_api_key: str = Field(..., description="Google Gemini API key")
    elevenlabs_api_key: str = Field(..., description="ElevenLabs API key")
    unsplash_access_key: str = Field(..., description="Unsplash API key")


class VideoGenerationState(BaseModel):
    """State for the video generation agent."""

    # Core state
    messages: List[BaseMessage] = Field(default_factory=list)

    # Required inputs
    user_request: str = Field(..., description="User's video generation request")

    # Video generation state
    video_quality: Optional[float] = Field(None, ge=0, le=10)
    ffmpeg_command: Optional[str] = Field(None, description="FFmpeg command used")
    assets_used: Optional[List[str]] = Field(None, description="Assets used in render")
    number_of_renders: Optional[int] = Field(
        None, description="Number of renders completed"
    )
    renders: Optional[List[RenderRecord]] = Field(None, description="Render records")

    # Iteration control
    current_iteration: Optional[int] = Field(
        None, description="Current iteration count"
    )
    quality_satisfied: Optional[bool] = Field(
        None, description="Whether quality target met"
    )
    max_iterations_reached: Optional[bool] = Field(
        None, description="Whether max iterations reached"
    )

    # Generated content paths
    generated_audio_path: Optional[str] = Field(
        None, description="Path to generated audio"
    )
    generated_subtitle_path: Optional[str] = Field(
        None, description="Path to generated subtitles"
    )
    downloaded_images: Optional[List[str]] = Field(
        None, description="Downloaded image paths"
    )
    final_video_path: Optional[str] = Field(None, description="Final video file path")

    # Quality assessment
    quality_feedback: Optional[str] = Field(None, description="Quality feedback text")
    improvement_suggestions: Optional[List[str]] = Field(
        None, description="Improvement suggestions"
    )


class VideoGenerationRequest(BaseModel):
    """Schema for video generation requests."""

    user_request: str = Field(..., description="Natural language video description")
    duration: Optional[float] = Field(
        30.0, description="Desired video duration in seconds", gt=0
    )
    resolution: VideoResolution = Field("1080p", description="Video resolution preset")
    include_audio: bool = Field(True, description="Whether to include generated audio")
    include_subtitles: bool = Field(True, description="Whether to include subtitles")
    max_iterations: int = Field(
        3, description="Maximum quality improvement iterations", ge=1, le=10
    )
    target_quality: float = Field(
        8.0, description="Target quality score (0-10)", ge=0, le=10
    )


class TextToSpeechRequest(BaseModel):
    """Schema for text-to-speech generation requests."""

    text: str = Field(..., description="Text to convert to speech", min_length=1)
    voice_id: str = Field("21m00Tcm4TlvDq8ikWAM", description="ElevenLabs voice ID")
    model: str = Field("eleven_monolingual_v1", description="ElevenLabs model ID")


class ImageSearchRequest(BaseModel):
    """Schema for Unsplash image search requests."""

    query: str = Field(..., description="Search query for images", min_length=1)
    count: int = Field(1, description="Number of images to download", ge=1, le=10)
    quality: ImageQuality = Field("regular", description="Image quality/size")


class SubtitleRequest(BaseModel):
    """Schema for subtitle generation requests."""

    subtitle_text: str = Field(..., description="Text to display as subtitle")
    start_time: float = Field(0.0, description="Start time in seconds", ge=0)
    duration: float = Field(5.0, description="Duration in seconds", gt=0)
    font_size: int = Field(24, description="Font size in pixels", ge=8, le=72)
    font_color: str = Field("white", description="Font color (name or hex)")


class VideoCreationRequest(BaseModel):
    """Schema for video creation requests."""

    input_files: List[str] = Field(
        ..., description="List of input file paths", min_length=1
    )
    output_path: str = Field(..., description="Output video file path")
    duration: float = Field(10.0, description="Video duration in seconds", gt=0)
    resolution: str = Field("1920x1080", description="Video resolution (WIDTHxHEIGHT)")
    fps: int = Field(30, description="Frames per second", ge=1, le=60)


class QualityAssessmentResult(BaseModel):
    """Schema for video quality assessment results."""

    quality_score: float = Field(..., description="Quality score (0-10)", ge=0, le=10)
    feedback: str = Field(..., description="Detailed quality feedback and suggestions")
    improvement_suggestions: List[str] = Field(
        default_factory=list, description="Specific improvement recommendations"
    )
    target_met: bool = Field(..., description="Whether target quality was achieved")


class AssetSearchRequest(BaseModel):
    """Schema for media library search requests."""

    query: str = Field(..., description="Search query", min_length=1)
    asset_type: Optional[AssetType] = Field(None, description="Filter by asset type")


class AssetSearchResult(BaseModel):
    """Schema for media library search results."""

    assets: List[MediaAsset] = Field(
        default_factory=list, description="Matching assets"
    )
    total_count: int = Field(..., description="Total number of matching assets")


class VideoGenerationSummary(BaseModel):
    """Schema for video generation process summary."""

    final_video_path: str = Field(..., description="Path to final video file")
    total_iterations: int = Field(
        ..., description="Total number of iterations performed"
    )
    total_renders: int = Field(..., description="Total number of video renders")
    final_quality_score: float = Field(
        ..., description="Final quality assessment score", ge=0, le=10
    )
    assets_used: List[str] = Field(
        default_factory=list, description="List of asset paths used"
    )
    generated_audio_path: Optional[str] = Field(
        None, description="Path to generated audio file"
    )
    generated_subtitle_path: Optional[str] = Field(
        None, description="Path to generated subtitle file"
    )
    process_duration: Optional[float] = Field(
        None, description="Total process duration in seconds"
    )
