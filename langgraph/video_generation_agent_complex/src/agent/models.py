"""Pydantic models for video generation agent data structures.

This module defines the data models used throughout the video generation agent
for type safety and validation of video creation requests, asset management,
and quality assessment responses.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# Asset type enum as Literal for strict typing
AssetType = Literal["image", "audio", "video"]

# Image quality options for Unsplash
ImageQuality = Literal["raw", "full", "regular", "small", "thumb"]

# Unsplash/stock media type options
UnsplashMediaType = Literal["photo", "video"]
UnsplashOrientation = Literal["landscape", "portrait", "squarish"]
UnsplashContentFilter = Literal["low", "high"]

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
    created_at: Optional[float] = Field(
        None, description="Unix timestamp when the render was created"
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


class OpenAITranscriptionWord(BaseModel):
    """Word-level timing as returned by OpenAI verbose_json with timestamp_granularities=["word"]."""

    word: str = Field(..., description="Tokenized word text")
    start: float = Field(..., ge=0, description="Word start time in seconds")
    end: float = Field(..., gt=0, description="Word end time in seconds")


class OpenAITranscriptionUsage(BaseModel):
    """Lightweight usage metadata returned by OpenAI for transcriptions."""

    type: Optional[str] = Field(None, description="Usage granularity type (e.g., duration)")
    seconds: Optional[int] = Field(None, ge=0, description="Billed seconds if provided")


class OpenAITranscriptionVerboseJson(BaseModel):
    """Subset of OpenAI verbose_json transcription response for convenience and typing."""

    task: Optional[str] = Field(None, description="Task performed (transcribe/translate)")
    language: Optional[str] = Field(None, description="Detected language")
    duration: Optional[float] = Field(None, ge=0, description="Audio duration in seconds")
    text: str = Field("", description="Full transcript text")
    words: Optional[List[OpenAITranscriptionWord]] = Field(
        None, description="Per-word timing information"
    )
    segments: Optional[List[Dict]] = Field(
        None, description="Optional per-segment info as opaque dicts"
    )
    usage: Optional[OpenAITranscriptionUsage] = Field(
        None, description="Optional usage metadata"
    )


class OpenAITranscriptionRequest(BaseModel):
    """Request for transcribing an audio file via OpenAI Whisper/4o-transcribe."""

    file_path: str = Field(..., description="Path to local audio file to transcribe")
    model: str = Field(
        "whisper-1",
        description="OpenAI transcription model (e.g., whisper-1, gpt-4o-transcribe)",
    )
    response_format: Literal["verbose_json"] = Field(
        "verbose_json", description="Always use verbose_json for timing info"
    )
    timestamp_granularities: List[Literal["word", "segment"]] = Field(
        default_factory=lambda: ["word"],
        description="Requested timestamp granularities",
    )
    prompt: Optional[str] = Field(None, description="Optional prompt to guide decoding")
    language: Optional[str] = Field(
        None, description="Optional explicit language code (e.g., en)"
    )


class ASSStyle(BaseModel):
    """ASS style definition for flexible subtitle rendering."""

    name: str = Field("Default", description="Style name")
    fontname: str = Field("Arial", description="Font family name")
    fontsize: int = Field(24, description="Font size in points", ge=8, le=96)
    primary_color: str = Field("white", description="Primary colour (name or hex)")
    secondary_color: str = Field(
        "yellow", description="Secondary (karaoke) colour (name or hex)"
    )
    outline_color: str = Field("black", description="Outline colour (name or hex)")
    back_color: str = Field("&H80000000", description="Background/box colour in ASS or hex")
    bold: bool = Field(False, description="Bold text")
    italic: bool = Field(False, description="Italic text")
    underline: bool = Field(False, description="Underline text")
    strikeout: bool = Field(False, description="Strikeout text")
    scale_x: int = Field(100, ge=1, le=400, description="X scaling percent")
    scale_y: int = Field(100, ge=1, le=400, description="Y scaling percent")
    spacing: int = Field(0, ge=-50, le=200, description="Letter spacing")
    angle: int = Field(0, ge=-359, le=359, description="Rotation angle")
    border_style: int = Field(1, ge=1, le=4, description="ASS border style")
    outline: int = Field(2, ge=0, le=10, description="Outline size")
    shadow: int = Field(2, ge=0, le=10, description="Shadow size")
    alignment: int = Field(2, ge=1, le=9, description="ASS alignment 1-9")
    margin_l: int = Field(10, ge=0, le=200, description="Left margin")
    margin_r: int = Field(10, ge=0, le=200, description="Right margin")
    margin_v: int = Field(10, ge=0, le=200, description="Vertical margin")
    encoding: int = Field(1, ge=0, le=65535, description="Code page encoding")


class WordTiming(BaseModel):
    """Generic word timing for ASS karaoke generation."""

    word: str = Field(..., description="Word text")
    start: float = Field(..., ge=0, description="Start time in seconds")
    end: float = Field(..., gt=0, description="End time in seconds")


class SubtitleRequest(BaseModel):
    """Schema for ASS subtitle generation with optional word timings and styles.

    Backwards-compatible with the previous simple schema but supports advanced
    styling and precise karaoke timings when provided.
    """

    # Content sources
    subtitle_text: Optional[str] = Field(
        None, description="Text to display as subtitle if words not provided"
    )
    words: Optional[List[WordTiming]] = Field(
        None, description="Optional per-word timings to drive karaoke"
    )

    # Timing (used when building a single Dialogue line)
    start_time: float = Field(0.0, description="Start time in seconds", ge=0)
    duration: float = Field(5.0, description="Duration in seconds", gt=0)

    # Simple style (legacy fields) — still honored
    font_size: int = Field(24, description="Font size in points", ge=8, le=96)
    font_color: str = Field("white", description="Primary font color (name or hex)")
    highlight_color: str = Field(
        "yellow", description="Karaoke highlight color (name or hex)"
    )

    # Advanced styles: either provide a style name and styles list, or rely on legacy
    style_name: str = Field(
        "Default", description="Style name to reference in Events"
    )
    styles: Optional[List[ASSStyle]] = Field(
        None, description="Optional list of styles to include in the ASS file"
    )

    # Event options
    layer: int = Field(0, ge=0, le=100, description="Dialogue layer")
    effect: Optional[str] = Field(
        None,
        description="Optional ASS Effect column (e.g., Banner; ignored for karaoke)",
    )
    karaoke: bool = Field(
        True,
        description="If true, add per-word karaoke timing tags (\\k) when words available",
    )


class VideoCreationRequest(BaseModel):
    """Schema for video creation requests."""

    input_files: List[str] = Field(
        ..., description="List of input file paths", min_length=1
    )
    output_path: str = Field(..., description="Output video file path")
    duration: float = Field(10.0, description="Video duration in seconds", gt=0)
    resolution: str = Field("1920x1080", description="Video resolution (WIDTHxHEIGHT)")
    fps: int = Field(30, description="Frames per second", ge=1, le=60)
    subtitle_path: Optional[str] = Field(
        None, description="Optional path to ASS/SRT subtitle file to overlay"
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


class KlingVideoRequest(BaseModel):
    """Schema for Kling text-to-video generation.

    Supports either pure text prompting or starting from a first frame image.
    The underlying provider can be configured via environment variable
    `KLING_PROVIDER` ("replicate" or "fal").
    """

    prompt: str = Field(..., description="Text prompt for video generation")
    negative_prompt: Optional[str] = Field(
        "", description="Things to avoid in the output"
    )
    duration: int = Field(5, ge=1, description="Duration in seconds (commonly 5 or 10)")
    aspect_ratio: Literal["16:9", "9:16", "1:1", "4:3"] = Field(
        "16:9", description="Aspect ratio of the output video"
    )
    start_image_path: Optional[str] = Field(
        None,
        description="Optional local image path to use as first frame",
    )

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

    query: Optional[str] = Field(None, description="Search query")
    asset_type: Optional[AssetType] = Field(None, description="Filter by asset type")
    tags_any: Optional[List[str]] = Field(
        None, description="Match assets containing ANY of these tags"
    )
    tags_all: Optional[List[str]] = Field(
        None, description="Match assets containing ALL of these tags"
    )
    ids_any: Optional[List[str]] = Field(None, description="Filter by specific IDs")


class AssetSearchResult(BaseModel):
    """Schema for media library search results."""

    assets: List[MediaAsset] = Field(
        default_factory=list, description="Matching assets"
    )
    total_count: int = Field(..., description="Total number of matching assets")


class MediaSearchRequest(BaseModel):
    """Schema for external media (e.g., Unsplash) search/download requests."""

    query: str = Field(..., description="Search query for media", min_length=1)
    count: int = Field(1, description="Number of items to download", ge=1, le=10)
    media_type: UnsplashMediaType = Field(
        "photo", description='Media type to search for ("photo" or "video")'
    )
    quality: ImageQuality = Field(
        "regular", description="Image quality for photos (ignored for video)"
    )
    page: int = Field(1, description="Pagination page (Unsplash)", ge=1)
    orientation: Optional[UnsplashOrientation] = Field(
        None, description="Filter by photo orientation (Unsplash)"
    )
    content_filter: UnsplashContentFilter = Field(
        "low", description="Unsplash content safety filter"
    )


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
    renders: Optional[List[RenderRecord]] = Field(
        None, description="Raw render records collected during the process"
    )
