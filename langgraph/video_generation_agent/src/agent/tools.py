"""Custom tools for video generation agent."""

import json
import os
from typing import Any, Dict, List, Optional

import aiofiles
import ffmpeg  # type: ignore[import-untyped]
import httpx
from elevenlabs import ElevenLabs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from .media_assets import MEDIA_LIBRARY_ASSETS
from .models import (
    SubtitleRequest,
    TextToSpeechRequest,
    VideoCreationRequest,
)


class ToolExecutionError(Exception):
    """Custom tool error to surface structured failures to the agent."""

    def __init__(self, message: str, *, command: str | None = None):
        """Initialize the error.

        Args:
            message: Human-readable error message.
            command: Optional ffmpeg command string for debugging.
        """
        super().__init__(message)
        self.command = command


@tool
async def search_unsplash_media(
    query: str, count: int = 1, media_type: str = "photo", quality: str = "regular"
) -> List[str]:
    """Search Unsplash for images or videos and download them to /tmp folder.

    Args:
        query: Search query for media
        count: Number of items to download (max 10)
        media_type: Type of media to search ("photo" or "video")
        quality: Image quality for photos (raw, full, regular, small, thumb)

    Returns:
        List of paths to downloaded media
    """
    try:
        access_key = os.getenv("UNSPLASH_ACCESS_KEY")
        if not access_key:
            raise ValueError("UNSPLASH_ACCESS_KEY environment variable not set")

        # Search for media
        async with httpx.AsyncClient() as client:
            endpoint = "search/photos" if media_type == "photo" else "search/videos"
            response = await client.get(
                f"https://api.unsplash.com/{endpoint}",
                params={
                    "query": query,
                    "per_page": min(count, 10),
                    "client_id": access_key,
                },
            )
            response.raise_for_status()
            data = response.json()

        # Create tmp directory
        media_dir = f"/tmp/unsplash_{media_type}s"
        os.makedirs(media_dir, exist_ok=True)

        downloaded_paths = []

        # Download media
        async with httpx.AsyncClient() as client:
            for i, item in enumerate(data["results"]):
                if media_type == "photo":
                    media_url = item["urls"][quality]
                    filename = (
                        f"unsplash_{query.replace(' ', '_')}_{i}_{item['id']}.jpg"
                    )
                else:  # video
                    media_url = item["video_files"][0]["link"]  # Get first video file
                    filename = (
                        f"unsplash_video_{query.replace(' ', '_')}_{i}_{item['id']}.mp4"
                    )

                filepath = f"{media_dir}/{filename}"

                async with client.stream("GET", media_url) as response:
                    response.raise_for_status()

                    async with aiofiles.open(filepath, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            await f.write(chunk)

                    downloaded_paths.append(filepath)

        return downloaded_paths

    except (ValueError, httpx.HTTPError, OSError) as e:
        return [f"Error downloading media: {str(e)}"]


@tool
def elevenlabs_text_to_speech(request: TextToSpeechRequest) -> str:
    """Generate speech from text using ElevenLabs API.

    Args:
        request: TextToSpeechRequest object containing text, voice_id, and model

    Returns:
        Path to the generated audio file
    """
    try:
        # Set API key from environment
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable not set")

        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=api_key)

        # Generate audio using the correct method
        audio = client.text_to_speech.convert(
            text=request.text, voice_id=request.voice_id, model_id=request.model
        )

        # Save to tmp directory
        os.makedirs("/tmp/audio", exist_ok=True)
        output_path = f"/tmp/audio/speech_{hash(request.text) % 10000}.mp3"

        # Save audio bytes to file
        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        return output_path

    except (ValueError, OSError, RuntimeError) as e:
        return f"Error generating speech: {str(e)}"


@tool
def generate_ass_file_tool(request: SubtitleRequest) -> str:
    """Generate an ASS (Advanced SubStation Alpha) subtitle file.

    Args:
        request: SubtitleRequest object containing subtitle parameters

    Returns:
        Path to the generated ASS file
    """
    try:
        os.makedirs("/tmp/subtitles", exist_ok=True)

        # Create ASS file content
        ass_content = (
            "[Script Info]\n"
            "Title: Generated Subtitle\n"
            "ScriptType: v4.00+\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,Arial,{request.font_size},{request.font_color},&H000000FF,&H00000000,"
            f"&H80000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
            "MarginV, Effect, Text\n"
            f"Dialogue: 0,{request.start_time:.2f},{request.start_time + request.duration:.2f},Default,,0,0,0,,"
            f"{request.subtitle_text}\n"
        )

        # Save ASS file
        filename = f"subtitle_{hash(request.subtitle_text) % 10000}.ass"
        filepath = f"/tmp/subtitles/{filename}"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(ass_content)

        return filepath

    except (OSError, ValueError) as e:
        return f"Error generating ASS file: {str(e)}"


@tool
def search_media_library(
    query: str, asset_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Search the hardcoded media library for assets.

    Args:
        query: Search query (searches in description and tags)
        asset_type: Filter by asset type (image, audio, video)

    Returns:
        List of matching assets with metadata
    """
    try:
        query_lower = query.lower()
        results = []

        for asset in MEDIA_LIBRARY_ASSETS:
            # Filter by type if specified
            if asset_type and asset.type != asset_type:
                continue

            # Search in description and tags
            searchable_text = (f"{asset.description} {' '.join(asset.tags)}").lower()

            if query_lower in searchable_text:
                results.append(asset)

        return results

    except (KeyError, TypeError) as e:
        return [{"error": f"Error searching media library: {str(e)}"}]


@tool
def analyze_video_quality(video_path: str) -> Dict[str, Any]:
    """Analyze video quality with an LLM and return combined scoring.

    Returns dict:
        - quality_score: float (0-10)
        - breakdown: dict with visual_quality, audio_quality, narrative_coherence
        - feedback: str
        - improvement_suggestions: List[str]
    """
    try:
        if not os.path.exists(video_path):
            return {"error": f"Video not found: {video_path}"}

        # Extract lightweight metadata (best-effort)
        meta: Dict[str, Any] = {}
        try:
            probe = ffmpeg.probe(video_path)  # type: ignore[no-untyped-call]
            streams = probe.get("streams", [])
            vstreams = [s for s in streams if s.get("codec_type") == "video"]
            astreams = [s for s in streams if s.get("codec_type") == "audio"]
            fmt = probe.get("format", {})
            if vstreams:
                v0 = vstreams[0]
                meta["video_codec"] = v0.get("codec_name")
                meta["width"] = v0.get("width")
                meta["height"] = v0.get("height")
                meta["avg_frame_rate"] = v0.get("avg_frame_rate")
            if astreams:
                a0 = astreams[0]
                meta["audio_codec"] = a0.get("codec_name")
                meta["audio_channels"] = a0.get("channels")
            meta["duration"] = fmt.get("duration")
            meta["bit_rate"] = fmt.get("bit_rate")
        except Exception:
            # Non-fatal
            meta = {}

        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            return {"error": "GOOGLE_API_KEY not set for quality analysis"}

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.2,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a strict video quality evaluator.
Return ONLY JSON with this schema:
{
  "visual_quality": number (0-10),
  "audio_quality": number (0-10),
  "narrative_coherence": number (0-10),
  "feedback": string,
  "improvement_suggestions": string[]
}
Scores must be integers.
                    """.strip(),
                ),
                (
                    "human",
                    """
Evaluate the video using the provided metadata. You cannot view the file; judge based on plausible expectations.
Video Path: {video_path}
Metadata: {metadata}
                    """.strip(),
                ),
            ]
        )

        messages = prompt.format_messages(
            video_path=video_path, metadata=json.dumps(meta)
        )
        resp = llm.invoke(messages)
        content = getattr(resp, "content", "")

        data: Dict[str, Any]
        try:
            # If model returns non-JSON text, try to find JSON block
            start = content.find("{")
            end = content.rfind("}")
            payload = content[start : end + 1] if start != -1 and end != -1 else content
            data = json.loads(payload)
        except Exception:
            return {"error": "Failed to parse JSON from LLM response", "raw": content}

        def _clamp(x: Any) -> int:
            try:
                return max(0, min(10, int(x)))
            except Exception:
                return 0

        v = _clamp(data.get("visual_quality"))
        a = _clamp(data.get("audio_quality"))
        n = _clamp(data.get("narrative_coherence"))
        total = v + a + n  # 0-30
        normalized = round(total / 3.0, 2)  # 0-10

        return {
            "quality_score": normalized,
            "breakdown": {
                "visual_quality": v,
                "audio_quality": a,
                "narrative_coherence": n,
                "total": total,
            },
            "feedback": data.get("feedback", ""),
            "improvement_suggestions": data.get("improvement_suggestions", []),
        }
    except Exception as e:
        return {"error": f"Quality analysis failed: {e}"}


@tool
def create_video(request: VideoCreationRequest) -> Dict[str, str]:
    """Create a video using ffmpeg with multiple input files.

    Args:
        request: VideoCreationRequest object containing video creation parameters

    Returns:
        Path to the created video or error message
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(request.output_path), exist_ok=True)

        # Separate input files by type
        images = []
        videos = []
        audio_files = []

        for file_path in request.input_files:
            if not os.path.exists(file_path):
                continue

            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                images.append(file_path)
            elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                videos.append(file_path)
            elif ext in [".mp3", ".wav", ".aac", ".m4a"]:
                audio_files.append(file_path)

        # Build ffmpeg command
        inputs = []

        # Add images
        for img in images:
            inputs.append(ffmpeg.input(img, loop=1, t=request.duration))

        # Add videos
        for vid in videos:
            inputs.append(ffmpeg.input(vid))

        # Add audio
        for audio in audio_files:
            inputs.append(ffmpeg.input(audio))

        if not inputs:
            raise ToolExecutionError("No valid input files found")

        # Create video stream
        if len(inputs) == 1:
            video_stream = inputs[0].video
        else:
            # Concatenate multiple inputs
            video_stream = ffmpeg.concat(*inputs, v=1, a=0)

        # Scale to desired resolution
        width, height = request.resolution.split("x")
        video_stream = video_stream.filter("scale", width, height)

        # Add audio if available
        if audio_files:
            audio_stream = ffmpeg.input(audio_files[0])
            output = ffmpeg.output(
                video_stream,
                audio_stream,
                request.output_path,
                vcodec="libx264",
                acodec="aac",
                r=request.fps,
                t=request.duration,
            )
        else:
            output = ffmpeg.output(
                video_stream,
                request.output_path,
                vcodec="libx264",
                r=request.fps,
                t=request.duration,
            )

        # Compose command string (best-effort) for observability
        try:
            command_str = ffmpeg.compile(output)
        except Exception:
            command_str = "ffmpeg <compiled>"

        # Run ffmpeg
        try:
            ffmpeg.run(output, overwrite_output=True, quiet=True)
        except Exception as e:
            raise ToolExecutionError(
                f"FFmpeg failed: {e}",
                command=command_str,  # type: ignore[str-bytes-safe]
            )

        return {"output_path": request.output_path, "ffmpeg_command": command_str}

    except ToolExecutionError:
        raise
    except (OSError, RuntimeError, ValueError) as e:
        raise ToolExecutionError(f"Error creating video: {str(e)}")


# Initialize tmp directories
def initialize_tmp_directories():
    """Initialize temporary directories for the video generation agent."""
    directories = [
        "/tmp/assets",
        "/tmp/audio",
        "/tmp/unsplash",
        "/tmp/subtitles",
        "/tmp/videos",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
