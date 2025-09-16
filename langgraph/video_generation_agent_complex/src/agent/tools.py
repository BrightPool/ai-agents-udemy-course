"""Custom tools for video generation agent."""

import json
import os
import shlex
import shutil
import subprocess
from typing import Any, Dict, List

import aiofiles
import ffmpeg  # type: ignore[import-untyped]
import httpx
from elevenlabs.client import ElevenLabs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from openai import OpenAI

# Optional third-party clients for external video generation providers
try:  # pragma: no cover - optional dependency at runtime
    import replicate  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - import fallback
    replicate = None  # type: ignore[assignment]
try:  # pragma: no cover - optional dependency at runtime
    import fal_client  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - import fallback
    fal_client = None  # type: ignore[assignment]

from .media_assets import MEDIA_LIBRARY_ASSETS
from .models import (
    AssetSearchRequest,
    AssetSearchResult,
    FFmpegExecuteRequest,
    KlingVideoRequest,
    MediaSearchRequest,
    OpenAITranscriptionRequest,
    OpenAITranscriptionVerboseJson,
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


def _map_aspect_ratio_to_resolution(ar: str) -> str:
    mapping = {
        "16:9": "1920x1080",
        "9:16": "1080x1920",
        "1:1": "1080x1080",
        "4:3": "1440x1080",
    }
    return mapping.get(ar, "1920x1080")


@tool
def kling_generate_video(request: KlingVideoRequest) -> Dict[str, Any]:
    """Generate a short video with Kling v2.1 via Replicate or fal.ai.

    Inputs:
      - prompt (str)
      - negative_prompt (optional str)
      - duration (int seconds, typically 5 or 10)
      - aspect_ratio ("16:9" | "9:16" | "1:1" | "4:3")
      - start_image_path (optional local file path)

    Returns dict with fields: provider, url, duration, aspect_ratio, prompt
    and possibly local_download_path if we downloaded the output.
    """
    provider = os.getenv("KLING_PROVIDER", "replicate").strip().lower()
    duration = int(getattr(request, "duration", 5) or 5)
    aspect_ratio = getattr(request, "aspect_ratio", "16:9") or "16:9"
    prompt = request.prompt
    negative_prompt = getattr(request, "negative_prompt", "") or ""
    start_image_path = getattr(request, "start_image_path", None)

    # Validate optional start image
    start_image_url: str | None = None
    if start_image_path:
        if not os.path.exists(start_image_path):
            return {"error": f"start_image not found: {start_image_path}"}
        try:
            # Upload file to get a temporary URL when using fal
            if provider == "fal" and fal_client is not None:
                start_image_url = fal_client.upload_file(start_image_path)
            else:
                # For replicate, file upload isn't needed; their model accepts a URL.
                # Users should provide a URL or skip start_image.
                start_image_url = None
        except Exception:
            start_image_url = None

    try:
        if provider == "fal":
            if fal_client is None:
                return {"error": "fal-client not installed"}
            api_key = os.getenv("FAL_KEY")
            if not api_key:
                return {"error": "FAL_KEY not set"}

            # Model name on fal for Kling can vary; commonly hosted as "kwaivgi/kling-v2.1-master"
            # If a different endpoint is desired, allow override via env
            endpoint = os.getenv("KLING_FAL_ENDPOINT", "kwaivgi/kling-v2.1-master")

            arguments: Dict[str, Any] = {
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
            }
            if negative_prompt:
                arguments["negative_prompt"] = negative_prompt
            if start_image_url:
                arguments["start_image"] = start_image_url

            result = fal_client.run(endpoint, arguments=arguments)
            # Expect result to have a .get("video") or similar URL
            # Normalize into url field
            url = None
            if isinstance(result, dict):
                url = (
                    result.get("video")
                    or result.get("url")
                    or result.get("output")
                )
            if not url:
                # As a fallback, try to stringify
                url = str(result)

            return {
                "provider": "fal",
                "url": url,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "prompt": prompt,
            }
        else:
            # Default: use Replicate
            if replicate is None:
                return {"error": "replicate not installed"}
            token = os.getenv("REPLICATE_API_TOKEN") or os.getenv("REPLICATE_API_KEY")
            if not token:
                return {"error": "REPLICATE_API_TOKEN not set"}

            os.environ["REPLICATE_API_TOKEN"] = token

            # Model identifier from the user's example
            model = os.getenv("KLING_REPLICATE_MODEL", "kwaivgi/kling-v2.1-master")

            inputs: Dict[str, Any] = {
                "prompt": prompt,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "negative_prompt": negative_prompt,
            }
            # Replicate input generally expects a URL for start_image; we only pass if we have a URL
            if start_image_url:
                inputs["start_image"] = start_image_url

            output = replicate.run(model, input=inputs)  # type: ignore[arg-type]

            # The example shows output.url() and output.read(); but replicate.run often returns a URL string or list
            try:
                if hasattr(output, "url") and callable(getattr(output, "url")):
                    url = output.url()  # type: ignore[call-arg]
                elif isinstance(output, (str, bytes)):
                    url = output.decode() if isinstance(output, bytes) else output
                elif isinstance(output, list) and output:
                    url = output[-1]
                else:
                    url = str(output)
            except Exception:
                url = str(output)

            return {
                "provider": "replicate",
                "url": url,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "prompt": prompt,
            }
    except Exception as e:
        return {"error": f"Kling generation failed: {e}"}


@tool
async def search_unsplash_media(request: MediaSearchRequest) -> List[str]:
    """Search Unsplash for photos and download them to /tmp folder.

    Args:
        request: MediaSearchRequest with query, count, media_type and quality

    Returns:
        List of paths to downloaded media
    """
    try:
        access_key = os.getenv("UNSPLASH_ACCESS_KEY")
        if not access_key:
            raise ValueError("UNSPLASH_ACCESS_KEY environment variable not set")

        # Unsplash official API does not support video search/download
        if request.media_type == "video":
            return [
                "Unsplash API does not provide videos. Set media_type='photo' for Unsplash."
            ]

        # Search for media
        async with httpx.AsyncClient() as client:
            endpoint = "search/photos"
            headers = {
                "Authorization": f"Client-ID {access_key}",
                "Accept-Version": "v1",
            }
            params: Dict[str, Any] = {
                "query": request.query,
                "per_page": min(request.count, 30),
                "page": request.page,
                "content_filter": request.content_filter,
            }
            if request.media_type == "photo" and request.orientation:
                params["orientation"] = request.orientation

            response = await client.get(
                f"https://api.unsplash.com/{endpoint}", params=params, headers=headers
            )
            response.raise_for_status()
            data = response.json()

        # Create tmp directory
        media_dir = "/tmp/unsplash_photos"
        os.makedirs(media_dir, exist_ok=True)

        downloaded_paths = []

        # Download media
        async with httpx.AsyncClient() as client:
            for i, item in enumerate(data.get("results", [])):
                # Track download event as per API guidelines
                try:
                    download_loc = item.get("links", {}).get("download_location")
                    if download_loc:
                        await client.get(download_loc, headers=headers)
                except Exception:
                    pass

                # Respect hotlinking and keep ixid params; use returned URLs
                media_url = item["urls"][request.quality]
                filename = (
                    f"unsplash_{request.query.replace(' ', '_')}_{i}_{item['id']}.jpg"
                )

                filepath = f"{media_dir}/{filename}"

                async with client.stream("GET", media_url, headers=headers) as resp2:
                    resp2.raise_for_status()
                    async with aiofiles.open(filepath, "wb") as f:
                        async for chunk in resp2.aiter_bytes():
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

        # Generate audio using the official client (streaming bytes)
        audio = client.text_to_speech.convert(
            text=request.text,
            voice_id=request.voice_id,
            model_id=request.model,
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

        # Build ASS with karaoke highlighting per-word
        def _fmt_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}:{minutes:02d}:{secs:05.2f}"

        # Build karaoke chunks: prefer precise word timings if provided
        karaoke_chunks = ""
        if getattr(request, "words", None):
            pieces = []
            for w in request.words:  # type: ignore[union-attr]
                dur_cs = max(
                    1, int(round(max(0.01, float(w.end) - float(w.start)) * 100))
                )
                pieces.append(f"{{\\k{dur_cs}}}{w.word}")
            karaoke_chunks = " ".join(pieces)
        else:
            base_text: str = request.subtitle_text or ""
            words = [w for w in base_text.split() if w]
            total_words = max(1, len(words))
            per_word_cs = int(round((request.duration / total_words) * 100))
            karaoke_chunks = "".join([f"{{\\k{per_word_cs}}}{w} " for w in words])

        start_ts = _fmt_time(request.start_time)
        end_ts = _fmt_time(request.start_time + request.duration)

        ass_content = (
            "[Script Info]\n"
            "Title: Generated Subtitle\n"
            "ScriptType: v4.00+\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: {request.style_name if hasattr(request, 'style_name') else 'Default'},Arial,{request.font_size},{request.font_color},{request.highlight_color},&H00000000,"
            f"&H80000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            f"Dialogue: 0,{start_ts},{end_ts},{request.style_name if hasattr(request, 'style_name') else 'Default'},,0,0,0,,{karaoke_chunks.strip()}\n"
        )

        # Save ASS file
        seed = request.subtitle_text or (
            " ".join([w.word for w in (request.words or [])])
            if getattr(request, "words", None)
            else "subtitle"
        )
        filename = f"subtitle_{abs(hash(seed)) % 100000}.ass"
        filepath = f"/tmp/subtitles/{filename}"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(ass_content)

        return filepath

    except (OSError, ValueError) as e:
        return f"Error generating ASS file: {str(e)}"


@tool
def search_media_library(request: AssetSearchRequest) -> AssetSearchResult:
    """Search the hardcoded media library for assets.

    Args:
        request: AssetSearchRequest with filters for type, tags, ids

    Returns:
        AssetSearchResult containing matching assets and total count
    """
    try:
        results: List[Any] = []

        for asset in MEDIA_LIBRARY_ASSETS:
            # Filter by type
            if request.asset_type and asset.type != request.asset_type:
                continue

            # Filter by ids
            if request.ids_any and asset.id not in set(request.ids_any):
                continue

            # Filter by tags_any
            if request.tags_any and not any(t in asset.tags for t in request.tags_any):
                continue

            # Filter by tags_all
            if request.tags_all and not all(t in asset.tags for t in request.tags_all):
                continue

            # Filter by free-text query
            if request.query:
                searchable_text = (
                    f"{asset.filename} {asset.description} {' '.join(asset.tags)}"
                ).lower()
                if request.query.lower() not in searchable_text:
                    continue

            results.append(asset)

        return AssetSearchResult(assets=list(results), total_count=len(results))

    except (KeyError, TypeError):
        return AssetSearchResult(assets=[], total_count=0)


@tool
def analyze_video_quality(
    video_path: str,
    recent_renders: List[Dict[str, Any]] | None = None,
    target_quality_score: float | None = None,
) -> Dict[str, Any]:
    """Analyze video quality with an LLM and return combined scoring.

    Inputs:
        - video_path: Path to the video to evaluate
        - recent_renders: Optional list of recent render dicts (<=3 recommended) providing
          prior ffmpeg command, output paths, quality scores/breakdowns, created_at, media_used
        - target_quality_score: Optional target score (0-10) for calibration context

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
            # Additional lightweight file info
            try:
                stat = os.stat(video_path)
                meta["file_size_bytes"] = stat.st_size
                meta["modified_time"] = getattr(stat, "st_mtime", None)
                meta["created_time"] = getattr(stat, "st_ctime", None)
            except Exception:
                pass
            try:
                base = os.path.basename(video_path)
                meta["file_name"] = base
                meta["file_extension"] = os.path.splitext(base)[1].lower()
            except Exception:
                pass
        except Exception:
            # Non-fatal
            meta = {}

        # Prepare prior renders context (most recent first, trimmed)
        prior_context: Dict[str, Any] = {"recent_renders": []}
        if target_quality_score is not None:
            prior_context["target_quality_score"] = float(target_quality_score)
        try:
            renders_list = list(recent_renders or [])
            try:
                renders_list = sorted(
                    renders_list,
                    key=lambda r: float(r.get("created_at", 0.0) or 0.0),
                    reverse=True,
                )
            except Exception:
                pass
            trimmed: List[Dict[str, Any]] = []
            for r in renders_list[:3]:
                try:
                    trimmed.append(
                        {
                            "generated_video_file_path": r.get(
                                "generated_video_file_path", ""
                            ),
                            "ffmpeg_command": r.get("ffmpeg_command", ""),
                            "quality_score": r.get("quality_score", None),
                            "quality_breakdown": r.get("quality_breakdown", None),
                            "created_at": r.get("created_at", None),
                            "media_used": r.get("media_used", []),
                        }
                    )
                except Exception:
                    continue
            prior_context["recent_renders"] = trimmed
        except Exception:
            prior_context["recent_renders"] = []

        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            return {"error": "GOOGLE_API_KEY not set for quality analysis"}

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
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

 Prior renders (most recent first) and target, for calibration:
 {prior_context}
                    """.strip(),
                ),
            ]
        )

        messages = prompt.format_messages(
            video_path=video_path,
            metadata=json.dumps(meta),
            prior_context=json.dumps(prior_context),
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
    """Create an ffmpeg command string for the requested video composition.

    This does not execute ffmpeg. It returns a single command string and
    optional notes. Use `execute_ffmpeg` to run the command separately.

    Returns: { ffmpeg_command, output_path, notes }
    """
    try:
        if not request.input_files:
            raise ToolExecutionError("No input files provided")

        # Partition input files by type and verify existence
        images: List[str] = []
        videos: List[str] = []
        audio_files: List[str] = []

        for file_path in request.input_files:
            if not isinstance(file_path, str):
                continue
            if not os.path.exists(file_path):
                continue
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                images.append(file_path)
            elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                videos.append(file_path)
            elif ext in [".mp3", ".wav", ".aac", ".m4a"]:
                audio_files.append(file_path)

        if not images and not videos and not audio_files:
            raise ToolExecutionError("No valid input files found")

        # Choose primary visual source: prefer first video, else first image
        primary_visual = videos[0] if videos else (images[0] if images else None)
        if primary_visual is None:
            raise ToolExecutionError("No visual input (image/video) provided")

        # Build an LLM plan for the ffmpeg command
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ToolExecutionError(
                "GOOGLE_API_KEY not set; cannot plan ffmpeg command with LLM"
            )

        # Use a stronger model for command generation
        llm = ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_VIDEO_CMD_MODEL", "gemini-2.5-pro"),
            google_api_key=api_key,
            temperature=0.2,
        )

        # Prepare prompt with strict JSON output
        planning_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an expert ffmpeg engineer. Generate a SINGLE safe ffmpeg command "
                        "that will run non-interactively and overwrite outputs (use -y). "
                        "Respect the requested duration, fps, and resolution, maintaining aspect via scale+pad. "
                        "If the primary visual is an image, loop it for the duration. If subtitles exist, overlay them. "
                        "Prefer H.264 video (libx264, CRF 18-20, preset veryfast) and AAC audio 192k. "
                        "Ensure cross-platform compatibility by adding -pix_fmt yuv420p. "
                        'Return ONLY compact JSON: {"command": string, "notes": string}. '
                        "Quote all file paths with single quotes. Do not include comments or backticks."
                    ),
                ),
                (
                    "human",
                    (
                        "Inputs JSON:\n{inputs}\n\n"
                        "Primary visual: '{primary}'.\n"
                        "Desired resolution: {resolution}, fps: {fps}, duration: {duration}s.\n"
                        "Subtitle file: {subtitle}.\n"
                        "Output path: '{output}'.\n"
                        "Generate the command starting with 'ffmpeg'."
                    ),
                ),
            ]
        )

        inputs_dict: Dict[str, Any] = {
            "images": images,
            "videos": videos,
            "audio": audio_files,
        }

        messages = planning_prompt.format_messages(
            inputs=json.dumps(inputs_dict),
            primary=primary_visual,
            resolution=request.resolution,
            fps=request.fps,
            duration=request.duration,
            subtitle=(request.subtitle_path or ""),
            output=request.output_path,
        )

        llm_resp = llm.invoke(messages)
        llm_text = getattr(llm_resp, "content", "")

        # Extract JSON and read command
        try:
            start = llm_text.find("{")
            end = llm_text.rfind("}")
            payload = (
                llm_text[start : end + 1] if start != -1 and end != -1 else llm_text
            )
            plan_obj = json.loads(payload)
            planned_command = str(plan_obj.get("command", "")).strip()
            if not planned_command:
                raise ValueError("No 'command' field produced by LLM")
        except Exception as e:
            raise ToolExecutionError(f"Failed to parse LLM plan: {e}")

        # Safety and normalization: ensure 'ffmpeg' prefix and '-y', ensure output path present
        if not planned_command.lower().startswith("ffmpeg"):
            planned_command = f"ffmpeg {planned_command}"

        if " -y " not in f" {planned_command} ":
            # insert after 'ffmpeg'
            parts = planned_command.split()
            if parts and parts[0].lower() == "ffmpeg":
                parts.insert(1, "-y")
                planned_command = " ".join(parts)

        if request.output_path not in planned_command:
            planned_command = f"{planned_command} '{request.output_path}'"

        return {
            "output_path": request.output_path,
            "ffmpeg_command": planned_command,
            "notes": str(plan_obj.get("notes", "")),
        }

    except ToolExecutionError:
        raise
    except (OSError, RuntimeError, ValueError) as e:
        raise ToolExecutionError(f"Error creating video: {str(e)}")


@tool
def execute_ffmpeg(request: FFmpegExecuteRequest) -> Dict[str, str]:
    """Execute a provided ffmpeg command via subprocess with timeout and logs.

    Ensures '-y' overwrite flag is present. If output_path is provided, creates
    its parent directory ahead of time. Returns stdout/stderr tails for debugging.
    """
    try:
        if not isinstance(request.command, str) or not request.command.strip():
            raise ToolExecutionError("Command must be a non-empty string")

        if not shutil.which("ffmpeg"):
            raise ToolExecutionError("ffmpeg not found on PATH. Please install ffmpeg.")

        # Ensure output directory exists if provided
        if request.output_path:
            out_dir = os.path.dirname(request.output_path) or "."
            os.makedirs(out_dir, exist_ok=True)

        planned_command = request.command.strip()
        if not planned_command.lower().startswith("ffmpeg"):
            planned_command = f"ffmpeg {planned_command}"

        if " -y " not in f" {planned_command} ":
            parts = planned_command.split()
            if parts and parts[0].lower() == "ffmpeg":
                parts.insert(1, "-y")
                planned_command = " ".join(parts)

        try:
            cmd_list = shlex.split(planned_command)
        except Exception as e:
            raise ToolExecutionError(f"Invalid command: {e}", command=planned_command)

        try:
            completed = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                check=False,
                timeout=max(10, int(getattr(request, "timeout_seconds", 180))),
            )
        except subprocess.TimeoutExpired:
            raise ToolExecutionError(
                f"FFmpeg timed out after {getattr(request, 'timeout_seconds', 180)}s",
                command=planned_command,
            )
        except OSError as e:
            raise ToolExecutionError(
                f"Failed to run ffmpeg: {e}", command=planned_command
            )

        stdout_tail = (completed.stdout or "")[-2000:]
        stderr_tail = (completed.stderr or "")[-4000:]

        if completed.returncode != 0:
            raise ToolExecutionError(
                f"FFmpeg failed with code {completed.returncode}: {stderr_tail[-800:]}",
                command=planned_command,
            )

        return {
            "output_path": request.output_path or "",
            "ffmpeg_command": planned_command,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
    except ToolExecutionError:
        raise
    except (OSError, RuntimeError, ValueError) as e:
        raise ToolExecutionError(f"Error executing ffmpeg: {str(e)}")


@tool
def list_recent_renders(
    renders: List[dict] | None = None, limit: int = 2
) -> Dict[str, Any]:
    """Return the most recent render records for LLM context.

    Provide agent state's renders as list[dict] (e.g., via model_dump). Returns up to
    `limit` items with ffmpeg command, output path, quality, and media used.
    """
    try:
        if not renders:
            return {"renders": []}
        try:
            sorted_items = sorted(
                renders,
                key=lambda r: float(r.get("created_at", 0.0) or 0.0),
                reverse=True,
            )
        except Exception:
            sorted_items = list(renders)

        trimmed: List[Dict[str, Any]] = []
        for r in sorted_items[: max(1, int(limit))]:
            trimmed.append(
                {
                    "ffmpeg_command": r.get("ffmpeg_command", ""),
                    "generated_video_file_path": r.get("generated_video_file_path", ""),
                    "quality_score": r.get("quality_score", None),
                    "quality_breakdown": r.get("quality_breakdown", None),
                    "created_at": r.get("created_at", None),
                    "media_used": r.get("media_used", []),
                }
            )

        return {"renders": trimmed}
    except Exception as e:
        return {"renders": [], "error": str(e)}


@tool
def transcribe_audio_openai(request: OpenAITranscriptionRequest) -> Dict[str, Any]:
    """Transcribe an audio file using OpenAI SDK (verbose_json with word timings)."""
    try:
        if not os.path.exists(request.file_path):
            return {"error": f"File not found: {request.file_path}"}

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"error": "OPENAI_API_KEY environment variable not set"}

        client = OpenAI(api_key=api_key)
        with open(request.file_path, "rb") as audio_file:
            kwargs: Dict[str, Any] = {
                "file": audio_file,
                "model": request.model,
                "response_format": request.response_format,
            }
            if request.timestamp_granularities:
                kwargs["timestamp_granularities"] = request.timestamp_granularities
            if request.prompt:
                kwargs["prompt"] = request.prompt
            if request.language:
                kwargs["language"] = request.language

            tr = client.audio.transcriptions.create(**kwargs)  # type: ignore[arg-type]

        # Normalize result to dict and validate
        try:
            if hasattr(tr, "model_dump"):
                payload = tr.model_dump()  # type: ignore[assignment]
            elif hasattr(tr, "to_dict"):
                payload = tr.to_dict()  # type: ignore[assignment]
            else:
                payload = json.loads(str(tr))
        except Exception:
            payload = {
                "text": getattr(tr, "text", ""),
                "words": getattr(tr, "words", None),
            }

        try:
            typed = OpenAITranscriptionVerboseJson(**payload)
            return typed.model_dump()
        except Exception:
            return payload
    except Exception as e:
        return {"error": f"Transcription failed: {e}"}


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


def _safe_remove_path(path: str) -> tuple[bool, str | None]:
    """Attempt to remove a file or directory tree safely.

    Returns (success, error_message_or_None).
    """
    try:
        if not os.path.exists(path):
            return True, None
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=False)
        else:
            os.remove(path)
        return True, None
    except Exception as e:
        return False, str(e)


def cleanup_tmp_directories(preserve_paths: List[str] | None = None) -> Dict[str, Any]:
    """Cleanup temporary files created by the video generation agent.

    This removes files from known temp directories while preserving any paths
    explicitly listed in `preserve_paths` (e.g., the final video output).

    It intentionally does NOT delete the media library directory `/tmp/assets`.

    Returns a summary dict with keys: removed_files, errors, preserved, directories_processed.
    """
    preserved: set[str] = set(preserve_paths or [])

    # Directories that hold ephemeral artifacts
    target_directories: List[str] = [
        "/tmp/unsplash",  # documented location
        "/tmp/unsplash_photos",  # actual download location used by tool
        "/tmp/audio",
        "/tmp/subtitles",
        "/tmp/videos",  # keep final video if preserved
    ]

    removed: List[str] = []
    errors: List[str] = []
    processed: List[str] = []

    for directory in target_directories:
        if not os.path.isdir(directory):
            continue
        processed.append(directory)

        try:
            for name in os.listdir(directory):
                candidate = os.path.join(directory, name)
                # Preserve any explicitly preserved file
                if candidate in preserved:
                    continue
                ok, err = _safe_remove_path(candidate)
                if ok:
                    removed.append(candidate)
                elif err:
                    errors.append(f"{candidate}: {err}")
        except Exception as e:
            errors.append(f"{directory}: {e}")

        # Optionally remove empty aux directories (but keep /tmp/videos for future runs)
        try:
            if directory != "/tmp/videos" and not os.listdir(directory):
                # best-effort remove empty dir
                os.rmdir(directory)
        except Exception:
            # non-fatal
            pass

    return {
        "removed_files": removed,
        "errors": errors,
        "preserved": sorted(list(preserved)),
        "directories_processed": processed,
    }
