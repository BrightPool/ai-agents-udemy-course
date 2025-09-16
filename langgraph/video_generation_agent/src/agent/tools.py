"""Custom tools for video generation agent."""

import json
import os
import shlex
import shutil
import subprocess
from io import BytesIO
from typing import Any, Dict, List, Optional

import fal_client  # type: ignore[import-untyped]
import ffmpeg  # type: ignore[import-untyped]
from google import genai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image

from .models import (
    FFmpegExecuteRequest,
    GenerateImageRequest,
    KlingVideoRequest,
    KlingVideoResult,
    Storyboard,
    StoryboardCreateRequest,
    StoryboardUpdateRequest,
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
def create_story_board(request: StoryboardCreateRequest) -> Storyboard:
    """Create a 3-scene storyboard (beginning, middle, end) for a product ad.

    Generates initial prompts and scene descriptions guided by the request.
    """
    try:
        product = request.product_name
        brand = request.brand
        tone = request.tone
        key_msg = request.key_message
        ta = request.target_audience
        scene_names = ["beginning", "middle", "end"]

        scenes = []
        for idx, sid in enumerate(scene_names[: request.scenes_count]):
            title = f"{sid.capitalize()}"
            desc = (
                f"{brand} {product} ad for {ta}. {sid} of story, tone: {tone}. "
                f"Key message: {key_msg}. Visual focus: product + person."
            )
            prompt = (
                f"Photorealistic {product} with a person interacting, cinematic lighting, "
                f"brand aesthetic for {brand}, clean composition, ad-ready. Scene: {sid}."
            )
            scenes.append(
                {
                    "id": sid,
                    "title": title,
                    "description": desc,
                    "prompt": prompt,
                    "duration_seconds": request.default_scene_duration_seconds,
                }
            )

        return Storyboard(
            product_name=product,
            brand=brand,
            target_audience=ta,
            key_message=key_msg,
            tone=tone,
            scenes=scenes,  # type: ignore[arg-type]
        )
    except Exception:
        # Return an empty storyboard on error
        return Storyboard(
            product_name=request.product_name,
            brand=request.brand,
            target_audience=request.target_audience,
            key_message=request.key_message,
            tone=request.tone,
            scenes=[],
        )


@tool
def update_story_board(request: StoryboardUpdateRequest) -> Storyboard:
    """Apply simple natural-language edits to a storyboard.

    For safety and determinism, performs lightweight rule-based updates:
    - If instructions mention duration like `duration=8`, apply to all scenes.
    - If instructions mention tone, update tone.
    - If instructions include 'emphasize <text>', append to key_message.
    """
    sb = request.storyboard
    instructions = request.instructions.lower()

    try:
        # Update duration for all scenes if specified
        import re

        dur_match = re.search(r"duration\s*=\s*(\d+(?:\.\d+)?)", instructions)
        if dur_match:
            new_dur = float(dur_match.group(1))
            for sc in sb.scenes:
                sc.duration_seconds = max(1.0, min(60.0, new_dur))

        # Update tone if mentioned as tone=...
        tone_match = re.search(r"tone\s*=\s*([a-zA-Z\- ]{3,40})", instructions)
        if tone_match:
            sb.tone = tone_match.group(1).strip()

        # Emphasize phrase in key message
        emph_match = re.search(r"emphasize\s+([\w \-]{3,60})", instructions)
        if emph_match:
            phrase = emph_match.group(1).strip()
            if phrase and phrase.lower() not in sb.key_message.lower():
                sb.key_message = f"{sb.key_message}. Emphasis: {phrase}."

        return sb
    except Exception:
        return sb


@tool
def generate_image(request: GenerateImageRequest) -> List[str]:
    """Generate one or more images using Gemini Image API and save to /tmp.

    Requires GOOGLE_API_KEY in env. Saves PNG files and returns their paths.
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return ["Error: GOOGLE_API_KEY not set"]

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[request.prompt],
        )

        os.makedirs("/tmp/images", exist_ok=True)
        saved: List[str] = []
        idx = 0
        # Extract images from interleaved parts
        for part in response.candidates[0].content.parts:  # type: ignore[index]
            if getattr(part, "inline_data", None) is not None:
                try:
                    image = Image.open(BytesIO(part.inline_data.data))
                    out_path = f"/tmp/images/{request.output_basename}_{idx}.png"
                    image.save(out_path)
                    saved.append(out_path)
                    idx += 1
                except Exception:
                    continue

        if not saved:
            return ["Error: No images in model response"]
        return saved[: request.num_images]
    except Exception as e:
        return [f"Error generating image: {e}"]


@tool
def kling_generate_video_from_image(request: KlingVideoRequest) -> KlingVideoResult:
    """Generate a short video from an image using fal.ai Kling endpoint.

    - If image_path is provided, upload it to obtain a URL.
    - Submits to the fal queue and waits for result with logs.
    Returns a KlingVideoResult with video_url if available.
    """
    try:
        # Configure fal client via env var FAL_KEY if present
        fal_api_key = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
        if fal_api_key:
            os.environ.setdefault("FAL_KEY", fal_api_key)

        image_url: Optional[str] = request.image_url
        if request.image_path and os.path.exists(request.image_path):
            try:
                image_url = fal_client.upload_file(request.image_path)
            except Exception:
                pass

        logs_collector: List[str] = []

        def on_queue_update(update):
            try:
                if isinstance(update, fal_client.InProgress):
                    for log in update.logs:
                        msg = log.get("message")
                        if isinstance(msg, str):
                            logs_collector.append(msg)
            except Exception:
                pass

        args: Dict[str, Any] = {
            "prompt": request.prompt,
            "duration": float(request.duration_seconds),
        }
        if image_url:
            args["image_url"] = image_url
        if request.seed is not None:
            args["seed"] = int(request.seed)

        result = fal_client.subscribe(
            request.endpoint,
            arguments=args,
            with_logs=True,
            on_queue_update=on_queue_update,
        )

        # Best-effort parsing of result
        video_url = None
        try:
            if isinstance(result, dict):
                # Common conventions: check for url fields
                video_url = (
                    result.get("video_url")
                    or result.get("url")
                    or result.get("output_url")
                )
        except Exception:
            video_url = None

        return KlingVideoResult(
            request_id=getattr(result, "request_id", None) if result else None,
            video_url=video_url,
            logs=logs_collector or None,
        )
    except Exception as e:
        return KlingVideoResult(video_url=None, logs=[f"error: {e}"])


@tool
def run_ffmpeg_binary(request: FFmpegExecuteRequest) -> Dict[str, str]:
    """Execute the ffmpeg binary with a provided full command string.

    Ensures -y flag and creates output directory if provided.
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
    except ToolExecutionError as e:
        # Expose as a JSON string for ToolMessage
        return {
            "output_path": request.output_path or "",
            "ffmpeg_command": getattr(e, "command", request.command),
            "stdout_tail": "",
            "stderr_tail": str(e),
        }


@tool
def score_video(
    video_path: str, target_quality_score: float | None = None
) -> Dict[str, Any]:
    """Score a video by probing metadata and using an LLM rubric (0-10)."""
    try:
        # Lightweight probe for metadata
        if not os.path.exists(video_path):
            return {"error": f"Video not found: {video_path}"}
        meta: Dict[str, Any] = {}
        try:
            probe = ffmpeg.probe(video_path)  # type: ignore[no-untyped-call]
            streams = probe.get("streams", [])
            vstreams = [s for s in streams if s.get("codec_type") == "video"]
            astreams = [s for s in streams if s.get("codec_type") == "audio"]
            fmt = probe.get("format", {})
            if vstreams:
                v0 = vstreams[0]
                meta["width"] = v0.get("width")
                meta["height"] = v0.get("height")
                meta["avg_frame_rate"] = v0.get("avg_frame_rate")
            if astreams:
                a0 = astreams[0]
                meta["audio_channels"] = a0.get("channels")
            meta["duration"] = fmt.get("duration")
        except Exception:
            pass

        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            return {"error": "GOOGLE_API_KEY not set for scoring"}

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
You are a strict video quality evaluator. Return ONLY compact JSON with fields:
{"visual_quality": int 0-10, "audio_quality": int 0-10, "narrative_coherence": int 0-10, "feedback": string}
                    """.strip(),
                ),
                (
                    "human",
                    """
Evaluate this video path with the provided metadata (you cannot open the file).
Video Path: {video_path}
Metadata: {metadata}
Target (optional): {target}
                    """.strip(),
                ),
            ]
        )
        messages = prompt.format_messages(
            video_path=video_path,
            metadata=json.dumps(meta),
            target=json.dumps({"target_quality_score": target_quality_score})
            if target_quality_score is not None
            else "{}",
        )
        resp = llm.invoke(messages)
        content = getattr(resp, "content", "")
        try:
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
        score = round((v + a + n) / 3.0, 2)
        return {
            "quality_score": score,
            "breakdown": {
                "visual_quality": v,
                "audio_quality": a,
                "narrative_coherence": n,
            },
            "feedback": data.get("feedback", ""),
        }
    except Exception as e:
        return {"error": f"Scoring failed: {e}"}


"""
Removed legacy external media search/download tool in favor of generate_image.
"""


"""
Removed legacy ElevenLabs TTS tool.
"""


"""
Removed legacy ASS subtitle generation tool.
"""


"""
Removed legacy media library search tool.
"""


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


"""
Removed legacy create_video planning tool.
"""


"""
Removed legacy execute_ffmpeg in favor of run_ffmpeg_binary alias.
"""


"""
Removed legacy list_recent_renders tool.
"""


"""
Removed legacy OpenAI transcription tool.
"""


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
