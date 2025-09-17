"""Custom tools for video generation agent."""

import base64
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import fal_client  # type: ignore[import-untyped]
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.types import Command

from .models import (
    FFmpegExecuteRequest,
    GenerateImageRequest,
    Storyboard,
    StoryboardCreateRequest,
    Veo3VideoRequest,
    VideoConcatRequest,
    VideoConcatResult,
)
from .utils import upload_media_to_google


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
            scene_id = str(uuid.uuid4())
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
                    "id": scene_id,
                    "title": title,
                    "description": desc,
                    "prompt": prompt,
                    "duration_seconds": request.default_scene_duration_seconds,
                }
            )

        return Storyboard(scenes=scenes)  # type: ignore[arg-type]
    except Exception:
        # Return an empty storyboard on error
        return Storyboard(scenes=[])


@tool()
def update_scene(
    tool_call_id: Annotated[str, InjectedToolCallId],
    storyboard: Storyboard,
    scene_index: int,
    config: RunnableConfig,
    title: Optional[str] = None,
    description: Optional[str] = None,
    prompt: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    quality_score: Optional[float] = None,
    feedback: Optional[str] = None,
) -> Command:
    """Update a specific scene in the storyboard using Command.

    Only updates the fields that are provided (not None). Returns a Command
    that updates the storyboard state.
    """
    # Validate scene index
    if scene_index >= len(storyboard.scenes):
        raise ToolExecutionError(
            f"Scene index {scene_index} is out of range. Storyboard has {len(storyboard.scenes)} scenes."
        )

    # Get the scene to update
    scene = storyboard.scenes[scene_index]

    # Update only the provided fields
    updated_fields = []
    if title is not None:
        scene.title = title
        updated_fields.append("title")
    if description is not None:
        scene.description = description
        updated_fields.append("description")
    if prompt is not None:
        scene.prompt = prompt
        updated_fields.append("prompt")
    if duration_seconds is not None:
        scene.duration_seconds = duration_seconds
        updated_fields.append("duration")
    if quality_score is not None:
        scene.quality_score = quality_score
        updated_fields.append("quality_score")
    if feedback is not None:
        scene.feedback = feedback
        updated_fields.append("feedback")

    # Create success message describing what was updated
    fields_str = ", ".join(updated_fields) if updated_fields else "no fields"
    success_message = f"Successfully updated scene {scene_index} ({fields_str})"

    # Return Command to update the storyboard state and message history
    return Command(
        update={
            # update the state keys
            "storyboard": storyboard,
            # update the message history with ToolMessage
            "messages": [ToolMessage(success_message, tool_call_id=tool_call_id)],
        }
    )


@tool
def generate_image(
    request: GenerateImageRequest,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """Generate image(s), upload to Google, and update chat via Command.

    Saves PNG files under `/tmp/images`, uploads each to Google (file storage),
    and posts a ToolMessage containing compact metadata (file_id/file_uri).
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ToolExecutionError("GOOGLE_API_KEY not set")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-image-preview",
            google_api_key=api_key,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        messages = []

        # If there are input image paths, add them to the prompt
        messages.append(HumanMessage(content=request.prompt))
        response = llm.invoke(
            [HumanMessage(content=request.prompt)],
            generation_config={"response_modalities": ["IMAGE"]},
        )

        os.makedirs("/tmp/images", exist_ok=True)
        images_payload: List[Dict[str, Any]] = []
        file_uploads = []
        for idx, block in enumerate(response.content):
            image_url: Optional[str] = None
            if isinstance(block, dict):
                val = block.get("image_url")
                if isinstance(val, str):
                    image_url = val
                elif isinstance(val, dict):
                    url_val = val.get("url")
                    if isinstance(url_val, str):
                        image_url = url_val
            elif isinstance(block, str) and block.startswith("data:"):
                image_url = block

            if not image_url or "," not in image_url:
                continue

            try:
                header, b64_data = image_url.split(",", 1)
            except ValueError:
                continue
            mime_type = "image/png"
            if ":" in header:
                mime_type = header.split(";", 1)[0].split(":", 1)[-1] or mime_type

            image_bytes = base64.b64decode(b64_data)
            out_path = f"/tmp/images/{request.output_basename}_{idx}.png"
            with open(out_path, "wb") as f:
                f.write(image_bytes)

            # Upload to Google file storage
            upload = upload_media_to_google(
                out_path, google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            payload_item: Dict[str, Any] = {
                "kind": "image",
                "path": out_path,
                "mime_type": mime_type,
                "file_id": upload.get("file_id"),
                "file_uri": upload.get("file_uri"),
            }
            if getattr(request, "inline_base64", False):
                payload_item["base64"] = b64_data
            images_payload.append(payload_item)
            file_uploads.append(upload)

        if not images_payload:
            payload = {"error": "No images in model response", "images": []}
            return Command(
                update={
                    "messages": [
                        ToolMessage(json.dumps(payload), tool_call_id=tool_call_id)
                    ]
                }
            )

        payload = {"images": images_payload, "prompt": request.prompt}
        happy_path_messages = []
        happy_path_messages.append(
            ToolMessage(json.dumps(payload), tool_call_id=tool_call_id)
        )
        for file in file_uploads:
            happy_path_messages.append(
                HumanMessage(
                    content=[
                        {
                            "type": "media",
                            "file_uri": file.get("file_uri"),
                            "mime_type": file.get("mime_type"),
                        },
                    ]
                )
            )

        return Command(update={"messages": happy_path_messages})
    except Exception as e:
        payload = {"error": f"Error generating image: {e}", "images": []}
        return Command(
            update={
                "messages": [
                    ToolMessage(json.dumps(payload), tool_call_id=tool_call_id)
                ]
            }
        )


@tool
def veo3_generate_video(
    request: Veo3VideoRequest,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """Generate a video using Google's Veo 3 Fast endpoint on fal.ai.

    Supports optional reference imagery via path, hosted URL, or base64 payload.
    Returns structured metadata including the resulting video URL.
    """
    try:
        fal_api_key = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
        if fal_api_key:
            os.environ.setdefault("FAL_KEY", fal_api_key)

        reference_url: Optional[str] = request.image_url
        if request.image_path and os.path.exists(request.image_path):
            try:
                reference_url = fal_client.upload_file(request.image_path)
            except Exception:
                reference_url = reference_url or None

        if request.image_base64:
            if request.image_base64.startswith("data:"):
                reference_url = request.image_base64
            else:
                mime = request.image_mime_type or "image/png"
                reference_url = f"data:{mime};base64,{request.image_base64}"

        # using synchronous run keeps things simple; no queue/log streaming

        arguments: Dict[str, Any] = {
            "prompt": request.prompt,
            "aspect_ratio": request.aspect_ratio,
            "duration": request.duration,
            "enhance_prompt": request.enhance_prompt,
            "auto_fix": request.auto_fix,
            "resolution": request.resolution,
            "generate_audio": request.generate_audio,
        }

        if reference_url:
            arguments["image_url"] = reference_url
        if request.negative_prompt:
            arguments["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            arguments["seed"] = int(request.seed)

        result = fal_client.run(
            "fal-ai/veo3/fast",
            arguments=arguments,
        )

        request_id = getattr(result, "request_id", None)
        video_url: Optional[str] = None

        if isinstance(result, dict):
            video = result.get("video")
            if isinstance(video, dict):
                url = video.get("url") or video.get("video_url")
                if isinstance(url, str):
                    video_url = url
            if not video_url:
                # Some responses may surface url at the top level
                possible_url = result.get("url") or result.get("video_url")
                if isinstance(possible_url, str):
                    video_url = possible_url

            request_id = request_id or result.get("request_id") or result.get("id")

        # Generate unique video identifier for easy reference
        video_identifier = f"veo3_video_{uuid.uuid4().hex[:8]}"

        # Attempt to download the remote video before upload
        local_path: Optional[str] = None
        try:
            if isinstance(video_url, str) and video_url.startswith("http"):
                import urllib.request

                os.makedirs("/tmp/videos", exist_ok=True)
                local_path = f"/tmp/videos/{video_identifier}.mp4"
                urllib.request.urlretrieve(video_url, local_path)  # nosec B310
        except Exception:
            local_path = None

        upload_info: Dict[str, Optional[str]] = {"file_id": None, "file_uri": None}
        if local_path and os.path.exists(local_path):
            upload_info = upload_media_to_google(
                local_path, google_api_key=os.getenv("GOOGLE_API_KEY")
            )

        payload: Dict[str, Any] = {
            "video": {
                "kind": "video",
                "path": local_path,
                "url": video_url,
                "mime_type": "video/mp4",
                "file_id": upload_info.get("file_id"),
                "file_uri": upload_info.get("file_uri"),
            },
            "request_id": request_id,
            "video_identifier": video_identifier,
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(json.dumps(payload), tool_call_id=tool_call_id),
                    HumanMessage(
                        content=[
                            {
                                "type": "media",
                                "file_uri": upload_info.get("file_uri"),
                                "mime_type": upload_info.get("mime_type"),
                            },
                        ]
                    ),
                ]
            }
        )
    except Exception as exc:
        payload = {"error": f"error: {exc}"}
        return Command(
            update={
                "messages": [
                    ToolMessage(json.dumps(payload), tool_call_id=tool_call_id)
                ]
            }
        )


@tool
def concat_videos(
    request: VideoConcatRequest,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """Concatenate multiple video files using ffmpeg concat demuxer."""
    temp_dir = Path(tempfile.mkdtemp(prefix="video_concat_"))
    logs: List[str] = []

    try:
        # Verify all input files exist
        for video_path in request.video_paths:
            if not os.path.exists(video_path):
                payload = VideoConcatResult(
                    video_path=None,
                    segments=None,
                    logs=[f"error: video file not found: {video_path}"],
                ).model_dump()
                return Command(
                    update={
                        "messages": [
                            ToolMessage(json.dumps(payload), tool_call_id=tool_call_id)
                        ]
                    }
                )

        # Create concat list file
        concat_list_path = temp_dir / "concat_list.txt"
        with concat_list_path.open("w", encoding="utf-8") as concat_file:
            for video_path in request.video_paths:
                concat_file.write(f"file '{video_path}'\n")

        # Setup output path
        output_dir = (
            Path(request.output_path).parent
            if request.output_path
            else Path("/tmp/videos")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            Path(request.output_path)
            if request.output_path
            else output_dir / f"{request.output_basename}_{uuid.uuid4().hex[:8]}.mp4"
        )

        # Run ffmpeg concat
        concat_command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c",
            "copy",
            str(output_path),
        ]

        proc = subprocess.run(
            concat_command,
            capture_output=True,
            text=True,
            check=False,
        )

        if proc.returncode != 0 or not output_path.exists():
            logs.append(f"ffmpeg concat failed with code {proc.returncode}")
            if proc.stderr:
                logs.append(proc.stderr.strip())
            payload = VideoConcatResult(
                video_path=None, segments=None, logs=logs
            ).model_dump()
            return Command(
                update={
                    "messages": [
                        ToolMessage(json.dumps(payload), tool_call_id=tool_call_id)
                    ]
                }
            )

        # Upload the concatenated video
        upload = upload_media_to_google(
            str(output_path), google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        payload: Dict[str, Any] = {
            "video": {
                "kind": "video",
                "path": str(output_path),
                "mime_type": "video/mp4",
                "file_id": upload.get("file_id"),
                "file_uri": upload.get("file_uri"),
            },
            "segments": request.video_paths,
            "logs": [log for log in logs if log],
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(json.dumps(payload), tool_call_id=tool_call_id),
                    HumanMessage(
                        content=[
                            {
                                "type": "media",
                                "file_uri": upload.get("file_uri"),
                                "mime_type": upload.get("mime_type"),
                            },
                        ]
                    ),
                ]
            }
        )

    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


@tool
def run_ffmpeg_binary(
    request: FFmpegExecuteRequest,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
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

        # Upload output if present
        file_info: Dict[str, Optional[str]] = {"file_id": None, "file_uri": None}
        if request.output_path and os.path.exists(request.output_path):
            file_info = upload_media_to_google(
                request.output_path, google_api_key=os.getenv("GOOGLE_API_KEY")
            )

        payload: Dict[str, Any] = {
            "output_path": request.output_path or "",
            "ffmpeg_command": planned_command,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "file_id": file_info.get("file_id"),
            "file_uri": file_info.get("file_uri"),
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(json.dumps(payload), tool_call_id=tool_call_id),
                    HumanMessage(
                        content=[
                            {
                                "type": "media",
                                "file_uri": file_info.get("file_uri"),
                                "mime_type": file_info.get("mime_type"),
                            },
                        ]
                    ),
                ]
            }
        )
    except ToolExecutionError as e:
        # Expose error via Command ToolMessage
        payload = {
            "output_path": request.output_path or "",
            "ffmpeg_command": getattr(e, "command", request.command),
            "stdout_tail": "",
            "stderr_tail": str(e),
            "error": True,
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(json.dumps(payload), tool_call_id=tool_call_id)
                ]
            }
        )


@tool
def analyze_video_quality(
    scene_index: int,
    score: float,
    feedback: Optional[str] = None,
    storyboard: Optional[Storyboard] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    config: RunnableConfig | None = None,
) -> Command:
    """Update a single scene's quality score and feedback via Command (KISS).

    Inputs:
      - scene_index: index of the scene to update
      - score: numeric score (0-10) provided by the LLM
      - feedback: optional short feedback string
      - storyboard: optional current storyboard to update; if omitted, no-op
    """
    try:
        if storyboard is None or not isinstance(scene_index, int):
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=json.dumps(
                                {
                                    "error": "Missing storyboard or invalid scene_index",
                                }
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        # Clamp score to [0, 10]
        try:
            normalized_score = max(0.0, min(10.0, float(score)))
        except Exception:
            normalized_score = 0.0

        if scene_index < 0 or scene_index >= len(storyboard.scenes):
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=json.dumps(
                                {
                                    "error": "scene_index out of range",
                                    "scene_index": scene_index,
                                    "num_scenes": len(storyboard.scenes),
                                }
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        # Update the targeted scene
        scene = storyboard.scenes[scene_index]
        scene.quality_score = normalized_score
        if feedback is not None:
            scene.feedback = feedback

        # Return Command to update state.storyboard and message history
        return Command(
            update={
                "storyboard": storyboard,
                "messages": [
                    ToolMessage(
                        content=json.dumps(
                            {
                                "updated": True,
                                "scene_index": scene_index,
                                "score": normalized_score,
                                "feedback": feedback or "",
                            }
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )
    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps({"error": f"Quality update failed: {e}"}),
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )
