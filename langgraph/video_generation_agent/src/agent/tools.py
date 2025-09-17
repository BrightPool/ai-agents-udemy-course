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
    Veo3VideoBatchRequest,
    Veo3VideoRequest,
    VideoConcatRequest,
    VideoConcatResult,
)
from .utils import download_file_to_path, upload_media_to_google


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


def _resolve_image_to_fal_url(
    *,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_base64: Optional[str] = None,
    gcs_or_file_id: Optional[str] = None,
    mime_type: Optional[str] = None,
) -> str:
    """Normalize any input image source into a fal-hosted URL.

    Rules (KISS):
    1) If local path -> upload via fal and return URL
    2) If Google file id/URL -> download with Google API -> upload via fal
    3) If http(s) URL -> download -> upload via fal
    4) If base64 -> write temp file -> upload via fal

    Raises ToolExecutionError if unable to produce a fal URL.
    """
    # 1) Local path
    if isinstance(image_path, str) and os.path.exists(image_path):
        try:
            return fal_client.upload_file(image_path)
        except Exception as e:
            raise ToolExecutionError(f"Failed to upload image_path to fal: {e}")

    # Prepare temp dir
    os.makedirs("/tmp/images", exist_ok=True)

    def _upload_temp(temp_path: str) -> str:
        try:
            return fal_client.upload_file(temp_path)
        except Exception as e:
            raise ToolExecutionError(f"Failed to upload image to fal: {e}")

    # 2) Google file id / URL
    candidate = gcs_or_file_id or image_url
    if isinstance(candidate, str) and (
        candidate.startswith("files/")
        or ("generativelanguage.googleapis.com" in candidate and "/files/" in candidate)
    ):
        tmp_path = f"/tmp/images/source_{uuid.uuid4().hex[:8]}"
        ok = download_file_to_path(
            candidate, tmp_path, google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        if not ok:
            raise ToolExecutionError(
                "Failed to download Google file; ensure GOOGLE_API_KEY is set and file id is valid"
            )
        return _upload_temp(tmp_path)

    # 3) Generic http(s) URL -> download then upload
    if isinstance(image_url, str) and (
        image_url.startswith("http://") or image_url.startswith("https://")
    ):
        try:
            import urllib.request

            tmp_path = f"/tmp/images/source_{uuid.uuid4().hex[:8]}"
            urllib.request.urlretrieve(image_url, tmp_path)  # nosec B310
            return _upload_temp(tmp_path)
        except Exception as e:
            raise ToolExecutionError(f"Failed to fetch image_url: {e}")

    # 4) Base64 -> write then upload
    if isinstance(image_base64, str) and image_base64:
        try:
            if image_base64.startswith("data:"):
                header, b64_data = image_base64.split(",", 1)
                mt = header.split(";", 1)[0].split(":", 1)[-1] or (
                    mime_type or "image/png"
                )
            else:
                b64_data = image_base64
                mt = mime_type or "image/png"

            raw = base64.b64decode(b64_data)
            ext = (
                ".png"
                if mt.endswith("png")
                else (".jpg" if "jpeg" in mt or "jpg" in mt else ".bin")
            )
            tmp_path = f"/tmp/images/source_{uuid.uuid4().hex[:8]}{ext}"
            with open(tmp_path, "wb") as f:
                f.write(raw)
            return _upload_temp(tmp_path)
        except Exception as e:
            raise ToolExecutionError(f"Failed to process base64 image: {e}")

    raise ToolExecutionError(
        "No valid image provided. Supply image_path, gcs_uri/files/<id>, image_url, or image_base64."
    )


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
            # Enriched prompt for maximum video vibes and storyboard guidance
            prompt = (
                f"Storyboard scene: {sid}. Create a hero frame to animate later. "
                f"Style: cinematic, ad-grade, brand aesthetic for {brand}. "
                f"Visuals: {product} centered with human interaction; clean composition; depth of field. "
                f"Lighting: soft key light with rim highlights; high dynamic range; realistic. "
                f"Camera: {('establishing wide to medium push-in' if sid == 'beginning' else ('handheld medium tracking' if sid == 'middle' else 'slow dolly-out reveal'))}. "
                f"Mood: {tone}. "
                f"Music: modern {('inspirational build-up' if sid == 'beginning' else ('energetic percussive mid-tempo' if sid == 'middle' else 'uplifting resolve'))}, instrumental only. "
                f"Shots: use a primary hero angle; ensure cohesive color grading; cinematic vibes. "
                f"Avoid text overlays. Maintain brand cleanliness."
            )
            # Align default scene duration to 8 seconds to match image-to-video clips
            duration_seconds = max(8.0, float(request.default_scene_duration_seconds))
            scenes.append(
                {
                    "id": scene_id,
                    "title": title,
                    "description": desc,
                    "prompt": prompt,
                    "duration_seconds": duration_seconds,
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
        messages.append(
            HumanMessage(
                content=f"""f Can you please just generate a single image for the following prompt: {request.prompt}"""
            )
        )
        response = llm.invoke(
            [HumanMessage(content=request.prompt)],
            generation_config={"response_modalities": ["TEXT", "IMAGE"]},
        )

        os.makedirs("/tmp/images", exist_ok=True)
        images_payload: List[Dict[str, Any]] = []
        file_uploads = []
        saved_count = 0
        for idx, block in enumerate(response.content, start=1):
            if saved_count >= max(1, int(getattr(request, "max_images", 1))):
                break
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
            saved_count += 1

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
    """Generate a video using fal.ai Veo 3 Image-to-Video endpoint.

    Aligns to schema: prompt, image_url (data URI or hosted URL), aspect_ratio,
    duration (8s), resolution (720p/1080p), generate_audio.

    Simplified: reads a local file under /tmp/images (request.image_filename),
    uploads to fal storage, then submits to the image-to-video endpoint.
    """
    try:
        fal_api_key = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
        if not fal_api_key:
            raise ToolExecutionError(
                "Missing FAL_KEY/FAL_API_KEY. Please set it in your environment or .env"
            )
        os.environ.setdefault("FAL_KEY", fal_api_key)

        # Resolve local path under /tmp/images
        local_path = os.path.join("/tmp/images", request.image_filename)
        if not os.path.exists(local_path):
            payload = {
                "error": f"Image not found: {local_path}",
                "hint": "Ensure the image was generated/saved under /tmp/images",
            }
            return Command(
                update={
                    "messages": [
                        ToolMessage(json.dumps(payload), tool_call_id=tool_call_id)
                    ]
                }
            )

        # Upload to fal and get a hosted URL
        try:
            reference_url = fal_client.upload_file(local_path)
        except Exception as e:
            payload = {"error": f"Failed to upload to fal: {e}"}
            return Command(
                update={
                    "messages": [
                        ToolMessage(json.dumps(payload), tool_call_id=tool_call_id)
                    ]
                }
            )

        # Fail fast if no valid reference
        # reference_url already a fal-hosted URL

        # Build arguments per image-to-video schema
        # Coerce duration to 8s for image-to-video (API constraint)
        arguments: Dict[str, Any] = {
            "prompt": request.prompt,
            "aspect_ratio": request.aspect_ratio,
            "duration": "8s",
            "resolution": request.resolution,
            "generate_audio": request.generate_audio,
            "image_url": reference_url,
        }

        # Use the image-to-video endpoint
        result = fal_client.run("fal-ai/veo3/image-to-video", arguments=arguments)

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
def veo3_generate_videos_batch(
    request: Veo3VideoBatchRequest,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """Generate multiple Veo3 Image-to-Video clips concurrently and optionally stitch.

    - Validates that each request has an image (image_url, image_base64, image_path, gcs_uri/gsc_uri)
    - Submits all requests concurrently using fal subscribe for better throughput
    - Waits for all to complete, returns list of local paths and URLs
    - If stitch_after=True, concatenates the videos and returns the final artifact
    """
    try:
        # Validate presence of image source in every request
        missing: List[int] = []
        for idx, r in enumerate(request.requests):
            has_image = any(
                [
                    bool(r.image_filename),
                ]
            )
            if not has_image:
                missing.append(idx)
        if missing:
            msg = {
                "error": "All scenes must include an image before generating videos",
                "missing_indices": missing,
            }
            return Command(
                update={
                    "messages": [
                        ToolMessage(json.dumps(msg), tool_call_id=tool_call_id)
                    ]
                }
            )

        fal_api_key = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
        if not fal_api_key:
            raise ToolExecutionError(
                "Missing FAL_KEY/FAL_API_KEY. Please set it in your environment or .env"
            )
        os.environ.setdefault("FAL_KEY", fal_api_key)

        # Helper to build arguments per single request (reuse single-tool logic partially)
        def build_args(r: Veo3VideoRequest) -> Dict[str, Any]:
            local_path = os.path.join("/tmp/images", r.image_filename)
            if not os.path.exists(local_path):
                raise ToolExecutionError(f"Image not found: {local_path}")
            try:
                reference_url = fal_client.upload_file(local_path)
            except Exception as e:
                raise ToolExecutionError(f"Failed to upload to fal: {e}")

            args: Dict[str, Any] = {
                "prompt": r.prompt,
                "aspect_ratio": r.aspect_ratio,
                "duration": "8s",
                "resolution": r.resolution,
                "generate_audio": r.generate_audio,
                "image_url": reference_url,
            }
            return args

        # Submit all requests concurrently with subscribe (with_logs disabled for brevity)
        handlers = []
        for r in request.requests:
            args = build_args(r)
            handler = fal_client.submit("fal-ai/veo3/image-to-video", arguments=args)
            handlers.append((r, handler))

        # Collect results
        local_paths: List[str] = []
        urls: List[str] = []
        for r, handler in handlers:
            res = fal_client.result("fal-ai/veo3/image-to-video", handler.request_id)
            video_url: Optional[str] = None
            if isinstance(res, dict):
                v = res.get("video")
                if isinstance(v, dict):
                    u = v.get("url") or v.get("video_url")
                    if isinstance(u, str):
                        video_url = u
                if not video_url:
                    u2 = res.get("url") or res.get("video_url")
                    if isinstance(u2, str):
                        video_url = u2
            urls.append(video_url or "")

            # download locally for stitching/upload
            local_path: Optional[str] = None
            try:
                if video_url and video_url.startswith("http"):
                    import urllib.request

                    os.makedirs("/tmp/videos", exist_ok=True)
                    local_path = f"/tmp/videos/veo3_scene_{uuid.uuid4().hex[:8]}.mp4"
                    urllib.request.urlretrieve(video_url, local_path)  # nosec B310
            except Exception:
                local_path = None
            if local_path:
                local_paths.append(local_path)

        # If stitching requested and we have 2+ clips, perform concat here
        if request.stitch_after and len(local_paths) >= 2:
            temp_dir = Path(tempfile.mkdtemp(prefix="video_concat_batch_"))
            logs: List[str] = []
            try:
                concat_list_path = temp_dir / "concat_list.txt"
                with concat_list_path.open("w", encoding="utf-8") as concat_file:
                    for vp in local_paths:
                        concat_file.write(f"file '{vp}'\n")

                output_dir = Path("/tmp/videos")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = (
                    output_dir / f"{request.output_basename}_{uuid.uuid4().hex[:8]}.mp4"
                )

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
                    concat_command, capture_output=True, text=True, check=False
                )

                if proc.returncode != 0 or not output_path.exists():
                    if proc.stderr:
                        logs.append(proc.stderr.strip())
                    payload = {
                        "video": None,
                        "segments": local_paths,
                        "logs": logs,
                    }
                    return Command(
                        update={
                            "messages": [
                                ToolMessage(
                                    json.dumps(payload), tool_call_id=tool_call_id
                                )
                            ]
                        }
                    )

                upload = upload_media_to_google(
                    str(output_path), google_api_key=os.getenv("GOOGLE_API_KEY")
                )
                payload = {
                    "video": {
                        "kind": "video",
                        "path": str(output_path),
                        "mime_type": "video/mp4",
                        "file_id": upload.get("file_id"),
                        "file_uri": upload.get("file_uri"),
                    },
                    "segments": local_paths,
                    "logs": logs,
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

        payload = {
            "videos": {
                "local_paths": local_paths,
                "urls": urls,
            }
        }
        return Command(
            update={
                "messages": [
                    ToolMessage(json.dumps(payload), tool_call_id=tool_call_id)
                ]
            }
        )
    except Exception as exc:
        payload = {"error": f"batch error: {exc}"}
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
