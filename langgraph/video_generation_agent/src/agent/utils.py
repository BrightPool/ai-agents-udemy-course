"""Utility functions for the video generation agent."""

import asyncio
import os
import shutil
from typing import Any, Dict, List, Optional


def _get_google_api_key(override: Optional[str] = None) -> Optional[str]:
    """Resolve Google API key from override or environment."""
    if override and isinstance(override, str) and override.strip():
        return override
    return os.getenv("GOOGLE_API_KEY")


def upload_media_to_google(
    file_path: str, *, google_api_key: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """Upload a local file to Google GenAI file storage.

    Returns a dict with keys: file_id, file_uri, mime_type. Values may be None on failure.
    """
    try:
        import mimetypes

        api_key = _get_google_api_key(google_api_key)
        if not api_key:
            return {"file_id": None, "file_uri": None, "mime_type": None}

        from google import genai  # type: ignore[import-untyped]

        # Detect MIME type from file extension
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            # Fallback based on file extension
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".mp4", ".mov", ".avi", ".mkv"]:
                mime_type = "video/mp4"
            elif ext in [".jpg", ".jpeg"]:
                mime_type = "image/jpeg"
            elif ext in [".png"]:
                mime_type = "image/png"
            elif ext in [".gif"]:
                mime_type = "image/gif"
            elif ext in [".mp3"]:
                mime_type = "audio/mpeg"
            elif ext in [".wav"]:
                mime_type = "audio/wav"
            else:
                mime_type = "application/octet-stream"

        client = genai.Client(api_key=api_key)
        uploaded = client.files.upload(file=file_path)
        file_id = getattr(uploaded, "name", None) or getattr(uploaded, "id", None)
        file_uri = getattr(uploaded, "uri", None)
        if not isinstance(file_id, str):
            file_id = None
        if not isinstance(file_uri, str):
            file_uri = None
        return {"file_id": file_id, "file_uri": file_uri, "mime_type": mime_type}
    except Exception:
        return {"file_id": None, "file_uri": None, "mime_type": None}


# Best-effort download for Google file storage URIs or HTTP(S) URLs
def download_file_to_path(
    uri_or_id: str, dest_path: str, *, google_api_key: Optional[str] = None
) -> bool:
    """Download a file identified by a Google file URI/ID or HTTP(S) URL.

    Returns True on success. Supports:
      - data URIs (writes decoded content)
      - http/https URLs
      - Google GenAI file ids like "files/abc123" via genai client
    """
    try:
        if not isinstance(uri_or_id, str) or not uri_or_id:
            return False

        # data URI
        if uri_or_id.startswith("data:"):
            import base64 as _b64

            try:
                header, b64_data = uri_or_id.split(",", 1)
            except ValueError:
                return False
            content = _b64.b64decode(b64_data)
            os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(content)
            return True

        # HTTP(S)
        if uri_or_id.startswith("http://") or uri_or_id.startswith("https://"):
            # Special-case Google GenAI file URLs and rewrite to files/{id}
            if (
                "generativelanguage.googleapis.com" in uri_or_id
                and "/files/" in uri_or_id
            ):
                try:
                    # Extract the ID after '/files/' and before any trailing segments or query
                    import urllib.parse as _up

                    parsed = _up.urlparse(uri_or_id)
                    parts = parsed.path.split("/files/")
                    file_id = parts[1].split("/")[0] if len(parts) > 1 else None
                    if file_id:
                        rewritten = f"files/{file_id}"
                        return download_file_to_path(
                            rewritten, dest_path, google_api_key=google_api_key
                        )
                except Exception:
                    pass

            import urllib.request as _url

            os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
            _url.urlretrieve(uri_or_id, dest_path)  # nosec B310
            return True

        # Google GenAI file id (e.g., files/xyz)
        if uri_or_id.startswith("files/"):
            api_key = _get_google_api_key(google_api_key)
            if not api_key:
                return False
            try:
                from google import genai  # type: ignore[import-untyped]

                client = genai.Client(api_key=api_key)
                # download API may vary by version; try common attributes
                downloaded = getattr(client.files, "download", None)
                if callable(downloaded):
                    resp = downloaded(name=uri_or_id)
                    content = getattr(resp, "content", None) or getattr(
                        resp, "data", None
                    )
                    if isinstance(content, (bytes, bytearray)):
                        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
                        with open(dest_path, "wb") as f:
                            f.write(content)
                        return True
            except Exception:
                return False

        # Fallback: unsupported scheme
        return False
    except Exception:
        return False


# Initialize tmp directories
async def initialize_tmp_directories():
    """Initialize temporary directories for the video generation agent."""
    directories = [
        "/tmp/assets",
        "/tmp/audio",
        "/tmp/unsplash",
        "/tmp/subtitles",
        "/tmp/videos",
    ]

    def _create_dirs():
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    await asyncio.to_thread(_create_dirs)


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
