"""Simple batch descriptor for videos using DSPy + Gemini 2.0 Flash.

Happy path:
- For each video in a given directory:
  1) Extract mono 16 kHz WAV audio (temporary file).
  2) Create a muted copy of the video (no audio track).
  3) Send the WAV to Gemini 2.0 Flash via DSPy using the Audio type.
  4) Save transcript and a two-line Bloom-style description next to the video.

Requirements at runtime:
- ffmpeg available on PATH
- Environment: GOOGLE_API_KEY set (used by DSPy for Gemini)

Usage:
    python desc_writer.py /absolute/path/to/videos

Notes for maintainers:
- Keep this script narrow and predictable; no fallbacks or variants.
- If DSPy.Audio is not available in your installed DSPy version, update DSPy.
"""

from __future__ import annotations

import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import dspy


# -----------------------------
# Configuration and constants
# -----------------------------

SUPPORTED_VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def _require_env(key: str) -> str:
    """Return required env var or raise with a clear message.

    We don't implement fallbacks; this script assumes GOOGLE_API_KEY is set.
    """

    value = os.getenv(key, "").strip()
    if not value:
        raise RuntimeError(
            f"Environment variable {key} must be set. Export GOOGLE_API_KEY before running."
        )
    return value


def _run_ffmpeg(args: List[str]) -> None:
    """Run ffmpeg with given args. Raises RuntimeError on non-zero exit.

    This helper keeps error messages short and actionable.
    """

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{proc.stderr}")


def extract_wav_16k_mono(input_video: Path, output_wav: Path) -> None:
    """Extract mono 16 kHz WAV using ffmpeg.

    We always overwrite existing outputs to keep the flow simple.
    """

    _run_ffmpeg([
        "-y",
        "-i",
        str(input_video),
        "-vn",  # no video
        "-ac",
        "1",  # mono
        "-ar",
        "16000",  # 16 kHz
        "-c:a",
        "pcm_s16le",  # 16-bit PCM WAV
        str(output_wav),
    ])


def create_muted_copy(input_video: Path, output_video: Path) -> None:
    """Create a muted copy (audio removed) without re-encoding the video stream.

    This uses stream copy for video (`-c copy`) and removes audio (`-an`).
    """

    _run_ffmpeg([
        "-y",
        "-i",
        str(input_video),
        "-c",
        "copy",
        "-an",
        str(output_video),
    ])


def configure_dspy_for_gemini() -> None:
    """Configure DSPy to use Google Gemini 2.0 Flash.

    This matches the pattern used elsewhere in this repo (see dspy/video_generation_agent).
    """

    api_key = _require_env("GOOGLE_API_KEY")
    lm = dspy.LM(
        "gemini/gemini-2.0-flash",
        api_key=api_key,
        temperature=0.2,
        max_tokens=16000,
    )
    dspy.configure(lm=lm)


class AudioDescribeSignature(dspy.Signature):
    """Transcribe the given audio and produce a concise two-line Bloom-style description.

    Output requirements:
    - transcript: near-verbatim transcript of the spoken audio.
    - description: exactly two lines; each line starts with a measurable Bloom's verb
      (e.g., Identify, Analyze, Compare, Design, Evaluate, Synthesize). Each line is a
      single concise sentence, objective and faithful to the audio.
    """

    audio: dspy.Audio = dspy.InputField(description="Mono 16kHz WAV extracted from the video.")
    transcript: str = dspy.OutputField(description="Verbatim transcript of the audio.")
    description: str = dspy.OutputField(
        description="Two lines total; each starts with a Bloom verb; one sentence per line."
    )


def _load_audio_as_dspy(audio_path: Path) -> dspy.Audio:
    """Load audio as a DSPy Audio object.

    We assume modern DSPy exposes `Audio.from_file(path, mime_type=...)`.
    If your installed DSPy lacks this API, update DSPy to a recent version.
    """

    AudioType = getattr(dspy, "Audio", None)
    if AudioType is None:
        raise RuntimeError(
            "DSPy.Audio not found. Please upgrade DSPy to a version that supports Audio."
        )
    if not hasattr(AudioType, "from_file"):
        raise RuntimeError(
            "DSPy.Audio.from_file(...) not available. Please upgrade DSPy to a recent version."
        )
    # Older DSPy versions do not accept a mime_type kwarg; rely on file extension.
    return AudioType.from_file(str(audio_path))


def _ensure_two_lines(text: str) -> str:
    """Return at most two non-empty lines, trimmed. If fewer provided, return as-is.

    We don't invent content; this is only defensive formatting to keep the contract.
    """

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[:2])


@dataclass
class ProcessResult:
    video_path: Path
    muted_video_path: Path
    wav_path: Path
    transcript_path: Path
    description_path: Path


def process_video(video_path: Path) -> ProcessResult:
    """Process a single video: extract WAV, create muted copy, run DSPy, save outputs."""

    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    stem = video_path.stem
    suffix = video_path.suffix
    parent = video_path.parent

    wav_path = parent / f"{stem}.mono16k.wav"
    muted_path = parent / f"{stem}.muted{suffix}"
    transcript_path = parent / f"{stem}.transcript.txt"
    description_path = parent / f"{stem}.description.txt"

    # 1) Extract audio (WAV mono 16kHz)
    extract_wav_16k_mono(video_path, wav_path)

    # 2) Create a muted copy of the video
    create_muted_copy(video_path, muted_path)

    # 3) DSPy inference: transcript + two-line description
    configure_dspy_for_gemini()
    predictor = dspy.Predict(AudioDescribeSignature)
    audio_obj = _load_audio_as_dspy(wav_path)
    pred = predictor(audio=audio_obj)

    transcript_text = getattr(pred, "transcript", "").strip()
    description_text = _ensure_two_lines(getattr(pred, "description", "").strip())

    # 4) Save outputs
    transcript_path.write_text(transcript_text, encoding="utf-8")
    description_path.write_text(description_text + ("\n" if not description_text.endswith("\n") else ""), encoding="utf-8")

    return ProcessResult(
        video_path=video_path,
        muted_video_path=muted_path,
        wav_path=wav_path,
        transcript_path=transcript_path,
        description_path=description_path,
    )


def iter_videos(directory: Path) -> Iterable[Path]:
    """Yield video files under directory (non-recursive) that match supported extensions."""

    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        # Skip any previously created muted copies to avoid re-processing videos with no audio.
        if ".muted" in p.stem:
            continue
        yield p


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python desc_writer.py /absolute/path/to/videos", file=sys.stderr)
        return 2

    input_dir = Path(argv[1]).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 2

    print(f"Processing directory: {input_dir}")
    videos = list(iter_videos(input_dir))
    if not videos:
        print("No supported videos found (.mp4, .mov, .mkv, .avi, .webm).")
        return 0

    for video in videos:
        print(f"\n→ Processing: {video}")
        try:
            result = process_video(video)
        except Exception as e:
            # Clear, actionable error for interns maintaining this script.
            print(f"FAILED: {video}\n{e}")
            continue

        print("✓ Muted video:", result.muted_video_path)
        print("✓ Transcript:", result.transcript_path)
        print("✓ Description:", result.description_path)
        # The temporary WAV is kept to make debugging/model re-runs simpler.

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))


