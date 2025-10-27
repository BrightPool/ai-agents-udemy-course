# Video Generation Agent

## What this agent does
- Creates short ad-style videos by chaining tools: storyboard → image → video → (optional) ffmpeg postprocess → scoring
- Uses ReAct to orchestrate multiple tools (Gemini for text/image, fal.ai Kling for video, ffmpeg for processing)
- Inputs: `user_message`
- Outputs: `answer` (progress/final), plus `action` and `tool_result` JSON from tools

---

## High-level workflow
```
[Start: Provide user_message]
     |
     v
[ReAct: plan]
     |
     +--> [create_story_board]
     |          |
     |          v
     |     [storyboard JSON]
     |
     +--> [update_story_board] (optional)
     |          |
     |          v
     |     [edited storyboard JSON]
     |
     +--> [generate_image]
     |          |
     |          v
     |     [image files under /tmp/images]
     |
     +--> [kling_generate_video_from_image]
     |          |
     |          v
     |     [video URL]
     |
     +--> [run_ffmpeg_binary] (optional)
     |          |
     |          v
     |     [processed video path]
     |
     +--> [score_video] (optional)
     |          |
     |          v
     |     [quality JSON]
     |
     v
[Output: action, tool_result, answer]
```
- Inputs: product brief in `user_message`
- Outputs: step-by-step `answer` updates and final result summary
- External services: Google Gemini (`GOOGLE_API_KEY`), fal.ai Kling (`FAL_KEY`/`FAL_API_KEY`), local ffmpeg

---

## Components and external services
- LLMs: `gemini/gemini-2.0-flash` (main), `gemini-2.5-flash-image-preview` (image gen), `gemini-2.5-flash` (scoring)
- Env: `GOOGLE_API_KEY`, optional `FAL_KEY` or `FAL_API_KEY`, optional `KLING_FAL_ENDPOINT`
- Tools/APIs:
  - Storyboard: `create_story_board`, `update_story_board`
  - Images: `generate_image` (Gemini)
  - Video: `kling_generate_video_from_image` (fal.ai)
  - Processing: `run_ffmpeg_binary` (local ffmpeg)
  - Scoring: `score_video` (Gemini text rubric + ffprobe metadata)
- Dependencies (pip): `dspy`, `python-dotenv`, `google-genai`, `fal-client`, `pillow`, `ffmpeg-python`
- Notes: Requires `ffmpeg` on PATH; temp dirs under `/tmp` are created/cleaned.

---

## Copy‑paste prompts

### 1) ReAct system schema (VideoReActSignature)
```
System/Instructions:
You are a product advertisement creative agent.

Tools:
- create_story_board(product_name, brand, target_audience, key_message, tone, scenes_count=3, default_scene_duration_seconds=6.0)
- update_story_board(storyboard_json, instructions)
- generate_image(prompt, num_images=1, output_basename="product_ad_image")
- kling_generate_video_from_image(prompt, image_path="", image_url="", duration_seconds=6.0, endpoint="fal-ai/kling/v1", seed=None)
- run_ffmpeg_binary(command, output_path="", timeout_seconds=180)
- score_video(video_path, target_quality_score=None)

Workflow: Create storyboard → update (optional) → generate image → generate video → (optional) ffmpeg postprocess → score video.
Outputs:
- action: one of {create_story_board, update_story_board, generate_image, kling_generate_video_from_image, run_ffmpeg_binary, score_video, answer_direct}
- tool_result: JSON string from the tool call (may be empty for answer_direct)
- answer: concise user-facing update or final summary
```

### 2) Scoring rubric (embedded in tool)
```
Return ONLY JSON with keys: visual_quality (0-10), audio_quality (0-10), narrative_coherence (0-10), feedback (string).
Inputs include video path and ffprobe metadata.
```

---

## Scoring/aggregation (code)
JS:
```javascript
// Aggregate a 0–10 quality score from rubric sub-scores
function qualityScore(v, a, n) {
  const clamp = x => Math.max(0, Math.min(10, Number(x) || 0));
  return Math.round(((clamp(v) + clamp(a) + clamp(n)) / 3) * 100) / 100;
}
```

Python:
```python
# Parse score JSON and compute average
import json

def compute_quality(avg_json: str) -> float:
    data = json.loads(avg_json)
    v = int(data.get("visual_quality", 0))
    a = int(data.get("audio_quality", 0))
    n = int(data.get("narrative_coherence", 0))
    return round((v + a + n) / 3.0, 2)
```

---

## Implementation steps
1) Install: `pip install dspy python-dotenv google-genai fal-client pillow ffmpeg-python`
2) Configure LM: `dspy.configure(lm=dspy.LM("gemini/gemini-2.0-flash", api_key=GOOGLE_API_KEY, temperature=0.6, max_tokens=16000))`
3) Ensure `ffmpeg` is installed and on PATH; set `FAL_KEY`/`FAL_API_KEY` for Kling, if used
4) Implement tools from the notebook (temp dir helpers, storyboard, image gen, Kling video, ffmpeg, scoring)
5) Define `VideoReActSignature` and wire tools into `VideoGenerationAgent` (set `max_iters`)
6) Call the agent with a product brief; monitor `action`, `tool_result`, and `answer` for progress

---

## Example I/O
- Input: `"Create a 3‑scene ad storyboard for EcoBottle by GreenLife (inspiring tone) and produce a 6s clip."`
- Output (abridged):
  - action: `generate_image`
  - tool_result: `{"images": ["/tmp/images/product_ad_image_0.png"]}`
  - answer: `"Generated image for the second scene. Now generating image for the last scene."`

---

## Files
- `dspy/video_generation_agent/video-generation-agent.ipynb`
- `dspy/video_generation_agent/example_video.mp4`
