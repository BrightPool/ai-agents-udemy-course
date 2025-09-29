# Video Response Generator DAG

## What this agent does
- Generates three realistic testimonial-style video clips for a selected persona and buying situations.
- Uses an LLM to (1) create buying situations and (2) assemble per‑scene prompts.
- Submits prompts to a video generation service, downloads the clips, and merges them into one MP4.

---

## High-level workflow
```
[Start: Provide persona name]
     |
     v
[LLM: Generate 3 buying situations (JSON)]
     |
     v
[LLM: Build testimonial prompt for each]
     |
     v
[Video Gen: Create 3 videos via FAL route]
     |
     v
[Code: Download MP4s + ffmpeg concat]
     |
     v
[Output: final MP4 URL + metadata]

- Inputs: persona_selection (string; must match a defined persona)
- Outputs: final video URL, execution_id, work dir, prompts, per‑clip URLs
- External services: OpenAI text model, FAL video generation, ffmpeg CLI
```

---

## Components and external services
- LLMs: openai/gpt-5-mini
- Env: OPENAI_API_KEY, FAL_API_KEY (or FAL_KEY), FAL_MODEL_ROUTE
- Notes: Requires `ffmpeg` available on PATH. Default FAL route: `fal-ai/veo3`.

---

## Copy‑paste prompts

### 1) Generation prompt (buying situations)
```
Role play as {name}, a {age} {gender} from {location}, works as a {occupation} earning {income}. {background}. You have been invited to review a new product concept and reveal deep, personal insights into when you would buy this offer.

Respond only with raw JSON (no markdown). Return only valid JSON with fields:
  - "persona": string
  - "buying_situations": object with exactly 3 keys. Each value must include:
    - situation (string)
    - location (string)
    - trigger (string)
    - evaluation (string)
    - conclusion (string)

Example shape (copy/edit values as needed):
{"persona":"Rhys UK Entrepreneur","buying_situations":{"celebrating_milestone":{"situation":"Celebrating a business milestone","location":"Private members' club in London","trigger":"Closed a major deal","evaluation":"Considered champagne and cocktails, but wanted tradition","conclusion":"Chose premium whisky"},"hosting_partners":{"situation":"Hosting international partners","location":"Home office in Manchester","trigger":"Inviting overseas investors","evaluation":"Compared various spirits; whisky is distinctly British","conclusion":"Bought respected Scottish whisky"},"relaxing_weekend":{"situation":"Relaxing after a long week","location":"Apartment balcony overlooking the city","trigger":"Needed a way to unwind","evaluation":"Looked at beer, gin, wine; whisky appealed","conclusion":"Chose whisky as a ritual drink"}}}
```

### 2) Prompt assembly schema (per video)
```
You are a director of a qualitative research agency that specialises in creating realistic simulated testimonials that capture buying situations for new product concepts.

Your tasks:
  1) Craft scene_description (persona name + short appearance)
  2) Write a short 6–8s quote
  3) Assemble final prompt_string for a video generator

Return a structured object strictly matching this JSON shape (no markdown around it):
{
  "prompt": {
    "scene_description": {
      "persona": "<Persona's name only, e.g., Omar Ali>",
      "appearance": "<brief appearance>"
    },
    "quote": "<short 6–8s quote>",
    "prompt_string": "<complete video prompt>"
  }
}

Important: scene_description.persona MUST be a single short string (name only), not an object.
Return exactly the keys shown above.
```

---

## Scoring/aggregation (code)
Python (ffmpeg concat of downloaded clips):
```python
from pathlib import Path
import subprocess

def prepare_concat_file(work_dir: str) -> Path:
    list_path = Path(work_dir) / "videos.txt"
    lines = [f"file '{p.name}'\n" for p in sorted(Path(work_dir).glob("*.mp4"))]
    list_path.write_text("".join(lines), encoding="utf-8")
    return list_path

def merge_videos_ffmpeg(work_dir: str) -> str:
    videos_txt = prepare_concat_file(work_dir)
    final_path = Path(work_dir) / "final_output.mp4"
    cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", str(videos_txt), "-c", "copy", "-y", str(final_path)
    ]
    subprocess.run(cmd, cwd=str(work_dir), check=True, capture_output=True)
    return str(final_path)
```

---

## Implementation steps
1) Trigger: input `persona_selection` (string; must match a predefined persona)
2) LLM: Generation
   - Use the Generation prompt above; fill persona fields; ensure exactly 3 buying_situations
3) LLM: Prompt assembly
   - For each situation, return the schema with `scene_description.persona` as a name string, plus `quote` and `prompt_string`
4) Video Gen
   - Submit each `prompt_string` to the FAL route (default `fal-ai/veo3`) and wait for completion
5) Code
   - Download each MP4, write `videos.txt`, and run ffmpeg concat to produce `final_output.mp4`
6) Output
   - Return execution_id, work_dir, prompts, per‑clip URLs, final video URL

Optional (advanced):
- Optimization loop: propose → evaluate → keep best

---

## Example I/O
- Input: `persona_selection = "Omar US Developer"`
- Output (abridged):
  - execution_id: 55b073a7f7fd408f8cfbd4b7f9b00919
  - work_dir: /tmp/n8n/55b073a7f7fd408f8cfbd4b7f9b00919
  - prompts: [ { scene_description.persona, appearance, quote, prompt_string }, ... ]
  - video_urls: ["https://.../output.mp4", "https://.../output.mp4", "https://.../output.mp4"]
  - final_path: /tmp/n8n/55b073a7f7fd408f8cfbd4b7f9b00919/final_output.mp4
  - result.videoUrl: http://localhost:3001/video/<execution_id>/final_output.mp4

---

## Files
- dspy/video_reaction_agent/video-response-generator-dag.ipynb


