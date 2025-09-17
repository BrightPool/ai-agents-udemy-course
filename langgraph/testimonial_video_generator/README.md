# Testimonial Video Generator DAG Agent

This project mirrors the "testimonial video generator" n8n workflow using LangGraph. It coordinates persona selection, structured prompt creation, video generation through the FAL queue API, and final video assembly with `ffmpeg` - all inside a single Directed Acyclic Graph (DAG) agent.

## High-Level Flow

1. **Init** - create a run-specific directory under `/tmp/n8n/<execution_id>` and clean stale files.
2. **Persona Setup** - define available personas, select the form submission choice, and materialize the persona profile.
3. **Phase 1 - Buying Situations** - ask Anthropic Claude to produce exactly three buying situations for the persona and validate them with Pydantic models.
4. **Phase 2 - Prompt Generation Loop** - iterate over each buying situation, request a structured Veo3 testimonial prompt, and cache both the structured output and the final prompt string.
5. **Phase 3 - Video Generation** - submit prompt strings to FAL (`/fal-ai/veo3` queue), poll until all statuses are `COMPLETED`, fetch the finished video URLs, download them locally, build a concat manifest, and merge them into `final_output.mp4` via `ffmpeg`.
6. **Completion** - return the local URL (`http://localhost:3001/video/<execution_id>/final_output.mp4`) alongside the execution id and a success message.

The LangGraph DAG exactly reproduces the control flow of the original n8n nodes (waiting loop, conditional branching, and concatenation pipeline), while keeping all intermediate artefacts on disk for inspection.

## Core Nodes & Responsibilities

- `init_and_create_directory` - allocate workspace and generate an `execution_id`.
- `define_personas` / `set_persona` - replicate n8n persona definition and selection logic.
- `generate_buying_situations` - call Claude 3.5 Sonnet and coerce the response into `BuyingSituationsOutput` (expects three entries).
- `generate_prompt_for_current` -> `next_prompt_or_submit` -> `increment_prompt_index` - structured prompt loop mirroring the split/batch cycle in n8n.
- `submit_fal_requests` -> `wait_30_seconds` -> `poll_statuses` - queue requests, throttle polling, and branch until everything completes.
- `fetch_video_urls` -> `download_videos` -> `prepare_concat_file` -> `merge_videos_ffmpeg` - materialize videos and create the final reel.
- `complete` - emit the JSON payload consumed by the React front-end.

## Prerequisites

- Python 3.9+
- `ffmpeg` available on your `PATH` (required for the concat step)
- API credentials:
  - `ANTHROPIC_API_KEY` (Claude 3.5 Sonnet)
  - FAL queue auth - either `FAL_API_KEY` or a header pair (`FAL_AUTH_HEADER_NAME` / `FAL_AUTH_HEADER_VALUE`)

Optional but recommended tools:

- [`uv`](https://github.com/astral-sh/uv) for environment management
- `langgraph` CLI (installed automatically via project dependencies)

## Setup

```bash
# Clone or switch into this project directory
cd langgraph/testimonial_video_generator

# Create a virtual environment (uv makes this quick)
uv venv
source .venv/bin/activate

# Install dependencies
uv sync

# Copy the environment template if present
cp .env.example .env  # edit values afterwards (file may not exist in repo)
```

Populate `.env` with the required keys:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
FAL_API_KEY=key-...
# or supply a custom header
FAL_AUTH_HEADER_NAME=Authorization
FAL_AUTH_HEADER_VALUE=Key key-...
FAL_QUEUE_URL=https://queue.fal.run/fal-ai/veo3
FAL_REQUEST_URL_BASE=https://queue.fal.run/fal-ai/veo3/requests
```

## Running the Agent

Start the LangGraph development server from the project root:

```bash
uv run langgraph dev
```

You should see output similar to:

```
Ready!

- API: http://127.0.0.1:2024
- Docs: http://127.0.0.1:2024/docs
```

### Testing with the SDK

`run_graph.py` streams a full execution using the LangGraph SDK:

```bash
uv run python run_graph.py
```

It sends `persona_selection="Omar US Developer"` and prints incremental events as the DAG progresses. You can adjust the payload in `run_graph.py` to match other personas or integrate your own frontend.

### Calling the Compiled Graph Directly

You can also script against the SDK yourself:

```python
from langgraph_sdk import get_client
import asyncio

client = get_client(url="http://127.0.0.1:2024")

async def main():
    async for event in client.runs.stream(
        None,
        "agent",
        input={"persona_selection": "Emily US Foodie"},
        stream_mode="messages-tuple",
    ):
        print(event.event, event.data)

asyncio.run(main())
```

The final chunk includes `result` with the public video URL and metadata.

## Verifying Output

- Intermediate videos land in `/tmp/n8n/<execution_id>/` (named `00.mp4`, `01.mp4`, ...) together with `videos.txt`.
- `final_output.mp4` combines them using the concat demuxer.
- The agent returns a JSON blob replicating the n8n `complete` node output, ready for the existing React UI.

## Troubleshooting

- **Missing videos** - ensure the prompts produce three buying situations; otherwise the loop short-circuits.
- **Polling never finishes** - verify FAL credentials and URLs; the agent expects `status` to become `COMPLETED`.
- **ffmpeg errors** - install `ffmpeg` locally (`brew install ffmpeg`, `apt install ffmpeg`, etc.) and confirm it is on `PATH`.

This README now reflects the LangGraph implementation used in the Udemy course and aligns with the original n8n testimonial video generator workflow.
