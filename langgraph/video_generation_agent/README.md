# Video Generation Agent

A LangGraph agent that plans, illustrates, and produces short product-advertisement videos by coordinating multiple Gemini and fal.ai capabilities.

## Highlights

- **Gemini 2.5 Flash orchestration** – the core planner/reasoner drives the creative workflow and conversations.
- **Inline hero imagery** – uses `gemini-2.5-flash-image-preview` to generate photorealistic stills and automatically injects them as base64 blocks into the running chat.
- **Native Veo 3 Fast video generation** – forwards storyboard prompts plus inline imagery directly to fal.ai’s `veo3/fast` endpoint and loops the resulting video URL back into the conversation.
- **Media-aware chat history** – the agent records images/videos in the message stream (base64 or URL) so downstream tools can reuse them scene-by-scene.
- **Automatic scene stitching** – once all clips are generated, `concat_videos` pulls the remote files (if needed) and uses ffmpeg to deliver a single polished mp4.
- **Optional post tools** – ffmpeg helpers, Kling fallback, and scoring utilities remain available for future extensions.

## Architecture Overview

1. **Initialize** – create temp directories, surface environment warnings, seed the system prompt.
2. **Agent loop** – Gemini 2.5 Flash reasons over the `VideoGenerationState.messages` history and decides which tool to call next.
3. **Storyboard tools** – `create_story_board` and `update_story_board` craft a three-scene concept for the ad.
4. **Image pass** – `generate_image` produces hero images, persists PNGs to `/tmp/images`, and returns base64 payloads.
5. **Video pass** – `veo3_generate_video` sends prompts plus optional reference imagery (path/url/base64) to fal.ai and returns a streaming-ready URL.
6. **Stitch** – `concat_videos` normalises segments (when needed) and concatenates them into a master mp4.
7. **Attach media** – `attach_media_to_chat` converts tool payloads into human messages so the main LLM can see the latest visuals.
8. **Finalize** – when no more tool calls are pending, a Gemini summary node reports on deliverables and clean-up runs.

## State Shape

`VideoGenerationState` keeps the loop minimal and chat-centric:

- `messages`: running LangChain message history (LLM + tool turns).
- `current_iteration`: iteration counter so Gemini stops at configured limits.
- `max_iterations_reached`: flag toggled when the loop hits the user-defined cap.
- `final_video_path`: optional local artifact recorded during ffmpeg steps.
- `user_image_paths` / `user_image_urls`: scratch fields for user-provided reference art.

## Tooling

- **`create_story_board` / `update_story_board`** – lightweight prompt constructors for three-scene arcs.
- **`generate_image`** – wraps `ChatGoogleGenerativeAI` (`gemini-2.5-flash-image-preview`), saves PNGs, and returns base64/mime/path metadata.
- **`veo3_generate_video`** – submits to `fal-ai/veo3/fast`, supporting optional reference imagery via path, hosted URL, or base64 data URI. Outputs logs, request id, and final video URL.
- **`concat_videos`** – downloads remote clips when necessary, normalises codecs with ffmpeg, and stitches scenes into a single mp4 ready for scoring or export.
- **`kling_generate_video_from_image`** – legacy Kling integration kept for experimentation.
- **`run_ffmpeg_binary`** – best-effort ffmpeg execution helper.
- **`score_video`** – Gemini-based rubric scoring of completed renders.

## Setup

### Requirements

- Python 3.10+
- FFmpeg available on PATH
- Google Gemini and fal.ai credentials

### Install dependencies

```bash
uv add "langgraph-cli[inmem]"
uv sync
```

Copy environment template and supply keys:

```bash
cp .env.example .env
```

Minimum variables:

- `GOOGLE_API_KEY` – for Gemini (LLM + image preview).
- `FAL_KEY` (or `FAL_API_KEY`) – for Veo 3 Fast.
- Optional: `LANGSMITH_API_KEY` if you want LangSmith tracing.

### Run locally

1. Start the dev server: `langgraph dev`
2. Open Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
3. Kick off a run from the UI or script (see below).

## Usage Example

```python
from langgraph_sdk import get_sync_client

client = get_sync_client(url="http://127.0.0.1:2024")

prompt = "Create a 3-scene launch teaser for a smart water bottle."

run = client.runs.stream(
    None,
    "agent",
    input={"messages": [{"role": "human", "content": prompt}]},
    stream_mode="messages-tuple",
)

for event in run:
    print(event.event, event.data)
```

To seed the flow with a custom hero image, drop a `HumanMessage` block containing `{"type": "image_url", "image_url": "data:image/png;base64,..."}`. The agent will forward that to Veo 3 automatically when constructing scenes.

## Directory Layout

```
video_generation_agent/
├── src/agent/
│   ├── graph.py          # LangGraph state machine and nodes
│   ├── models.py         # Pydantic schemas for requests/results
│   └── tools.py          # Tool implementations (Gemini, fal.ai, ffmpeg)
├── example_video.mp4     # Reference artifact (optional)
├── langgraph.json        # Config for CLI
├── main.py               # Convenience runner
├── pyproject.toml        # Poetry/uv project definition
└── README.md             # You are here
```

Temporary outputs live under `/tmp/images` for stills and whatever directory a video tool reports. The attachment node promotes any saved file (or returned URL) into the chat history so downstream steps—including Veo 3 or ffmpeg refinements—can reuse them without extra plumbing.

## Quality Assessment

The agent performs iterative quality improvement:

1. **Initial Creation**: Creates video with basic settings
2. **Quality Assessment**: Evaluates video quality (0-10 scale)
3. **Feedback Generation**: Provides specific improvement suggestions
4. **Iteration Decision**: Continues if quality target not met and iterations remain
5. **Improvement**: Addresses specific quality issues
6. **Final Assessment**: Provides final quality score and summary

## Error Handling

The agent includes comprehensive error handling:

- API key validation
- File system operations
- Network requests
- FFmpeg operations
- Tool execution

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:

1. Check the LangGraph documentation
2. Review the tool implementations
3. Test individual components
4. Check API key configuration
5. Verify FFmpeg installation

## Roadmap

- [ ] Enhanced video quality assessment
- [ ] More sophisticated asset management
- [ ] Advanced subtitle styling options
- [ ] Batch video processing
- [ ] Custom voice training integration
- [ ] Real-time video preview
- [ ] Advanced FFmpeg filters
- [ ] Multi-language support
