# Video Generation Agent

A sophisticated LangGraph agent for creating high-quality videos with iterative quality improvement using Google Gemini, ElevenLabs, Unsplash, and FFmpeg.

## Features

- **Multi-modal AI**: Uses Google Gemini 1.5 Pro for video planning and quality assessment
- **Text-to-Speech**: ElevenLabs integration for professional voiceovers
- **Image Assets**: Unsplash API integration for high-quality images
- **Subtitle Generation**: ASS file creation for video subtitles
- **Media Library**: Hardcoded library of 10 curated assets
- **Video Creation**: FFmpeg wrapper for professional video production
- **Iterative Improvement**: Quality-based iteration until target quality is met
- **State Management**: Comprehensive state tracking for video generation process

## Architecture

The agent follows a sophisticated workflow:

1. **Initialize**: Set up directories and LLM
2. **Plan**: Analyze user request and create video production plan
3. **Gather Assets**: Search media library and Unsplash for required assets
4. **Create Content**: Generate audio, subtitles, and combine assets
5. **Assess Quality**: Evaluate video quality and provide feedback
6. **Iterate**: Improve video based on feedback (if needed)
7. **Finalize**: Provide final summary and recommendations

## State Management

The agent tracks comprehensive state including:

- `video_quality`: Current quality score (0-10)
- `ffmpeg_command`: FFmpeg command used for video creation
- `assets_used`: List of assets utilized in the video
- `number_of_renders`: Number of video renders completed
- `current_iteration`: Current iteration count
- `quality_satisfied`: Whether target quality is achieved
- `max_iterations_reached`: Whether max iterations limit reached

## Tools

### 1. ElevenLabs Text-to-Speech
- Generates high-quality speech from text
- Supports multiple voices and models
- Saves audio to `/tmp/audio/` directory

### 2. Unsplash Search & Download
- Searches Unsplash for images based on query
- Downloads images to `/tmp/unsplash/` directory
- Supports quality selection and count limits

### 3. ASS Subtitle Generation
- Creates Advanced SubStation Alpha subtitle files
- Customizable font size, color, timing
- Saves to `/tmp/subtitles/` directory

### 4. Media Library Search
- Hardcoded library of 10 curated assets
- Includes images, audio, and video clips
- Tag-based search functionality

### 5. Video Creation (FFmpeg)
- Professional video creation using FFmpeg
- Supports multiple input formats
- Configurable resolution, duration, and quality

## Setup

### Prerequisites

- Python 3.9+
- FFmpeg installed on system
- API keys for required services

### Installation

1. **Install LangGraph CLI**:
   ```bash
   uv add "langgraph-cli[inmem]"
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set up environment variables**:
   Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

   Required API keys:
   - `LANGSMITH_API_KEY`: LangSmith API key (free)
   - `GOOGLE_API_KEY`: Google Gemini API key
   - `ELEVENLABS_API_KEY`: ElevenLabs API key
   - `UNSPLASH_ACCESS_KEY`: Unsplash API key

### Running the Agent

1. **Start LangGraph Server**:
   ```bash
   langgraph dev
   ```

2. **Test individual tools**:
   ```bash
   python main.py
   ```

3. **Access LangGraph Studio**:
   Visit: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

## Usage

### Basic Usage

```python
from langgraph_sdk import get_sync_client

client = get_sync_client(url="http://localhost:2024")

# Create a video request
user_request = """
Create a 30-second promotional video about sustainable energy. 
Include images of solar panels and wind turbines, 
a professional voiceover, and subtitles.
"""

# Run the agent
for chunk in client.runs.stream(
    None,
    "agent",
    input={
        "messages": [{"role": "human", "content": user_request}],
        "user_request": user_request,
    },
    stream_mode="messages-tuple",
):
    print(f"Event: {chunk.event}")
    print(f"Data: {chunk.data}")
```

### Configuration

The agent accepts context parameters:

- `max_iterations`: Maximum number of quality improvement iterations (default: 3)
- `target_quality_score`: Target quality score to achieve (default: 8.0)
- `google_api_key`: Google Gemini API key
- `elevenlabs_api_key`: ElevenLabs API key
- `unsplash_access_key`: Unsplash API key

## Directory Structure

```
video_generation_agent/
├── src/agent/
│   ├── __init__.py
│   ├── graph.py          # Main LangGraph agent
│   └── tools.py          # Custom tools
├── tests/
├── .env                  # Environment variables
├── .env.example          # Environment template
├── langgraph.json        # LangGraph configuration
├── main.py              # Entry point and testing
├── pyproject.toml       # Project dependencies
└── README.md            # This file
```

## Temporary Directories

The agent creates and uses several temporary directories:

- `/tmp/assets/`: Hardcoded media library assets
- `/tmp/audio/`: Generated speech files
- `/tmp/unsplash/`: Downloaded Unsplash images
- `/tmp/subtitles/`: Generated subtitle files
- `/tmp/videos/`: Final video outputs

## Media Library Assets

The agent includes 10 hardcoded assets:

1. **Images**: Sunset beach, mountain landscape, city skyline, forest path, desert dunes, abstract pattern
2. **Audio**: Ambient nature sounds, upbeat music
3. **Videos**: Waterfall clip, clouds timelapse

Each asset includes metadata: ID, type, filename, description, tags, and path.

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