"""Main entry point for the Video Generation Agent."""

import logging
import os

from dotenv import load_dotenv
from langgraph_sdk import get_sync_client

from src.agent.tools import (
    generate_ass_file_tool,
    kling_generate_video,
    search_media_library,
)

# Load environment variables
load_dotenv()


async def run_video_generation_agent():
    """Run the video generation agent with a sample request."""
    # Initialize the client
    client = get_sync_client(url="http://localhost:2024")

    # Sample video generation request
    user_request = """
    Create a 30-second promotional video about sustainable energy.
    The video should include:
    - Images of solar panels and wind turbines
    - A professional voiceover explaining the benefits
    - Subtitles for accessibility
    - High-quality 1080p output
    """

    # Run the agent
    logging.info("🎬 Starting Video Generation Agent...")
    logging.info("📝 Request: %s", user_request)
    logging.info("%s", "-" * 50)

    try:
        for chunk in client.runs.stream(
            None,  # Threadless run
            "agent",  # Name of assistant defined in langgraph.json
            input={
                "messages": [
                    {
                        "role": "human",
                        "content": user_request,
                    }
                ],
                "user_request": user_request,
            },
            stream_mode="messages-tuple",
        ):
            logging.info("📊 Event: %s", getattr(chunk, "event", None))
            if hasattr(chunk, "data") and chunk.data:
                logging.info("📄 Data: %s", chunk.data)
            logging.info("%s", "-" * 30)

    except Exception as e:
        logging.exception("❌ Error running agent: %s", e)
        logging.error("Make sure the LangGraph server is running with: langgraph dev")


def test_tools_individually():
    """Test individual tools to ensure they work correctly."""
    logging.info("🔧 Testing individual tools...")

    # Test media library search
    result = search_media_library.invoke(
        {
            "request": {
                "query": "nature",
                "asset_type": None,
                "tags_any": None,
                "tags_all": None,
                "ids_any": None,
            }
        }
    )
    # result is an AssetSearchResult pydantic model or dict-like
    try:
        count = getattr(result, "total_count", None)
        assets = getattr(result, "assets", None)
    except Exception:
        count = None
        assets = None
    if count is None or assets is None:
        logging.warning(
            "📚 Media library search returned unexpected format: %s", result
        )
    else:
        logging.info("📚 Media library search results: %s assets found", count)
        for asset in list(assets)[:3]:
            logging.info(
                "  - %s: %s",
                getattr(asset, "type", "?"),
                getattr(asset, "description", "?"),
            )

    # Test ASS file generation
    subtitle_path = generate_ass_file_tool.invoke(
        {
            "request": {
                "subtitle_text": "Welcome to our sustainable energy video",
                "start_time": 0.0,
                "duration": 5.0,
            }
        }
    )
    logging.info("📝 Generated subtitle file: %s", subtitle_path)

    logging.info("✅ Tool tests completed")

    # Optional: quick Kling smoke test (skips if no keys)
    if os.getenv("REPLICATE_API_TOKEN") or os.getenv("FAL_KEY"):
        try:
            resp = kling_generate_video.invoke(
                {
                    "request": {
                        "prompt": "a woman walking in a park",
                        "duration": 5,
                        "aspect_ratio": "16:9",
                        "negative_prompt": "",
                    }
                }
            )
            logging.info("🎞️ Kling response: %s", resp)
        except Exception as e:
            logging.warning("Kling test skipped/failed: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("🎥 Video Generation Agent")
    logging.info("%s", "=" * 50)

    # Test tools first
    test_tools_individually()
    logging.info("")

    # Run the full agent (requires LangGraph server to be running)
    logging.info("🚀 To run the full agent:")
    logging.info("1. Make sure you have set your API keys in .env file")
    logging.info("2. Start the LangGraph server: langgraph dev")
    logging.info("3. Run this script again")
    logging.info("")

    # Uncomment the line below to run the agent when server is ready
    # import asyncio; asyncio.run(run_video_generation_agent())
