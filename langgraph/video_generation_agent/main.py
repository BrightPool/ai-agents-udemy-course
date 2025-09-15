"""Main entry point for the Video Generation Agent."""

import asyncio
import os
from dotenv import load_dotenv
from langgraph_sdk import get_sync_client

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
    print("🎬 Starting Video Generation Agent...")
    print(f"📝 Request: {user_request}")
    print("-" * 50)
    
    try:
        for chunk in client.runs.stream(
            None,  # Threadless run
            "agent",  # Name of assistant defined in langgraph.json
            input={
                "messages": [{
                    "role": "human",
                    "content": user_request,
                }],
                "user_request": user_request,
            },
            stream_mode="messages-tuple",
        ):
            print(f"📊 Event: {chunk.event}")
            if hasattr(chunk, 'data') and chunk.data:
                print(f"📄 Data: {chunk.data}")
            print("-" * 30)
            
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        print("Make sure the LangGraph server is running with: langgraph dev")


def test_tools_individually():
    """Test individual tools to ensure they work correctly."""
    
    print("🔧 Testing individual tools...")
    
    # Test media library search
    from src.agent.tools import search_media_library
    results = search_media_library.invoke({"query": "nature"})
    print(f"📚 Media library search results: {len(results)} assets found")
    for result in results[:3]:  # Show first 3 results
        print(f"  - {result['type']}: {result['description']}")
    
    # Test ASS file generation
    from src.agent.tools import generate_ass_file_tool
    subtitle_path = generate_ass_file_tool.invoke({
        "subtitle_text": "Welcome to our sustainable energy video",
        "start_time": 0.0,
        "duration": 5.0
    })
    print(f"📝 Generated subtitle file: {subtitle_path}")
    
    print("✅ Tool tests completed")


if __name__ == "__main__":
    print("🎥 Video Generation Agent")
    print("=" * 50)
    
    # Test tools first
    test_tools_individually()
    print()
    
    # Run the full agent (requires LangGraph server to be running)
    print("🚀 To run the full agent:")
    print("1. Make sure you have set your API keys in .env file")
    print("2. Start the LangGraph server: langgraph dev")
    print("3. Run this script again")
    print()
    
    # Uncomment the line below to run the agent when server is ready
    # asyncio.run(run_video_generation_agent())