"""Run the testimonial video generator DAG agent with LangGraph SDK."""

import asyncio
import os

from langgraph_sdk import get_client


async def main() -> None:
    """Run the video generator agent."""
    base_url = os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:2024")
    client = get_client(url=base_url)

    # Required inputs for this graph
    input_payload = {
        # Choose a persona key defined in graph.py define_personas
        "persona_selection": "Omar US Developer",
        # Optional: path to an uploaded image if needed in future
        # "image_path": "/path/to/image.png",
    }

    # Note: Using print for demo purposes - consider using logging in production
    print(f"Streaming run to {base_url}...\n")  # noqa: T201
    async for chunk in client.runs.stream(
        None,
        "agent",
        input=input_payload,
        stream_mode="messages-tuple",
    ):
        print(f"Event: {chunk.event}")  # noqa: T201
        print(f"Data: {chunk.data}\n")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
