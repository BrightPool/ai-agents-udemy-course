import asyncio
import os

from langgraph_sdk import get_client


async def main() -> None:
    base_url = os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:2024")
    client = get_client(url=base_url)

    # Inputs expected by the Mem0 Coaching Agent
    input_payload = {
        "messages": [
            {
                "role": "human",
                "content": "I'm training for a marathon and struggling to stay motivated. Any quick advice?",
            }
        ],
        "user_id": "demo-user-123",
        "k": 3,
        "enable_graph": True,
    }

    print(f"Streaming run to {base_url}...\n")
    async for chunk in client.runs.stream(
        input_payload["user_id"],
        "agent",
        input=input_payload,
        stream_mode="messages-tuple",
    ):
        print(f"Event: {chunk.event}")
        print(f"Data: {chunk.data}\n")


if __name__ == "__main__":
    asyncio.run(main())
