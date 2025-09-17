import asyncio
import os

from langgraph_sdk import get_client


async def main() -> None:
    base_url = os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:2024")
    client = get_client(url=base_url)

    # Minimal required inputs
    input_payload = {
        "messages": [
            {
                "role": "human",
                "content": "What's the status of order ORD-001?",
            }
        ],
        "customer_email": "john.doe@email.com",
        "order_id": "ORD-001",
    }

    print(f"Streaming run to {base_url}...\n")
    async for chunk in client.runs.stream(
        None,
        "agent",
        input=input_payload,
        stream_mode="messages-tuple",
    ):
        print(f"Event: {chunk.event}")
        print(f"Data: {chunk.data}\n")


if __name__ == "__main__":
    asyncio.run(main())
