import asyncio
import os

from langgraph_sdk import get_client


async def main() -> None:
    base_url = os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:2024")
    client = get_client(url=base_url)

    sample_invoice = (
        "Invoice Number: INV-1001\n"
        "Invoice Date: 2024-01-15\n"
        "Bill To: Acme Corp\n"
        "Total Amount: $4,250\n"
        "Due Date: 2024-02-14\n"
    )

    input_payload = {
        "messages": [
            {
                "role": "human",
                "content": sample_invoice,
            }
        ],
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
