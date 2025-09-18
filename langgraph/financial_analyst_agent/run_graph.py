import asyncio
import os

from langgraph_sdk import get_client


async def main() -> None:
    base_url = os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:2024")
    client = get_client(url=base_url)

    input_payload = {
        "messages": [],
        "user_message": "Compare the 5-day percentage return for AAPL and MSFT and report which outperformed.",
    }

    try:
        max_iters = int(os.getenv("FINANCIAL_AGENT_MAX_ITERS", "5"))
    except ValueError:
        max_iters = 5

    context = {
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "max_iterations": max_iters,
    }

    print(f"Streaming run to {base_url}...\n")
    async for chunk in client.runs.stream(
        None,
        "agent",
        input=input_payload,
        config={"context": context},
        stream_mode="messages-tuple",
    ):
        print(f"Event: {chunk.event}")
        print(f"Data: {chunk.data}\n")


if __name__ == "__main__":
    asyncio.run(main())
