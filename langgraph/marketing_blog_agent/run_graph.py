import asyncio
import os

from langgraph_sdk import get_client


async def main() -> None:
    base_url = os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:2024")
    client = get_client(url=base_url)

    # Minimal required inputs for marketing blog agent
    input_payload = {
        "messages": [
            {
                "role": "human",
                "content": (
                    "Please draft a launch blog for Nimbus Scale EU customers covering "
                    "SOC 2, ISO 27001, Frankfurt data residency, private Slack support hours, "
                    "Salesforce and Snowflake integrations, Concierge Migration, Growth Summit SF details, "
                    "Acme Logistics case study results, and promo code BUILD25, in a practical, no-hype tone."
                ),
            }
        ],
        "topic": "Launch blog for Nimbus Scale EU customers",
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
