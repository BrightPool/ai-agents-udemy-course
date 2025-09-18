import asyncio
import os
from typing import Optional

import pandas as pd
from langgraph_sdk import get_client


async def process_csv_data(csv_path: str, max_rows: Optional[int] = None) -> None:
    """Process CSV data through the LangGraph invoice extraction agent."""
    base_url = os.getenv("LANGGRAPH_BASE_URL", "http://127.0.0.1:2024")
    client = get_client(url=base_url)

    print(f"Loading CSV data from: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")

        # Check if 'Input' column exists
        if "Input" not in df.columns:
            print("Error: 'Input' column not found in CSV file")
            return

        # Limit rows if specified
        if max_rows:
            df = df.head(max_rows)
            print(f"Processing first {max_rows} rows")

        results = []

        for row_idx in range(len(df)):
            row = df.iloc[row_idx]
            invoice_text = str(row["Input"]).strip()

            if not invoice_text:
                print(f"Row {row_idx}: Skipping empty input")
                continue

            print(f"\nProcessing row {row_idx + 1}/{len(df)}...")
            print(f"Invoice text preview: {invoice_text[:100]}...")

            input_payload = {
                "messages": [
                    {
                        "role": "human",
                        "content": invoice_text,
                    }
                ],
            }

            try:
                print(f"Streaming extraction for row {row_idx + 1}...")
                full_response = ""
                async for chunk in client.runs.stream(
                    None,
                    "agent",
                    input=input_payload,
                    stream_mode="messages-tuple",
                ):
                    if chunk.event == "messages":
                        for message_data in chunk.data:
                            if hasattr(message_data, "content"):
                                full_response += message_data.content

                results.append(
                    {
                        "row_index": row_idx,
                        "input_text": invoice_text,
                        "extraction_result": full_response,
                    }
                )

                print(f"✓ Row {row_idx + 1} processed successfully")

            except Exception as e:
                print(f"✗ Error processing row {row_idx + 1}: {e}")
                results.append(
                    {"row_index": row_idx, "input_text": invoice_text, "error": str(e)}
                )

        # Save results to CSV
        results_df = pd.DataFrame(results)
        output_path = csv_path.replace(".csv", "_extraction_results.csv")
        results_df.to_csv(output_path, index=False)

        print("\nProcessing complete!")
        print(f"Results saved to: {output_path}")
        print(
            f"Successfully processed: {len([r for r in results if 'extraction_result' in r])}/{len(results)} rows"
        )

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
    except Exception as e:
        print(f"Error processing CSV: {e}")


async def main() -> None:
    # Path to the invoice CSV file
    csv_path = "./src/agent/data/invoice_ner_clean.csv"

    # Process first 5 rows for testing (set to None to process all)
    max_rows = 5

    await process_csv_data(csv_path, max_rows)


if __name__ == "__main__":
    asyncio.run(main())
