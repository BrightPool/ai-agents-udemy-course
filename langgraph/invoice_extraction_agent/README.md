# LangGraph Invoice Extraction Agent

A LangGraph-driven agent that extracts structured invoice fields from raw text
with OpenAI models. The workflow relies on LangChain's structured output
support and pydantic schemas, making the JSON contract explicit and easy to use
in downstream systems.

## Highlights

- **Structured Output** – The agent calls `ChatOpenAI.with_structured_output`
  with a pydantic schema so every response arrives as validated JSON.
- **Functional Helpers** – Lightweight utility functions expose sync/async
  access to the extractor outside LangGraph while reusing the cached model
  chain.
- **LangGraph Interface** – A minimal graph turns human messages into formatted
  summaries plus the raw JSON payload for tooling or automations.

## Prerequisites

1. **Environment variables** – Provide your OpenAI API key in `.env`:

   ```bash
   cp .env.example .env
   echo "OPENAI_API_KEY=sk-your-key" >> .env
   ```

2. **Python environment** – Python 3.9+ is required. The project uses `uv` for
   dependency management, but you can substitute your preferred tooling.

## Setup

```bash
# Install uv if needed
brew install uv  # or: pipx install uv

cd invoice_extraction_agent
uv venv
source .venv/bin/activate  # optional; you can also prefix commands with `uv run`
uv sync
```

The sync step installs LangGraph, LangChain Core, the OpenAI client bindings,
Pydantic, and other dependencies.

## Running the LangGraph Agent

1. Start the local LangGraph server:

   ```bash
   uv run langgraph dev
   ```

2. Interact via LangGraph Studio or run the bundled script:

   ```bash
   uv run python run_graph.py
   ```

   The script streams a sample invoice through the agent and prints the
   structured output.

### Input Contract

Provide the invoice text as the latest human message. The agent responds with a
human-readable summary plus prettified JSON that mirrors the pydantic
`InvoiceExtractionResult` schema.

## Structured Schema

Key pieces of the output include:

- `rationale` – short explanation of how the model identified the fields.
- `invoice_number`, `invoice_date`, `due_date`, `total_amount`, etc. –
  top-level string fields containing the most common invoice attributes.
- `line_items` – optional list of objects containing `description`, `quantity`,
  `unit_price`, and `total`.
- `notes` and `payment_terms` – additional freeform snippets preserved when
  available.

Any field that cannot be recovered from the source text is returned as `null`.

## Development Tips

- The extractor caches the underlying chat model, so repeated calls are cheap.
- `src/agent/utils.py` exposes small helpers for invoking the extractor from a
  REPL or notebook without using LangGraph.
- If you change dependencies, rerun `uv sync` to refresh the lock file.

Happy hacking!
