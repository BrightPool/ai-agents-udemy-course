# Competitor Monitoring Agent with LangGraph

This project recreates the "competitor monitoring" n8n workflow in LangGraph. It polls a shared Google Sheet for competitor sources, collects fresh article URLs, stores them in a local database, generates Sparse Priming Representations (SPRs) and human-ready summaries with OpenAI models, and prepares an automated competitor digest email.

## Workflow Overview

1. **Fetch Competitor List** – Read the shared Google Sheet with `pygsheets` and parse competitor metadata.
2. **Discover URLs** – Crawl each competitor blog root, extract anchor tags, normalize absolute URLs, and filter to the competitor domain.
3. **Diff Against History** – Persist normalized URLs to Postgres (`competitor_urls` table), skipping duplicates to find only newly discovered links.
4. **Generate Insights** – For each new article, fetch the page content, produce an SPR JSON payload, and craft a reader-friendly competitor summary.
5. **Executive Briefing** – Aggregate the summaries and ask the model for a 2–3 sentence executive overview highlighting major trends.
6. **Email Assembly** – Render a digest email body (HTML) that mirrors the n8n Gmail node. The email is written to `outbox/latest_competitor_digest.html` for manual review or downstream delivery.

The LangGraph pipeline can run on demand (via Studio or the SDK) and is easy to wire into an external scheduler if you want to mirror the cron trigger from n8n.

## Project Structure

```
competitor_monitoring_agent/
├── docker-compose.yml        # Local Postgres service for diffing URLs
├── langgraph.json            # LangGraph configuration for the competitor graph
├── pyproject.toml            # Project metadata and dependencies
├── README.md                 # This document
├── run_graph.py              # Convenience script for running the graph via SDK
├── outbox/
│   └── latest_competitor_digest.html  # Last rendered email payload
└── src/
    └── agent/
        ├── __init__.py
        ├── graph.py          # State graph orchestrating the full workflow
        ├── models.py         # Typed payloads for URLs, SPRs, summaries, email
        └── tools.py          # Reusable helpers (HTTP fetch, HTML parsing, persistence)
```

## Environment Variables

Create a `.env` file (copy `.env.example`) and set the following values:

```
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4.1-mini

# Data sources
COMPETITOR_SHEET_URL=https://docs.google.com/.../edit#gid=0
COMPETITOR_SHEET_WORKSHEET="Sheet1"
USER_AGENT="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Google Sheets authentication (choose one)
# GOOGLE_SERVICE_ACCOUNT_FILE=/absolute/path/to/service-account.json
# GOOGLE_SERVICE_ACCOUNT_JSON='{"type": "service_account", ... }'
# GOOGLE_SERVICE_ACCOUNT_ENV_VAR=PYGSHEETS_SERVICE_ACCOUNT_JSON

# Email routing
DIGEST_RECIPIENT=rhys@unvanity.com
DIGEST_FROM=monitor@yourcompany.com

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/competitors
```

`OPENAI_MODEL_NAME` defaults to `gpt-4.1-mini` if omitted. `USER_AGENT` mirrors the headers configured in the n8n HTTP nodes. `pygsheets` still requires Google credentials; supply a service-account JSON file (or paste its contents into `GOOGLE_SERVICE_ACCOUNT_JSON`) even if the sheet is shared read-only.

Once credentials are in place, share the Google Sheet with the service account email address (found in the JSON file under `client_email`) so it can read the rows.

## Setup

1. **Install uv**

   ```bash
   brew install uv
   # or
   pipx install uv
   ```

2. **Create virtual environment**

   ```bash
   cd competitor_monitoring_agent
   uv venv
   source .venv/bin/activate
   # You can also prefix commands with `uv run` instead of activating.
   ```

3. **Start Postgres**

   ```bash
   docker compose up -d postgres
   ```

4. **Share the Google Sheet (one-time)**

   Grant read access to the sheet for your service account email (or whichever Google identity you authorised pygsheets with). Without this the `load_competitors` node will warn that it cannot open the spreadsheet.

5. **Install dependencies**

   ```bash
   uv sync
   # Include dev extras (ruff, pytest, mypy) if desired:
   # uv sync --group dev
   ```

6. **Launch LangGraph server**

   ```bash
   uv run langgraph dev
   ```

   Sample output:

   ```
   >    Ready!
   >
   >    - API: http://localhost:2024
   >
   >    - Docs: http://localhost:2024/docs
   >
   >    - LangGraph Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
   ```

7. **Execute the agent**

   - **Via LangGraph Studio**: Use the “competitor_monitoring” graph, provide optional overrides (e.g., alternative CSV URL), and click *Execute*.
   - **Via script**: `uv run python run_graph.py` streams the run and prints state transitions.

## Key Implementation Notes

- **Google Sheets Intake** – `tools.fetch_competitor_records()` uses `pygsheets` so the agent reads the shared spreadsheet directly (service-account credentials recommended).
- **Postgres Persistence** – `tools.ensure_database()` provisions a `competitor_urls` table in Postgres using the same `ON CONFLICT DO NOTHING` behaviour as the original workflow.
- **HTML Parsing** – A lightweight `html.parser`-based extractor replicates the HTML node that collected anchor tags. We normalize links with `urllib.parse.urljoin` and restrict output to the competitor’s root domain and blog domain.
- **SPR Generation** – The `generate_sprs` state uses `ChatOpenAI` with the same system prompt as the n8n LangChain node. Responses are parsed into typed `SPRDocument` models, and parsing errors surface in the final report.
- **Summaries & Aggregation** – Two additional LLM nodes reproduce the “Summary” and “Executive summary” n8n nodes, ensuring the same JSON-only outputs.
- **Email Rendering** – Instead of calling Gmail directly, the graph writes the HTML body to disk and returns it in the final state. Hook this into your email service or KEEP as manual copy/paste.
- **Scheduling** – Hook this compiled graph into your preferred scheduler (cron, Airflow, LangGraph Platform, etc.) to emulate the n8n schedule trigger; manual executions remain available through Studio or `run_graph.py`.

## Extending the Agent

- Swap `COMPETITOR_SHEET_URL` for another sheet or backend (e.g., Airtable) without modifying code.
- Point `DATABASE_URL` at your production database (or adjust `tools.ensure_database()` for alternative storage).
- Integrate your preferred email API by replacing `write_email_to_outbox()` with an SMTP/send service call.
- Adjust prompts or models by updating `models.py` and the constants at the top of `graph.py`.

This LangGraph port stays faithful to the original n8n graph while taking advantage of Python tooling, typed models, and straightforward deployment with LangGraph Studio.
