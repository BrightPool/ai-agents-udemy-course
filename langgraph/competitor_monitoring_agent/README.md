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
├── .env.example              # Template for environment configuration
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
        └── utils.py          # Reusable helpers (HTTP fetch, HTML parsing, persistence)
```

## Environment Variables

Create a `.env` file (copy `.env.example`) and set the following values:

```
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4o-mini

# Data sources
COMPETITOR_SHEET_URL=https://docs.google.com/.../edit#gid=0
COMPETITOR_SHEET_WORKSHEET="Sheet1"
USER_AGENT="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Processing Limits
MAX_LINKS_TO_PROCESS=10

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

### Configuration Options

- **`OPENAI_API_KEY`**: Required. Your OpenAI API key for generating SPRs and summaries
- **`OPENAI_MODEL_NAME`**: Model to use (defaults to `gpt-4o-mini` for cost efficiency)
- **`COMPETITOR_SHEET_URL`**: Google Sheets URL containing competitor information
- **`MAX_LINKS_TO_PROCESS`**: Limits articles processed per run (default: 10) to prevent excessive API costs and processing time
- **`USER_AGENT`**: HTTP user agent string for web scraping
- **Google Sheets Auth**: Choose one authentication method for accessing the competitor sheet
- **Email Settings**: Configure where digest emails are sent
- **`DATABASE_URL`**: PostgreSQL connection string for storing discovered URLs

### Authentication Strategy

The agent prioritizes **public access** for Google Sheets:

1. **First**: Attempts CSV export URL (no authentication required)
2. **Fallback**: Uses pygsheets with service account credentials

This approach allows the agent to work with public sheets without requiring Google Cloud setup, while still supporting private sheets when needed.

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

   - **Via LangGraph Studio**: Use the “competitor*monitoring” graph, provide optional overrides (e.g., alternative CSV URL), and click \_Execute*.
   - **Via script**: `uv run python run_graph.py` streams the run and prints state transitions.

## Database Management

### Reset the Database

To clear all stored competitor URLs (useful for testing or starting fresh):

```bash
# One-line command to truncate the table
PGPASSWORD=postgres psql -h localhost -U postgres -d competitors -c "TRUNCATE TABLE competitor_urls;"
```

### Interactive Database Access

To explore the database interactively:

```bash
# Connect to the database
PGPASSWORD=postgres psql -h localhost -U postgres -d competitors

# Once connected, you can run SQL commands:
# \dt                           -- List all tables
# SELECT * FROM competitor_urls; -- View all stored URLs
# TRUNCATE TABLE competitor_urls; -- Delete all rows
# \q                            -- Exit
```

## Architecture & Key Decisions

### 🏗️ **Core Architecture**

The agent follows a **LangGraph state machine** pattern with 7 sequential nodes:

1. **`load_competitors`** → Parse Google Sheet → CompetitorSource models
2. **`crawl_and_diff`** → Discover URLs → Store in Postgres → DiscoveredLink models
3. **`process_new_links`** → Generate SPRs → SPRDocument models (parallelized)
4. **`summaries`** → Create summaries → ArticleSummary models (parallelized)
5. **`executive_summary`** → Aggregate overview → Executive summary text
6. **`compose_digest`** → Render HTML email → DigestEmail model
7. **`create_report`** → Final structured output → PipelineReport model

### 🔧 **Key Implementation Decisions**

#### **Parallel Processing Strategy**

- **SPR Generation**: Uses `llm.with_structured_output(SPRDocument)` + `.batch()` for parallel processing
- **Summary Generation**: Uses `llm.with_structured_output(SummaryText)` + `.batch()` for parallel processing
- **Benefit**: Dramatically faster execution, especially with multiple articles
- **Configurable Limit**: `MAX_LINKS_TO_PROCESS` prevents excessive API costs

#### **Authentication Hierarchy**

- **Primary**: CSV export URLs (no auth required) for public sheets
- **Fallback**: pygsheets with service account credentials
- **Benefit**: Works with public sheets without Google Cloud setup
- **Flexibility**: Still supports private sheets when needed

#### **Data Persistence Strategy**

- **Postgres**: `competitor_urls` table with `ON CONFLICT DO NOTHING`
- **Deduping**: Only processes newly discovered URLs
- **Benefit**: Efficient diffing prevents reprocessing existing content

#### **Error Handling & Resilience**

- **Structured Outputs**: Pydantic models ensure type safety and validation
- **Graceful Degradation**: Failed individual articles don't stop the pipeline
- **Warnings Collection**: All issues logged in final report for debugging

#### **Configuration Philosophy**

- **Environment-Driven**: All settings via `.env` file
- **Sensible Defaults**: Works out-of-the-box with minimal setup
- **KISS Principle**: Removed complex n8n-specific configurations

### 📊 **Performance Optimizations**

- **Batch Processing**: Parallel LLM calls reduce total execution time
- **Link Limiting**: `MAX_LINKS_TO_PROCESS` prevents runaway processing
- **Early Termination**: Conditional edges skip unnecessary steps when no new links
- **Efficient Storage**: Normalized URLs prevent duplicate processing

### 🔒 **Security & Reliability**

- **No Direct Email**: HTML output written to disk for manual review
- **Structured Data**: Type-safe Pydantic models throughout
- **Error Boundaries**: Individual failures don't crash the entire pipeline
- **Configurable Limits**: Prevent excessive resource usage

### 🚀 **Extensibility**

- **Modular Design**: Each node is independently testable and replaceable
- **Configuration-Driven**: Easy to customize behavior without code changes
- **Type Safety**: Full Pydantic coverage makes refactoring safe
- **Clean Interfaces**: Clear separation between data models and business logic

## Extending the Agent

### 🚀 **Performance Tuning**

- **Scale Processing**: Increase `MAX_LINKS_TO_PROCESS` for larger batches (monitor API costs)
- **Model Selection**: Upgrade to `gpt-4o` for higher quality outputs at increased cost
- **Batch Optimization**: Adjust batch sizes in `utils.batch()` calls for your infrastructure

### 🔧 **Customization Options**

- **Alternative Data Sources**: Replace Google Sheets with Airtable, Notion, or custom APIs
- **Database Flexibility**: Point `DATABASE_URL` at production databases or cloud services
- **Email Integration**: Replace `write_email_to_outbox()` with SMTP, SendGrid, or other services
- **Model Customization**: Modify prompts in `graph.py` or extend Pydantic models in `models.py`

### 📈 **Production Deployment**

- **Monitoring**: Add logging/metrics for production observability
- **Caching**: Implement Redis for faster duplicate detection
- **Rate Limiting**: Add delays between requests to respect site policies
- **Error Recovery**: Enhance retry logic for network failures

### 🎯 **Key Advantages Over n8n**

- **Type Safety**: Full Pydantic validation prevents runtime errors
- **Parallel Processing**: 10x+ faster execution with batched LLM calls
- **Cost Control**: Configurable limits prevent unexpected API bills
- **Maintainability**: Clean separation of concerns and modular design
- **Extensibility**: Easy to add new data sources, models, or processing steps

This LangGraph implementation provides a solid foundation for production competitor intelligence while maintaining the simplicity and reliability of the original n8n workflow.
