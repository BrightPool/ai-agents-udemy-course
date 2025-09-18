# Competitor Monitoring Agent (DSPy)

## What this agent does
- Discovers new in‑domain blog/article URLs from a published competitors CSV (site + blog root)
- Generates SPR (Sparse Priming Representation) for each new page and writes 2–3 sentence per‑item summaries
- Produces a concise executive summary from all item summaries
- Assembles a ready‑to‑send HTML email digest; persists seen URLs to avoid duplicates

---

## High-level workflow
```
[Start: Provide csv_url, to_email]
      |
      v
[Tool: get_competitors(csv_url)]
      |
      v
[Tool: discover_new_urls()]
      |
      v
[Code: diff + persist -> state/competitor_urls.json]
      |
      v
[Loop over new URLs]
      |
      +--> [Tool: generate_spr(max_pages cap)]
      |          |
      |          v
      |     [LLM: SPR JSON per page]
      |
      +--> [Tool: summarize_items()]
                 |
                 v
            [LLM: 2–3 sentence summary per page]
      |
      v
[Tool: exec_summary()] -> [LLM: concise trends/themes/outliers]
      |
      v
[Code: build_email_html(to_email, subject)]
      |
      v
[Output: email_html (ready to send)]

Optional:
  [Tool: send_email_smtp(to_email, subject)] (stubbed in repo)
```
- Inputs: `csv_url` (string), `to_email` (string)
- Outputs: `email_html` (string), plus counts for new URLs and summaries
- External services: OpenAI LLM (see below)

---

## Components and external services
- LLMs: `openai/gpt-5-mini`
- Env: `OPENAI_API_KEY`
  - Optional tuning: `COMP_MONITOR_LOG_LEVEL`, `COMP_MONITOR_HTTP_TIMEOUT`, `COMP_MONITOR_LM_TIMEOUT`, `COMP_MONITOR_MAX_SPR_PAGES`, `COMP_MONITOR_MAX_LINKS_PER_SITE`, `COMP_MONITOR_TO_EMAIL`
- Dependencies (pip): `dspy`, `python-dotenv`, `requests`, `beautifulsoup4`, `lxml`, `pandas`
- State: persists URL index at `dspy/competitor_monitoring_agent/state/competitor_urls.json`
- Notes: Do not change API keys in `.env`.

---

## Copy‑paste prompts

### 1) ReAct system schema (CompetitorAgentSignature)
```
System/Instructions:
You are a competitor monitoring agent. Given csv_url and to_email, do the following:
1) Fetch competitors CSV.
2) Crawl each blog root, collect in‑domain article‑like links, and diff against a persisted URL store.
3) For newly discovered URLs (capped), generate an SPR for each and write a brief per‑item summary.
4) Aggregate an executive summary and build a ready‑to‑send HTML email digest.
Return the final HTML in email_html_out.

Inputs:
- csv_url (string)
- to_email (string)

Available tools:
- get_competitors(csv_url: string)
- discover_new_urls()
- generate_spr(max_pages: integer = env COMP_MONITOR_MAX_SPR_PAGES)
- summarize_items()
- exec_summary()
- build_email_html(to_email: string, subject: string = "daily competitor digest")
- send_email_smtp(to_email: string, subject: string)  # stub (no actual send)

Outputs:
- reasoning (string)
- email_html_out (string)
```

### 2) SPR generation prompt (SPRSignature)
```
System/Instructions:
Render page content as a Sparse Priming Representation (SPR) for downstream LLM use.
Output a JSON array string with objects that capture: source_url, title, date (if any), and spr (list of short statements).

Inputs:
- source_url (string)
- page_text (string)
- title_hint (string)

Output:
- spr_json (string; JSON array as specified)
```

### 3) Item summary prompt (ItemSummarySignature)
```
System/Instructions:
Summarize a single competitor page based on its SPR into 2–3 sentences for a business owner. Reference the source and label the competitor.

Inputs:
- competitor_name (string)
- source_url (string)
- spr_json (string)

Output:
- summary (string)
```

### 4) Executive summary prompt (ExecSummarySignature)
```
System/Instructions:
Aggregate multiple item summaries and write a concise 2–3 sentence executive summary highlighting trends, themes, and outliers. Output JSON only.

Inputs:
- summaries_json (string; JSON array of item summaries)

Output:
- executive_summary_json (string)
```

---

## Scoring/aggregation (code)
JS:
```javascript
// Summarize counts from an agent run
function competitorDigestSummary(state) {
  const newCount = Array.isArray(state?.new_urls) ? state.new_urls.length : 0;
  const items = Array.isArray(state?.item_summaries) ? state.item_summaries.length : 0;
  const htmlLength = typeof state?.email_html === "string" ? state.email_html.length : 0;
  return { newCount, items, htmlLength };
}
```

Python:
```python
# Summarize counts from an agent run
def competitor_digest_summary(state: dict) -> dict:
    new_count = len(state.get("new_urls", []))
    items = len(state.get("item_summaries", []))
    html_length = len(state.get("email_html", ""))
    return {"newCount": new_count, "items": items, "htmlLength": html_length}
```

---

## Implementation steps
1) Install: `pip install dspy python-dotenv requests beautifulsoup4 lxml pandas`
2) Configure LM:
```
import os, dspy
lm = dspy.LM("openai/gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=1, max_tokens=16000)
dspy.configure(lm=lm)
```
3) Provide input params: `csv_url` (published CSV with columns `Url`, `blog urls`) and `to_email`
4) Orchestrate tools in order:
   - `get_competitors(csv_url)` → `discover_new_urls()` → (store diff under `state/competitor_urls.json`)
   - `generate_spr(max_pages)` → `summarize_items()` → `exec_summary()`
   - `build_email_html(to_email, subject)`
5) Output: return/save `email_html`. Optional: call `send_email_smtp(to_email, subject)` (stub)

---

## Example I/O
- Input:
  - `csv_url`: published Google Sheets CSV of competitors (site + blog root)
  - `to_email`: "youremailhere"
- Output (abridged):
  - newCount: `10`
  - items: `10`
  - htmlLength: `6990`
  - executive_summary_json: `{"executive_summary_json": "<concise trends/themes/outliers>"}`

---

## Files
- `dspy/competitor_monitoring_agent/competitor-monitoring-agent.ipynb`
- `dspy/competitor_monitoring_agent/state/competitor_urls.json`


