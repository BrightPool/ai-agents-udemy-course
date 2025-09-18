"""Competitor monitoring agent implemented with LangGraph."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, TypedDict, cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.utils import convert_to_secret_str
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent.models import (
    ArticleSummary,
    CompetitorSource,
    DigestEmail,
    DiscoveredLink,
    PipelineReport,
    SPRDocument,
    SummaryText,
)
from agent.utils import (
    DEFAULT_DB_URL,
    DEFAULT_OUTBOX_DIR,
    DEFAULT_SUBJECT,
    ensure_database,
    extract_links,
    fetch_competitor_records,
    fetch_text,
    format_digest_html,
    html_to_text,
    normalize_links,
    parse_competitor_records,
    persist_links,
    write_email_to_outbox,
)

# Load environment variables from .env when available for local development.
_project_root = Path(__file__).parent.parent.parent
_env_file = _project_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-mini")
DEFAULT_SHEET_URL = os.getenv(
    "COMPETITOR_SHEET_URL",
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vTvE_qRDWRu5hjZ45yY6juAc4i7iT3DIrJT9q3cz29uIpGpz0IRRzHPdqcKge8obrTjwNS7qC3TGGg-/pub?gid=0&single=true&output=csv",
)
DEFAULT_USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
)
DEFAULT_RECIPIENT = os.getenv("DIGEST_RECIPIENT", "rhys@unvanity.com")
DEFAULT_SENDER = os.getenv("DIGEST_FROM", "monitor@yourcompany.com")
DEFAULT_MAX_LINKS = int(os.getenv("MAX_LINKS_TO_PROCESS", "10"))

SPR_SYSTEM_PROMPT = """# MISSION
You are a Sparse Priming Representation (SPR) writer. An SPR is a particular kind of use of language for advanced NLP, NLU, and NLG tasks, particularly useful for the latest generation of Large Language Models (LLMs). You will be given information by the USER which you are to render as an SPR.

# THEORY
LLMs are a kind of deep neural network. They have been demonstrated to embed knowledge, abilities, and concepts, ranging from reasoning to planning, and even to theory of mind. These are called latent abilities and latent content, collectively referred to as latent space. The latent space of an LLM can be activated with the correct series of words as inputs, which will create a useful internal state of the neural network. This is not unlike how the right shorthand cues can prime a human mind to think in a certain way. Like human minds, LLMs are associative, meaning you only need to use the correct associations to "prime" another model to think in the same way.

# METHODOLOGY
Render the input as a distilled list of succinct statements, assertions, associations, concepts, analogies, and metaphors. The idea is to capture as much, conceptually, as possible but with as few words as possible. Write it in a way that makes sense to you, as the future audience will be another language model, not a human. Use complete sentences.
"""

SUMMARY_SYSTEM_PROMPT = """You are an analyst that summarises competitor activity.
The input you will receive is some information on recent key themes, assertions, and associations found in competitor blog posts.

Your task:
- Read the content carefully.
- Write a short, clear, and helpful summary of what is being talked about.
- Focus on the most important trends, announcements, or themes.
- Keep the summary concise (2–3 sentences).
- Write for a human business owner audience (note any signals of product, strategy, or marketing teams).
- Avoid jargon and technical details unless essential for clarity.
– Always make sure to reference the source and label the specific competitor.
"""

EXEC_SUMMARY_SYSTEM_PROMPT = """You are an analyst writing a daily executive briefing on competitor news.
Input: a JSON array of competitor summaries from multiple sources. Each summary captures key product updates, feature launches, and strategic themes.

Task:
- Read across all summaries.
- Identify common trends, repeated themes, or emerging patterns.
- Highlight any notable differences or outliers.
- Write a concise 2–3 sentence executive summary that captures the "big picture" of what competitors are focusing on.
- Audience: senior executives. Use plain business language (strategy, positioning, direction).
- Output only the executive summary as JSON.
"""


class CompetitorMonitoringState(TypedDict, total=False):
    """State tracked across the LangGraph workflow."""

    competitors: List[CompetitorSource]
    newly_discovered: List[DiscoveredLink]
    spr_documents: List[SPRDocument]
    summaries: List[ArticleSummary]
    executive_summary: str
    email: DigestEmail
    warnings: List[str]
    report: PipelineReport


def _extract_configurable(config: RunnableConfig | None) -> Mapping[str, Any]:
    if config and isinstance(config, Mapping):
        maybe = config.get("configurable")  # type: ignore[assignment]
        if isinstance(maybe, Mapping):
            return maybe
    if config and hasattr(config, "context"):
        ctx = getattr(config, "context")
        if isinstance(ctx, Mapping):
            return ctx
    return {}


def _get_from_config(
    config: RunnableConfig | None,
    key: str,
    env_key: str,
    default: str | None = None,
) -> str | None:
    configurable = _extract_configurable(config)
    value = configurable.get(key)
    if isinstance(value, str) and value:
        return value
    env_value = os.getenv(env_key)
    if env_value:
        return env_value
    return default


def _build_openai_client(
    config: RunnableConfig | None, temperature: float = 0.2
) -> ChatOpenAI:
    api_key = _get_from_config(config, "openai_api_key", "OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required to run the competitor monitoring agent."
        )
    model_name = _get_from_config(
        config, "openai_model_name", "OPENAI_MODEL_NAME", DEFAULT_OPENAI_MODEL
    )
    return ChatOpenAI(
        model=model_name or DEFAULT_OPENAI_MODEL,
        api_key=convert_to_secret_str(api_key),
        temperature=temperature,
    )


def load_competitors(
    state: CompetitorMonitoringState, config: RunnableConfig
) -> Dict[str, object]:
    """Fetch competitor rows from Google Sheets and parse into structured records."""
    _ = state  # Explicitly acknowledge unused state in this node.
    sheet_url = _get_from_config(
        config, "competitor_sheet_url", "COMPETITOR_SHEET_URL", DEFAULT_SHEET_URL
    )
    worksheet_name = _get_from_config(
        config, "competitor_sheet_worksheet", "COMPETITOR_SHEET_WORKSHEET"
    )
    service_account_file = _get_from_config(
        config, "google_service_account_file", "GOOGLE_SERVICE_ACCOUNT_FILE"
    )
    service_account_env_var = _get_from_config(
        config, "google_service_account_env_var", "GOOGLE_SERVICE_ACCOUNT_ENV_VAR"
    )
    service_account_json = _get_from_config(
        config, "google_service_account_json", "GOOGLE_SERVICE_ACCOUNT_JSON"
    )

    warnings: List[str] = []
    competitors: List[CompetitorSource] = []
    try:
        records = fetch_competitor_records(
            sheet_url or DEFAULT_SHEET_URL,
            worksheet=worksheet_name,
            service_account_file=service_account_file,
            service_account_env_var=service_account_env_var,
            service_account_json=service_account_json,
        )
        competitors = parse_competitor_records(records)
        if not competitors:
            warnings.append(
                "Competitor sheet fetched successfully but no rows were parsed."
            )
    except Exception as exc:  # noqa: BLE001
        hint = ""
        if sheet_url and "output=csv" in sheet_url:
            hint = " (pygsheets requires the standard /spreadsheets/d/<ID>/ URL rather than the published CSV export.)"
        warnings.append(f"Failed to load competitor sheet via pygsheets: {exc!s}{hint}")

    return {"competitors": competitors, "warnings": warnings}


def crawl_and_diff(
    state: CompetitorMonitoringState, config: RunnableConfig
) -> Dict[str, object]:
    """Crawl competitor blogs, persist URLs, and return newly discovered links."""
    competitors = state.get("competitors", []) or []
    user_agent = _get_from_config(
        config, "user_agent", "USER_AGENT", DEFAULT_USER_AGENT
    )
    database_url = _get_from_config(
        config, "database_url", "DATABASE_URL", DEFAULT_DB_URL
    )
    connection = ensure_database(database_url or DEFAULT_DB_URL)

    new_links: List[DiscoveredLink] = []
    warnings = state.get("warnings", []).copy() if state.get("warnings") else []

    for competitor in competitors:
        try:
            html = fetch_text(
                competitor.blog_url,
                headers={"User-Agent": user_agent or DEFAULT_USER_AGENT},
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Failed to fetch blog for {competitor.company}: {exc!s}")
            continue

        raw_links = extract_links(html)
        crawled = normalize_links(competitor, raw_links)
        discovered = persist_links(connection, crawled)
        new_links.extend(discovered)

    connection.close()

    return {"newly_discovered": new_links, "warnings": warnings}


def has_new_links(
    state: CompetitorMonitoringState,
) -> Literal["process_new_links", "send_no_updates"]:
    """Route depending on whether we have new articles to process."""
    if state.get("newly_discovered"):
        return "process_new_links"
    return "send_no_updates"


def generate_sprs(
    state: CompetitorMonitoringState, config: RunnableConfig
) -> Dict[str, object]:
    """Generate SPR payloads for each new article using parallel batch processing."""
    llm = _build_openai_client(config, temperature=0.1)
    user_agent = _get_from_config(
        config, "user_agent", "USER_AGENT", DEFAULT_USER_AGENT
    )
    max_links = int(
        _get_from_config(
            config,
            "max_links_to_process",
            "MAX_LINKS_TO_PROCESS",
            str(DEFAULT_MAX_LINKS),
        )
        or DEFAULT_MAX_LINKS
    )

    spr_documents: List[SPRDocument] = []
    warnings = state.get("warnings", []).copy() if state.get("warnings") else []

    # Limit the number of links to process to avoid massive crawls
    all_links = state.get("newly_discovered", []) or []
    links_to_process = all_links[:max_links]

    if len(all_links) > max_links:
        warnings.append(
            f"Limited processing to {max_links} links out of {len(all_links)} discovered. "
            "Increase MAX_LINKS_TO_PROCESS to process more."
        )

    # Prepare all the content in advance
    processed_links = []
    for link in links_to_process:
        try:
            html = fetch_text(
                link.normalized_url,
                headers={"User-Agent": user_agent or DEFAULT_USER_AGENT},
            )
            article_text = html_to_text(html)
            processed_links.append((link, article_text))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Failed to fetch article {link.normalized_url}: {exc!s}")

    if not processed_links:
        return {"spr_documents": spr_documents, "warnings": warnings}

    # Batch process all SPR generations
    structured_llm = llm.with_structured_output(SPRDocument)

    prompt_texts: List[str] = []
    for link, article_text in processed_links:
        prompt = (
            "Open the link, get the content and generate an SPR from it. Your output must be valid JSON.\n\n"
            "Always include metadata about the source:\n"
            f"- `source_url` → the page where the content came from {link.normalized_url}\n"
            '- `title` → the article/blog title if available (or "unknown" if not)\n'
            "- `spr` → the actual distilled SPR list\n\n"
            "# OUTPUT FORMAT\n"
            '[\n  {\n    "source_url": "<SOURCE_URL>",\n    "title": "<TITLE_OR_UNKNOWN>",\n    "date": "<DATE_OR_UNKNOWN>",\n     "spr": [\n      "First distilled SPR statement.",\n      "Second SPR statement.",\n      "Third SPR statement."\n    ]\n  }\n]\n\n'
            "# CONTENT\n"
            f"Source URL: {link.normalized_url}\n"
            f"Competitor: {link.competitor}\n"
            f"Content: {article_text}"
        )
        prompt_texts.append(prompt)

    try:
        raw_results = structured_llm.batch(prompt_texts) if prompt_texts else []  # type: ignore[arg-type]
        results: List[SPRDocument] = cast(List[SPRDocument], raw_results)

        # Process results and ensure source_url is set
        for (link, _), result in zip(processed_links, results):
            if result and not result.source_url:
                result = SPRDocument(
                    source_url=link.normalized_url,
                    title=result.title,
                    date=result.date,
                    spr=list(result.spr or []),
                )
            if result:
                spr_documents.append(result)

    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Failed to batch process SPR generations: {exc!s}")

    return {"spr_documents": spr_documents, "warnings": warnings}


def summarise_articles(
    state: CompetitorMonitoringState, config: RunnableConfig
) -> Dict[str, object]:
    """Produce analyst summaries for each SPR document."""
    llm = _build_openai_client(config, temperature=0.2)
    summaries: List[ArticleSummary] = []
    warnings = state.get("warnings", []).copy() if state.get("warnings") else []

    documents = state.get("spr_documents", []) or []
    discovered_links = {
        link.normalized_url: link for link in state.get("newly_discovered", []) or []
    }

    structured_summary_llm = llm.with_structured_output(SummaryText)
    prompt_texts: List[Any] = [
        f"{SUMMARY_SYSTEM_PROMPT}\n\nInput: {json.dumps(doc.model_dump(), ensure_ascii=False)}"
        for doc in documents
    ]

    raw_results = (
        structured_summary_llm.batch(cast("List[Any]", prompt_texts))
        if prompt_texts
        else []
    )
    results: List[SummaryText] = cast(List[SummaryText], raw_results)

    for spr, result in zip(documents, results):
        link_meta = discovered_links.get(spr.source_url)
        competitor_name = link_meta.competitor if link_meta else "Competitor"
        content_text = getattr(result, "content", "") or ""
        summaries.append(
            ArticleSummary(
                competitor=competitor_name,
                source_url=spr.source_url,
                content=content_text.strip(),
            )
        )

    return {"summaries": summaries, "warnings": warnings}


def generate_executive_summary(
    state: CompetitorMonitoringState, config: RunnableConfig
) -> Dict[str, object]:
    """Aggregate summaries into an executive overview."""
    summaries = state.get("summaries", []) or []
    if not summaries:
        return {"executive_summary": "No competitor activity detected today."}

    llm = _build_openai_client(config, temperature=0.2)

    payload = json.dumps(
        [summary.model_dump() for summary in summaries], ensure_ascii=False
    )
    response = llm.invoke(
        [
            SystemMessage(content=EXEC_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=payload),
        ]
    )
    content = (
        response.content
        if isinstance(response.content, str)
        else "".join(  # type: ignore[arg-type]
            part["text"]
            for part in response.content
            if isinstance(part, dict) and part.get("type") == "text"  # type: ignore[index]
        )
    )

    executive_summary = content.strip() or "No competitor activity detected today."
    return {"executive_summary": executive_summary}


def compose_digest(
    state: CompetitorMonitoringState, config: RunnableConfig
) -> Dict[str, object]:
    """Create the email payload and persist it to disk."""
    executive_summary = state.get(
        "executive_summary", "No competitor activity detected today."
    )
    summaries = state.get("summaries", []) or []

    body_html = format_digest_html(executive_summary, summaries)

    recipient = _get_from_config(
        config, "digest_recipient", "DIGEST_RECIPIENT", DEFAULT_RECIPIENT
    )
    sender = _get_from_config(config, "digest_from", "DIGEST_FROM", DEFAULT_SENDER)

    email = DigestEmail(
        subject=DEFAULT_SUBJECT,
        recipient=recipient or DEFAULT_RECIPIENT,
        body_html=body_html,
        metadata={
            "from": sender or DEFAULT_SENDER,
            "generated_at": datetime.utcnow().isoformat(),
        },
    )
    outbox_path = write_email_to_outbox(email, DEFAULT_OUTBOX_DIR)
    email.metadata["outbox_path"] = str(outbox_path)

    return {"email": email}


def compose_no_updates_email(
    state: CompetitorMonitoringState, config: RunnableConfig
) -> Dict[str, object]:
    """Send the fallback email when no new links were discovered."""
    recipient = _get_from_config(
        config, "digest_recipient", "DIGEST_RECIPIENT", DEFAULT_RECIPIENT
    )
    sender = _get_from_config(config, "digest_from", "DIGEST_FROM", DEFAULT_SENDER)
    body_html = """
<p>Hi Rhys,</p>

<p>Nothing to report today.</p>

<p>Best,</p>
    """
    email = DigestEmail(
        subject=f"{DEFAULT_SUBJECT} ",
        recipient=recipient or DEFAULT_RECIPIENT,
        body_html=body_html,
        metadata={
            "from": sender or DEFAULT_SENDER,
            "generated_at": datetime.utcnow().isoformat(),
        },
    )
    outbox_path = write_email_to_outbox(email, DEFAULT_OUTBOX_DIR)
    email.metadata["outbox_path"] = str(outbox_path)
    return {"email": email}


def create_report(state: CompetitorMonitoringState) -> Dict[str, object]:
    """Package the final report for downstream consumers."""
    report = PipelineReport(
        newly_discovered=state.get("newly_discovered", []) or [],
        spr_documents=state.get("spr_documents", []) or [],
        summaries=state.get("summaries", []) or [],
        executive_summary=state.get("executive_summary"),
        email=state.get("email"),
        warnings=state.get("warnings", []) or [],
    )
    return {"report": report}


workflow = (
    StateGraph(CompetitorMonitoringState)
    .add_node("load_competitors", load_competitors)
    .add_node("crawl_and_diff", crawl_and_diff)
    .add_node("process_new_links", generate_sprs)
    .add_node("summaries", summarise_articles)
    .add_node("executive_summary", generate_executive_summary)
    .add_node("compose_digest", compose_digest)
    .add_node("send_no_updates", compose_no_updates_email)
    .add_node("create_report", create_report)
    .add_edge(START, "load_competitors")
    .add_edge("load_competitors", "crawl_and_diff")
    .add_conditional_edges(
        "crawl_and_diff", has_new_links, ["process_new_links", "send_no_updates"]
    )
    .add_edge("process_new_links", "summaries")
    .add_edge("summaries", "executive_summary")
    .add_edge("executive_summary", "compose_digest")
    .add_edge("compose_digest", "create_report")
    .add_edge("send_no_updates", "create_report")
    .add_edge("create_report", END)
)


graph = workflow.compile(name="Competitor Monitoring Agent")

__all__ = ["graph"]
