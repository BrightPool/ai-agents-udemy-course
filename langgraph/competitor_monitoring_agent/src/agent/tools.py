"""Helper functions for the competitor monitoring agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
import pygsheets
import psycopg
import re

from .models import ArticleSummary, CompetitorSource, DigestEmail, DiscoveredLink


DEFAULT_DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/competitors")
DEFAULT_OUTBOX_DIR = Path("outbox")
DEFAULT_SUBJECT = "daily competitor digest"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
HTTP_TIMEOUT = httpx.Timeout(30.0)
HTML_TAG_RE = re.compile(r"<[^>]+>")


class AnchorExtractor(HTMLParser):
    """HTML parser that collects href attributes from anchor tags."""

    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: Sequence[tuple[str, str | None]]) -> None:  # noqa: D401
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value.strip())


@dataclass
class CrawledLinks:
    competitor: CompetitorSource
    normalized_links: List[str]


def get_request_headers(extra_headers: Dict[str, str] | None = None) -> Dict[str, str]:
    """Merge default headers with optional overrides."""

    headers = DEFAULT_HEADERS.copy()
    if extra_headers:
        headers.update({k: v for k, v in extra_headers.items() if v})
    return headers


def fetch_text(url: str, headers: Dict[str, str] | None = None) -> str:
    """Fetch raw text content from a URL using httpx."""

    request_headers = get_request_headers(headers)
    with httpx.Client(follow_redirects=True, timeout=HTTP_TIMEOUT) as client:
        response = client.get(url, headers=request_headers)
        response.raise_for_status()
        return response.text


def _build_pygsheets_client(
    *,
    service_account_file: str | None = None,
    service_account_env_var: str | None = None,
    service_account_json: str | None = None,
) -> pygsheets.Client:
    """Create and return a pygsheets client using the provided credentials."""

    authorize_kwargs: Dict[str, object] = {"retries": 3, "local": False}

    if service_account_json:
        env_var = service_account_env_var or "PYGSHEETS_SERVICE_ACCOUNT_JSON"
        os.environ[env_var] = service_account_json
        service_account_env_var = env_var

    if service_account_env_var:
        return pygsheets.authorize(service_account_env_var=service_account_env_var, **authorize_kwargs)

    if service_account_file:
        return pygsheets.authorize(service_account_file=service_account_file, **authorize_kwargs)

    # Fall back to default authorization flow (expects client_secret.json or cached creds).
    return pygsheets.authorize(**authorize_kwargs)


def fetch_competitor_records(
    sheet_url: str,
    *,
    worksheet: str | None = None,
    service_account_file: str | None = None,
    service_account_env_var: str | None = None,
    service_account_json: str | None = None,
) -> List[Mapping[str, object]]:
    """Fetch competitor rows from Google Sheets via pygsheets."""

    client = _build_pygsheets_client(
        service_account_file=service_account_file,
        service_account_env_var=service_account_env_var,
        service_account_json=service_account_json,
    )

    try:
        spreadsheet = client.open_by_url(sheet_url)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unable to open Google Sheet at {sheet_url}: {exc!s}") from exc

    try:
        worksheet_obj = spreadsheet.worksheet_by_title(worksheet) if worksheet else spreadsheet.sheet1
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Unable to load worksheet '{worksheet}' from sheet {sheet_url}: {exc!s}"
        ) from exc

    return worksheet_obj.get_all_records(empty_value="")


def parse_competitor_records(records: Sequence[Mapping[str, object]]) -> List[CompetitorSource]:
    """Convert Google Sheet rows into structured competitor sources."""

    competitors: List[CompetitorSource] = []
    for row in records:
        normalized_row = {
            str(key).strip().lower(): (str(value).strip() if value is not None else "")
            for key, value in row.items()
            if key is not None
        }

        root_url = normalized_row.get("url") or normalized_row.get("root_url")
        blog_url = normalized_row.get("blog urls") or normalized_row.get("blog_url")
        company = (
            normalized_row.get("company")
            or normalized_row.get("name")
            or normalized_row.get("competitor")
            or "Unknown"
        )

        if not root_url or not blog_url:
            continue

        competitors.append(
            CompetitorSource(
                company=company,
                root_url=root_url,
                blog_url=blog_url,
            )
        )

    return competitors


def extract_links(html: str) -> List[str]:
    """Extract raw anchor href values from HTML."""

    parser = AnchorExtractor()
    parser.feed(html)
    return parser.links


def normalize_links(competitor: CompetitorSource, raw_links: Iterable[str]) -> CrawledLinks:
    """Normalize links relative to the competitor blog URL and restrict to known domains."""

    root_host = urlparse(competitor.root_url).netloc
    blog_host = urlparse(competitor.blog_url).netloc or root_host

    normalized: List[str] = []
    seen: set[str] = set()

    for href in raw_links:
        candidate = href.strip()
        if not candidate or candidate.startswith("javascript:"):
            continue
        normalized_url = urljoin(competitor.blog_url, candidate)
        parsed = urlparse(normalized_url)
        if not parsed.scheme.startswith("http"):
            continue
        host = parsed.netloc
        if host not in {root_host, blog_host}:
            continue
        cleaned = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", parsed.query, ""))
        if cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)

    return CrawledLinks(competitor=competitor, normalized_links=normalized)


def html_to_text(html: str, max_characters: int = 8000) -> str:
    """Strip tags from HTML, collapse whitespace, and truncate for prompt safety."""

    text = HTML_TAG_RE.sub(" ", html)
    text = unescape(text)
    text = " ".join(text.split())
    if len(text) > max_characters:
        return text[:max_characters] + " ..."
    return text


def ensure_database(db_url: str = DEFAULT_DB_URL) -> psycopg.Connection:
    """Create (if needed) and return a Postgres connection for storing URLs."""

    connection = psycopg.connect(db_url, autocommit=True)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS competitor_urls (
                id SERIAL PRIMARY KEY,
                normalized_url TEXT UNIQUE NOT NULL,
                source_site TEXT NOT NULL,
                date_found TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
    return connection


def persist_links(
    connection: psycopg.Connection,
    crawled: CrawledLinks,
) -> List[DiscoveredLink]:
    """Insert normalized links into storage and return only newly seen entries."""

    new_links: List[DiscoveredLink] = []
    discovered_at = datetime.utcnow()

    with connection.cursor() as cursor:
        for normalized_url in crawled.normalized_links:
            cursor.execute(
                """
                INSERT INTO competitor_urls (normalized_url, source_site, date_found)
                VALUES (%s, %s, %s)
                ON CONFLICT (normalized_url) DO NOTHING
                """,
                (normalized_url, crawled.competitor.root_url, discovered_at),
            )

            if cursor.rowcount == 1:
                new_links.append(
                    DiscoveredLink(
                        normalized_url=normalized_url,
                        source_site=crawled.competitor.root_url,
                        competitor=crawled.competitor.company,
                        discovered_at=discovered_at,
                    )
                )

    return new_links


def write_email_to_outbox(email: DigestEmail, outbox_dir: Path = DEFAULT_OUTBOX_DIR) -> Path:
    """Persist the rendered email to disk and return its path."""

    outbox_dir.mkdir(parents=True, exist_ok=True)
    output_path = outbox_dir / "latest_competitor_digest.html"
    html_template = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{email.subject}</title>
</head>
<body>
{email.body_html}
</body>
</html>
"""
    output_path.write_text(html_template, encoding="utf-8")
    return output_path


def format_digest_html(
    executive_summary: str,
    summaries: List[ArticleSummary],
    greeting: str = "Hi Rhys,",
) -> str:
    """Render the HTML body used by the Gmail node in the original workflow."""

    summary_items = "\n".join(
        f"<li><strong>{entry.competitor}</strong>: {entry.content} (<a href=\"{entry.source_url}\">source</a>)"  # noqa: E501
        for entry in summaries
    )

    if not summary_items:
        summary_items = "<li>No competitor activity detected today.</li>"

    return f"""
<p>{greeting}</p>

<p>Here’s the daily scoop:</p>

<h3>Executive Summary</h3>
<p>{executive_summary}</p>

<h3>Competitor Updates</h3>
<ul>
{summary_items}
</ul>

<p>Have a great day,</p>

<hr />
<p><em>This email was sent automatically with LangGraph</em></p>
"""
