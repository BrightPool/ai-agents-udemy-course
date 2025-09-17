"""Helper functions for the competitor monitoring agent."""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
import re

from .models import ArticleSummary, CompetitorSource, DigestEmail, DiscoveredLink


DEFAULT_DB_PATH = Path("data/competitor_monitoring.db")
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


def parse_competitor_csv(csv_text: str) -> List[CompetitorSource]:
    """Convert CSV text to structured competitor sources."""

    reader = csv.DictReader(StringIO(csv_text))
    competitors: List[CompetitorSource] = []
    for row in reader:
        normalized_row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
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


def ensure_database(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create (if needed) and return a SQLite connection for storing URLs."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS competitor_urls (
            normalized_url TEXT PRIMARY KEY,
            source_site TEXT NOT NULL,
            date_found TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def persist_links(
    connection: sqlite3.Connection,
    crawled: CrawledLinks,
) -> List[DiscoveredLink]:
    """Insert normalized links into storage and return only newly seen entries."""

    new_links: List[DiscoveredLink] = []
    timestamp = datetime.utcnow().isoformat()
    with connection:
        for normalized_url in crawled.normalized_links:
            row = connection.execute(
                """
                INSERT OR IGNORE INTO competitor_urls (normalized_url, source_site, date_found)
                VALUES (?, ?, ?)
                """,
                (normalized_url, crawled.competitor.root_url, timestamp),
            )
            if row.rowcount == 1:
                new_links.append(
                    DiscoveredLink(
                        normalized_url=normalized_url,
                        source_site=crawled.competitor.root_url,
                        competitor=crawled.competitor.company,
                        discovered_at=datetime.fromisoformat(timestamp),
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
