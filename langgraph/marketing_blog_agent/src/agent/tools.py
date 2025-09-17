"""Marketing Blog Agent Tools.

Tools supporting a ReAct-style marketing blog writer workflow:
- Vector search over example marketing corpus (FAISS, with numpy fallback)
- Outline management (persist outline across tool calls)
- Section persistence and editing
- Final blog assembly
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from langchain_core.tools import tool
from dotenv import load_dotenv


# --- Environment ---
load_dotenv()


# --- Example marketing corpus (can be replaced with your own docs) ---
EXAMPLE_DOCS: List[dict] = [
    {"id": "company_vision_1", "text": "Nimbus is the revenue automation platform for RevOps and Data teams. We replace brittle spreadsheets with governed, AI-assisted workflows."},
    {"id": "pricing_tiers_1", "text": "Pricing: Starter $99/mo up to 5 seats; Pro $499/mo up to 25 seats; Scale $1,999/mo unlimited seats with SSO and SAML."},
    {"id": "compliance_1", "text": "Compliance: SOC 2 Type II and ISO 27001 certified. GDPR compliant. HIPAA not supported."},
    {"id": "data_residency_1", "text": "Data residency: EU customers can pin data to Frankfurt (eu-central-1). Default region us-east-1."},
    {"id": "sla_support_1", "text": "SLA: 99.9% uptime. Support first response under 4 business hours; Scale gets 30-minute critical SLA."},
    {"id": "support_channels_1", "text": "Support: Private Slack 9-5 PT on weekdays; 24/7 on-call for P1 incidents via PagerDuty."},
    {"id": "integrations_crm_1", "text": "Integrations: Native connectors for Salesforce and HubSpot including bidirectional sync and custom objects."},
    {"id": "integrations_warehouse_1", "text": "Warehouses: Snowflake and BigQuery supported; Redshift in private beta."},
    {"id": "integrations_streaming_1", "text": "Streaming: Kafka and Segment sources supported; exactly-once event delivery with idempotency keys."},
    {"id": "personas_1", "text": "Personas: RevOps needs pipeline visibility; Data Engineering needs reliable ingestion; Marketing Ops needs attribution sanity."},
    {"id": "brand_voice_1", "text": "Voice: practical, no-hype, crisp verbs, short sentences. Avoid exclamation marks."},
    {"id": "style_guide_1", "text": "Style: use US English, Oxford comma, and sentence case for headings."},
    {"id": "product_features_1", "text": "Features: Rules Engine, Playbooks, and Workflows. Rules Engine executes row-level policies with audit logs."},
    {"id": "security_1", "text": "Security: PII redaction enabled by default; customer-managed keys available on Scale."},
    {"id": "programs_migration_1", "text": "Concierge Migration: free one-time program up to 20 hours; includes schema mapping and QA."},
    {"id": "event_growth_summit_1", "text": "Growth Summit SF: Oct 14-16; booth B12; CEO Maya Chen keynote Oct 15 at 10:00am."},
    {"id": "offer_code_1", "text": "Promo: BUILD25 gives 25% off the first year for contracts signed before Dec 31."},
    {"id": "case_study_1", "text": "Case study: Acme Logistics increased lead-to-opportunity by 23% and cut churn 12% after adopting Nimbus."},
    {"id": "limits_api_1", "text": "API limits: 600 requests/min per org and 10 requests/sec per user. 429 means back off."},
    {"id": "data_retention_1", "text": "Data retention: logs stored 30 days by default; retention can be extended on Scale."},
    {"id": "roadmap_1", "text": "Roadmap: AI Forecasting open beta in Q4; Redwood Plugin GA in Q1."},
    {"id": "naming_1", "text": "Naming: use 'Nimbus' in external copy; avoid the internal codename 'AcmeCloud'."},
    {"id": "billing_1", "text": "Billing: Annual contracts only; invoices net-30; procurement often requests a security questionnaire."},
]


# --- Embeddings + Index (OpenAI -> FAISS; with numpy fallback) ---
_OPENAI_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")


def _embed_texts(texts: List[str]) -> np.ndarray:
    """Embed texts using OpenAI. Fallback to deterministic random vectors if unavailable."""
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(model=_OPENAI_EMBED_MODEL, input=texts)
        vectors = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
        return np.vstack(vectors)
    except Exception:
        # Deterministic fallback for offline/dev environments
        dim = 1536
        fallback = []
        for text in texts:
            rng = random.Random(hash(text) % (2**32))
            vec = np.array([rng.random() for _ in range(dim)], dtype=np.float32)
            fallback.append(vec)
        return np.vstack(fallback)


def _build_index(texts: List[str]):
    """Build cosine-sim FAISS index or numpy fallback."""
    X = _embed_texts(texts)
    # Normalize for cosine-like search via L2 on unit vectors
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatL2(X_norm.shape[1])
        index.add(X_norm.astype("float32"))
        backend = "faiss"
    except Exception:
        index = X_norm.astype("float32")  # store raw matrix for numpy fallback
        backend = "numpy"

    return X_norm.astype("float32"), index, backend


_TEXTS = [d["text"] for d in EXAMPLE_DOCS]
_X, _INDEX, _INDEX_BACKEND = _build_index(_TEXTS)
_ID_LOOKUP = {i: EXAMPLE_DOCS[i]["id"] for i in range(len(EXAMPLE_DOCS))}


def _search(query: str, k: int = 3) -> list[dict]:
    """Semantic search over example docs returning list of dicts with id/text/score."""
    q = _embed_texts([query])[0]
    q = q / (np.linalg.norm(q) + 1e-12)

    if _INDEX_BACKEND == "faiss":
        import faiss  # type: ignore

        distances, indices = _INDEX.search(q.reshape(1, -1), k)
        idxs, dists = indices[0], distances[0]
    else:
        # numpy fallback: brute force L2 on unit vectors
        diffs = _X - q[None, :]
        l2 = np.sum(diffs * diffs, axis=1)
        idxs = np.argsort(l2)[:k]
        dists = l2[idxs]

    results = []
    for idx, dist in zip(idxs, dists):
        if int(idx) < 0 or int(idx) >= len(EXAMPLE_DOCS):
            continue
        results.append(
            {
                "id": _ID_LOOKUP[int(idx)],
                "text": EXAMPLE_DOCS[int(idx)]["text"],
                # similarity ~ 1 - (L2 / 2) for unit vectors
                "score": float(1.0 - float(dist) / 2.0),
            }
        )
    return results


# --- In-memory working state ---
@dataclass
class BlogStateMemory:
    topic: Optional[str] = None
    outline: List[str] = None  # type: ignore[assignment]
    sections: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:  # noqa: D401
        if self.outline is None:
            self.outline = []
        if self.sections is None:
            self.sections = {}


_BLOG_STATE = BlogStateMemory()


# --- Tools ---
@tool
def search_context(query: str, k: int = 4) -> str:
    """Vector search in marketing corpus; returns JSON with top-k results.

    Args:
        query: Search query
        k: Number of results to return (default 4)

    Returns:
        JSON string: {"tool": "search_context", "results": [{id,text,score}, ...]}
    """
    hits = _search(query, k=max(1, min(int(k), 10)))
    return json.dumps({"tool": "search_context", "results": hits})


@tool
def change_outline(new_outline: List[str]) -> str:
    """Replace the current outline with a new one. Returns updated outline as JSON."""
    _BLOG_STATE.outline = list(new_outline)
    return json.dumps({"tool": "change_outline", "outline": _BLOG_STATE.outline})


@tool
def write_section(section_title: str, draft: str) -> str:
    """Persist a section draft authored by the model. Returns the saved section.

    Note: The LLM should generate the draft and pass it to this tool.
    """
    _BLOG_STATE.sections[section_title] = draft
    return json.dumps({"tool": "write_section", "section_title": section_title, "saved": True})


@tool
def edit_section(section_title: str, new_draft: str) -> str:
    """Replace an existing section draft with an edited version. Returns confirmation."""
    _BLOG_STATE.sections[section_title] = new_draft
    return json.dumps({"tool": "edit_section", "section_title": section_title, "saved": True})


@tool
def assemble_blog() -> str:
    """Assemble the final blog from the current outline and sections. Returns the blog text."""
    parts: List[str] = []
    for title in _BLOG_STATE.outline:
        body = _BLOG_STATE.sections.get(title, "")
        parts.append(f"# {title}\n\n{body}".strip())
    final_blog = "\n\n".join([p for p in parts if p])
    return json.dumps({"tool": "assemble_blog", "final_blog": final_blog})


__all__ = [
    "search_context",
    "change_outline",
    "write_section",
    "edit_section",
    "assemble_blog",
]

