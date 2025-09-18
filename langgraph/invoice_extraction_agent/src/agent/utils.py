"""Functional helpers for working with the structured invoice extractor."""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig

from agent.invoice_program import (
    InvoiceExtractionResult,
    aextract_invoice,
    extract_invoice,
)


def run_extract_invoice(
    invoice_text: str,
    *,
    config: Optional[RunnableConfig] = None,
) -> InvoiceExtractionResult:
    """Run the shared invoice extractor against arbitrary text."""
    return extract_invoice(invoice_text, config=config)


async def arun_extract_invoice(
    invoice_text: str,
    *,
    config: Optional[RunnableConfig] = None,
) -> InvoiceExtractionResult:
    """Asynchronous helper mirroring :func:`run_extract_invoice`."""
    return await aextract_invoice(invoice_text, config=config)


def extract_invoice_dict(
    invoice_text: str,
    *,
    config: Optional[RunnableConfig] = None,
) -> Dict[str, Any]:
    """Convenience helper that returns the structured output as a dictionary."""
    result = run_extract_invoice(invoice_text, config=config)
    return result.model_dump(exclude_none=True, exclude_defaults=True)


__all__ = [
    "arun_extract_invoice",
    "extract_invoice_dict",
    "run_extract_invoice",
]
