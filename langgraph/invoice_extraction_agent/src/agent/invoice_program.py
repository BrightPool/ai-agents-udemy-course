"""Structured invoice extraction helpers for the LangGraph agent."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2_000


class LineItem(BaseModel):
    """Itemised charge extracted from an invoice."""

    description: Optional[str] = Field(
        None,
        description="Short description of the goods or services.",
    )
    quantity: Optional[str] = Field(
        None,
        description="Quantity or units for the line item.",
    )
    unit_price: Optional[str] = Field(
        None,
        description="Price per unit. Include the currency symbol or code when present.",
    )
    total: Optional[str] = Field(
        None,
        description="Total amount for this line item, including currency if available.",
    )


class InvoiceExtractionResult(BaseModel):
    """Structured representation of key invoice fields."""

    rationale: str = Field(
        ..., description="Brief explanation of how the fields were identified."
    )
    invoice_number: Optional[str] = Field(
        None, description="Invoice identifier exactly as written on the document."
    )
    invoice_date: Optional[str] = Field(
        None,
        description="Date the invoice was issued. Prefer ISO 8601 format when possible.",
    )
    due_date: Optional[str] = Field(
        None, description="Payment due date taken from the invoice."
    )
    purchase_order: Optional[str] = Field(
        None, description="Related purchase order number if provided."
    )
    bill_to: Optional[str] = Field(
        None, description="Party responsible for paying the invoice."
    )
    bill_to_address: Optional[str] = Field(
        None, description="Mailing address of the billed party, if present."
    )
    ship_to: Optional[str] = Field(
        None, description="Shipping recipient or location when specified."
    )
    seller: Optional[str] = Field(
        None, description="Business or individual issuing the invoice."
    )
    seller_address: Optional[str] = Field(None, description="Address of the seller.")
    subtotal_amount: Optional[str] = Field(
        None,
        description="Subtotal before taxes or additional charges, including currency if present.",
    )
    tax_amount: Optional[str] = Field(
        None, description="Total tax amount on the invoice."
    )
    total_amount: Optional[str] = Field(
        None, description="Total amount due, including currency when provided."
    )
    payment_terms: Optional[str] = Field(
        None, description="Payment terms or instructions listed on the invoice."
    )
    notes: Optional[str] = Field(
        None, description="Additional notes, adjustments, or comments that matter."
    )
    line_items: List[LineItem] = Field(
        default_factory=list,
        description="List of itemised charges recovered from the invoice.",
    )


def _build_prompt() -> ChatPromptTemplate:
    """Create the prompt template used to guide structured extraction."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert financial operations analyst. "
                    "Extract the requested invoice information. "
                    "Always respond using the structured schema provided by the tool. "
                    "Return null for any field that is not explicitly stated. "
                    "Normalise dates to ISO 8601 when practical and keep amounts coupled "
                    "with their currency symbols or codes."
                ),
            ),
            (
                "human",
                ("Invoice text:\n----------------\n{invoice_text}\n----------------"),
            ),
        ]
    )


@lru_cache(maxsize=1)
def _structured_chain() -> Runnable:
    """Create (and cache) the runnable graph that yields structured output."""
    prompt = _build_prompt()
    llm = ChatOpenAI(
        model=DEFAULT_MODEL_NAME,
        temperature=DEFAULT_TEMPERATURE,
    )
    return prompt | llm.with_structured_output(InvoiceExtractionResult)


def extract_invoice(
    invoice_text: str,
    *,
    config: Optional[RunnableConfig] = None,
) -> InvoiceExtractionResult:
    """Extract invoice fields synchronously."""
    if not invoice_text or not invoice_text.strip():
        raise ValueError("invoice_text must be a non-empty string.")
    chain = _structured_chain()
    return chain.invoke({"invoice_text": invoice_text}, config=config)


async def aextract_invoice(
    invoice_text: str,
    *,
    config: Optional[RunnableConfig] = None,
) -> InvoiceExtractionResult:
    """Extract invoice fields asynchronously."""
    if not invoice_text or not invoice_text.strip():
        raise ValueError("invoice_text must be a non-empty string.")
    chain = _structured_chain()
    return await chain.ainvoke({"invoice_text": invoice_text}, config=config)


def format_extraction_result(result: InvoiceExtractionResult) -> str:
    """Human-readable summary for downstream messaging."""
    lines: List[str] = ["Invoice Extraction Result", "========================"]
    rationale = result.rationale.strip()
    lines.append("Rationale:")
    lines.append(rationale if rationale else "Model did not provide a rationale.")

    payload = result.model_dump(exclude_none=True, exclude_defaults=True)
    payload.pop("rationale", None)
    line_items = payload.pop("line_items", [])

    if payload:
        lines.append("")
        lines.append("Extracted Fields:")
        for key, value in sorted(payload.items()):
            label = key.replace("_", " ").title()
            lines.append(f"- {label}: {value}")

    if line_items:
        lines.append("")
        lines.append("Line Items:")
        for index, item in enumerate(line_items, start=1):
            lines.append(f"  Item {index}:")
            for sub_key, sub_value in item.items():
                sub_label = sub_key.replace("_", " ").title()
                lines.append(f"    - {sub_label}: {sub_value}")

    return "\n".join(lines)


__all__ = [
    "InvoiceExtractionResult",
    "LineItem",
    "aextract_invoice",
    "extract_invoice",
    "format_extraction_result",
]
