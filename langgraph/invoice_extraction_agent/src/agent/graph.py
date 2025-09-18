"""LangGraph interface for the structured invoice extraction workflow."""

from __future__ import annotations

import json
from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.invoice_program import extract_invoice, format_extraction_result


class InvoiceExtractionState(TypedDict, total=False):
    """LangGraph state for invoice extraction conversations."""

    messages: Annotated[List[AnyMessage], add_messages]


def run_invoice_extraction(
    state: InvoiceExtractionState,
) -> Dict[str, List[AnyMessage]]:
    """Execute structured invoice extraction and return an AI response."""
    invoice_text: Optional[str] = None
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            invoice_text = msg.content
            break

    if not invoice_text:
        return {
            "messages": [
                AIMessage(
                    content="Please provide the raw invoice text so I can extract fields."
                ),
            ]
        }

    try:
        result = extract_invoice(invoice_text)
    except Exception as exc:  # pragma: no cover - surfaced to the user at runtime
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Invoice extraction failed while contacting the language model.\n"
                        f"Reason: {exc}"
                    )
                )
            ]
        }

    formatted = format_extraction_result(result)
    payload = result.model_dump(exclude_none=True, exclude_defaults=True)
    content = formatted
    if payload:
        content += "\n\nJSON Output:\n" + json.dumps(payload, indent=2, sort_keys=True)

    message = AIMessage(
        content=content,
        additional_kwargs={"structured_output": payload} if payload else {},
    )

    return {"messages": [message]}


graph = (
    StateGraph(InvoiceExtractionState)
    .add_node("extract", run_invoice_extraction)
    .add_edge(START, "extract")
    .add_edge("extract", END)
    .compile(name="Invoice Extraction Agent")
)


__all__ = ["graph", "InvoiceExtractionState"]
