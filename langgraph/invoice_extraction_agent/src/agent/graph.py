from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.models import InvoiceExtractionResult
from agent.utils import create_structured_chain


class State(TypedDict, total=False):
    """State for the chatbot."""

    messages: Annotated[List[AnyMessage], add_messages]
    extraction_result: Optional[InvoiceExtractionResult]


def chatbot(state: State) -> State:
    """Chatbot node."""
    # Get the most recent human message:
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
                )
            ]
        }

    extraction_result = create_structured_chain().invoke({"invoice_text": invoice_text})

    return {
        "messages": [
            AIMessage(
                content=f"""The invoice has been extracted successfully. The formatted result is: {extraction_result.model_dump_json()}"""
            )
        ],
        "extraction_result": extraction_result,
    }  # type: ignore


graph = (
    StateGraph(State)
    .add_node("chatbot", chatbot)
    .add_edge(START, "chatbot")
    .add_edge("chatbot", END)
    .compile(name="LangGraph Cloud Template")
)
