from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from agent.models import InvoiceExtractionResult


def build_prompt() -> ChatPromptTemplate:
    """Build the prompt for the chatbot."""
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
            ("Invoice text:\n----------------\n{invoice_text}\n----------------"),
        ]
    )


def create_structured_chain() -> Runnable:
    """Create the structured chain."""
    return build_prompt() | ChatOpenAI(model="gpt-5-nano").with_structured_output(
        InvoiceExtractionResult
    )
