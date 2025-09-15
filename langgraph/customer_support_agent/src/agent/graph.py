"""Customer Service LangGraph Agent.

A sophisticated agent for handling e-commerce customer service inquiries
with order management, refunds, documentation search, and support.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.utils import convert_to_secret_str
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from agent.tools import (
    get_order_status_tool,
    refund_customer_tool,
    search_documentation_tool,
    search_orders_tool,
)

# Load environment variables from .env file if it exists
# This allows developers to run the graph directly with python-dotenv support
_current_file = Path(__file__)
_project_root = _current_file.parent.parent.parent  # Navigate to project root
_env_file = _project_root / ".env"

if _env_file.exists():
    load_dotenv(_env_file)
    # Note: Using print here for initialization feedback - consider using logging in production
    print(f"✅ Loaded environment variables from {_env_file}")  # noqa: T201
else:
    print(f"ℹ️  No .env file found at {_env_file}")  # noqa: T201
    print(  # noqa: T201
        "   Environment variables will be loaded from system environment or runtime context"
    )


class Context(TypedDict):
    """Context parameters for the customer service agent."""

    anthropic_api_key: str
    max_iterations: int


class CustomerServiceState(TypedDict):
    """State for the customer service agent."""

    # Core state
    messages: Annotated[List[AnyMessage], add_messages]

    # Required inputs
    customer_email: str
    order_id: str


def is_relevant_query(query: str) -> bool:
    """Check if a query is relevant to e-commerce customer service.

    Args:
        query: The customer's query

    Returns:
        True if relevant, False otherwise
    """
    query_lower = query.lower()

    # Define relevant keywords
    relevant_keywords = [
        # Order related
        "order",
        "purchase",
        "buy",
        "bought",
        "ordered",
        "order id",
        "order number",
        "tracking",
        "delivery",
        "shipped",
        "delivered",
        "status",
        # Product related
        "product",
        "item",
        "product",
        "specification",
        "compatibility",
        "warranty",
        "return",
        "refund",
        "exchange",
        "defective",
        "broken",
        "damaged",
        # Account related
        "account",
        "login",
        "password",
        "profile",
        "billing",
        "payment",
        "credit card",
        "paypal",
        "invoice",
        "receipt",
        # Shipping related
        "shipping",
        "express",
        "standard",
        "overnight",
        "cost",
        "free shipping",
        "address",
        "delivery",
        "package",
        "shipment",
        # General customer service
        "help",
        "support",
        "customer service",
        "assistance",
        "problem",
        "issue",
        "question",
        "inquiry",
        "complaint",
        "concern",
    ]

    # Check if query contains relevant keywords
    return any(keyword in query_lower for keyword in relevant_keywords)


def llm_call(state: CustomerServiceState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """LLM decides whether to call a tool or not."""
    # Set up the LLM with Anthropic Claude
    ctx = getattr(runtime, "context", None)
    api_key_value = None
    if isinstance(ctx, dict):
        api_key_value = ctx.get("anthropic_api_key")  # type: ignore[assignment]
    if not api_key_value:
        api_key_value = os.getenv("ANTHROPIC_API_KEY")

    llm = ChatAnthropic(
        model_name="claude-3-5-sonnet-20241022",
        api_key=convert_to_secret_str(api_key_value or ""),
        temperature=0.1,  # Low temperature for consistent responses
        timeout=30,
        stop=None,
    )

    # Define tools
    tools = [
        search_documentation_tool,
        search_orders_tool,
        refund_customer_tool,
        get_order_status_tool,
    ]

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create system message with injected context
    customer_email = state.get("customer_email", "")
    order_id = state.get("order_id", "")

    context_info = ""
    if customer_email or order_id:
        context_info = f"""
CUSTOMER CONTEXT:
- Customer Email: {customer_email}
- Order ID: {order_id}

Use this information to provide personalized assistance. When the customer asks about "my order" or similar, refer to the provided order ID and customer email.
"""

    system_message = SystemMessage(
        content=f"""
You are a professional customer service representative for TechStore, an e-commerce company specializing in electronics and tech accessories.
{context_info}
YOUR ROLE:
- Provide helpful, accurate, and friendly customer service
- Assist with order inquiries, product information, shipping, returns, and refunds
- Maintain a professional and empathetic tone
- Always prioritize customer satisfaction

WHAT YOU CAN HELP WITH:
1. Order Status & Tracking
   - Check order status using order ID or customer email
   - Provide delivery information and tracking details

2. Product Information
   - Product specifications and compatibility
   - Warranty information
   - Availability and stock status

3. Shipping & Delivery
   - Shipping options and costs
   - Delivery timeframes
   - Shipping policies

4. Returns & Refunds
   - Return policy information
   - Process refunds for eligible orders
   - Exchange procedures

5. Account & Payment
   - Account management assistance
   - Payment method information
   - Billing inquiries

6. General Support
   - Company policies
   - Technical support guidance
   - Troubleshooting assistance

IMPORTANT GUIDELINES:
- ALWAYS verify order IDs and customer information before processing refunds
- Use the search_documentation_tool for policy and general information questions
- Use search_orders_tool to look up customer orders by email or order ID
- Use refund_customer_tool only for legitimate refund requests
- Use get_order_status_tool for quick status checks
- If a customer asks about something unrelated to e-commerce, politely redirect them
- Always be helpful but maintain professional boundaries
- If you cannot resolve an issue, offer to escalate to a human representative

REJECTION CRITERIA:
- Questions about topics completely unrelated to e-commerce, orders, products, or customer service
- Requests for personal information about other customers
- Inappropriate or offensive language
- Requests that violate company policies

Remember: Your goal is to provide excellent customer service while maintaining efficiency and accuracy.
"""
    )

    # Derive the user's textual request for relevance filtering
    user_request: str | None = None
    # Try to extract from the latest HumanMessage
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_request = msg.content if isinstance(msg.content, str) else None
            break

    # Check if query is relevant (only if we have text to evaluate)
    if user_request and not is_relevant_query(user_request):
        return {
            "messages": [
                AIMessage(
                    content="""I'm sorry, but I can only assist with e-commerce related inquiries such as:
- Order status and tracking
- Product information and specifications
- Shipping and delivery questions
- Returns and refunds
- Account and payment issues
- General customer service support

Please feel free to ask me about any of these topics, and I'll be happy to help!"""
                )
            ],
        }

    # Invoke LLM with tools
    response = llm_with_tools.invoke([system_message] + state["messages"])

    return {"messages": [response]}


tool_node = ToolNode(
    tools=[
        search_documentation_tool,
        search_orders_tool,
        refund_customer_tool,
        get_order_status_tool,
    ]
)


def should_continue(state: CustomerServiceState) -> Literal["tool_node", END]:  # type: ignore
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call."""
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM (AIMessage) makes a tool call, then perform an action
    if isinstance(last_message, AIMessage):
        tool_calls = getattr(last_message, "tool_calls", None)
        if tool_calls:
            return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


# Define the graph
graph = (
    StateGraph(CustomerServiceState, context_schema=Context)
    .add_node("llm_call", llm_call)
    .add_node("tool_node", tool_node)
    # Add edges to connect nodes
    .add_edge(START, "llm_call")
    .add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    .add_edge("tool_node", "llm_call")
    .compile(name="Customer Service Agent")
)
