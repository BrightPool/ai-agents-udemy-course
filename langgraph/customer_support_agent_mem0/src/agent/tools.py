"""Customer Service Tools for LangGraph Agent.

Tools for handling e-commerce customer service operations including
order lookup, refunds, documentation search, and order status checks.
"""

from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from .models import (
    DocumentationCategory,
    DocumentationSearchRequest,
    DocumentationSearchResult,
)

# Mock orders database
ORDERS_DATABASE = [
    {
        "order_id": "ORD-001",
        "customer_email": "john.doe@email.com",
        "customer_name": "John Doe",
        "product": "Wireless Headphones",
        "price": 99.99,
        "status": "delivered",
        "order_date": "2024-01-15",
        "delivery_date": "2024-01-18",
    },
    {
        "order_id": "ORD-002",
        "customer_email": "jane.smith@email.com",
        "customer_name": "Jane Smith",
        "product": "Smart Watch",
        "price": 199.99,
        "status": "shipped",
        "order_date": "2024-01-20",
        "delivery_date": None,
    },
    {
        "order_id": "ORD-003",
        "customer_email": "bob.wilson@email.com",
        "customer_name": "Bob Wilson",
        "product": "Laptop Stand",
        "price": 49.99,
        "status": "processing",
        "order_date": "2024-01-22",
        "delivery_date": None,
    },
    {
        "order_id": "ORD-004",
        "customer_email": "alice.brown@email.com",
        "customer_name": "Alice Brown",
        "product": "Bluetooth Speaker",
        "price": 79.99,
        "status": "delivered",
        "order_date": "2024-01-10",
        "delivery_date": "2024-01-13",
    },
    {
        "order_id": "ORD-005",
        "customer_email": "charlie.davis@email.com",
        "customer_name": "Charlie Davis",
        "product": "Phone Case",
        "price": 24.99,
        "status": "cancelled",
        "order_date": "2024-01-25",
        "delivery_date": None,
    },
]

# Documentation directory path
DOCUMENTATION_DIR = Path(__file__).parent / "documentation"


def _load_documentation_file(category: DocumentationCategory) -> str:
    """Load documentation content from a .txt file for the given category.

    Args:
        category: The documentation category to load

    Returns:
        The content of the documentation file, or empty string if file not found
    """
    file_path = DOCUMENTATION_DIR / f"{category}.txt"

    try:
        if file_path.exists():
            with open(file_path, encoding="utf-8") as file:
                return file.read().strip()
        else:
            return f"No documentation file found for category: {category}"
    except Exception as e:
        return f"Error reading documentation file for {category}: {str(e)}"


def _get_document_chunk(content: str, start_line: int = 0, end_line: int = 100) -> str:
    """Demonstrate how to implement document pagination for large files.

    DEVELOPER NOTE: This is an example of how you could implement start/end parameters
    for handling large documentation files in production systems.

    Args:
        content: Full document content
        start_line: Starting line number (0-indexed)
        end_line: Ending line number (exclusive)

    Returns:
        Chunk of document content between start_line and end_line

    Example usage for large files:
        # In your tool function, you could accept these parameters:
        # def search_documentation_tool(
        #     request: DocumentationSearchRequest,
        #     start_line: int = 0,
        #     end_line: int = 100
        # ) -> str:
        #     content = _load_documentation_file(category)
        #     chunk = _get_document_chunk(content, start_line, end_line)
        #     # Process chunk...
    """
    lines = content.split("\n")
    chunk_lines = lines[start_line:end_line]

    metadata = f"""
---
DOCUMENT CHUNK INFO:
- Total lines in document: {len(lines)}
- Showing lines {start_line} to {min(end_line, len(lines))}
- Chunk size: {len(chunk_lines)} lines
---
"""

    return metadata + "\n".join(chunk_lines)


def _classify_query_to_category(query: str) -> DocumentationCategory:
    """Heuristic classification of a query into a documentation category."""
    q = query.lower()

    shipping_keywords = [
        "shipping",
        "delivery",
        "express",
        "standard",
        "overnight",
        "tracking",
    ]
    returns_keywords = ["return", "refund", "exchange", "policy", "rma"]
    products_keywords = [
        "product",
        "warranty",
        "compatibility",
        "specification",
        "specs",
        "feature",
    ]
    account_keywords = ["account", "password", "login", "profile", "dashboard", "reset"]
    payment_keywords = [
        "payment",
        "billing",
        "credit",
        "card",
        "paypal",
        "apple pay",
        "invoice",
        "receipt",
    ]

    if any(k in q for k in shipping_keywords):
        return "shipping"
    if any(k in q for k in returns_keywords):
        return "returns"
    if any(k in q for k in products_keywords):
        return "products"
    if any(k in q for k in account_keywords):
        return "account"
    if any(k in q for k in payment_keywords):
        return "payment"

    # Default to products if unknown; could also pick most common category
    return "products"


@tool
def search_documentation_tool(request: DocumentationSearchRequest) -> str:
    """Search through the company documentation/knowledge base using agentic RAG.

    This tool uses agentic file reading to dynamically load the ENTIRE documentation
    file and inject it into the tool response for the LLM to perform RAG search.

    Args:
        request: DocumentationSearchRequest object containing query and category

    Returns:
        JSON string of DocumentationSearchResult with resolved category and full content
    """
    resolved_category: DocumentationCategory
    if request.category == "auto":
        resolved_category = _classify_query_to_category(request.query)
    else:
        resolved_category = request.category  # type: ignore[assignment]

    # Load ENTIRE documentation content from file using agentic file reading
    full_content = _load_documentation_file(resolved_category)

    if (
        not full_content
        or "No documentation file found" in full_content
        or "Error reading" in full_content
    ):
        result = DocumentationSearchResult(
            category=resolved_category,
            content=(
                f"I couldn't find documentation for the '{resolved_category}' category. "
                "Please try a different category or contact our support team."
            ),
        )
        return result.model_dump_json()

    # IMPORTANT: Handle large documentation files
    # For production systems with 50k+ line documentation files, you'll need to implement:
    # 1. Document chunking strategies (semantic, paragraph, or fixed-size chunks)
    # 2. Vector embeddings for similarity search
    # 3. Hybrid search (keyword + semantic)
    # 4. Document summarization for overview sections
    # 5. Progressive disclosure (show relevant sections first)
    #
    # Current implementation: Simple character limit for LLM context window
    MAX_CONTENT_LENGTH = 8000  # Adjust based on your LLM's context window

    if len(full_content) > MAX_CONTENT_LENGTH:
        # For large files, trim content and provide guidance
        trimmed_content = full_content[:MAX_CONTENT_LENGTH]

        # Find the last complete sentence/paragraph to avoid cutting mid-sentence
        last_period = trimmed_content.rfind(".")
        last_newline = trimmed_content.rfind("\n")
        cut_point = (
            max(last_period, last_newline) if last_period > 0 else MAX_CONTENT_LENGTH
        )

        trimmed_content = full_content[:cut_point]

        # Add developer guidance for handling large files
        large_file_notice = f"""

---
⚠️  LARGE DOCUMENTATION FILE DETECTED ⚠️
Original file size: {len(full_content)} characters
Trimmed to: {len(trimmed_content)} characters

DEVELOPER NOTE: This file exceeds the recommended size for direct LLM processing.
For production systems with large documentation (50k+ lines), consider implementing:

1. **Document Chunking**: Break large docs into semantic chunks
2. **Vector Search**: Use embeddings for similarity-based retrieval
3. **Hybrid Search**: Combine keyword + semantic search
4. **Progressive Disclosure**: Return start/end parameters for pagination
5. **Document Summarization**: Provide overview + detailed sections

Example implementation parameters you could add:
- start_line: int = 0
- end_line: int = 100
- chunk_size: int = 1000
- search_strategy: Literal["semantic", "keyword", "hybrid"] = "hybrid"

For now, showing first {len(trimmed_content)} characters of documentation.
---
"""

        full_content = trimmed_content + large_file_notice

    # Return the document content for agentic RAG search
    # The LLM will perform the search and extraction based on the user's query
    result = DocumentationSearchResult(
        category=resolved_category,
        content=f"""FULL DOCUMENTATION FOR {resolved_category.upper()} CATEGORY:

{full_content}

---
USER QUERY: {request.query}

Please search through the above documentation and provide relevant information to answer the user's query about {request.query}.""",
    )
    return result.model_dump_json()


@tool
def search_orders_tool(
    customer_email: Optional[str] = None, order_id: Optional[str] = None
) -> str:
    """Search for customer orders by email or order ID.

    Args:
        customer_email: Customer's email address
        order_id: Specific order ID

    Returns:
        Order information or message if not found
    """
    if not customer_email and not order_id:
        return (
            "Please provide either a customer email or order ID to search for orders."
        )

    found_orders = []

    for order in ORDERS_DATABASE:
        if customer_email and order["customer_email"].lower() == customer_email.lower():
            found_orders.append(order)
        elif order_id and order["order_id"].upper() == order_id.upper():
            found_orders.append(order)

    if not found_orders:
        if customer_email:
            return f"No orders found for email: {customer_email}"
        else:
            return f"No orders found for order ID: {order_id}"

    result = []
    for order in found_orders:
        order_info = f"""
Order ID: {order["order_id"]}
Customer: {order["customer_name"]} ({order["customer_email"]})
Product: {order["product"]}
Price: ${order["price"]}
Status: {order["status"].title()}
Order Date: {order["order_date"]}
Delivery Date: {order["delivery_date"] if order["delivery_date"] else "Not delivered yet"}
"""
        result.append(order_info.strip())

    return "\n\n".join(result)


@tool
def refund_customer_tool(order_id: str, reason: str = "Customer request") -> str:
    """Process a refund for a customer order.

    Args:
        order_id: The order ID to refund
        reason: Reason for the refund

    Returns:
        Confirmation message or error
    """
    # Find the order
    order = None
    for o in ORDERS_DATABASE:
        if o["order_id"].upper() == order_id.upper():
            order = o
            break

    if not order:
        return f"Order {order_id} not found. Please verify the order ID."

    # Check if order is eligible for refund
    if order["status"] == "cancelled":
        return f"Order {order_id} has already been cancelled and cannot be refunded."

    if order["status"] == "processing":
        return (
            f"Order {order_id} is still being processed. Refunds can only be processed "
            "for shipped or delivered orders."
        )

    # Simulate refund processing
    refund_amount = order["price"]

    # Update order status (in a real system, this would update the database)
    order["status"] = "refunded"

    return f"""
Refund processed successfully!

Order ID: {order_id}
Customer: {order["customer_name"]} ({order["customer_email"]})
Product: {order["product"]}
Refund Amount: ${refund_amount}
Reason: {reason}
Status: Refunded

The refund will be credited to the original payment method within 5-7 business days.
"""


@tool
def get_order_status_tool(order_id: str) -> str:
    """Get the current status of an order.

    Args:
        order_id: The order ID to check

    Returns:
        Order status information
    """
    for order in ORDERS_DATABASE:
        if order["order_id"].upper() == order_id.upper():
            status_info = f"""
Order ID: {order["order_id"]}
Status: {order["status"].title()}
Customer: {order["customer_name"]}
Product: {order["product"]}
Order Date: {order["order_date"]}
"""
            if order["delivery_date"]:
                status_info += f"Delivery Date: {order['delivery_date']}"
            else:
                status_info += "Delivery Date: Not delivered yet"

            return status_info.strip()

    return f"Order {order_id} not found. Please verify the order ID."
