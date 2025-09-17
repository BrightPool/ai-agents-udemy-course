"""Customer Service Agent package."""

from .graph import graph
from .tools import (
    get_order_status_tool,
    refund_customer_tool,
    search_documentation_tool,
    search_orders_tool,
)

# Define tools list for easy access
tools = [
    search_documentation_tool,
    search_orders_tool,
    refund_customer_tool,
    get_order_status_tool
]

__all__ = [
    "graph",
    "tools",
    "search_documentation_tool",
    "search_orders_tool", 
    "refund_customer_tool",
    "get_order_status_tool"
]