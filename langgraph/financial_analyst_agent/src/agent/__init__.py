"""Financial Analyst Agent package."""

from .graph import graph
from .tools import (
    code_interpreter_tool,
    run_finance_analysis,
    yf_get_history,
    yf_get_price,
)

# Expose the toolkit for external runners/inspectors
TOOLS = [
    yf_get_price,
    yf_get_history,
    run_finance_analysis,
    code_interpreter_tool,
]

__all__ = [
    "graph",
    "TOOLS",
    "yf_get_price",
    "yf_get_history",
    "run_finance_analysis",
    "code_interpreter_tool",
]
