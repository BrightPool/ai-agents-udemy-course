"""Tool implementations for the financial analyst agent."""

from __future__ import annotations

import os
from typing import Any, Iterable, List

import yfinance as yf
from langchain_core.tools import tool
from openai import OpenAI
from pydantic import ValidationError

from .models import CodeInterpreterRequest, FinanceAnalysisRequest, HistoryRequest, PriceRequest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _format_currency(value: float | None, currency: str | None) -> str:
    if value is None:
        return "Unknown"
    if currency:
        return f"{value:.2f} {currency}"
    return f"{value:.2f}"


def _sanitize_ticker(raw: str) -> str:
    return raw.strip().upper()


def _fmt_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    if value is None:
        return ""
    return str(value)


def _fmt_volume(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{int(value)}"
    if value is None:
        return ""
    return str(value)


def _env_value(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


_CLIENT: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set for financial analyst tools.")
        _CLIENT = OpenAI(api_key=api_key)
    return _CLIENT


def _extract_output_text(resp: Any) -> str:
    """Best-effort extraction of assistant text from a Responses API payload."""

    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text

    output: Iterable[Any] | None = getattr(resp, "output", None)
    if not output:
        return ""

    fragments: List[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if text:
                        fragments.append(text)
                elif isinstance(part, str):
                    fragments.append(part)
        elif isinstance(content, str):
            fragments.append(content)
    return "\n".join(fragments)


# ---------------------------------------------------------------------------
# Yahoo Finance tools
# ---------------------------------------------------------------------------


@tool
def yf_get_price(ticker: str) -> str:
    """Return the latest available price snapshot for the provided ticker."""

    try:
        PriceRequest(ticker=ticker)
    except ValidationError as exc:  # pragma: no cover - defensive guard
        return f"Invalid ticker request: {exc}"

    symbol = _sanitize_ticker(ticker)
    if not symbol:
        return "Ticker must be a non-empty string."

    try:
        security = yf.Ticker(symbol)

        price: float | None = None
        currency: str | None = None

        fast_info = getattr(security, "fast_info", None)
        if fast_info:
            price = getattr(fast_info, "last_price", None) or fast_info.get("last_price")
            currency = getattr(fast_info, "currency", None) or fast_info.get("currency")

        if price is None:
            history = security.history(period="1d")
            if history is not None and not history.empty:
                price = float(history["Close"].iloc[-1])

        if price is None:
            return f"No price data available for {symbol}."

        return f"{symbol} price: {_format_currency(price, currency)}"
    except Exception as exc:  # pragma: no cover - network/data errors
        return f"Error fetching price for {symbol}: {exc}"


@tool
def yf_get_history(ticker: str, period: str = "5d", interval: str = "1d") -> str:
    """Return up to the last 10 OHLCV rows for a ticker as plain text."""

    try:
        HistoryRequest(ticker=ticker, period=period, interval=interval)
    except ValidationError as exc:  # pragma: no cover - defensive guard
        return f"Invalid history request: {exc}"

    symbol = _sanitize_ticker(ticker)
    if not symbol:
        return "Ticker must be a non-empty string."

    try:
        security = yf.Ticker(symbol)
        frame = security.history(period=period, interval=interval, auto_adjust=False)
        if frame is None or frame.empty:
            return f"No history for {symbol}."

        tail = frame.tail(10)
        lines: List[str] = [f"History {symbol} ({period}, {interval}):"]
        for index, row in tail.iterrows():
            lines.append(
                " ".join(
                    [
                        str(index.date()),
                        f"O={_fmt_float(row.get('Open'))}",
                        f"H={_fmt_float(row.get('High'))}",
                        f"L={_fmt_float(row.get('Low'))}",
                        f"C={_fmt_float(row.get('Close'))}",
                        f"V={_fmt_volume(row.get('Volume'))}",
                    ]
                )
            )
        return "\n".join(lines)
    except Exception as exc:  # pragma: no cover
        return f"Error fetching history for {symbol}: {exc}"


# ---------------------------------------------------------------------------
# OpenAI-powered analysis tools
# ---------------------------------------------------------------------------


@tool
def run_finance_analysis(question: str, context: str = "") -> str:
    """Call OpenAI to produce a low-temperature financial analysis answer."""

    try:
        FinanceAnalysisRequest(question=question, context=context)
    except ValidationError as exc:  # pragma: no cover - defensive guard
        return f"Invalid analysis request: {exc}"

    try:
        client = _get_openai_client()
    except Exception as exc:  # pragma: no cover - missing credentials
        return f"Analysis unavailable: {exc}"

    prompt_context = context.strip()
    input_text = (
        f"Context:\n{prompt_context}\n\nQuestion:\n{question}" if prompt_context else question
    )

    instructions = (
        "You are a financial analyst. Perform deterministic reasoning over the provided context, "
        "summarise key numeric results, and end with a direct answer. If context is insufficient, "
        "explicitly request the missing data."
    )

    model_name = _env_value("FINANCIAL_ANALYST_REASONING_MODEL", "gpt-5-mini")

    try:
        response = client.responses.create(
            model=model_name,
            instructions=instructions,
            input=input_text,
            temperature=0.0,
        )
    except Exception as exc:  # pragma: no cover - network/API errors
        return f"Analysis error: {exc}"

    return _extract_output_text(response).strip()


@tool
def code_interpreter_tool(question: str, context: str = "") -> str:
    """Execute Python inside the native OpenAI Code Interpreter."""

    try:
        CodeInterpreterRequest(question=question, context=context)
    except ValidationError as exc:  # pragma: no cover
        return f"Invalid code interpreter request: {exc}"

    try:
        client = _get_openai_client()
    except Exception as exc:  # pragma: no cover
        return f"Code Interpreter unavailable: {exc}"

    prompt_context = context.strip()
    input_text = (
        f"Context:\n{prompt_context}\n\nQuestion:\n{question}" if prompt_context else question
    )

    instructions = (
        "You are a financial data analyst. Use the python tool to perform precise calculations, "
        "format numeric results, and return a concise textual summary. Only print the final answer."
    )

    model_name = _env_value("FINANCIAL_ANALYST_CODE_MODEL", "gpt-4.1")

    try:
        response = client.responses.create(
            model=model_name,
            instructions=instructions,
            input=input_text,
            tool_choice="required",
            tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        )
    except Exception as exc:  # pragma: no cover - network/API errors
        return f"Code Interpreter error: {exc}"

    return _extract_output_text(response).strip()


__all__ = [
    "yf_get_price",
    "yf_get_history",
    "run_finance_analysis",
    "code_interpreter_tool",
]
