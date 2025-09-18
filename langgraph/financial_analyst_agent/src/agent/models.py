"""Pydantic models for the financial analyst agent tool inputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PriceRequest(BaseModel):
    """Input schema for the Yahoo Finance price tool."""

    ticker: str = Field(..., description="Equity ticker symbol (e.g., AAPL, MSFT)")


class HistoryRequest(BaseModel):
    """Input schema for the historical OHLCV tool."""

    ticker: str = Field(..., description="Equity ticker symbol (e.g., AAPL, MSFT)")
    period: str = Field(
        "5d",
        description="Lookback window as understood by yfinance (e.g., 5d, 1mo, 6mo)",
    )
    interval: str = Field(
        "1d",
        description="Sampling interval (e.g., 1d, 1h, 5m). Keep it coarse for small contexts.",
    )


class FinanceAnalysisRequest(BaseModel):
    """Input schema for deterministic finance analysis routines."""

    question: str = Field(..., description="Question or instruction about the data")
    context: str = Field(
        "",
        description="Optional contextual data such as prior tool outputs, formatted as text.",
    )


class CodeInterpreterRequest(BaseModel):
    """Input schema for invoking the native OpenAI code interpreter tool."""

    question: str = Field(..., description="Task the python tool should solve")
    context: str = Field(
        "",
        description="Optional tabular or textual data the tool should consider.",
    )
