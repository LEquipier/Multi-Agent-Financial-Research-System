from __future__ import annotations

from typing import Any

import httpx

from src.core.config import settings
from src.models.schemas import ToolCategory
from src.tools.registry import register_tool

_ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
_FINNHUB_BASE = "https://finnhub.io/api/v1"


@register_tool(
    name="get_stock_price",
    description="Get the current stock price for a given symbol",
    category=ToolCategory.MARKET_DATA,
    input_schema={
        "type": "object",
        "properties": {"symbol": {"type": "string", "description": "Stock ticker symbol"}},
        "required": ["symbol"],
    },
)
async def get_stock_price(symbol: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            _ALPHA_VANTAGE_BASE,
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": settings.alpha_vantage_api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    quote = data.get("Global Quote", {})
    if not quote:
        return {"symbol": symbol, "price": None, "error": "No data returned"}

    return {
        "symbol": symbol,
        "price": float(quote.get("05. price", 0)),
        "open": float(quote.get("02. open", 0)),
        "high": float(quote.get("03. high", 0)),
        "low": float(quote.get("04. low", 0)),
        "volume": int(quote.get("06. volume", 0)),
        "previous_close": float(quote.get("08. previous close", 0)),
        "change_pct": quote.get("10. change percent", "0%"),
    }


@register_tool(
    name="get_price_history",
    description="Get daily OHLCV price history for a stock symbol",
    category=ToolCategory.MARKET_DATA,
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker symbol"},
            "period": {
                "type": "string",
                "description": "compact (100 days) or full",
                "default": "compact",
            },
        },
        "required": ["symbol"],
    },
)
async def get_price_history(
    symbol: str, period: str = "compact"
) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            _ALPHA_VANTAGE_BASE,
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": period,
                "apikey": settings.alpha_vantage_api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    time_series = data.get("Time Series (Daily)", {})
    history = []
    for date_str, values in sorted(time_series.items()):
        history.append({
            "date": date_str,
            "open": float(values["1. open"]),
            "high": float(values["2. high"]),
            "low": float(values["3. low"]),
            "close": float(values["4. close"]),
            "volume": int(values["5. volume"]),
        })
    return history


@register_tool(
    name="get_company_overview",
    description="Get fundamental company data including P/E, market cap, etc.",
    category=ToolCategory.MARKET_DATA,
    input_schema={
        "type": "object",
        "properties": {"symbol": {"type": "string", "description": "Stock ticker symbol"}},
        "required": ["symbol"],
    },
)
async def get_company_overview(symbol: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            _ALPHA_VANTAGE_BASE,
            params={
                "function": "OVERVIEW",
                "symbol": symbol,
                "apikey": settings.alpha_vantage_api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    if not data or "Symbol" not in data:
        return {"symbol": symbol, "error": "No overview data"}

    return {
        "symbol": data.get("Symbol"),
        "name": data.get("Name"),
        "sector": data.get("Sector"),
        "industry": data.get("Industry"),
        "market_cap": data.get("MarketCapitalization"),
        "pe_ratio": data.get("PERatio"),
        "eps": data.get("EPS"),
        "dividend_yield": data.get("DividendYield"),
        "52_week_high": data.get("52WeekHigh"),
        "52_week_low": data.get("52WeekLow"),
        "beta": data.get("Beta"),
        "profit_margin": data.get("ProfitMargin"),
    }
