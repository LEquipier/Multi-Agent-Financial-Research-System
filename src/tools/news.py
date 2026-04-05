from __future__ import annotations

from typing import Any

import httpx

from src.core.config import settings
from src.models.schemas import ToolCategory
from src.tools.registry import register_tool

_FINNHUB_BASE = "https://finnhub.io/api/v1"


@register_tool(
    name="get_market_news",
    description="Get general market news articles",
    category=ToolCategory.NEWS,
    input_schema={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "News category: general, forex, crypto, merger",
                "default": "general",
            }
        },
    },
)
async def get_market_news(category: str = "general") -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_FINNHUB_BASE}/news",
            params={
                "category": category,
                "token": settings.finnhub_api_key,
            },
        )
        resp.raise_for_status()
        articles = resp.json()

    return [
        {
            "headline": a.get("headline", ""),
            "summary": a.get("summary", ""),
            "source": a.get("source", ""),
            "url": a.get("url", ""),
            "published_at": a.get("datetime", 0),
            "category": a.get("category", ""),
        }
        for a in articles[:20]
    ]


@register_tool(
    name="get_company_news",
    description="Get news articles for a specific company within a date range",
    category=ToolCategory.NEWS,
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker symbol"},
            "from_date": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD format",
            },
            "to_date": {
                "type": "string",
                "description": "End date in YYYY-MM-DD format",
            },
        },
        "required": ["symbol", "from_date", "to_date"],
    },
)
async def get_company_news(
    symbol: str, from_date: str, to_date: str
) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_FINNHUB_BASE}/company-news",
            params={
                "symbol": symbol,
                "from": from_date,
                "to": to_date,
                "token": settings.finnhub_api_key,
            },
        )
        resp.raise_for_status()
        articles = resp.json()

    return [
        {
            "headline": a.get("headline", ""),
            "summary": a.get("summary", ""),
            "source": a.get("source", ""),
            "url": a.get("url", ""),
            "published_at": a.get("datetime", 0),
            "related": a.get("related", ""),
        }
        for a in articles[:20]
    ]
