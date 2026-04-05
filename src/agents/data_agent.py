from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

from src.agents.base import AgentState, BaseAgent
from src.models.schemas import ToolCategory


class DataAgent(BaseAgent):
    name = "data_agent"
    role = "Market data and news fetcher"
    system_prompt = """You are a financial data agent. Your job is to fetch real-time market data
including stock prices, price history, company fundamentals, and relevant news.
You fetch data in parallel for efficiency and validate that sufficient data was returned."""

    async def process(self, state: AgentState) -> AgentState:
        symbol = state.get("metadata", {}).get("symbol", "")
        if not symbol:
            state["data_sufficient"] = False
            return state

        today = datetime.utcnow().strftime("%Y-%m-%d")
        month_ago = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Parallel fetch
        results = await asyncio.gather(
            self._fetch_price(symbol),
            self._fetch_history(symbol),
            self._fetch_news(symbol, month_ago, today),
            self._fetch_overview(symbol),
            return_exceptions=True,
        )

        price_data, history_data, news_data, overview_data = results

        # Validate results
        market_data: dict[str, Any] = {
            "symbol": symbol,
            "current_price": None,
            "price_history": [],
            "news": [],
            "company_overview": {},
            "fetched_at": datetime.utcnow().isoformat(),
        }

        data_sufficient = True

        if isinstance(price_data, dict) and price_data.get("price"):
            market_data["current_price"] = price_data
        else:
            data_sufficient = False

        if isinstance(history_data, list) and len(history_data) > 0:
            market_data["price_history"] = history_data
        else:
            data_sufficient = False

        if isinstance(news_data, list):
            market_data["news"] = news_data

        if isinstance(overview_data, dict) and "error" not in overview_data:
            market_data["company_overview"] = overview_data

        state["market_data"] = market_data
        state["data_sufficient"] = data_sufficient

        return state

    async def _fetch_price(self, symbol: str) -> Any:
        try:
            return await self.tool_registry.execute("get_stock_price", symbol=symbol)
        except Exception:
            return {"error": "Failed to fetch price"}

    async def _fetch_history(self, symbol: str) -> Any:
        try:
            return await self.tool_registry.execute(
                "get_price_history", symbol=symbol, period="compact"
            )
        except Exception:
            return []

    async def _fetch_news(self, symbol: str, from_date: str, to_date: str) -> Any:
        try:
            return await self.tool_registry.execute(
                "get_company_news", symbol=symbol, from_date=from_date, to_date=to_date
            )
        except Exception:
            return []

    async def _fetch_overview(self, symbol: str) -> Any:
        try:
            return await self.tool_registry.execute(
                "get_company_overview", symbol=symbol
            )
        except Exception:
            return {"error": "Failed to fetch overview"}
