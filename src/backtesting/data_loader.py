from __future__ import annotations

from typing import Any

from src.tools.registry import get_global_registry


async def load_historical_data(
    symbol: str,
    period: str = "compact",
) -> list[dict[str, Any]]:
    """Load historical price data using the registered market data tool."""
    registry = get_global_registry()
    try:
        data = await registry.execute(
            "get_price_history", symbol=symbol, period=period
        )
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []
