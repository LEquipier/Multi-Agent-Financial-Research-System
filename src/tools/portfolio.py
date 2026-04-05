from __future__ import annotations

from typing import Any

from src.models.schemas import (
    OrderSide,
    PortfolioState,
    ToolCategory,
)
from src.tools.registry import register_tool


@register_tool(
    name="get_portfolio_value",
    description="Calculate the total portfolio value given current market prices",
    category=ToolCategory.PORTFOLIO,
    input_schema={
        "type": "object",
        "properties": {
            "portfolio": {"type": "object", "description": "Portfolio state"},
            "current_prices": {
                "type": "object",
                "description": "Mapping of symbol to current price",
            },
        },
        "required": ["portfolio", "current_prices"],
    },
)
def get_portfolio_value(
    portfolio: PortfolioState, current_prices: dict[str, float]
) -> float:
    total = portfolio.cash
    for symbol, position in portfolio.positions.items():
        price = current_prices.get(symbol, position.current_price)
        total += position.quantity * price
    return round(total, 2)


@register_tool(
    name="get_positions",
    description="Get current portfolio positions with market values",
    category=ToolCategory.PORTFOLIO,
    input_schema={
        "type": "object",
        "properties": {
            "portfolio": {"type": "object", "description": "Portfolio state"},
        },
        "required": ["portfolio"],
    },
)
def get_positions(portfolio: PortfolioState) -> list[dict[str, Any]]:
    return [
        {
            "symbol": pos.symbol,
            "quantity": pos.quantity,
            "avg_cost": pos.avg_cost,
            "current_price": pos.current_price,
            "market_value": pos.market_value,
            "unrealized_pnl": pos.unrealized_pnl,
        }
        for pos in portfolio.positions.values()
    ]


@register_tool(
    name="check_position_limit",
    description="Check if a proposed trade would exceed position concentration limits",
    category=ToolCategory.PORTFOLIO,
    input_schema={
        "type": "object",
        "properties": {
            "portfolio": {"type": "object"},
            "symbol": {"type": "string"},
            "side": {"type": "string", "enum": ["BUY", "SELL"]},
            "quantity": {"type": "integer"},
            "price": {"type": "number"},
            "max_position_pct": {"type": "number", "default": 0.2},
        },
        "required": ["portfolio", "symbol", "side", "quantity", "price"],
    },
)
def check_position_limit(
    portfolio: PortfolioState,
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    max_position_pct: float = 0.2,
) -> dict[str, Any]:
    total_value = portfolio.total_value
    if total_value <= 0:
        return {"allowed": False, "reason": "Portfolio value is zero or negative"}

    current_qty = 0
    if symbol in portfolio.positions:
        current_qty = portfolio.positions[symbol].quantity

    if side == OrderSide.BUY:
        new_qty = current_qty + quantity
    else:
        new_qty = current_qty - quantity

    new_position_value = abs(new_qty * price)
    position_pct = new_position_value / total_value

    return {
        "allowed": position_pct <= max_position_pct,
        "position_pct": round(position_pct, 4),
        "max_allowed_pct": max_position_pct,
        "reason": (
            f"Position would be {position_pct:.1%} of portfolio"
            if position_pct > max_position_pct
            else "Within limits"
        ),
    }
