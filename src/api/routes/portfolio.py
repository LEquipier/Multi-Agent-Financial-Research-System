from __future__ import annotations

from fastapi import APIRouter

from src.api.dependencies import get_trading_engine

router = APIRouter()


@router.get("/portfolio")
async def get_portfolio():
    """Get current portfolio state."""
    engine = get_trading_engine()
    return {
        "portfolio": engine.portfolio.model_dump(),
        "pnl": engine.get_pnl(),
        "max_drawdown": engine.get_max_drawdown(),
    }


@router.get("/portfolio/history")
async def get_portfolio_history():
    """Get portfolio value history for charting."""
    engine = get_trading_engine()
    return [s.model_dump() for s in engine.portfolio_history]
