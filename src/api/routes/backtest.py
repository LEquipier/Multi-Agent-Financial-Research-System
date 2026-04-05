from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.backtesting.data_loader import load_historical_data
from src.backtesting.engine import BacktestEngine
from src.models.schemas import BacktestRequest

router = APIRouter()


@router.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """Run a backtest over historical price data."""
    price_history = await load_historical_data(request.symbol, period="full")

    if not price_history:
        raise HTTPException(status_code=400, detail="Failed to load historical data")

    # Trim to requested days
    if request.days and len(price_history) > request.days:
        price_history = price_history[-request.days :]

    engine = BacktestEngine()
    result = await engine.run(request.symbol, price_history)

    return result
