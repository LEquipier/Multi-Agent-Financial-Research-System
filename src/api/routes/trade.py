from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.dependencies import get_trading_engine
from src.models.schemas import TradeRequest

router = APIRouter()


@router.post("/trade")
async def execute_trade(request: TradeRequest):
    """Manually execute a trade (buy/sell) on the simulation engine."""
    engine = get_trading_engine()

    if request.action.upper() == "BUY":
        record = engine.execute_buy(request.symbol, request.quantity, request.price)
    elif request.action.upper() == "SELL":
        record = engine.execute_sell(request.symbol, request.quantity, request.price)
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")

    if record is None:
        raise HTTPException(status_code=400, detail="Trade execution failed (insufficient funds or shares)")

    engine.save()
    return record.model_dump()
