from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.agents.workflow import run_analysis_pipeline
from src.api.dependencies import get_rag_pipeline, get_trading_engine, get_tracer
from src.models.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_symbol(request: AnalyzeRequest):
    """Run the multi-agent analysis pipeline for a stock symbol."""
    engine = get_trading_engine()
    rag = get_rag_pipeline()

    try:
        result = await run_analysis_pipeline(
            symbol=request.symbol,
            trading_engine=engine,
            rag_pipeline=rag,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AnalyzeResponse(
        symbol=request.symbol.upper(),
        trade_signal=result.get("trade_signal"),
        risk_assessment=result.get("risk_assessment"),
        analysis=result.get("analysis", {}),
        trace_id=result.get("metadata", {}).get("trace_id"),
    )


@router.get("/trace/{trace_id}")
async def get_trace(trace_id: str):
    """Retrieve an execution trace by run ID."""
    tracer = get_tracer()
    trace = await tracer.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace
