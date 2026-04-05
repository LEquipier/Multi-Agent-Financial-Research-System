from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import analyze, backtest, portfolio, trade
from src.core.config import settings
from src.core.logging import get_logger, setup_logging

# Import tool modules to trigger @register_tool decorators
import src.tools.indicators  # noqa: F401
import src.tools.market_data  # noqa: F401
import src.tools.news  # noqa: F401
import src.tools.portfolio  # noqa: F401

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("app_startup", version="0.1.0")
    yield
    logger.info("app_shutdown")


app = FastAPI(
    title="Multi-Agent Financial Research System",
    description="Production-grade multi-agent system for financial research & trading simulation",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(analyze.router, prefix="/api/v1", tags=["analysis"])
app.include_router(trade.router, prefix="/api/v1", tags=["trading"])
app.include_router(portfolio.router, prefix="/api/v1", tags=["portfolio"])
app.include_router(backtest.router, prefix="/api/v1", tags=["backtest"])


@app.get("/health")
async def health():
    return {"status": "ok"}
