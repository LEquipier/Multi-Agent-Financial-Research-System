from __future__ import annotations

from functools import lru_cache

from src.core.config import settings
from src.models.schemas import SimulationConfig
from src.observability.decision_log import DecisionLogger
from src.observability.tracer import ExecutionTracer
from src.services.rag import RAGPipeline
from src.services.trading_engine import TradingEngine
from src.tools.registry import get_global_registry


@lru_cache
def get_trading_engine() -> TradingEngine:
    config = SimulationConfig(
        slippage_pct=settings.slippage_pct,
        commission_per_trade=settings.commission_per_trade,
        commission_pct=settings.commission_pct,
        max_position_pct=settings.max_position_pct,
    )
    engine = TradingEngine(initial_capital=settings.initial_capital, config=config)
    engine.load()
    return engine


@lru_cache
def get_rag_pipeline() -> RAGPipeline:
    return RAGPipeline()


@lru_cache
def get_tracer() -> ExecutionTracer:
    return ExecutionTracer()


@lru_cache
def get_decision_logger() -> DecisionLogger:
    return DecisionLogger()
