from __future__ import annotations

import json
from typing import Any

from src.agents.workflow import run_analysis_pipeline
from src.core.config import settings
from src.core.logging import get_logger
from src.models.schemas import PortfolioState, SimulationConfig
from src.services.evaluation import EvaluationEngine
from src.services.trading_engine import TradingEngine

logger = get_logger(__name__)


class BacktestEngine:
    """Run the agent pipeline over historical price data (day by day)."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig(
            slippage_pct=settings.slippage_pct,
            commission_per_trade=settings.commission_per_trade,
            commission_pct=settings.commission_pct,
            max_position_pct=settings.max_position_pct,
        )
        self.evaluator = EvaluationEngine()

    async def run(
        self,
        symbol: str,
        price_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Simulate trading over a sequence of daily prices.

        Each day the pipeline runs, receives a trade signal, and that signal is
        executed on the simulated trading engine.
        """
        engine = TradingEngine(self.config)
        decisions: list[dict[str, Any]] = []

        if not price_history:
            return {"error": "Empty price history"}

        sorted_history = sorted(price_history, key=lambda x: x.get("date", ""))
        first_price = sorted_history[0].get("close", 0)

        for day in sorted_history:
            price = day.get("close", 0)
            date = day.get("date", "unknown")

            if price <= 0:
                continue

            # Update portfolio prices
            engine.update_prices({symbol: price})

            portfolio = engine.portfolio
            portfolio_before = portfolio.model_dump()

            # Run agent pipeline
            try:
                result = await run_analysis_pipeline(
                    symbol=symbol,
                    portfolio=portfolio,
                    trading_engine=engine,
                )
            except Exception as e:
                logger.error("backtest_day_error", date=date, error=str(e))
                continue

            trade_signal = result.get("trade_signal", {})
            action = trade_signal.get("action", "HOLD") if trade_signal else "HOLD"
            risk = result.get("risk_assessment", {})
            approved = risk.get("approved", False) if risk else False

            # Execute approved trades
            if trade_signal and approved and action != "HOLD":
                qty = trade_signal.get("suggested_quantity", 0)
                if action == "BUY" and qty > 0:
                    record = engine.execute_buy(symbol, qty, price)
                    if record:
                        logger.info("backtest_buy", date=date, symbol=symbol, qty=qty, price=price)
                elif action == "SELL" and qty > 0:
                    record = engine.execute_sell(symbol, qty, price)
                    if record:
                        logger.info("backtest_sell", date=date, symbol=symbol, qty=qty, price=price)

            portfolio_after = engine.portfolio.model_dump()

            decisions.append(
                {
                    "date": date,
                    "price": price,
                    "action": action,
                    "approved": approved,
                    "confidence": trade_signal.get("confidence", 0) if trade_signal else 0,
                    "portfolio_value": engine.portfolio.total_value,
                }
            )

        # Final metrics
        pnl = engine.get_pnl()
        max_dd = engine.get_max_drawdown()

        last_price = sorted_history[-1].get("close", first_price) if sorted_history else first_price
        baseline_return = (last_price - first_price) / first_price if first_price > 0 else 0
        comparison = self.evaluator.compare_vs_baseline(
            agent_return_pct=pnl.get("return_pct", 0),
            baseline_return_pct=baseline_return,
        )

        return {
            "symbol": symbol,
            "days_simulated": len(decisions),
            "final_portfolio_value": pnl.get("total_value", 0),
            "total_return_pct": pnl.get("return_pct", 0),
            "max_drawdown": max_dd,
            "vs_baseline": comparison,
            "decisions": decisions,
            "portfolio_history": [s.model_dump() for s in engine.portfolio_history],
        }
