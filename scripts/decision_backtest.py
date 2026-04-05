"""
Decision-Level Backtest — AAPL 2024-01-01 to 2024-01-15

Runs the full agent pipeline (Analysis → Trading → Risk) on each trading day
using ONLY historical data available up to that date (no future leakage).

Usage:
    conda activate finagent
    python scripts/decision_backtest.py
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
SYMBOL = "AAPL"
# We use the last ~10 trading days from compact history (free tier).
# The exact dates are determined dynamically after fetching.
BACKTEST_DAYS = 10          # how many trading days to simulate
INITIAL_CAPITAL = 100_000.0
MAX_ITERATIONS = 2          # risk re-loop max
LOOKBACK_DAYS = 60          # how many prior days for indicator computation

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _sep(title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def _mini_sep(title: str) -> None:
    print(f"\n  ── {title} {'─'*(56 - len(title))}")


# --------------------------------------------------------------------------- #
# Step 1 — Fetch historical prices (one API call)
# --------------------------------------------------------------------------- #

async def fetch_full_history(symbol: str) -> list[dict[str, Any]]:
    """Fetch full daily price history from Alpha Vantage."""
    import httpx
    from src.core.config import settings

    _sep("FETCHING HISTORICAL PRICE DATA")
    print(f"  Symbol: {symbol}  (Alpha Vantage TIME_SERIES_DAILY, compact)")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "compact",
                "apikey": settings.alpha_vantage_api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    time_series = data.get("Time Series (Daily)", {})
    if not time_series:
        print(f"  ❌ No data returned. Response: {json.dumps(data)[:300]}")
        # Alpha Vantage free tier might return a note or info
        if "Note" in data or "Information" in data:
            msg = data.get("Note") or data.get("Information", "")
            print(f"  ⚠️  {msg}")
            print("  Tip: Alpha Vantage free tier allows 25 requests/day. Wait and retry.")
        sys.exit(1)

    history = []
    for date_str, values in sorted(time_series.items()):
        history.append({
            "date": date_str,
            "open": float(values["1. open"]),
            "high": float(values["2. high"]),
            "low": float(values["3. low"]),
            "close": float(values["4. close"]),
            "volume": int(values["5. volume"]),
        })

    print(f"  ✅ Fetched {len(history)} daily bars  "
          f"({history[0]['date']} → {history[-1]['date']})")
    return history


# --------------------------------------------------------------------------- #
# Step 2 — Compute indicators for a price window
# --------------------------------------------------------------------------- #

def compute_indicators(closing_prices: list[float]) -> dict[str, Any]:
    """Compute all technical indicators locally (no API call)."""
    from src.tools.indicators import (
        compute_bollinger_bands,
        compute_macd,
        compute_rsi,
        compute_sma,
    )

    return {
        "rsi": compute_rsi(closing_prices),
        "sma_20": compute_sma(closing_prices, window=20),
        "sma_50": compute_sma(closing_prices, window=50),
        "macd": compute_macd(closing_prices),
        "bollinger_bands": compute_bollinger_bands(closing_prices),
    }


# --------------------------------------------------------------------------- #
# Step 3 — Run agents for a single day
# --------------------------------------------------------------------------- #

async def run_day_pipeline(
    symbol: str,
    date: str,
    windowed_history: list[dict[str, Any]],
    trading_engine: "TradingEngine",
    iteration: int = 0,
) -> dict[str, Any]:
    """
    Run Analysis → Trading → Risk for one day using only
    historical data up to `date`. Returns the full pipeline result.
    """
    from src.agents.analysis_agent import AnalysisAgent
    from src.agents.base import AgentState
    from src.agents.risk_agent import RiskAgent
    from src.agents.trading_agent import TradingAgent
    from src.observability.tracer import ExecutionTracer
    from src.services.memory import ShortTermMemory
    from src.tools.registry import get_global_registry

    registry = get_global_registry()
    tracer = ExecutionTracer()
    memory = ShortTermMemory()

    today_bar = windowed_history[-1]
    price = today_bar["close"]
    closing_prices = [bar["close"] for bar in windowed_history]

    # Build market_data structure the analysis agent expects
    market_data: dict[str, Any] = {
        "symbol": symbol,
        "current_price": {
            "price": price,
            "open": today_bar["open"],
            "high": today_bar["high"],
            "low": today_bar["low"],
            "volume": today_bar["volume"],
        },
        "price_history": windowed_history,
        "news": [],           # no news API for historical dates
        "company_overview": {},
        "fetched_at": date,
    }

    # Initial state
    state: AgentState = {
        "messages": [],
        "market_data": market_data,
        "analysis": {},
        "trade_signal": None,
        "risk_assessment": None,
        "portfolio": trading_engine.portfolio.model_dump(),
        "metadata": {"symbol": symbol, "backtest_date": date},
        "execution_plan": None,
        "iteration_count": 0,
        "data_sufficient": True,
        "risk_feedback": None,
        "run_id": f"bt-{date}",
        "current_agent": "",
    }

    # Agent instances
    analysis_agent = AnalysisAgent(
        tool_registry=registry, tracer=tracer, memory=memory
    )
    trading_agent = TradingAgent(
        tool_registry=registry, tracer=tracer, memory=memory
    )
    risk_agent = RiskAgent(
        trading_engine=trading_engine,
        tool_registry=registry,
        tracer=tracer,
        memory=memory,
    )

    # Run pipeline with risk feedback loop
    for i in range(MAX_ITERATIONS):
        state = await analysis_agent.process(state)
        state = await trading_agent.process(state)
        state = await risk_agent.process(state)

        risk = state.get("risk_assessment", {})
        decision = risk.get("decision", "approved" if risk.get("approved") else "rejected")

        # If approved or modified, stop iterating
        if decision in ("approved", "modified"):
            break
        # If rejected AND we have iterations left, the risk_agent already
        # set risk_feedback in state → next loop will incorporate it
        if i == MAX_ITERATIONS - 1:
            break  # exhausted iterations

    return state


# --------------------------------------------------------------------------- #
# Step 4 — Main backtest loop
# --------------------------------------------------------------------------- #

async def main() -> None:
    t_start = time.time()

    # --- imports after sys.path is set ---
    import src.tools.market_data   # noqa: F401  (register tools)
    import src.tools.news          # noqa: F401
    import src.tools.indicators    # noqa: F401
    import src.tools.portfolio     # noqa: F401

    from src.core.config import settings
    from src.models.schemas import SimulationConfig
    from src.services.trading_engine import TradingEngine

    # 1. Fetch all price data
    full_history = await fetch_full_history(SYMBOL)

    # 2. Select trading window from available data
    # Use the last BACKTEST_DAYS trading days; the rest is lookback for indicators
    if len(full_history) < LOOKBACK_DAYS + BACKTEST_DAYS:
        print(f"  ⚠️  Only {len(full_history)} bars available. "
              f"Need {LOOKBACK_DAYS + BACKTEST_DAYS}. Reducing lookback.")

    trading_days = full_history[-(BACKTEST_DAYS):]
    start_date = trading_days[0]["date"]
    end_date = trading_days[-1]["date"]

    if not trading_days:
        print(f"  ❌ No trading days found in data")
        sys.exit(1)

    print(f"\n  Backtest window: {start_date} → {end_date}  ({len(trading_days)} days)")
    for td in trading_days:
        print(f"    {td['date']}  close=${td['close']:.2f}")

    # 3. Setup trading engine
    config = SimulationConfig(
        slippage_pct=settings.slippage_pct,
        commission_per_trade=settings.commission_per_trade,
        commission_pct=settings.commission_pct,
        max_position_pct=settings.max_position_pct,
    )
    engine = TradingEngine(initial_capital=INITIAL_CAPITAL, config=config)

    # 4. Run day-by-day
    _sep("DECISION-LEVEL BACKTEST")
    print(f"  Symbol: {SYMBOL}  |  Capital: ${INITIAL_CAPITAL:,.0f}  |  Days: {len(trading_days)}")

    decision_log: list[dict[str, Any]] = []
    daily_equity: list[float] = [INITIAL_CAPITAL]

    for day_bar in trading_days:
        date = day_bar["date"]

        # Window: all history up to and including this day (for indicator lookback)
        windowed = [b for b in full_history if b["date"] <= date]
        # Keep only last LOOKBACK_DAYS+1 bars for memory efficiency
        windowed = windowed[-(LOOKBACK_DAYS + 1):]

        _mini_sep(f"{date}  close=${day_bar['close']:.2f}")

        # Update engine prices BEFORE running pipeline
        engine.update_prices({SYMBOL: day_bar["close"]})

        # Record position before
        pos_before = 0
        if SYMBOL in engine.portfolio.positions:
            pos_before = engine.portfolio.positions[SYMBOL].quantity
        cash_before = engine.portfolio.cash
        equity_before = engine.portfolio.total_value

        # Run agent pipeline
        state = await run_day_pipeline(
            symbol=SYMBOL,
            date=date,
            windowed_history=windowed,
            trading_engine=engine,
        )

        trade_signal = state.get("trade_signal") or {}
        risk_assessment = state.get("risk_assessment") or {}
        action = trade_signal.get("action", "HOLD")
        confidence = trade_signal.get("confidence", 0)
        strategy = trade_signal.get("strategy", "neutral")
        signal_strength = trade_signal.get("signal_strength", "weak")
        risk_decision = risk_assessment.get("decision", "approved" if risk_assessment.get("approved") else "rejected")
        approved = risk_assessment.get("approved", False)
        adjustments = risk_assessment.get("adjustments", {})

        # Execute trade if approved and not HOLD
        trade_record = None
        qty = adjustments.get("position_size", trade_signal.get("suggested_quantity", 0))

        if approved and action == "BUY" and qty > 0:
            trade_record = engine.execute_buy(SYMBOL, qty, day_bar["close"])
        elif approved and action == "SELL" and qty > 0:
            trade_record = engine.execute_sell(SYMBOL, qty, day_bar["close"])

        # Record position after
        pos_after = 0
        if SYMBOL in engine.portfolio.positions:
            pos_after = engine.portfolio.positions[SYMBOL].quantity
        equity_after = engine.portfolio.total_value

        # Build decision record
        record = {
            "date": date,
            "price": day_bar["close"],
            "action": action,
            "confidence": confidence,
            "strategy": strategy,
            "signal_strength": signal_strength,
            "risk_decision": risk_decision,
            "risk_score": risk_assessment.get("risk_score", 0),
            "adjustments": adjustments,
            "position_before": pos_before,
            "position_after": pos_after,
            "cash": round(engine.portfolio.cash, 2),
            "equity": round(equity_after, 2),
            "executed": trade_record is not None,
            "executed_price": trade_record.executed_price if trade_record else None,
            "commission": trade_record.commission if trade_record else 0,
            "trace_id": state.get("run_id", ""),
            "reason": trade_signal.get("reason", "")[:200],
        }
        decision_log.append(record)
        daily_equity.append(equity_after)

        # Print summary
        exec_str = ""
        if trade_record:
            exec_str = f"  EXECUTED @ ${trade_record.executed_price:.2f} (comm ${trade_record.commission:.2f})"
        elif action != "HOLD" and approved:
            exec_str = "  (qty=0, no execution)"
        elif action != "HOLD" and not approved:
            exec_str = f"  BLOCKED by risk ({risk_decision})"

        print(f"    Signal  : {action}  conf={confidence:.2f}  strat={strategy}  strength={signal_strength}")
        print(f"    Risk    : {risk_decision}  score={risk_assessment.get('risk_score', 0):.2f}"
              f"{'  adj=' + str(adjustments) if adjustments else ''}")
        print(f"    Position: {pos_before} → {pos_after}  |  Equity: ${equity_after:,.2f}{exec_str}")

    # ------------------------------------------------------------------ #
    # Step 5 — Compute metrics
    # ------------------------------------------------------------------ #
    _sep("BACKTEST METRICS")

    # Basic returns
    first_price = trading_days[0]["close"]
    last_price = trading_days[-1]["close"]
    buy_hold_return = (last_price - first_price) / first_price * 100
    final_equity = daily_equity[-1]
    agent_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # Max drawdown from equity curve
    peak = daily_equity[0]
    max_dd = 0.0
    for eq in daily_equity:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

    # Sharpe ratio (annualized, using daily returns)
    daily_returns = []
    for i in range(1, len(daily_equity)):
        if daily_equity[i - 1] > 0:
            daily_returns.append(
                (daily_equity[i] - daily_equity[i - 1]) / daily_equity[i - 1]
            )
    if daily_returns:
        import numpy as np
        avg_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 1e-9
        sharpe = (avg_ret / std_ret) * math.sqrt(252) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0

    # Trade-level metrics
    executed_trades = [d for d in decision_log if d["executed"]]
    num_trades = len(executed_trades)

    # Win rate — compare equity change on trade days
    wins = 0
    for i, d in enumerate(decision_log):
        if d["executed"] and i + 1 < len(decision_log):
            next_eq = decision_log[i + 1]["equity"]
            if next_eq > d["equity"]:
                wins += 1
    win_rate = wins / num_trades * 100 if num_trades > 0 else 0

    # Signal accuracy — did the direction call match next-day price movement?
    correct_signals = 0
    total_directional = 0
    for i, d in enumerate(decision_log):
        if d["action"] in ("BUY", "SELL") and i + 1 < len(decision_log):
            total_directional += 1
            next_price = decision_log[i + 1]["price"]
            price_moved_up = next_price > d["price"]
            if (d["action"] == "BUY" and price_moved_up) or (
                d["action"] == "SELL" and not price_moved_up
            ):
                correct_signals += 1
    signal_accuracy = correct_signals / total_directional * 100 if total_directional > 0 else 0

    # Average trade return
    trade_returns = []
    for tr in engine.trade_history:
        if tr.order.side.value == "SELL":
            # realized trade return
            trade_returns.append(
                (tr.executed_price - tr.order.price) / tr.order.price * 100
            )
    avg_trade_return = sum(trade_returns) / len(trade_returns) if trade_returns else 0

    print(f"  {'Metric':<30} {'Value':>15}")
    print(f"  {'─'*46}")
    print(f"  {'Agent Return':<30} {agent_return:>14.2f}%")
    print(f"  {'Buy & Hold Return':<30} {buy_hold_return:>14.2f}%")
    print(f"  {'Alpha (vs B&H)':<30} {agent_return - buy_hold_return:>14.2f}%")
    print(f"  {'Max Drawdown':<30} {max_dd * 100:>14.2f}%")
    print(f"  {'Sharpe Ratio (ann.)':<30} {sharpe:>14.2f}")
    print(f"  {'Win Rate':<30} {win_rate:>14.1f}%")
    print(f"  {'Signal Accuracy':<30} {signal_accuracy:>14.1f}%")
    print(f"  {'Avg Trade Return':<30} {avg_trade_return:>14.2f}%")
    print(f"  {'Number of Trades':<30} {num_trades:>15}")
    print(f"  {'Final Equity':<30} {'$' + f'{final_equity:,.2f}':>15}")
    print(f"  {'First Price':<30} {'$' + f'{first_price:.2f}':>15}")
    print(f"  {'Last Price':<30} {'$' + f'{last_price:.2f}':>15}")

    # ------------------------------------------------------------------ #
    # Step 6 — Decision analysis
    # ------------------------------------------------------------------ #
    _sep("DECISION ANALYSIS")

    # 1. HOLD frequency
    hold_count = sum(1 for d in decision_log if d["action"] == "HOLD")
    buy_count = sum(1 for d in decision_log if d["action"] == "BUY")
    sell_count = sum(1 for d in decision_log if d["action"] == "SELL")
    total_days = len(decision_log)

    _mini_sep("Action Distribution")
    print(f"    HOLD : {hold_count}/{total_days} ({hold_count/total_days*100:.0f}%)")
    print(f"    BUY  : {buy_count}/{total_days} ({buy_count/total_days*100:.0f}%)")
    print(f"    SELL : {sell_count}/{total_days} ({sell_count/total_days*100:.0f}%)")

    # 2. Confidence distribution
    _mini_sep("Confidence Distribution")
    confs = [d["confidence"] for d in decision_log]
    if confs:
        import numpy as np
        print(f"    Min    : {min(confs):.2f}")
        print(f"    Max    : {max(confs):.2f}")
        print(f"    Mean   : {np.mean(confs):.2f}")
        print(f"    Median : {np.median(confs):.2f}")
        # Buckets
        low = sum(1 for c in confs if c < 0.55)
        mid = sum(1 for c in confs if 0.55 <= c < 0.70)
        high = sum(1 for c in confs if c >= 0.70)
        print(f"    <0.55 (weak)     : {low}")
        print(f"    0.55–0.70 (mod)  : {mid}")
        print(f"    ≥0.70 (strong)   : {high}")

    # 3. Risk rejections
    _mini_sep("Risk Decisions")
    risk_approved = sum(1 for d in decision_log if d["risk_decision"] == "approved")
    risk_rejected = sum(1 for d in decision_log if d["risk_decision"] == "rejected")
    risk_modified = sum(1 for d in decision_log if d["risk_decision"] == "modified")
    print(f"    Approved : {risk_approved}")
    print(f"    Rejected : {risk_rejected}")
    print(f"    Modified : {risk_modified}")

    rejected_details = [d for d in decision_log if d["risk_decision"] == "rejected"]
    if rejected_details:
        print(f"\n    Rejection details:")
        for rd in rejected_details:
            print(f"      {rd['date']}: {rd['action']} conf={rd['confidence']:.2f} "
                  f"strength={rd['signal_strength']} risk_score={rd['risk_score']:.2f}")

    # 4. High-confidence trade performance
    _mini_sep("High-Confidence vs Low-Confidence")
    high_conf_trades = [d for d in decision_log if d["executed"] and d["confidence"] >= 0.65]
    low_conf_trades = [d for d in decision_log if d["executed"] and d["confidence"] < 0.65]

    def _next_day_return(log: list, idx: int) -> float | None:
        if idx + 1 < len(decision_log):
            return (decision_log[idx + 1]["price"] - log[idx]["price"]) / log[idx]["price"] * 100
        return None

    if high_conf_trades:
        returns_h = []
        for ht in high_conf_trades:
            idx = decision_log.index(ht)
            r = _next_day_return(decision_log, idx)
            if r is not None:
                returns_h.append(r)
        if returns_h:
            print(f"    High-conf (≥0.65) trades: {len(high_conf_trades)}  "
                  f"avg next-day return: {sum(returns_h)/len(returns_h):.2f}%")
    else:
        print(f"    High-conf (≥0.65) trades: 0")

    if low_conf_trades:
        returns_l = []
        for lt in low_conf_trades:
            idx = decision_log.index(lt)
            r = _next_day_return(decision_log, idx)
            if r is not None:
                returns_l.append(r)
        if returns_l:
            print(f"    Low-conf  (<0.65) trades: {len(low_conf_trades)}  "
                  f"avg next-day return: {sum(returns_l)/len(returns_l):.2f}%")
    else:
        print(f"    Low-conf  (<0.65) trades: 0")

    # ------------------------------------------------------------------ #
    # Step 7 — Summary analysis
    # ------------------------------------------------------------------ #
    _sep("SUMMARY ANALYSIS")

    strengths = []
    weaknesses = []
    patterns = []

    # Assess decision quality
    if hold_count / total_days > 0.6:
        strengths.append("Conservative: system avoids trading when signals are unclear")
    if hold_count / total_days > 0.9:
        weaknesses.append("Overly passive: HOLD on >90% of days — may miss opportunities")

    if risk_rejected > 0:
        strengths.append(f"Risk discipline: rejected {risk_rejected} marginal trades")
    if risk_modified > 0:
        strengths.append(f"Adaptive risk: modified {risk_modified} trades (position/SL adjustments)")

    if num_trades > 0 and win_rate > 50:
        strengths.append(f"Positive edge: win rate {win_rate:.0f}%")
    elif num_trades > 0 and win_rate <= 50:
        weaknesses.append(f"Below-average win rate: {win_rate:.0f}%")

    if agent_return > buy_hold_return:
        strengths.append(f"Outperformed buy-and-hold by {agent_return - buy_hold_return:.2f}%")
    else:
        weaknesses.append(f"Underperformed buy-and-hold by {buy_hold_return - agent_return:.2f}%")

    if max_dd < 0.02:
        strengths.append(f"Low drawdown: {max_dd*100:.2f}%")
    elif max_dd > 0.05:
        weaknesses.append(f"Significant drawdown: {max_dd*100:.2f}%")

    if signal_accuracy > 60:
        strengths.append(f"Directional accuracy: {signal_accuracy:.0f}%")
    elif total_directional > 0 and signal_accuracy < 50:
        weaknesses.append(f"Poor directional accuracy: {signal_accuracy:.0f}%")

    # Patterns
    consecutive_holds = 0
    max_consecutive_holds = 0
    for d in decision_log:
        if d["action"] == "HOLD":
            consecutive_holds += 1
            max_consecutive_holds = max(max_consecutive_holds, consecutive_holds)
        else:
            consecutive_holds = 0
    if max_consecutive_holds >= 3:
        patterns.append(f"Max {max_consecutive_holds} consecutive HOLD days")

    strategies_used = set(d["strategy"] for d in decision_log if d["strategy"] != "neutral")
    if strategies_used:
        patterns.append(f"Strategies deployed: {', '.join(strategies_used)}")
    else:
        patterns.append("Only neutral/no-trade strategy observed — no directional conviction")

    print("\n  STRENGTHS:")
    for s in strengths:
        print(f"    ✅  {s}")
    if not strengths:
        print("    (none identified)")

    print("\n  WEAKNESSES:")
    for w in weaknesses:
        print(f"    ⚠️   {w}")
    if not weaknesses:
        print("    (none identified)")

    print("\n  PATTERNS:")
    for p in patterns:
        print(f"    📊  {p}")

    # ------------------------------------------------------------------ #
    # Step 8 — Full decision log JSON
    # ------------------------------------------------------------------ #
    _sep("FULL DECISION LOG")
    for rec in decision_log:
        # Compact single-line per day
        adj_str = f" adj={rec['adjustments']}" if rec["adjustments"] else ""
        exec_str = f" EXEC@${rec['executed_price']:.2f}" if rec["executed"] else ""
        print(
            f"  {rec['date']}  ${rec['price']:>7.2f}  "
            f"{rec['action']:>4}  conf={rec['confidence']:.2f}  "
            f"strat={rec['strategy']:<15}  "
            f"risk={rec['risk_decision']:<9} "
            f"pos={rec['position_before']}→{rec['position_after']}  "
            f"eq=${rec['equity']:>10,.2f}"
            f"{adj_str}{exec_str}"
        )

    # Save to file
    output_path = Path(__file__).parent.parent / "src" / "data" / "backtest_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "symbol": SYMBOL,
        "period": f"{start_date} → {end_date}",
        "initial_capital": INITIAL_CAPITAL,
        "metrics": {
            "agent_return_pct": round(agent_return, 4),
            "buy_hold_return_pct": round(buy_hold_return, 4),
            "alpha_pct": round(agent_return - buy_hold_return, 4),
            "max_drawdown_pct": round(max_dd * 100, 4),
            "sharpe_ratio": round(sharpe, 4),
            "win_rate_pct": round(win_rate, 2),
            "signal_accuracy_pct": round(signal_accuracy, 2),
            "avg_trade_return_pct": round(avg_trade_return, 4),
            "number_of_trades": num_trades,
            "final_equity": round(final_equity, 2),
        },
        "analysis": {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "patterns": patterns,
        },
        "decision_log": decision_log,
    }
    output_path.write_text(json.dumps(output_data, indent=2, default=str))
    print(f"\n  💾 Results saved to {output_path}")

    elapsed = time.time() - t_start
    print(f"\n  ⏱  Total backtest time: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
