"""
Strategy Selector — deterministic market-regime classifier.

Runs AFTER AnalysisAgent and BEFORE TradingAgent.
Reads technical indicators from state["analysis"]["technical_indicators"]
and writes state["analysis"]["strategy_context"] with:

  - regime: trend_up | trend_down | range | breakout | unknown
  - recommended_strategy: momentum | mean_reversion | breakout | neutral
  - regime_signals: dict of evidence used
  - re_entry_eligible: bool (whether re-entry conditions are met)
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AgentState, BaseAgent


class StrategySelector(BaseAgent):
    name = "strategy_selector"
    role = "Market regime detection and strategy selection"
    system_prompt = ""  # Not used — this is a pure rules-based component.

    async def process(self, state: AgentState) -> AgentState:
        analysis = state.get("analysis", {})
        indicators = analysis.get("technical_indicators", {})
        market_data = state.get("market_data", {})
        portfolio = state.get("portfolio", {})

        current_price_data = market_data.get("current_price", {})
        price = (
            current_price_data.get("price", 0)
            if isinstance(current_price_data, dict)
            else 0
        )

        regime, signals = self._detect_regime(indicators, price)
        strategy = self._select_strategy(regime)
        re_entry = self._check_re_entry(indicators, price, portfolio, market_data)

        analysis["strategy_context"] = {
            "regime": regime,
            "recommended_strategy": strategy,
            "regime_signals": signals,
            "re_entry_eligible": re_entry["eligible"],
            "re_entry_reason": re_entry["reason"],
        }
        state["analysis"] = analysis
        return state

    # ------------------------------------------------------------------ #
    # Regime detection
    # ------------------------------------------------------------------ #

    def _detect_regime(
        self, indicators: dict[str, Any], price: float
    ) -> tuple[str, dict[str, Any]]:
        """Classify market regime from technical indicators."""

        rsi = indicators.get("rsi")
        sma_20 = indicators.get("sma_20")
        sma_50 = indicators.get("sma_50")
        macd = indicators.get("macd") or {}
        bb = indicators.get("bollinger_bands") or {}

        macd_hist = macd.get("histogram")
        macd_line = macd.get("macd_line")
        macd_signal = macd.get("signal_line")
        bb_upper = bb.get("upper")
        bb_lower = bb.get("lower")
        bb_mid = bb.get("middle")

        signals: dict[str, Any] = {}
        scores = {"trend_up": 0, "trend_down": 0, "range": 0, "breakout": 0}

        # --- 1. Moving average alignment ---
        if sma_20 is not None and sma_50 is not None and price > 0:
            if price > sma_20 > sma_50:
                scores["trend_up"] += 2
                signals["ma_alignment"] = "bullish_stack"
            elif price < sma_20 < sma_50:
                scores["trend_down"] += 2
                signals["ma_alignment"] = "bearish_stack"
            elif abs(sma_20 - sma_50) / sma_50 < 0.01 if sma_50 else False:
                scores["range"] += 2
                signals["ma_alignment"] = "converged"
            else:
                scores["range"] += 1
                signals["ma_alignment"] = "mixed"

        # --- 2. RSI regime ---
        if rsi is not None:
            if rsi > 60:
                scores["trend_up"] += 1
                signals["rsi_regime"] = "bullish_momentum"
            elif rsi < 40:
                scores["trend_down"] += 1
                signals["rsi_regime"] = "bearish_momentum"
            else:
                scores["range"] += 1
                signals["rsi_regime"] = "neutral"

            # Extreme RSI → potential reversal/breakout
            if rsi > 75 or rsi < 25:
                scores["breakout"] += 1
                signals["rsi_extreme"] = True

        # --- 3. MACD momentum ---
        if macd_hist is not None and macd_line is not None:
            if macd_line > 0 and macd_hist > 0:
                scores["trend_up"] += 1
                signals["macd_regime"] = "bullish_accelerating"
            elif macd_line < 0 and macd_hist < 0:
                scores["trend_down"] += 1
                signals["macd_regime"] = "bearish_accelerating"
            elif macd_line < 0 and macd_hist > 0:
                # MACD turning up from below → potential reversal
                scores["trend_up"] += 1
                scores["breakout"] += 1
                signals["macd_regime"] = "bullish_crossover_pending"
            elif macd_line > 0 and macd_hist < 0:
                scores["trend_down"] += 1
                signals["macd_regime"] = "bearish_divergence"
            else:
                scores["range"] += 1
                signals["macd_regime"] = "flat"

        # --- 4. Bollinger Band position ---
        if bb_upper is not None and bb_lower is not None and price > 0:
            bb_width = bb_upper - bb_lower
            bb_pct = (price - bb_lower) / bb_width if bb_width > 0 else 0.5

            signals["bb_position"] = round(bb_pct, 2)

            if bb_pct > 0.95:
                scores["breakout"] += 1
                scores["trend_up"] += 1
                signals["bb_regime"] = "upper_breakout"
            elif bb_pct < 0.05:
                scores["breakout"] += 1
                scores["trend_down"] += 1
                signals["bb_regime"] = "lower_breakout"
            elif 0.3 < bb_pct < 0.7:
                scores["range"] += 1
                signals["bb_regime"] = "mid_range"
            elif bb_pct >= 0.7:
                scores["trend_up"] += 1
                signals["bb_regime"] = "upper_half"
            else:
                scores["trend_down"] += 1
                signals["bb_regime"] = "lower_half"

            # Narrow bands → squeeze → breakout potential
            if bb_mid and bb_mid > 0:
                bb_pct_width = bb_width / bb_mid
                if bb_pct_width < 0.04:
                    scores["breakout"] += 2
                    signals["bb_squeeze"] = True

        # --- 5. Price momentum (price vs SMA-20 as proxy) ---
        if sma_20 and price > 0:
            momentum_pct = (price - sma_20) / sma_20
            signals["price_momentum_pct"] = round(momentum_pct * 100, 2)
            if momentum_pct > 0.03:
                scores["trend_up"] += 1
            elif momentum_pct < -0.03:
                scores["trend_down"] += 1
            else:
                scores["range"] += 1

        # --- Determine winner ---
        regime = max(scores, key=scores.get)  # type: ignore[arg-type]

        # Require minimum signal conviction
        max_score = scores[regime]
        if max_score < 2:
            regime = "unknown"

        signals["regime_scores"] = scores
        return regime, signals

    def _select_strategy(self, regime: str) -> str:
        """Map regime to recommended strategy."""
        mapping = {
            "trend_up": "momentum",
            "trend_down": "momentum",  # momentum works both directions
            "range": "mean_reversion",
            "breakout": "breakout",
            "unknown": "neutral",
        }
        return mapping.get(regime, "neutral")

    # ------------------------------------------------------------------ #
    # Re-entry logic
    # ------------------------------------------------------------------ #

    def _check_re_entry(
        self,
        indicators: dict[str, Any],
        price: float,
        portfolio: dict[str, Any],
        market_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Detect strong reversal conditions that warrant re-entry after exit.

        Triggers when:
        - No current position (exited recently)
        - RSI bounced from oversold (< 35) back above 40
        - MACD histogram turned positive (bullish crossover)
        - Price reclaimed SMA-20
        """
        positions = portfolio.get("positions", {})
        symbol = market_data.get("symbol", "")

        # Only relevant if we have no position
        has_position = symbol in positions and positions[symbol].get("quantity", 0) > 0
        if has_position:
            return {"eligible": False, "reason": "Already holding position"}

        rsi = indicators.get("rsi")
        sma_20 = indicators.get("sma_20")
        macd = indicators.get("macd") or {}
        macd_hist = macd.get("histogram")
        macd_line = macd.get("macd_line")

        reversal_signals = 0
        reasons = []

        # RSI bounce from oversold
        if rsi is not None and 40 <= rsi <= 55:
            reversal_signals += 1
            reasons.append(f"RSI {rsi:.1f} recovered from oversold zone")

        # MACD bullish crossover
        if macd_hist is not None and macd_line is not None:
            if macd_hist > 0 and macd_line < 0:
                reversal_signals += 1
                reasons.append(
                    f"MACD histogram +{macd_hist:.4f} turning bullish while line still negative"
                )

        # Price reclaimed SMA-20
        if sma_20 is not None and price > 0:
            if price > sma_20 and (price - sma_20) / sma_20 < 0.02:
                reversal_signals += 1
                reasons.append(f"Price ${price:.2f} just reclaimed SMA-20 ${sma_20:.2f}")

        eligible = reversal_signals >= 2
        reason = "; ".join(reasons) if reasons else "No reversal signals detected"

        return {"eligible": eligible, "reason": reason}
