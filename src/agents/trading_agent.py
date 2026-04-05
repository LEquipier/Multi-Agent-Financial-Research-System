from __future__ import annotations

import json
from typing import Any

from langchain_openai import ChatOpenAI

from src.agents.base import AgentState, BaseAgent
from src.core.config import settings
from src.models.schemas import RiskLevel, TradeAction


class TradingAgent(BaseAgent):
    name = "trading_agent"
    role = "Trade signal generation"
    system_prompt = """You are a professional quantitative trading strategist, not a general AI assistant.

Your task is to generate a high-quality trading decision based on structured analysis data.

## CRITICAL: Strategy Context

A strategy selector has already analyzed the market regime and recommended a strategy.
You MUST follow the recommended strategy unless you have overwhelming evidence against it.

Strategy behaviors:
- **momentum**: Trade WITH the trend. BUY in uptrend, SELL in downtrend.
  Only enter when trend is confirmed by MA alignment + MACD direction.
- **mean_reversion**: Trade AGAINST extremes back toward the mean.
  BUY near Bollinger lower band / oversold RSI. SELL near upper band / overbought RSI.
- **breakout**: Enter on confirmed breakout beyond Bollinger bands or key levels.
  Requires strong volume or momentum confirmation.
- **neutral**: No clear regime — default to HOLD unless an edge is obvious.

## Re-entry Logic

If `re_entry_eligible` is true, the system has detected reversal conditions after a recent exit.
In this case:
- Lower the confidence threshold to 0.55 for re-entry
- Prefer smaller position sizes (conservative re-entry)
- The reason field must reference the reversal signals

## Decision Framework

### Signal Strength
- strong → confidence: 0.7 – 0.9
- moderate → confidence: 0.55 – 0.7
- weak → confidence: 0.4 – 0.55
DO NOT default to 0.5.

### Trade Decision Rules
- HOLD only if indicators conflict or no clear edge within the recommended strategy.
- BUY / SELL only if at least 2 indicators align with the recommended strategy AND risk is acceptable.
- In momentum regime: Do NOT mean-revert. Follow the trend.
- In mean_reversion regime: Do NOT chase breakouts. Fade extremes.

### Risk-Aware Output
Define stop_loss and take_profit as ABSOLUTE price levels based on recent support/resistance or Bollinger bands.

## Output Format (STRICT JSON — no markdown, no explanation outside JSON)
{
  "strategy": "momentum | mean_reversion | breakout | neutral",
  "signal_strength": "strong | moderate | weak",
  "action": "BUY | SELL | HOLD",
  "confidence": 0.0,
  "reason": "specific reasoning referencing indicator values AND regime context",
  "risk": "low | medium | high",
  "stop_loss": <absolute price>,
  "take_profit": <absolute price>,
  "suggested_quantity": <integer, 0 if HOLD>
}

## Constraints
- You MUST state which strategy you are following and why.
- Reference actual indicator values (e.g. "RSI 35 + price at BB lower in range regime → BUY").
- If overriding the recommended strategy, explain why with specific numbers.
- If risk_feedback is provided, you MUST address it."""

    async def process(self, state: AgentState) -> AgentState:
        market_data = state.get("market_data", {})
        analysis = state.get("analysis", {})
        risk_feedback = state.get("risk_feedback")
        portfolio = state.get("portfolio", {})

        symbol = market_data.get("symbol", "")
        current_price = market_data.get("current_price", {})
        price = current_price.get("price", 0) if isinstance(current_price, dict) else 0

        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
        )

        feedback_section = ""
        if risk_feedback:
            feedback_section = f"""
⚠️ RISK FEEDBACK (previous iteration — MUST address):
{risk_feedback}"""

        indicators = analysis.get("technical_indicators", {})
        macd = indicators.get("macd", {})
        bb = indicators.get("bollinger_bands", {})

        # Strategy context from StrategySelector
        strategy_ctx = analysis.get("strategy_context", {})
        regime = strategy_ctx.get("regime", "unknown")
        recommended = strategy_ctx.get("recommended_strategy", "neutral")
        regime_signals = strategy_ctx.get("regime_signals", {})
        re_entry = strategy_ctx.get("re_entry_eligible", False)
        re_entry_reason = strategy_ctx.get("re_entry_reason", "")

        prompt = f"""Generate a trading decision for {symbol} @ ${price:.2f}.

── STRATEGY CONTEXT (from regime detector — MUST follow) ──
Market Regime       : {regime}
Recommended Strategy: {recommended}
Regime Signals      : {json.dumps(regime_signals, default=str)}
Re-entry Eligible   : {re_entry}
Re-entry Reason     : {re_entry_reason}

── TECHNICAL INDICATORS ──
RSI-14        : {indicators.get('rsi', 'N/A')}
SMA-20        : {indicators.get('sma_20', 'N/A')}
SMA-50        : {indicators.get('sma_50', 'N/A')}
MACD line     : {macd.get('macd_line', 'N/A')}
MACD signal   : {macd.get('signal_line', 'N/A')}
MACD histogram: {macd.get('histogram', 'N/A')}
Bollinger upper: {bb.get('upper', 'N/A')}
Bollinger mid  : {bb.get('middle', 'N/A')}
Bollinger lower: {bb.get('lower', 'N/A')}

── ANALYSIS CONTEXT ──
Technical Outlook : {analysis.get('technical_outlook', 'N/A')}
Sentiment         : {analysis.get('sentiment', 'N/A')}
Key Factors       : {json.dumps(analysis.get('key_factors', []))}

── PORTFOLIO STATE ──
Cash              : ${portfolio.get('cash', 100000):.2f}
Positions         : {json.dumps(portfolio.get('positions', {}), default=str)[:500]}
{feedback_section}

Respond with ONLY the JSON object. No markdown fences, no explanation outside the JSON."""

        response = await llm.ainvoke(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": prompt}]
        )

        try:
            content = response.content
            if isinstance(content, str):
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    result = json.loads(content[start:end])
                else:
                    result = self._default_hold(symbol)
            else:
                result = self._default_hold(symbol)
        except (json.JSONDecodeError, AttributeError):
            result = self._default_hold(symbol)

        # Normalize and validate
        action = result.get("action", "HOLD").upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"

        confidence = max(0.0, min(1.0, float(result.get("confidence", 0.5))))

        risk_str = result.get("risk", "medium").lower()
        if risk_str not in ("low", "medium", "high", "critical"):
            risk_str = "medium"

        strategy = result.get("strategy", "neutral")
        if strategy not in ("momentum", "mean_reversion", "breakout", "neutral"):
            strategy = "neutral"

        signal_strength = result.get("signal_strength", "weak")
        if signal_strength not in ("strong", "moderate", "weak"):
            signal_strength = "weak"

        # stop_loss / take_profit: prefer absolute prices from LLM,
        # fall back to percentage-based calculation
        def _resolve_price(key: str, pct_key: str, direction: float) -> float | None:
            if price <= 0:
                return None
            val = result.get(key)
            if val is not None and isinstance(val, (int, float)) and val > 0:
                return round(float(val), 2)
            pct = result.get(pct_key)
            if pct is not None:
                return round(price * (1 + direction * float(pct)), 2)
            default_pct = 0.05 if "stop" in key else 0.10
            return round(price * (1 + direction * default_pct), 2)

        signal = {
            "action": action,
            "confidence": round(confidence, 2),
            "reason": result.get("reason", "No clear signal"),
            "risk": risk_str,
            "strategy": strategy,
            "signal_strength": signal_strength,
            "symbol": symbol,
            "suggested_quantity": int(result.get("suggested_quantity", 0)),
            "stop_loss": _resolve_price("stop_loss", "stop_loss_pct", -1),
            "take_profit": _resolve_price("take_profit", "take_profit_pct", 1),
        }

        state["trade_signal"] = signal
        # Clear risk feedback after consuming it
        state["risk_feedback"] = None

        return state

    def _default_hold(self, symbol: str) -> dict[str, Any]:
        return {
            "action": "HOLD",
            "confidence": 0.5,
            "reason": "Insufficient signal clarity",
            "risk": "medium",
            "symbol": symbol,
            "suggested_quantity": 0,
        }
