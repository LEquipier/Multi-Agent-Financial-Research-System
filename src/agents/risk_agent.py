from __future__ import annotations

import json
from typing import Any

from src.agents.base import AgentState, BaseAgent
from src.core.config import settings
from src.models.schemas import OrderSide, PortfolioState
from src.services.trading_engine import TradingEngine


class RiskAgent(BaseAgent):
    name = "risk_agent"
    role = "Risk assessment and trade signal validation"
    system_prompt = """You are an institutional risk management system enforcing trading discipline.

You are NOT allowed to generate new trades. You can ONLY: approve, reject, or modify.

Given a proposed trade signal, evaluate whether it should be executed."""

    def __init__(self, trading_engine: TradingEngine | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.trading_engine = trading_engine

    async def process(self, state: AgentState) -> AgentState:
        trade_signal = state.get("trade_signal")
        portfolio_data = state.get("portfolio", {})
        market_data = state.get("market_data", {})

        if not trade_signal:
            state["risk_assessment"] = {
                "approved": False,
                "risk_score": 1.0,
                "notes": "No trade signal to assess",
                "rejection_reason": "Missing trade signal",
            }
            return state

        action = trade_signal.get("action", "HOLD")
        confidence = trade_signal.get("confidence", 0)
        symbol = trade_signal.get("symbol", "")
        risk_level = trade_signal.get("risk", "medium")
        signal_strength = trade_signal.get("signal_strength", "weak")
        suggested_qty = trade_signal.get("suggested_quantity", 0)
        stop_loss = trade_signal.get("stop_loss")
        take_profit = trade_signal.get("take_profit")

        current_price_data = market_data.get("current_price", {})
        price = current_price_data.get("price", 0) if isinstance(current_price_data, dict) else 0

        # --- Risk checks ---
        rejection_reasons: list[str] = []
        adjustments: dict[str, Any] = {}
        risk_score = 0.0

        # 1. HOLD signals auto-approve
        if action == "HOLD":
            state["risk_assessment"] = {
                "approved": True,
                "decision": "approved",
                "risk_score": 0.0,
                "adjustments": {},
                "notes": "HOLD signal — no risk to assess",
            }
            return state

        # 2. Signal strength gate — reject weak signals outright
        if signal_strength == "weak":
            rejection_reasons.append(
                f"Signal strength is WEAK. Institutional policy: do not trade weak signals."
            )
            risk_score += 0.35

        # 3. Confidence threshold (stricter: 0.55 minimum)
        if confidence < 0.55:
            rejection_reasons.append(
                f"Confidence {confidence:.2f} below minimum 0.55. "
                "Need stronger conviction to execute."
            )
            risk_score += 0.3

        # 4. Risk level check
        if risk_level == "critical":
            rejection_reasons.append(
                "Risk level CRITICAL — trade rejected. "
                "Reduce position or wait for lower-risk entry."
            )
            risk_score += 0.4
        elif risk_level == "high":
            risk_score += 0.2
            # Borderline: modify rather than reject
            if not rejection_reasons:
                # Tighten stop-loss by 30%
                if stop_loss and price > 0:
                    tighter_sl = round(price - (price - stop_loss) * 0.7, 2)
                    adjustments["stop_loss"] = tighter_sl
                # Reduce position by 40%
                if suggested_qty > 0:
                    reduced_qty = max(1, int(suggested_qty * 0.6))
                    adjustments["position_size"] = reduced_qty

        # 5. Position concentration check
        if self.trading_engine and price > 0 and suggested_qty > 0:
            portfolio_value = self.trading_engine.portfolio.total_value
            position_value = suggested_qty * price

            if portfolio_value > 0:
                position_pct = position_value / portfolio_value
                max_pct = settings.max_position_pct

                if position_pct > max_pct:
                    # Calculate compliant quantity instead of outright rejecting
                    max_value = portfolio_value * max_pct
                    compliant_qty = max(1, int(max_value / price))
                    adjustments["position_size"] = min(
                        adjustments.get("position_size", compliant_qty),
                        compliant_qty,
                    )
                    if position_pct > max_pct * 1.5:
                        # Far over limit → reject
                        rejection_reasons.append(
                            f"Position {position_pct:.1%} of portfolio exceeds "
                            f"1.5× limit ({max_pct * 1.5:.1%}). Too concentrated."
                        )
                        risk_score += 0.3
                    else:
                        risk_score += 0.15

        # 6. Portfolio drawdown check
        if self.trading_engine:
            max_dd = self.trading_engine.get_max_drawdown()
            dd_limit = settings.max_position_pct * 0.5
            if max_dd > dd_limit:
                rejection_reasons.append(
                    f"Drawdown {max_dd:.1%} exceeds limit {dd_limit:.1%}. "
                    "Reduce exposure — no new risk until drawdown recovers."
                )
                risk_score += 0.25

        # 7. Insufficient funds check (BUY only)
        if action == "BUY" and self.trading_engine and price > 0 and suggested_qty > 0:
            effective_qty = adjustments.get("position_size", suggested_qty)
            slippage_price = self.trading_engine.apply_slippage(price, OrderSide.BUY)
            total_cost = effective_qty * slippage_price + self.trading_engine.compute_commission(
                effective_qty, slippage_price
            )
            if total_cost > self.trading_engine.portfolio.cash:
                affordable_qty = max(
                    1,
                    int(
                        self.trading_engine.portfolio.cash
                        / (slippage_price * 1.001)  # small buffer for commission
                    ),
                )
                if affordable_qty < 1:
                    rejection_reasons.append(
                        f"Insufficient funds: need ${total_cost:.2f}, "
                        f"have ${self.trading_engine.portfolio.cash:.2f}."
                    )
                    risk_score += 0.3
                else:
                    adjustments["position_size"] = min(
                        adjustments.get("position_size", affordable_qty),
                        affordable_qty,
                    )

        risk_score = min(1.0, round(risk_score, 2))

        # --- Decision logic ---
        # Strict rejection if any hard blockers
        if rejection_reasons:
            decision = "rejected"
            approved = False
        elif adjustments:
            decision = "modified"
            approved = True  # approved with modifications
        else:
            decision = "approved"
            approved = True

        # Compute max allowed position size
        max_position_size = 0
        if self.trading_engine and price > 0:
            max_value = self.trading_engine.portfolio.total_value * settings.max_position_pct
            max_position_size = int(max_value / price)

        # Build notes string
        if decision == "rejected":
            notes = "; ".join(rejection_reasons)
        elif decision == "modified":
            mod_parts = []
            if "position_size" in adjustments:
                mod_parts.append(f"qty {suggested_qty}→{adjustments['position_size']}")
            if "stop_loss" in adjustments:
                mod_parts.append(f"SL tightened to ${adjustments['stop_loss']}")
            notes = f"Approved with modifications: {', '.join(mod_parts)}"
        else:
            notes = "All risk checks passed"

        assessment: dict[str, Any] = {
            "approved": approved,
            "decision": decision,
            "risk_score": risk_score,
            "max_position_size": max_position_size,
            "adjustments": adjustments,
            "notes": notes,
        }

        if not approved:
            assessment["rejection_reason"] = "; ".join(rejection_reasons)
            state["risk_feedback"] = (
                f"RISK REJECTION for {symbol} {action}: " + "; ".join(rejection_reasons)
            )
            state["iteration_count"] = state.get("iteration_count", 0) + 1
        elif adjustments:
            # Apply modifications back to the trade signal
            if "position_size" in adjustments:
                trade_signal["suggested_quantity"] = adjustments["position_size"]
            if "stop_loss" in adjustments:
                trade_signal["stop_loss"] = adjustments["stop_loss"]
            state["trade_signal"] = trade_signal

        state["risk_assessment"] = assessment
        return state
