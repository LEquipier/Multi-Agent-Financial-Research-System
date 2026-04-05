from __future__ import annotations

from typing import Any


class EvaluationEngine:
    """Evaluate agent pipeline outputs against quantitative metrics."""

    def evaluate(
        self,
        trade_signal: dict[str, Any],
        risk_assessment: dict[str, Any],
        portfolio_before: dict[str, Any],
        portfolio_after: dict[str, Any],
    ) -> dict[str, Any]:
        signal_quality = self._score_signal(trade_signal)
        risk_quality = self._score_risk(risk_assessment, trade_signal)
        portfolio_impact = self._compute_portfolio_impact(portfolio_before, portfolio_after)

        overall = round(0.4 * signal_quality + 0.3 * risk_quality + 0.3 * portfolio_impact, 3)

        return {
            "signal_quality_score": round(signal_quality, 3),
            "risk_quality_score": round(risk_quality, 3),
            "portfolio_impact_score": round(portfolio_impact, 3),
            "overall_score": overall,
        }

    def _score_signal(self, signal: dict[str, Any]) -> float:
        if not signal:
            return 0.0

        confidence = signal.get("confidence", 0)
        has_stop_loss = signal.get("stop_loss") is not None
        has_take_profit = signal.get("take_profit") is not None
        has_reason = bool(signal.get("reason", ""))

        score = confidence * 0.5
        if has_stop_loss:
            score += 0.15
        if has_take_profit:
            score += 0.15
        if has_reason:
            score += 0.2

        return min(1.0, score)

    def _score_risk(self, risk: dict[str, Any], signal: dict[str, Any]) -> float:
        if not risk:
            return 0.0

        risk_score = risk.get("risk_score", 1.0)
        approved = risk.get("approved", False)
        action = signal.get("action", "HOLD")

        # HOLD with low risk is good
        if action == "HOLD" and approved:
            return 0.7

        # Approved with low risk score is great
        if approved:
            return 1.0 - risk_score * 0.5

        # Rejected trade is still a valid risk outcome
        return 0.3

    def _compute_portfolio_impact(
        self, before: dict[str, Any], after: dict[str, Any]
    ) -> float:
        if not before or not after:
            return 0.5

        total_before = before.get("total_value", 0) or (
            before.get("cash", 0) + sum(
                p.get("shares", 0) * p.get("avg_cost", 0)
                for p in before.get("positions", {}).values()
            )
        )
        total_after = after.get("total_value", 0) or (
            after.get("cash", 0) + sum(
                p.get("shares", 0) * p.get("avg_cost", 0)
                for p in after.get("positions", {}).values()
            )
        )

        if total_before == 0:
            return 0.5

        change_pct = (total_after - total_before) / total_before
        # Normalize: -0.1 → 0, 0 → 0.5, +0.1 → 1.0
        return max(0.0, min(1.0, 0.5 + change_pct * 5))

    def compare_vs_baseline(
        self,
        agent_return_pct: float,
        baseline_return_pct: float,
    ) -> dict[str, Any]:
        """Compare agent performance vs. buy-and-hold baseline."""
        alpha = agent_return_pct - baseline_return_pct
        outperformed = alpha > 0

        return {
            "agent_return_pct": round(agent_return_pct, 4),
            "baseline_return_pct": round(baseline_return_pct, 4),
            "alpha": round(alpha, 4),
            "outperformed": outperformed,
        }
