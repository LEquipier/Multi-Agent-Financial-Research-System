from __future__ import annotations

from src.services.evaluation import EvaluationEngine


class TestEvaluation:
    def setup_method(self):
        self.engine = EvaluationEngine()

    def test_evaluate_basic(self):
        signal = {"action": "BUY", "confidence": 0.8, "reason": "test", "stop_loss": 140, "take_profit": 170}
        risk = {"approved": True, "risk_score": 0.1}
        before = {"cash": 100000, "positions": {}}
        after = {"cash": 85000, "positions": {"AAPL": {"shares": 100, "avg_cost": 150}}}

        result = self.engine.evaluate(signal, risk, before, after)
        assert "overall_score" in result
        assert 0 <= result["overall_score"] <= 1

    def test_evaluate_no_signal(self):
        result = self.engine.evaluate({}, {}, {}, {})
        assert result["signal_quality_score"] == 0.0

    def test_compare_vs_baseline_outperform(self):
        result = self.engine.compare_vs_baseline(0.15, 0.10)
        assert result["outperformed"] is True
        assert result["alpha"] > 0

    def test_compare_vs_baseline_underperform(self):
        result = self.engine.compare_vs_baseline(0.05, 0.10)
        assert result["outperformed"] is False
        assert result["alpha"] < 0
