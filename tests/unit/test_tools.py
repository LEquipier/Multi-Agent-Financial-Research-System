from __future__ import annotations

from src.tools.indicators import (
    compute_bollinger_bands,
    compute_ema,
    compute_macd,
    compute_rsi,
    compute_sma,
)


class TestRSI:
    def test_rsi_returns_float(self, closing_prices):
        rsi = compute_rsi(closing_prices)
        assert isinstance(rsi, float)

    def test_rsi_in_range(self, closing_prices):
        rsi = compute_rsi(closing_prices)
        assert 0 <= rsi <= 100

    def test_rsi_insufficient_data(self):
        assert compute_rsi([100, 101]) is None


class TestSMA:
    def test_sma_returns_float(self, closing_prices):
        sma = compute_sma(closing_prices, window=20)
        assert isinstance(sma, float)

    def test_sma_window_larger_than_data(self):
        assert compute_sma([1, 2, 3], window=50) is None


class TestEMA:
    def test_ema_returns_float(self, closing_prices):
        ema = compute_ema(closing_prices, window=12)
        assert isinstance(ema, float)

    def test_ema_insufficient_data(self):
        assert compute_ema([1, 2], window=10) is None


class TestMACD:
    def test_macd_dict(self, closing_prices):
        macd = compute_macd(closing_prices)
        assert "macd_line" in macd
        assert "signal_line" in macd
        assert "histogram" in macd

    def test_macd_insufficient_data(self):
        result = compute_macd([100] * 10)
        assert result is None


class TestBollingerBands:
    def test_bollinger_dict(self, closing_prices):
        bb = compute_bollinger_bands(closing_prices)
        assert "upper" in bb
        assert "middle" in bb
        assert "lower" in bb

    def test_bollinger_ordering(self, closing_prices):
        bb = compute_bollinger_bands(closing_prices)
        assert bb is not None
        assert bb["upper"] >= bb["middle"] >= bb["lower"]

    def test_bollinger_insufficient_data(self):
        assert compute_bollinger_bands([1, 2, 3], window=20) is None
