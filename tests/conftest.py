from __future__ import annotations

import os

import pytest

# Override env vars for testing
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("ALPHA_VANTAGE_API_KEY", "test-key")
os.environ.setdefault("FINNHUB_API_KEY", "test-key")


@pytest.fixture
def sample_price_history() -> list[dict]:
    """100 days of synthetic OHLCV data."""
    import random

    random.seed(42)
    data = []
    price = 150.0
    for i in range(100):
        change = random.uniform(-3, 3)
        price = max(50, price + change)
        data.append(
            {
                "date": f"2024-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}",
                "open": round(price - 0.5, 2),
                "high": round(price + 2, 2),
                "low": round(price - 2, 2),
                "close": round(price, 2),
                "volume": random.randint(1_000_000, 10_000_000),
            }
        )
    return data


@pytest.fixture
def closing_prices(sample_price_history) -> list[float]:
    return [d["close"] for d in sample_price_history]


@pytest.fixture
def trading_engine():
    from src.models.schemas import SimulationConfig
    from src.services.trading_engine import TradingEngine

    config = SimulationConfig(
        slippage_pct=0.001,
        commission_per_trade=1.0,
        commission_pct=0.0005,
        max_position_pct=0.20,
    )
    return TradingEngine(initial_capital=100_000, config=config)
