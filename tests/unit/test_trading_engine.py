from __future__ import annotations

import pytest


class TestTradingEngine:
    def test_initial_state(self, trading_engine):
        assert trading_engine.portfolio.cash == 100_000
        assert len(trading_engine.portfolio.positions) == 0

    def test_buy_success(self, trading_engine):
        record = trading_engine.execute_buy("AAPL", 10, 150.0)
        assert record is not None
        assert record.order.symbol == "AAPL"
        assert record.order.quantity == 10
        assert "AAPL" in trading_engine.portfolio.positions

    def test_buy_insufficient_funds(self, trading_engine):
        record = trading_engine.execute_buy("AAPL", 10_000, 150.0)
        assert record is None

    def test_sell_success(self, trading_engine):
        trading_engine.execute_buy("AAPL", 10, 150.0)
        record = trading_engine.execute_sell("AAPL", 5, 160.0)
        assert record is not None
        assert record.order.quantity == 5

    def test_sell_insufficient_shares(self, trading_engine):
        trading_engine.execute_buy("AAPL", 10, 150.0)
        record = trading_engine.execute_sell("AAPL", 20, 160.0)
        assert record is None

    def test_sell_no_position(self, trading_engine):
        record = trading_engine.execute_sell("AAPL", 5, 150.0)
        assert record is None

    def test_slippage(self, trading_engine):
        buy_price = trading_engine.apply_slippage(100.0, "BUY")
        sell_price = trading_engine.apply_slippage(100.0, "SELL")
        assert buy_price > 100.0
        assert sell_price < 100.0

    def test_commission(self, trading_engine):
        commission = trading_engine.compute_commission(10, 150.0)
        assert commission > 0

    def test_pnl(self, trading_engine):
        pnl = trading_engine.get_pnl()
        assert pnl["total_value"] == 100_000
        assert pnl["return_pct"] == 0.0

    def test_max_drawdown_initial(self, trading_engine):
        dd = trading_engine.get_max_drawdown()
        assert dd == 0.0

    def test_update_prices(self, trading_engine):
        trading_engine.execute_buy("AAPL", 10, 150.0)
        trading_engine.update_prices({"AAPL": 160.0})
        pos = trading_engine.portfolio.positions["AAPL"]
        assert pos.current_price == 160.0

    def test_position_sizing(self, trading_engine):
        size = trading_engine.compute_position_size("AAPL", 150.0, 0.7)
        assert isinstance(size, int)
        assert size > 0
