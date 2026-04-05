from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from src.core.config import settings
from src.models.schemas import (
    Order,
    OrderSide,
    PortfolioSnapshot,
    PortfolioState,
    Position,
    SimulationConfig,
    TradeRecord,
)


class TradingEngine:
    def __init__(
        self,
        initial_capital: float | None = None,
        config: SimulationConfig | None = None,
        persistence_path: Path | None = None,
    ) -> None:
        capital = initial_capital or settings.initial_capital
        self.config = config or SimulationConfig(
            slippage_pct=settings.slippage_pct,
            commission_per_trade=settings.commission_per_trade,
            commission_pct=settings.commission_pct,
            max_position_pct=settings.max_position_pct,
        )
        self.portfolio = PortfolioState(
            cash=capital, positions={}, initial_capital=capital
        )
        self.trade_history: list[TradeRecord] = []
        self.portfolio_history: list[PortfolioSnapshot] = []
        self._peak_value: float = capital
        self._persistence_path = persistence_path or settings.portfolio_path

        self._record_snapshot()

    # ── Slippage & commission ──────────────────

    def apply_slippage(self, price: float, side: OrderSide) -> float:
        if side == OrderSide.BUY:
            return round(price * (1 + self.config.slippage_pct), 4)
        return round(price * (1 - self.config.slippage_pct), 4)

    def compute_commission(self, quantity: int, price: float) -> float:
        pct_fee = quantity * price * self.config.commission_pct
        return round(self.config.commission_per_trade + pct_fee, 4)

    # ── Position sizing ────────────────────────

    def compute_position_size(
        self, symbol: str, confidence: float, price: float
    ) -> int:
        if price <= 0 or confidence <= 0:
            return 0

        max_value = self.portfolio.total_value * self.config.max_position_pct
        current_value = 0.0
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            current_value = pos.quantity * price

        available = max_value - current_value
        if available <= 0:
            return 0

        risk_adjusted = available * min(confidence, 1.0)
        kelly_fraction = max(0.25, confidence - 0.1)
        budget = risk_adjusted * kelly_fraction

        quantity = int(budget / price)
        max_affordable = int(self.portfolio.cash / (price * (1 + self.config.slippage_pct) + self.config.commission_per_trade / max(quantity, 1)))
        return max(0, min(quantity, max_affordable))

    # ── Order execution ────────────────────────

    def execute_buy(self, symbol: str, quantity: int, price: float) -> TradeRecord | None:
        executed_price = self.apply_slippage(price, OrderSide.BUY)
        commission = self.compute_commission(quantity, executed_price)
        total_cost = quantity * executed_price + commission
        slippage = (executed_price - price) * quantity

        if total_cost > self.portfolio.cash:
            return None

        position_value = quantity * executed_price
        if self.portfolio.total_value > 0:
            existing = 0.0
            if symbol in self.portfolio.positions:
                existing = self.portfolio.positions[symbol].quantity * executed_price
            total_pct = (existing + position_value) / self.portfolio.total_value
            if total_pct > self.config.max_position_pct:
                return None

        self.portfolio.cash -= total_cost

        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            total_quantity = pos.quantity + quantity
            avg_cost = (pos.avg_cost * pos.quantity + executed_price * quantity) / total_quantity
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=total_quantity,
                avg_cost=round(avg_cost, 4),
                current_price=executed_price,
            )
        else:
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=executed_price,
                current_price=executed_price,
            )

        order = Order(symbol=symbol, side=OrderSide.BUY, quantity=quantity, price=price)
        record = TradeRecord(
            order=order,
            executed_price=executed_price,
            commission=commission,
            slippage=round(slippage, 4),
            total_cost=round(total_cost, 4),
        )
        self.trade_history.append(record)
        self._record_snapshot()
        return record

    def execute_sell(self, symbol: str, quantity: int, price: float) -> TradeRecord | None:
        if symbol not in self.portfolio.positions:
            return None

        position = self.portfolio.positions[symbol]
        if quantity > position.quantity:
            return None

        executed_price = self.apply_slippage(price, OrderSide.SELL)
        commission = self.compute_commission(quantity, executed_price)
        proceeds = quantity * executed_price - commission
        slippage = (price - executed_price) * quantity

        self.portfolio.cash += proceeds

        remaining = position.quantity - quantity
        if remaining == 0:
            del self.portfolio.positions[symbol]
        else:
            self.portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=remaining,
                avg_cost=position.avg_cost,
                current_price=executed_price,
            )

        order = Order(symbol=symbol, side=OrderSide.SELL, quantity=quantity, price=price)
        record = TradeRecord(
            order=order,
            executed_price=executed_price,
            commission=commission,
            slippage=round(slippage, 4),
            total_cost=round(proceeds, 4),
        )
        self.trade_history.append(record)
        self._record_snapshot()
        return record

    # ── PnL & metrics ─────────────────────────

    def get_pnl(self) -> dict:
        realized = 0.0
        for trade in self.trade_history:
            if trade.order.side == OrderSide.SELL:
                realized += trade.total_cost - trade.commission

        unrealized = sum(pos.unrealized_pnl for pos in self.portfolio.positions.values())
        total_value = self.portfolio.total_value
        total_return = total_value - self.portfolio.initial_capital

        return {
            "total_value": round(total_value, 2),
            "cash": round(self.portfolio.cash, 2),
            "unrealized_pnl": round(unrealized, 2),
            "total_return": round(total_return, 2),
            "return_pct": round(
                (total_return / self.portfolio.initial_capital) * 100, 2
            )
            if self.portfolio.initial_capital > 0
            else 0.0,
        }

    def get_max_drawdown(self) -> float:
        if not self.portfolio_history:
            return 0.0

        peak = self.portfolio_history[0].total_value
        max_dd = 0.0

        for snapshot in self.portfolio_history:
            if snapshot.total_value > peak:
                peak = snapshot.total_value
            if peak > 0:
                dd = (peak - snapshot.total_value) / peak
                max_dd = max(max_dd, dd)

        return round(max_dd, 4)

    def get_trade_history(self) -> list[TradeRecord]:
        return self.trade_history

    # ── Snapshot & persistence ─────────────────

    def _record_snapshot(self) -> None:
        positions_value = sum(
            p.quantity * p.current_price for p in self.portfolio.positions.values()
        )
        snapshot = PortfolioSnapshot(
            total_value=round(self.portfolio.cash + positions_value, 2),
            cash=round(self.portfolio.cash, 2),
            positions_value=round(positions_value, 2),
        )
        self.portfolio_history.append(snapshot)

    def update_prices(self, current_prices: dict[str, float]) -> None:
        for symbol, price in current_prices.items():
            if symbol in self.portfolio.positions:
                pos = self.portfolio.positions[symbol]
                self.portfolio.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=pos.quantity,
                    avg_cost=pos.avg_cost,
                    current_price=price,
                )

    def save(self) -> None:
        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "portfolio": self.portfolio.model_dump(mode="json"),
            "trade_count": len(self.trade_history),
        }
        self._persistence_path.write_text(json.dumps(data, indent=2, default=str))

    def load(self) -> None:
        if not self._persistence_path.exists():
            return
        data = json.loads(self._persistence_path.read_text())
        self.portfolio = PortfolioState(**data["portfolio"])
