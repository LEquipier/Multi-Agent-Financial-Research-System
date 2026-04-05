from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class TradeAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class MessageType(str, Enum):
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    FEEDBACK = "feedback"


class PlanStepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ToolCategory(str, Enum):
    MARKET_DATA = "market_data"
    NEWS = "news"
    INDICATOR = "indicator"
    PORTFOLIO = "portfolio"


# ──────────────────────────────────────────────
# Agent communication
# ──────────────────────────────────────────────

class AgentMessage(BaseModel):
    sender: str
    receiver: str
    content: dict[str, Any] = Field(default_factory=dict)
    message_type: MessageType = MessageType.RESULT
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────
# Market data
# ──────────────────────────────────────────────

class MarketData(BaseModel):
    symbol: str
    current_price: float | None = None
    price_history: list[dict[str, Any]] = Field(default_factory=list)
    company_overview: dict[str, Any] = Field(default_factory=dict)
    news: list[dict[str, Any]] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────

class TechnicalIndicators(BaseModel):
    rsi: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    ema_12: float | None = None
    ema_26: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bollinger_upper: float | None = None
    bollinger_lower: float | None = None
    bollinger_middle: float | None = None


class AnalysisResult(BaseModel):
    symbol: str
    technical: TechnicalIndicators = Field(default_factory=TechnicalIndicators)
    fundamental_summary: str = ""
    research_summary: str = ""
    sentiment: str = ""  # bullish / bearish / neutral
    key_factors: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────
# Trading signals
# ──────────────────────────────────────────────

class TradeSignal(BaseModel):
    action: TradeAction
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    risk: RiskLevel
    symbol: str = ""
    suggested_quantity: int = 0
    stop_loss: float | None = None
    take_profit: float | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────
# Risk assessment
# ──────────────────────────────────────────────

class RiskAssessment(BaseModel):
    approved: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    max_position_size: int = 0
    notes: str = ""
    rejection_reason: str | None = None
    adjustments: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────
# Portfolio & trading
# ──────────────────────────────────────────────

class Position(BaseModel):
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.quantity


class PortfolioState(BaseModel):
    cash: float
    positions: dict[str, Position] = Field(default_factory=dict)
    initial_capital: float = 100_000.0

    @property
    def total_value(self) -> float:
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value


class Order(BaseModel):
    symbol: str
    side: OrderSide
    quantity: int = Field(gt=0)
    price: float = Field(gt=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TradeRecord(BaseModel):
    order: Order
    executed_price: float
    commission: float
    slippage: float
    total_cost: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PortfolioSnapshot(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_value: float
    cash: float
    positions_value: float


# ──────────────────────────────────────────────
# Simulation config
# ──────────────────────────────────────────────

class SimulationConfig(BaseModel):
    slippage_pct: float = 0.001
    commission_per_trade: float = 1.0
    commission_pct: float = 0.0005
    max_position_pct: float = 0.20
    max_drawdown_pct: float = 0.10


# ──────────────────────────────────────────────
# Execution planning (Planner Agent)
# ──────────────────────────────────────────────

class PlanStep(BaseModel):
    agent: str
    objective: str
    depends_on: list[str] = Field(default_factory=list)
    status: PlanStepStatus = PlanStepStatus.PENDING
    tools_hint: list[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    symbol: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    current_step_index: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────
# Observability / tracing
# ──────────────────────────────────────────────

class ToolCallRecord(BaseModel):
    tool_name: str
    input_data: dict[str, Any] = Field(default_factory=dict)
    output_summary: str = ""
    latency_ms: float = 0.0
    error: str | None = None


class TraceSpan(BaseModel):
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_name: str
    input_summary: str = ""
    output_summary: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    start_ts: datetime = Field(default_factory=datetime.utcnow)
    end_ts: datetime | None = None
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        if self.end_ts and self.start_ts:
            return (self.end_ts - self.start_ts).total_seconds() * 1000
        return 0.0


class ExecutionTrace(BaseModel):
    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:8]}")
    spans: list[TraceSpan] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

class EvaluationMetrics(BaseModel):
    signal_accuracy: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    avg_latency_ms: float = 0.0
    decision_consistency: float = 0.0


class BacktestResult(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_portfolio_value: float
    total_return_pct: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    baseline_return_pct: float
    total_trades: int
    trades: list[TradeRecord] = Field(default_factory=list)
    equity_curve: list[PortfolioSnapshot] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Decision log
# ──────────────────────────────────────────────

class DecisionRecord(BaseModel):
    run_id: str
    agent_name: str
    decision_type: str
    input_summary: str
    output_summary: str
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────
# API request/response
# ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"


class AnalyzeResponse(BaseModel):
    symbol: str
    trade_signal: dict[str, Any] | None = None
    risk_assessment: dict[str, Any] | None = None
    analysis: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None


class TradeRequest(BaseModel):
    symbol: str
    action: str
    quantity: int = Field(gt=0)
    price: float = Field(gt=0)


class BacktestRequest(BaseModel):
    symbol: str
    days: int | None = None
    initial_capital: float = 100_000.0
