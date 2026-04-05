from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict

from src.models.schemas import (
    AnalysisResult,
    ExecutionPlan,
    ExecutionTrace,
    MarketData,
    PortfolioState,
    RiskAssessment,
    TradeSignal,
)
from src.observability.logger import bind_agent_context, unbind_agent_context
from src.observability.tracer import ExecutionTracer
from src.services.memory import ShortTermMemory
from src.tools.registry import ToolRegistry


class AgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    market_data: dict[str, Any]
    analysis: dict[str, Any]
    trade_signal: dict[str, Any] | None
    risk_assessment: dict[str, Any] | None
    portfolio: dict[str, Any]
    metadata: dict[str, Any]
    # v2 additions
    execution_plan: dict[str, Any] | None
    iteration_count: int
    data_sufficient: bool
    risk_feedback: str | None
    run_id: str
    current_agent: str


class BaseAgent(ABC):
    name: str = "base"
    role: str = ""
    system_prompt: str = ""

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        tracer: ExecutionTracer | None = None,
        memory: ShortTermMemory | None = None,
    ) -> None:
        self.tool_registry = tool_registry or ToolRegistry()
        self.tracer = tracer or ExecutionTracer()
        self.memory = memory or ShortTermMemory()

    async def invoke(self, state: AgentState) -> AgentState:
        run_id = state.get("run_id", "")
        span_id = self.tracer.start_span(self.name, input_summary=self._summarize_input(state))
        bind_agent_context(self.name, run_id, span_id)

        try:
            result = await self.process(state)
            self.tracer.end_span(span_id, output_summary=self._summarize_output(result))
            return result
        except Exception as e:
            self.tracer.end_span(span_id, error=str(e))
            raise
        finally:
            unbind_agent_context()

    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        ...

    def _summarize_input(self, state: AgentState) -> str:
        symbol = state.get("metadata", {}).get("symbol", "unknown")
        iteration = state.get("iteration_count", 0)
        return f"symbol={symbol}, iteration={iteration}"

    def _summarize_output(self, state: AgentState) -> str:
        return f"agent={self.name} completed"
