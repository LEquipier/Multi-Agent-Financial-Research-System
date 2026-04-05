from __future__ import annotations

import uuid
from typing import Any

from langgraph.graph import END, StateGraph

from src.agents.analysis_agent import AnalysisAgent
from src.agents.base import AgentState
from src.agents.data_agent import DataAgent
from src.agents.planner import PlannerAgent
from src.agents.research_agent import ResearchAgent
from src.agents.risk_agent import RiskAgent
from src.agents.strategy_selector import StrategySelector
from src.agents.trading_agent import TradingAgent
from src.core.config import settings
from src.core.logging import get_logger
from src.models.schemas import PortfolioState, SimulationConfig
from src.observability.decision_log import DecisionLogger
from src.observability.tracer import ExecutionTracer
from src.services.memory import ShortTermMemory
from src.services.rag import RAGPipeline
from src.services.trading_engine import TradingEngine
from src.tools.registry import get_global_registry

logger = get_logger(__name__)


def build_workflow(
    trading_engine: TradingEngine | None = None,
    rag_pipeline: RAGPipeline | None = None,
) -> StateGraph:
    """Build the LangGraph hub-and-spoke workflow."""
    registry = get_global_registry()
    tracer = ExecutionTracer()
    memory = ShortTermMemory()

    # Agents
    planner = PlannerAgent(tool_registry=registry, tracer=tracer, memory=memory)
    data_agent = DataAgent(tool_registry=registry, tracer=tracer, memory=memory)
    research = ResearchAgent(
        rag_pipeline=rag_pipeline or RAGPipeline(),
        tool_registry=registry,
        tracer=tracer,
        memory=memory,
    )
    analysis = AnalysisAgent(tool_registry=registry, tracer=tracer, memory=memory)
    strategy = StrategySelector(tool_registry=registry, tracer=tracer, memory=memory)
    trading = TradingAgent(tool_registry=registry, tracer=tracer, memory=memory)
    risk = RiskAgent(
        trading_engine=trading_engine, tool_registry=registry, tracer=tracer, memory=memory
    )

    # Build graph
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("planner", planner.invoke)
    graph.add_node("data", data_agent.invoke)
    graph.add_node("research", research.invoke)
    graph.add_node("analysis", analysis.invoke)
    graph.add_node("strategy", strategy.invoke)
    graph.add_node("trading", trading.invoke)
    graph.add_node("risk", risk.invoke)
    graph.add_node("execute", _execute_trade)

    # Entry: always start at planner
    graph.set_entry_point("planner")

    # Hub-and-spoke: planner routes to agents, agents return to planner
    graph.add_conditional_edges(
        "planner",
        _route_from_planner,
        {
            "data": "data",
            "research": "research",
            "analysis": "analysis",
            "strategy": "strategy",
            "trading": "trading",
            "risk": "risk",
            "execute": "execute",
            "end": END,
        },
    )

    # All agents route back to planner (hub-and-spoke)
    graph.add_edge("data", "planner")
    graph.add_edge("research", "planner")
    graph.add_edge("analysis", "planner")
    graph.add_edge("strategy", "planner")
    graph.add_edge("trading", "planner")
    graph.add_edge("risk", "planner")
    graph.add_edge("execute", END)

    return graph


def _route_from_planner(state: AgentState) -> str:
    """Conditional edge: read the current_agent field set by the planner."""
    return state.get("current_agent", "end")


async def _execute_trade(state: AgentState) -> AgentState:
    """Terminal node: log the approved trade for execution."""
    trade_signal = state.get("trade_signal", {})
    risk_assessment = state.get("risk_assessment", {})

    logger.info(
        "trade_execution_approved",
        signal=trade_signal,
        risk_assessment=risk_assessment,
    )

    messages = state.get("messages", [])
    messages.append(
        {
            "role": "system",
            "content": f"Trade approved: {trade_signal.get('action')} "
            f"{trade_signal.get('symbol')} — confidence {trade_signal.get('confidence')}, "
            f"risk score {risk_assessment.get('risk_score')}",
        }
    )
    state["messages"] = messages
    return state


async def run_analysis_pipeline(
    symbol: str,
    portfolio: PortfolioState | None = None,
    trading_engine: TradingEngine | None = None,
    rag_pipeline: RAGPipeline | None = None,
) -> AgentState:
    """Run the full multi-agent analysis pipeline for a stock symbol."""
    run_id = str(uuid.uuid4())
    tracer = ExecutionTracer()
    tracer.start_run(run_id, metadata={"symbol": symbol})

    if portfolio is None:
        portfolio = PortfolioState(
            cash=settings.initial_capital,
            positions={},
        )

    if trading_engine is None:
        config = SimulationConfig(
            slippage_pct=settings.slippage_pct,
            commission_per_trade=settings.commission_per_trade,
            commission_pct=settings.commission_pct,
            max_position_pct=settings.max_position_pct,
        )
        trading_engine = TradingEngine(initial_capital=settings.initial_capital, config=config)

    graph = build_workflow(trading_engine=trading_engine, rag_pipeline=rag_pipeline)
    compiled = graph.compile()

    initial_state: AgentState = {
        "messages": [],
        "market_data": {},
        "analysis": {},
        "trade_signal": None,
        "risk_assessment": None,
        "portfolio": portfolio.model_dump(),
        "metadata": {"symbol": symbol.upper()},
        "execution_plan": None,
        "iteration_count": 0,
        "data_sufficient": False,
        "risk_feedback": None,
        "run_id": run_id,
        "current_agent": "",
    }

    logger.info("pipeline_start", symbol=symbol, run_id=run_id)
    final_state = await compiled.ainvoke(initial_state)
    logger.info("pipeline_complete", symbol=symbol, run_id=run_id)

    # Save trace
    trace = tracer.end_run()
    await tracer.save_trace(trace)

    final_state["metadata"]["trace_id"] = trace.run_id

    return final_state
