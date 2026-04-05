from __future__ import annotations

import json
from typing import Any

from langchain_openai import ChatOpenAI

from src.agents.base import AgentState, BaseAgent
from src.core.config import settings
from src.models.schemas import ExecutionPlan, PlanStep, PlanStepStatus


class PlannerAgent(BaseAgent):
    name = "planner"
    role = "Task planner and dynamic workflow router"
    system_prompt = """You are a financial analysis planning agent. Your role is to:
1. Decompose financial analysis tasks into steps
2. Decide which agent to call next based on current state
3. Handle re-routing when risk assessment rejects a trade signal

Available agents and their capabilities:
- data_agent: Fetch stock prices, price history, company news, company overview
- research_agent: RAG-based research, news summarization, semantic retrieval
- analysis_agent: Technical indicators (RSI, MACD, SMA, Bollinger), fundamental analysis
- strategy_selector: Detect market regime, select optimal strategy, check re-entry eligibility
- trading_agent: Generate BUY/SELL/HOLD signals with confidence scores
- risk_agent: Validate trading signals against risk parameters, position limits

Routing rules:
- Standard flow: data → research → analysis → strategy → trading → risk
- If data_sufficient is False: route back to data_agent
- If risk rejects and iterations remain: route to analysis or trading with feedback
- If max iterations reached: route to "end"
- If risk approves: route to "execute"

Respond with JSON: {"next_agent": "<agent_name>", "reasoning": "<why>"}
Valid next_agent values: data, research, analysis, strategy, trading, risk, execute, end
"""

    async def process(self, state: AgentState) -> AgentState:
        execution_plan = state.get("execution_plan")
        iteration_count = state.get("iteration_count", 0)
        data_sufficient = state.get("data_sufficient", False)
        risk_assessment = state.get("risk_assessment")
        risk_feedback = state.get("risk_feedback")
        max_iterations = settings.max_agent_iterations

        # Determine next step using deterministic logic with LLM as tiebreaker
        next_agent = self._determine_next_agent(
            execution_plan=execution_plan,
            iteration_count=iteration_count,
            data_sufficient=data_sufficient,
            risk_assessment=risk_assessment,
            risk_feedback=risk_feedback,
            max_iterations=max_iterations,
            state=state,
        )

        # Build or update execution plan
        if execution_plan is None:
            plan = ExecutionPlan(
                symbol=state.get("metadata", {}).get("symbol", ""),
                steps=[
                    PlanStep(agent="data", objective="Fetch market data and news"),
                    PlanStep(
                        agent="research",
                        objective="Retrieve relevant context via RAG",
                        depends_on=["data"],
                    ),
                    PlanStep(
                        agent="analysis",
                        objective="Technical and fundamental analysis",
                        depends_on=["research"],
                    ),
                    PlanStep(
                        agent="strategy",
                        objective="Detect market regime and select strategy",
                        depends_on=["analysis"],
                    ),
                    PlanStep(
                        agent="trading",
                        objective="Generate trade signal",
                        depends_on=["strategy"],
                    ),
                    PlanStep(
                        agent="risk",
                        objective="Validate and assess risk",
                        depends_on=["trading"],
                    ),
                ],
            )
            state["execution_plan"] = plan.model_dump()
        else:
            plan_obj = ExecutionPlan(**execution_plan)
            # Mark completed steps
            for step in plan_obj.steps:
                if step.agent == state.get("current_agent") and step.status == PlanStepStatus.RUNNING:
                    step.status = PlanStepStatus.DONE
            # Mark next step as running
            for step in plan_obj.steps:
                if step.agent == next_agent and step.status in (PlanStepStatus.PENDING, PlanStepStatus.DONE):
                    step.status = PlanStepStatus.RUNNING
                    break
            state["execution_plan"] = plan_obj.model_dump()

        state["current_agent"] = next_agent

        return state

    def _determine_next_agent(
        self,
        execution_plan: dict | None,
        iteration_count: int,
        data_sufficient: bool,
        risk_assessment: dict | None,
        risk_feedback: str | None,
        max_iterations: int,
        state: AgentState,
    ) -> str:
        # First run: start with data
        if execution_plan is None:
            return "data"

        current = state.get("current_agent", "")

        # Data insufficient: retry
        if current == "data" and not data_sufficient:
            if iteration_count < max_iterations:
                return "data"
            return "end"

        # Risk rejected: re-route
        if current == "risk" and risk_assessment:
            if risk_assessment.get("approved", False):
                return "execute"
            if iteration_count < max_iterations:
                return "trading"  # retry with feedback
            return "end"

        # Follow standard flow
        flow = ["data", "research", "analysis", "strategy", "trading", "risk"]
        if current in flow:
            idx = flow.index(current)
            if idx + 1 < len(flow):
                return flow[idx + 1]

        # After execute or unknown state
        return "end"
