from __future__ import annotations

import json
from typing import Any

from langchain_openai import ChatOpenAI

from src.agents.base import AgentState, BaseAgent
from src.core.config import settings


class AnalysisAgent(BaseAgent):
    name = "analysis_agent"
    role = "Technical and fundamental financial analysis"
    system_prompt = """You are a senior financial analyst agent. Your job is to:
1. Compute technical indicators (RSI, MACD, SMA, Bollinger Bands)
2. Combine technical analysis with fundamental data and research
3. Produce a comprehensive analysis summary

If risk_feedback is provided from a previous iteration, incorporate it to adjust your analysis
(e.g., consider tighter stop-loss levels, different timeframes, or additional risk factors).

Output JSON: {"analysis_summary": "...", "technical_outlook": "bullish|bearish|neutral", "confidence_factors": [...]}"""

    async def process(self, state: AgentState) -> AgentState:
        market_data = state.get("market_data", {})
        analysis = state.get("analysis", {})
        risk_feedback = state.get("risk_feedback")

        history = market_data.get("price_history", [])
        closing_prices = [h["close"] for h in history if "close" in h]

        # Compute technical indicators
        indicators: dict[str, Any] = {}

        if closing_prices:
            rsi = await self.tool_registry.execute("compute_rsi", prices=closing_prices)
            indicators["rsi"] = rsi

            sma_20 = await self.tool_registry.execute(
                "compute_sma", prices=closing_prices, window=20
            )
            indicators["sma_20"] = sma_20

            sma_50 = await self.tool_registry.execute(
                "compute_sma", prices=closing_prices, window=50
            )
            indicators["sma_50"] = sma_50

            macd = await self.tool_registry.execute("compute_macd", prices=closing_prices)
            indicators["macd"] = macd

            bollinger = await self.tool_registry.execute(
                "compute_bollinger_bands", prices=closing_prices
            )
            indicators["bollinger_bands"] = bollinger

        analysis["technical_indicators"] = indicators
        analysis["current_price"] = market_data.get("current_price", {}).get("price")

        # Use LLM with all context for deeper analysis
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
        )

        feedback_section = ""
        if risk_feedback:
            feedback_section = f"""
IMPORTANT - Risk Feedback from previous iteration:
{risk_feedback}
Adjust your analysis to address these concerns."""

        prompt = f"""Analyze {market_data.get('symbol', 'unknown')} with the following data:

Technical Indicators:
{json.dumps(indicators, indent=2, default=str)}

Research Summary:
{analysis.get('research_summary', 'N/A')}

Sentiment: {analysis.get('sentiment', 'N/A')}

Key Factors: {json.dumps(analysis.get('key_factors', []))}

Company Overview:
{json.dumps(market_data.get('company_overview', {}), indent=2)[:800]}
{feedback_section}

Provide JSON with:
- analysis_summary: Comprehensive paragraph combining technical + fundamental analysis
- technical_outlook: "bullish", "bearish", or "neutral"
- confidence_factors: List of 3-5 factors supporting your analysis
- suggested_stop_loss_pct: Percentage for stop-loss (e.g., 0.05 for 5%)
- suggested_take_profit_pct: Percentage for take-profit"""

        response = await llm.ainvoke(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": prompt}]
        )

        try:
            content = response.content
            if isinstance(content, str):
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    result = json.loads(content[start:end])
                else:
                    result = {"analysis_summary": content, "technical_outlook": "neutral"}
            else:
                result = {"analysis_summary": str(content), "technical_outlook": "neutral"}
        except (json.JSONDecodeError, AttributeError):
            result = {"analysis_summary": str(response.content), "technical_outlook": "neutral"}

        analysis["analysis_summary"] = result.get("analysis_summary", "")
        analysis["technical_outlook"] = result.get("technical_outlook", "neutral")
        analysis["confidence_factors"] = result.get("confidence_factors", [])
        analysis["suggested_stop_loss_pct"] = result.get("suggested_stop_loss_pct", 0.05)
        analysis["suggested_take_profit_pct"] = result.get("suggested_take_profit_pct", 0.10)
        state["analysis"] = analysis

        return state
