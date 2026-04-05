from __future__ import annotations

import json
from typing import Any

from langchain_openai import ChatOpenAI

from src.agents.base import AgentState, BaseAgent
from src.core.config import settings
from src.services.rag import RAGPipeline


class ResearchAgent(BaseAgent):
    name = "research_agent"
    role = "RAG-powered research and news summarization"
    system_prompt = """You are a financial research agent. Your job is to:
1. Ingest and index news articles using hybrid retrieval (vector + keyword)
2. Retrieve relevant context for the current analysis
3. Summarize key findings from news and market data

Produce a concise research summary focusing on:
- Key news sentiment (bullish/bearish/neutral)
- Notable events (earnings, partnerships, regulatory changes)
- Market trends relevant to the stock

Output a JSON object: {"research_summary": "...", "sentiment": "...", "key_factors": [...]}"""

    def __init__(self, rag_pipeline: RAGPipeline | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rag_pipeline = rag_pipeline or RAGPipeline()

    async def process(self, state: AgentState) -> AgentState:
        market_data = state.get("market_data", {})
        symbol = market_data.get("symbol", "")
        news = market_data.get("news", [])

        # Ingest news into RAG system
        if news:
            await self.rag_pipeline.ingest_news(news)

        # Retrieve context
        query = f"Latest financial analysis and news for {symbol} stock market outlook"
        context = await self.rag_pipeline.retrieve_context(query, k=5, recency_bias=0.1)

        # Use LLM to summarize
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
        )

        news_text = ""
        for n in news[:10]:
            news_text += f"- {n.get('headline', '')}: {n.get('summary', '')}\n"

        prompt = f"""Analyze the following information for {symbol}:

Retrieved Context:
{context}

Recent News:
{news_text}

Company Overview:
{json.dumps(market_data.get('company_overview', {}), indent=2)[:1000]}

Provide a JSON response with:
- research_summary: A concise paragraph summarizing key findings
- sentiment: "bullish", "bearish", or "neutral"
- key_factors: List of 3-5 key factors influencing the stock"""

        response = await llm.ainvoke(
            [{"role": "system", "content": self.system_prompt},
             {"role": "user", "content": prompt}]
        )

        # Parse response
        try:
            content = response.content
            if isinstance(content, str):
                # Try to extract JSON from response
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    result = json.loads(content[start:end])
                else:
                    result = {
                        "research_summary": content,
                        "sentiment": "neutral",
                        "key_factors": [],
                    }
            else:
                result = {
                    "research_summary": str(content),
                    "sentiment": "neutral",
                    "key_factors": [],
                }
        except (json.JSONDecodeError, AttributeError):
            result = {
                "research_summary": str(response.content),
                "sentiment": "neutral",
                "key_factors": [],
            }

        analysis = state.get("analysis", {})
        analysis["research_summary"] = result.get("research_summary", "")
        analysis["sentiment"] = result.get("sentiment", "neutral")
        analysis["key_factors"] = result.get("key_factors", [])
        state["analysis"] = analysis

        return state
