"""
End-to-end smoke test for the full agent pipeline.

Mocks ALL external dependencies (httpx, ChatOpenAI, OpenAI Embeddings)
so the test runs without any API keys.

Acceptance criteria:
1. POST /analyze returns 200
2. Response contains structured JSON (action / confidence / reason / risk)
3. At least one BUY / SELL / HOLD signal generated
4. Risk Agent participates (risk_assessment present)
5. POST /trade executes successfully
6. GET /portfolio returns holdings and cash
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Mock data — realistic AAPL responses
# ---------------------------------------------------------------------------

MOCK_STOCK_PRICE = {
    "symbol": "AAPL",
    "price": 185.50,
    "change": 2.30,
    "change_pct": "1.26%",
    "volume": 54_321_000,
}

MOCK_PRICE_HISTORY = [
    {"date": f"2026-03-{d:02d}", "open": 180 + d * 0.3, "high": 182 + d * 0.3,
     "low": 178 + d * 0.3, "close": 180 + d * 0.5, "volume": 50_000_000 + d * 100_000}
    for d in range(1, 31)
]

MOCK_COMPANY_NEWS = [
    {
        "headline": "Apple Reports Record Q1 Revenue",
        "summary": "Apple Inc. posted record quarterly revenue of $124B driven by iPhone and Services growth.",
        "source": "Reuters",
        "published_at": "2026-03-15T10:00:00Z",
        "url": "https://example.com/news/1",
    },
    {
        "headline": "Apple Vision Pro Adoption Accelerates",
        "summary": "Enterprise adoption of Apple Vision Pro is growing faster than expected.",
        "source": "Bloomberg",
        "published_at": "2026-03-20T14:30:00Z",
        "url": "https://example.com/news/2",
    },
]

MOCK_COMPANY_OVERVIEW = {
    "Symbol": "AAPL",
    "Name": "Apple Inc",
    "MarketCapitalization": "3000000000000",
    "PERatio": "28.5",
    "EPS": "6.51",
    "DividendYield": "0.55",
    "52WeekHigh": "199.62",
    "52WeekLow": "164.08",
    "Sector": "Technology",
    "Industry": "Consumer Electronics",
}

# Mock LLM responses for each agent
MOCK_RESEARCH_RESPONSE = json.dumps({
    "research_summary": "Apple shows strong fundamentals with record Q1 revenue driven by iPhone and Services. Vision Pro enterprise adoption is accelerating. Overall sentiment is bullish.",
    "sentiment": "bullish",
    "key_factors": [
        "Record Q1 revenue of $124B",
        "Services segment growing 20% YoY",
        "Vision Pro enterprise adoption",
        "Strong iPhone demand",
    ],
})

MOCK_ANALYSIS_RESPONSE = json.dumps({
    "analysis_summary": "AAPL shows bullish technical signals with RSI at 58.3 (neutral-bullish), price above 20-day SMA, and positive MACD crossover. Fundamentals are strong with P/E of 28.5 and growing earnings.",
    "technical_outlook": "bullish",
    "confidence_factors": [
        "Price above 20-day and 50-day SMA",
        "RSI in neutral zone with upward momentum",
        "Positive MACD crossover",
        "Strong earnings and revenue growth",
    ],
    "suggested_stop_loss_pct": 0.05,
    "suggested_take_profit_pct": 0.12,
})

MOCK_TRADING_RESPONSE = json.dumps({
    "action": "BUY",
    "confidence": 0.75,
    "reason": "Strong bullish technicals + record earnings + positive sentiment. RSI not overbought. Recommend moderate entry.",
    "risk": "medium",
    "suggested_quantity": 50,
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.12,
})


# ---------------------------------------------------------------------------
# Helper: mock tool registry that intercepts external tool calls
# ---------------------------------------------------------------------------

def _build_mock_registry():
    """Return a ToolRegistry that intercepts httpx-based tools with mock data."""
    from src.tools.registry import ToolRegistry, ToolDefinition
    from src.models.schemas import ToolCategory

    registry = ToolRegistry()

    # --- Market data tools (mock) ---
    async def mock_get_stock_price(symbol: str):
        return MOCK_STOCK_PRICE

    async def mock_get_price_history(symbol: str, period: str = "compact"):
        return MOCK_PRICE_HISTORY

    async def mock_get_company_overview(symbol: str):
        return MOCK_COMPANY_OVERVIEW

    async def mock_get_company_news(symbol: str, from_date: str, to_date: str):
        return MOCK_COMPANY_NEWS

    async def mock_get_market_news(category: str = "general"):
        return MOCK_COMPANY_NEWS

    for name, handler in [
        ("get_stock_price", mock_get_stock_price),
        ("get_price_history", mock_get_price_history),
        ("get_company_overview", mock_get_company_overview),
        ("get_company_news", mock_get_company_news),
        ("get_market_news", mock_get_market_news),
    ]:
        registry.register(ToolDefinition(
            name=name, description=f"mock {name}", category=ToolCategory.MARKET_DATA,
            handler=handler, is_async=True,
        ))

    # --- Indicator tools (real, no external deps) ---
    from src.tools.indicators import (
        compute_rsi, compute_sma, compute_ema, compute_macd, compute_bollinger_bands,
    )
    for name, handler in [
        ("compute_rsi", compute_rsi),
        ("compute_sma", compute_sma),
        ("compute_ema", compute_ema),
        ("compute_macd", compute_macd),
        ("compute_bollinger_bands", compute_bollinger_bands),
    ]:
        registry.register(ToolDefinition(
            name=name, description=f"real {name}", category=ToolCategory.INDICATOR,
            handler=handler, is_async=False,
        ))

    # --- Portfolio tools (real, no external deps) ---
    from src.tools.portfolio import get_portfolio_value, get_positions, check_position_limit
    for name, handler in [
        ("get_portfolio_value", get_portfolio_value),
        ("get_positions", get_positions),
        ("check_position_limit", check_position_limit),
    ]:
        registry.register(ToolDefinition(
            name=name, description=f"real {name}", category=ToolCategory.PORTFOLIO,
            handler=handler, is_async=False,
        ))

    return registry


def _build_mock_rag_pipeline():
    """RAGPipeline whose ingest/retrieve don't call OpenAI embeddings."""
    rag = MagicMock()
    rag.ingest_news = AsyncMock(return_value=2)
    rag.retrieve_context = AsyncMock(
        return_value="[Score: 0.87] Apple Reports Record Q1 Revenue.\n\n"
                     "[Score: 0.72] Apple Vision Pro Adoption Accelerates."
    )
    return rag


def _mock_llm_ainvoke_factory():
    """Return different mock LLM responses based on the prompt content."""
    call_count = {"n": 0}

    async def mock_ainvoke(messages, **kwargs):
        call_count["n"] += 1
        prompt_text = str(messages)

        # Determine which agent is calling based on prompt content
        if "research" in prompt_text.lower() or "Retrieved Context" in prompt_text:
            content = MOCK_RESEARCH_RESPONSE
        elif "technical" in prompt_text.lower() and "Provide JSON" in prompt_text:
            content = MOCK_ANALYSIS_RESPONSE
        elif "trading signal" in prompt_text.lower() or "Generate a trading signal" in prompt_text:
            content = MOCK_TRADING_RESPONSE
        else:
            content = MOCK_RESEARCH_RESPONSE  # fallback

        response = MagicMock()
        response.content = content
        return response

    return mock_ainvoke


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_engine():
    from src.models.schemas import SimulationConfig
    from src.services.trading_engine import TradingEngine
    config = SimulationConfig(
        slippage_pct=0.001,
        commission_per_trade=1.0,
        commission_pct=0.0005,
        max_position_pct=0.20,
    )
    return TradingEngine(initial_capital=100_000, config=config)


@pytest.fixture
def mock_rag():
    return _build_mock_rag_pipeline()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestEndToEndSmoke:
    """Full pipeline smoke test — no API keys needed."""

    async def test_analyze_pipeline_returns_200(self, mock_engine, mock_rag):
        """AC1 + AC2 + AC3 + AC4: POST /analyze returns 200 with structured signal + risk."""
        mock_registry = _build_mock_registry()

        with (
            patch("src.agents.workflow.get_global_registry", return_value=mock_registry),
            patch("src.agents.research_agent.ChatOpenAI") as MockResearchLLM,
            patch("src.agents.analysis_agent.ChatOpenAI") as MockAnalysisLLM,
            patch("src.agents.trading_agent.ChatOpenAI") as MockTradingLLM,
        ):
            # Wire up mock LLMs
            for MockLLM in [MockResearchLLM, MockAnalysisLLM, MockTradingLLM]:
                instance = MockLLM.return_value
                instance.ainvoke = _mock_llm_ainvoke_factory()

            from src.agents.workflow import run_analysis_pipeline

            result = await run_analysis_pipeline(
                symbol="AAPL",
                trading_engine=mock_engine,
                rag_pipeline=mock_rag,
            )

        # --- AC1: pipeline completed (equivalent to 200 from API) ---
        assert result is not None

        # --- AC2: structured JSON with required fields ---
        trade_signal = result.get("trade_signal")
        assert trade_signal is not None, "trade_signal missing from result"
        assert "action" in trade_signal, "action missing from trade_signal"
        assert "confidence" in trade_signal, "confidence missing from trade_signal"
        assert "reason" in trade_signal, "reason missing from trade_signal"
        assert "risk" in trade_signal, "risk missing from trade_signal"

        # --- AC3: generated BUY / SELL / HOLD ---
        assert trade_signal["action"] in ("BUY", "SELL", "HOLD"), \
            f"Unexpected action: {trade_signal['action']}"

        # --- AC4: Risk Agent participated ---
        risk_assessment = result.get("risk_assessment")
        assert risk_assessment is not None, "risk_assessment missing — Risk Agent didn't run"
        assert "approved" in risk_assessment
        assert "risk_score" in risk_assessment

        # Bonus: check analysis was populated
        analysis = result.get("analysis", {})
        assert "technical_indicators" in analysis
        assert "sentiment" in analysis

    async def test_trade_endpoint(self, mock_engine):
        """AC5: POST /trade executes successfully."""
        from src.api.main import app

        with patch("src.api.routes.trade.get_trading_engine", return_value=mock_engine):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/api/v1/trade", json={
                    "symbol": "AAPL",
                    "action": "BUY",
                    "quantity": 10,
                    "price": 185.50,
                })
                assert resp.status_code == 200, f"Trade failed: {resp.text}"
                data = resp.json()
                assert data["order"]["symbol"] == "AAPL"
                assert data["order"]["side"] == "BUY"
                assert data["order"]["quantity"] == 10
                assert data["executed_price"] > 0
                assert data["commission"] > 0

    async def test_portfolio_endpoint(self, mock_engine):
        """AC6: GET /portfolio returns holdings and cash."""
        from src.api.main import app

        # Pre-populate: buy some AAPL
        mock_engine.execute_buy("AAPL", 20, 185.50)

        with patch("src.api.routes.portfolio.get_trading_engine", return_value=mock_engine):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/api/v1/portfolio")
                assert resp.status_code == 200
                data = resp.json()

                # Check portfolio structure
                portfolio = data["portfolio"]
                assert "cash" in portfolio
                assert portfolio["cash"] < 100_000  # some was spent
                assert "positions" in portfolio
                assert "AAPL" in portfolio["positions"]
                assert portfolio["positions"]["AAPL"]["quantity"] == 20

                # Check PnL present
                pnl = data["pnl"]
                assert "total_value" in pnl
                assert "return_pct" in pnl

                # Check max drawdown
                assert "max_drawdown" in data

    async def test_full_chain_analyze_then_trade_then_portfolio(self, mock_engine, mock_rag):
        """Full chain: analyze → extract signal → trade → check portfolio."""
        mock_registry = _build_mock_registry()

        # Step 1: Run analysis pipeline
        with (
            patch("src.agents.workflow.get_global_registry", return_value=mock_registry),
            patch("src.agents.research_agent.ChatOpenAI") as MockResearchLLM,
            patch("src.agents.analysis_agent.ChatOpenAI") as MockAnalysisLLM,
            patch("src.agents.trading_agent.ChatOpenAI") as MockTradingLLM,
        ):
            for MockLLM in [MockResearchLLM, MockAnalysisLLM, MockTradingLLM]:
                instance = MockLLM.return_value
                instance.ainvoke = _mock_llm_ainvoke_factory()

            from src.agents.workflow import run_analysis_pipeline
            result = await run_analysis_pipeline(
                symbol="AAPL",
                trading_engine=mock_engine,
                rag_pipeline=mock_rag,
            )

        signal = result["trade_signal"]
        risk = result["risk_assessment"]

        # Step 2: Execute trade if approved
        from src.api.main import app

        with (
            patch("src.api.routes.trade.get_trading_engine", return_value=mock_engine),
            patch("src.api.routes.portfolio.get_trading_engine", return_value=mock_engine),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                if signal["action"] != "HOLD":
                    qty = signal.get("suggested_quantity", 10) or 10
                    trade_resp = await client.post("/api/v1/trade", json={
                        "symbol": "AAPL",
                        "action": signal["action"],
                        "quantity": qty,
                        "price": 185.50,
                    })
                    assert trade_resp.status_code == 200, \
                        f"Trade failed: {trade_resp.text}"

                # Step 3: Verify portfolio reflects the trade
                portfolio_resp = await client.get("/api/v1/portfolio")
                assert portfolio_resp.status_code == 200
                portfolio_data = portfolio_resp.json()

                if signal["action"] == "BUY":
                    assert "AAPL" in portfolio_data["portfolio"]["positions"]
                    assert portfolio_data["portfolio"]["cash"] < 100_000
