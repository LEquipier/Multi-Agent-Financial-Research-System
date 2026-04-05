from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app


@pytest.mark.asyncio
class TestAPIEndpoints:
    async def test_health(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    async def test_portfolio(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/portfolio")
            assert resp.status_code == 200
            data = resp.json()
            assert "portfolio" in data
            assert "pnl" in data

    async def test_trade_invalid_action(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/trade",
                json={"symbol": "AAPL", "action": "INVALID", "quantity": 10, "price": 150},
            )
            assert resp.status_code in (400, 422)
