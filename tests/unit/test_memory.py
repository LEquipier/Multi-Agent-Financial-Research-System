from __future__ import annotations

import pytest

from src.services.memory import LongTermMemory, ShortTermMemory
from src.models.schemas import AgentMessage, MessageType


class TestShortTermMemory:
    def test_add_and_get(self):
        mem = ShortTermMemory(max_size=10)
        msg = AgentMessage(
            sender="test",
            receiver="planner",
            content={"text": "hello"},
            message_type=MessageType.RESULT,
        )
        mem.add_message(msg)
        messages = mem.get_recent_messages()
        assert len(messages) == 1
        assert messages[0].content == {"text": "hello"}

    def test_max_messages(self):
        mem = ShortTermMemory(max_size=3)
        for i in range(5):
            mem.add_message(
                AgentMessage(
                    sender="a",
                    receiver="b",
                    content={"n": i},
                    message_type=MessageType.RESULT,
                )
            )
        assert len(mem.get_recent_messages(10)) == 3

    def test_context(self):
        mem = ShortTermMemory()
        mem.set_context("key", "value")
        assert mem.get_context("key") == "value"
        assert mem.get_context("nonexistent") is None


class TestLongTermMemory:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        mem = LongTermMemory(db_path=db_path)

        await mem.store("decision", {"symbol": "AAPL", "action": "BUY", "confidence": 0.8, "reason": "bullish", "outcome": ""})
        decisions = await mem.retrieve("decisions:AAPL", limit=5)
        assert len(decisions) == 1
        assert decisions[0]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_market_trends(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        mem = LongTermMemory(db_path=db_path)

        await mem.store("trend", {"symbol": "AAPL", "data": {"sentiment": "bullish", "confidence": 0.8}})
        trends = await mem.get_market_trends("AAPL")
        assert len(trends) == 1
        assert trends[0]["data"]["sentiment"] == "bullish"
