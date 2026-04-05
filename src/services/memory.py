from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Any

import aiosqlite

from src.core.config import settings
from src.models.schemas import AgentMessage


class BaseMemory(ABC):
    @abstractmethod
    async def store(self, key: str, value: Any) -> None: ...

    @abstractmethod
    async def retrieve(self, key: str, limit: int = 10) -> list[Any]: ...

    @abstractmethod
    async def clear(self) -> None: ...


class ShortTermMemory(BaseMemory):
    def __init__(self, max_size: int = 50) -> None:
        self._buffer: deque[AgentMessage] = deque(maxlen=max_size)
        self._context: dict[str, Any] = {}

    async def store(self, key: str, value: Any) -> None:
        if isinstance(value, AgentMessage):
            self._buffer.append(value)
        else:
            self._context[key] = value

    async def retrieve(self, key: str, limit: int = 10) -> list[Any]:
        if key == "messages":
            return list(self._buffer)[-limit:]
        val = self._context.get(key)
        return [val] if val is not None else []

    async def clear(self) -> None:
        self._buffer.clear()
        self._context.clear()

    def add_message(self, message: AgentMessage) -> None:
        self._buffer.append(message)

    def get_recent_messages(self, n: int = 10) -> list[AgentMessage]:
        return list(self._buffer)[-n:]

    def set_context(self, key: str, value: Any) -> None:
        self._context[key] = value

    def get_context(self, key: str) -> Any:
        return self._context.get(key)


class LongTermMemory(BaseMemory):
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or str(settings.db_path)

    async def _ensure_tables(self, db: aiosqlite.Connection) -> None:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL,
                reason TEXT,
                outcome TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS market_trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                trend_data TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        await db.commit()

    async def store(self, key: str, value: Any) -> None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_tables(db)
            if key == "decision":
                await db.execute(
                    "INSERT INTO decisions (symbol, action, confidence, reason, outcome, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        value.get("symbol", ""),
                        value.get("action", ""),
                        value.get("confidence", 0),
                        value.get("reason", ""),
                        value.get("outcome", ""),
                        datetime.utcnow().isoformat(),
                    ),
                )
                await db.commit()
            elif key == "trend":
                await db.execute(
                    "INSERT INTO market_trends (symbol, trend_data, timestamp) VALUES (?, ?, ?)",
                    (
                        value.get("symbol", ""),
                        json.dumps(value.get("data", {})),
                        datetime.utcnow().isoformat(),
                    ),
                )
                await db.commit()

    async def retrieve(self, key: str, limit: int = 10) -> list[Any]:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_tables(db)
            if key.startswith("decisions:"):
                symbol = key.split(":", 1)[1]
                cursor = await db.execute(
                    "SELECT symbol, action, confidence, reason, outcome, timestamp "
                    "FROM decisions WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                    (symbol, limit),
                )
            else:
                cursor = await db.execute(
                    "SELECT symbol, action, confidence, reason, outcome, timestamp "
                    "FROM decisions ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
            rows = await cursor.fetchall()
            return [
                {
                    "symbol": r[0],
                    "action": r[1],
                    "confidence": r[2],
                    "reason": r[3],
                    "outcome": r[4],
                    "timestamp": r[5],
                }
                for r in rows
            ]

    async def get_past_decisions(self, symbol: str, limit: int = 10) -> list[dict]:
        return await self.retrieve(f"decisions:{symbol}", limit)

    async def get_market_trends(self, symbol: str) -> list[dict]:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_tables(db)
            cursor = await db.execute(
                "SELECT symbol, trend_data, timestamp FROM market_trends "
                "WHERE symbol = ? ORDER BY timestamp DESC LIMIT 20",
                (symbol,),
            )
            rows = await cursor.fetchall()
            return [
                {"symbol": r[0], "data": json.loads(r[1]), "timestamp": r[2]}
                for r in rows
            ]

    async def clear(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_tables(db)
            await db.execute("DELETE FROM decisions")
            await db.execute("DELETE FROM market_trends")
            await db.commit()
