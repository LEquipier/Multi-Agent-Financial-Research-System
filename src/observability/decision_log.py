from __future__ import annotations

import json
from datetime import datetime

import aiosqlite

from src.core.config import settings
from src.models.schemas import DecisionRecord


class DecisionLogger:
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or str(settings.traces_db_path)

    async def _ensure_table(self, db: aiosqlite.Connection) -> None:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS decision_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                decision_type TEXT NOT NULL,
                input_summary TEXT,
                output_summary TEXT,
                reasoning TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        await db.commit()

    async def log_decision(
        self,
        run_id: str,
        agent_name: str,
        decision_type: str,
        input_summary: str = "",
        output_summary: str = "",
        reasoning: str = "",
    ) -> None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_table(db)
            await db.execute(
                "INSERT INTO decision_log "
                "(run_id, agent_name, decision_type, input_summary, output_summary, reasoning, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    agent_name,
                    decision_type,
                    input_summary,
                    output_summary,
                    reasoning,
                    datetime.utcnow().isoformat(),
                ),
            )
            await db.commit()

    async def get_decisions(self, run_id: str) -> list[DecisionRecord]:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_table(db)
            cursor = await db.execute(
                "SELECT run_id, agent_name, decision_type, input_summary, "
                "output_summary, reasoning, timestamp "
                "FROM decision_log WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            )
            rows = await cursor.fetchall()
            return [
                DecisionRecord(
                    run_id=r[0],
                    agent_name=r[1],
                    decision_type=r[2],
                    input_summary=r[3],
                    output_summary=r[4],
                    reasoning=r[5],
                    timestamp=datetime.fromisoformat(r[6]),
                )
                for r in rows
            ]

    async def get_decision_chain(self, run_id: str) -> str:
        decisions = await self.get_decisions(run_id)
        if not decisions:
            return "No decisions recorded for this run."

        parts = []
        for d in decisions:
            parts.append(f"{d.agent_name}: {d.output_summary}")
        return " → ".join(parts)
