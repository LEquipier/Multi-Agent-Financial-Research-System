from __future__ import annotations

import json
import time
import uuid
from datetime import datetime

import aiosqlite

from src.core.config import settings
from src.models.schemas import (
    ExecutionTrace,
    ToolCallRecord,
    TraceSpan,
)


class ExecutionTracer:
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or str(settings.traces_db_path)
        self._current_run_id: str | None = None
        self._spans: dict[str, TraceSpan] = {}
        self._run_spans: list[TraceSpan] = []
        self._run_start: float = 0.0

    def start_run(self, run_id: str | None = None, metadata: dict | None = None) -> str:
        self._current_run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        self._spans.clear()
        self._run_spans.clear()
        self._run_start = time.perf_counter()
        self._metadata = metadata or {}
        return self._current_run_id

    def start_span(self, agent_name: str, input_summary: str = "") -> str:
        span = TraceSpan(
            agent_name=agent_name,
            input_summary=input_summary,
        )
        self._spans[span.span_id] = span
        return span.span_id

    def record_tool_call(
        self,
        span_id: str,
        tool_name: str,
        input_data: dict | None = None,
        output_summary: str = "",
        latency_ms: float = 0.0,
        error: str | None = None,
    ) -> None:
        span = self._spans.get(span_id)
        if not span:
            return
        span.tool_calls.append(
            ToolCallRecord(
                tool_name=tool_name,
                input_data=input_data or {},
                output_summary=output_summary,
                latency_ms=latency_ms,
                error=error,
            )
        )

    def end_span(
        self,
        span_id: str,
        output_summary: str = "",
        error: str | None = None,
    ) -> None:
        span = self._spans.get(span_id)
        if not span:
            return
        span.end_ts = datetime.utcnow()
        span.output_summary = output_summary
        span.error = error
        self._run_spans.append(span)
        del self._spans[span_id]

    def end_run(self) -> ExecutionTrace:
        total_ms = (time.perf_counter() - self._run_start) * 1000
        trace = ExecutionTrace(
            run_id=self._current_run_id or "",
            spans=list(self._run_spans),
            total_latency_ms=round(total_ms, 2),
        )
        self._current_run_id = None
        return trace

    async def save_trace(self, trace: ExecutionTrace) -> None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS execution_traces (
                    run_id TEXT PRIMARY KEY,
                    trace_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            await db.execute(
                "INSERT OR REPLACE INTO execution_traces (run_id, trace_json, created_at) "
                "VALUES (?, ?, ?)",
                (
                    trace.run_id,
                    trace.model_dump_json(),
                    trace.created_at.isoformat(),
                ),
            )
            await db.commit()

    async def get_trace(self, run_id: str) -> ExecutionTrace | None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS execution_traces (
                    run_id TEXT PRIMARY KEY,
                    trace_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cursor = await db.execute(
                "SELECT trace_json FROM execution_traces WHERE run_id = ?",
                (run_id,),
            )
            row = await cursor.fetchone()
            if row:
                return ExecutionTrace.model_validate_json(row[0])
            return None
