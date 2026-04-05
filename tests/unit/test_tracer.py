from __future__ import annotations

import pytest

from src.observability.tracer import ExecutionTracer


class TestTracer:
    def test_span_lifecycle(self):
        tracer = ExecutionTracer()
        tracer.start_run("test-run-123")

        span_id = tracer.start_span("data_agent", input_summary="AAPL")
        assert span_id is not None

        tracer.record_tool_call(span_id, "get_stock_price", {"symbol": "AAPL"}, output_summary="price=150")
        tracer.end_span(span_id, output_summary="price fetched")

        trace = tracer.end_run()
        assert trace.run_id == "test-run-123"
        assert len(trace.spans) == 1
        assert trace.spans[0].agent_name == "data_agent"
        assert len(trace.spans[0].tool_calls) == 1

    def test_end_run_total_latency(self):
        tracer = ExecutionTracer()
        tracer.start_run("dur-test")
        trace = tracer.end_run()
        assert trace.total_latency_ms is not None

    @pytest.mark.asyncio
    async def test_save_and_get_trace(self, tmp_path):
        tracer = ExecutionTracer(db_path=str(tmp_path / "traces.db"))
        tracer.start_run("persist-test")
        span_id = tracer.start_span("agent_a")
        tracer.end_span(span_id)
        trace = tracer.end_run()

        await tracer.save_trace(trace)
        loaded = await tracer.get_trace("persist-test")
        assert loaded is not None
        assert loaded.run_id == "persist-test"
