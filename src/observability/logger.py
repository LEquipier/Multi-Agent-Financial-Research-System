from __future__ import annotations

import structlog


def bind_agent_context(
    agent_name: str, run_id: str, span_id: str | None = None
) -> None:
    structlog.contextvars.bind_contextvars(
        agent_name=agent_name,
        run_id=run_id,
        span_id=span_id or "",
    )


def unbind_agent_context() -> None:
    structlog.contextvars.unbind_contextvars(
        "agent_name", "run_id", "span_id"
    )
