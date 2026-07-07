"""Lightweight structured logs for Agent observability."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger("app.observability")


def new_trace_id(prefix: str) -> str:
    """Return a short correlation id for one Agent request."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def elapsed_ms(started_at: float) -> int:
    """Milliseconds elapsed since a ``time.perf_counter()`` timestamp."""
    return int((time.perf_counter() - started_at) * 1000)


def payload_size(value: Any) -> int | None:
    """Best-effort size metric that avoids logging the payload itself."""
    if value is None:
        return None
    if isinstance(value, (str, bytes, list, tuple, set, dict)):
        return len(value)
    if hasattr(value, "model_dump"):
        try:
            return len(value.model_dump())
        except Exception:  # noqa: BLE001
            return None
    return None


def log_agent_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one JSON log line with stable keys for filtering and dashboards."""
    record = {"event": event}
    record.update({key: value for key, value in fields.items() if value is not None})
    logger.log(level, "agent_observe %s", json.dumps(record, ensure_ascii=False, default=str))
