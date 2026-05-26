from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TelemetryEvent:
    event_type: str
    context: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    provider: str | None = None
    model_name: str | None = None
    session_id: str | None = None
    raw_type: str | None = None
    tool_use_id: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_result_excerpt: str | None = None
    text_excerpt: str | None = None
    is_error: bool | None = None
    usage: dict[str, Any] | None = None
    model_usage: dict[str, Any] | None = None
    total_cost_usd: float | None = None
    duration_ms: int | None = None
    duration_api_ms: int | None = None
    num_turns: int | None = None
    stop_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: _json_safe(value) for key, value in payload.items() if value is not None}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)