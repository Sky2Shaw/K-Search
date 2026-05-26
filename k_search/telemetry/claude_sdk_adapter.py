from __future__ import annotations

import os
from typing import Any

from k_search.telemetry.events import TelemetryEvent, _json_safe


def event_from_claude_message(message: Any) -> list[TelemetryEvent]:
    try:
        if hasattr(message, "result"):
            return [_result_event(message)]
        content = getattr(message, "content", None)
        if isinstance(content, list):
            events: list[TelemetryEvent] = []
            for block in content:
                events.append(_event_from_block(block))
            return events
        if isinstance(content, str) and content.strip():
            return [TelemetryEvent(event_type="assistant_text", raw_type=type(message).__name__, text_excerpt=_truncate(content))]
        text = getattr(message, "text", None)
        if isinstance(text, str) and text.strip():
            return [TelemetryEvent(event_type="assistant_text", raw_type=type(message).__name__, text_excerpt=_truncate(text))]
        return [TelemetryEvent(event_type="system_message", raw_type=type(message).__name__, text_excerpt=_truncate(str(message)))]
    except Exception as exc:
        return [
            TelemetryEvent(
                event_type="system_message",
                raw_type=type(message).__name__,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        ]


def _event_from_block(block: Any) -> TelemetryEvent:
    block_type = _block_type(block)
    raw_type = f"{type(block).__name__}:{block_type}" if block_type else type(block).__name__
    if block_type == "tool_use":
        return TelemetryEvent(
            event_type="tool_use",
            raw_type=raw_type,
            tool_use_id=_get(block, "id"),
            tool_name=_get(block, "name"),
            tool_input=_safe_mapping(_get(block, "input")),
        )
    if block_type == "tool_result":
        return TelemetryEvent(
            event_type="tool_result",
            raw_type=raw_type,
            tool_use_id=_get(block, "tool_use_id"),
            tool_result_excerpt=_tool_result_excerpt(_get(block, "content")),
            is_error=bool(_get(block, "is_error")) if _get(block, "is_error") is not None else None,
        )
    if block_type == "text":
        text = _get(block, "text")
        return TelemetryEvent(event_type="assistant_text", raw_type=raw_type, text_excerpt=_assistant_text_excerpt(text))
    if block_type == "thinking":
        thinking = str(_get(block, "thinking") or "")
        return TelemetryEvent(
            event_type="assistant_thinking_metadata",
            raw_type=raw_type,
            text_excerpt=f"thinking_chars={len(thinking)}",
        )
    return TelemetryEvent(event_type="system_message", raw_type=raw_type, text_excerpt=_truncate(str(block)))


def _result_event(message: Any) -> TelemetryEvent:
    return TelemetryEvent(
        event_type="llm_result",
        raw_type=type(message).__name__,
        session_id=_get(message, "session_id"),
        duration_ms=_get(message, "duration_ms"),
        duration_api_ms=_get(message, "duration_api_ms"),
        num_turns=_get(message, "num_turns"),
        total_cost_usd=_get(message, "total_cost_usd"),
        usage=_safe_mapping(_get(message, "usage")),
        model_usage=_safe_mapping(_get(message, "model_usage")),
        stop_reason=_get(message, "subtype") or _get(message, "stop_reason"),
        is_error=bool(_get(message, "is_error")) if _get(message, "is_error") is not None else None,
        text_excerpt=_truncate(str(_get(message, "result") or "")),
    )


def _block_type(block: Any) -> str | None:
    value = _get(block, "type")
    if value:
        return str(value)
    name = type(block).__name__.lower()
    if "tooluse" in name or "tool_use" in name:
        return "tool_use"
    if "toolresult" in name or "tool_result" in name:
        return "tool_result"
    if "thinking" in name:
        return "thinking"
    if "text" in name:
        return "text"
    return None


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _safe_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return {str(key): _json_safe(item) for key, item in dumped.items()}
    return {"value": _json_safe(value)}


def _assistant_text_excerpt(text: Any) -> str | None:
    if os.getenv("KSEARCH_TELEMETRY_RECORD_ASSISTANT_TEXT", "").strip().lower() in {"0", "false", "no", "off"}:
        return None
    return _truncate(str(text or ""))


def _tool_result_excerpt(content: Any) -> str | None:
    if os.getenv("KSEARCH_TELEMETRY_RECORD_TOOL_RESULTS", "").strip().lower() in {"0", "false", "no", "off"}:
        return None
    return _truncate(str(content or ""))


def _truncate(text: str) -> str:
    limit_raw = os.getenv("KSEARCH_TELEMETRY_MAX_TEXT_CHARS", "").strip()
    try:
        limit = int(limit_raw) if limit_raw else 4000
    except ValueError:
        limit = 4000
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 24)].rstrip() + "\n[truncated telemetry]"