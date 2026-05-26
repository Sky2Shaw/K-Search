from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

from k_search.telemetry.events import TelemetryEvent


class TelemetrySink(Protocol):
    def write_event(self, event: TelemetryEvent) -> None: ...
    def close(self) -> None: ...


class JsonlSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write_event(self, event: TelemetryEvent) -> None:
        self._fh.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class MarkdownTimelineSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._started_ms: int | None = None
        self._summary: dict[str, object] = {}
        self._fh = self.path.open("w", encoding="utf-8")
        self._fh.write("# Claude Agent Timeline\n\n## Timeline\n\n")
        self._fh.flush()

    def write_event(self, event: TelemetryEvent) -> None:
        if self._started_ms is None:
            self._started_ms = event.ts_ms
        if event.event_type == "llm_result":
            self._summary = _summary_from_event(event)
        self._fh.write(_format_timeline_event(event, started_ms=self._started_ms))
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
        if self._summary:
            body = self.path.read_text(encoding="utf-8")
            summary = "\n".join(f"- {key}: {value}" for key, value in self._summary.items())
            self.path.write_text(
                "# Claude Agent Timeline\n\n## Summary\n\n" + summary + "\n\n" + body.replace("# Claude Agent Timeline\n\n", ""),
                encoding="utf-8",
            )


class CostJsonSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._latest_result: TelemetryEvent | None = None

    def write_event(self, event: TelemetryEvent) -> None:
        if event.event_type == "llm_result":
            self._latest_result = event

    def close(self) -> None:
        event = self._latest_result
        payload = {
            "summary": _summary_from_event(event) if event is not None else {},
            "usage": event.usage if event is not None and event.usage is not None else {},
            "model_usage": event.model_usage if event is not None and event.model_usage is not None else {},
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary_from_event(event: TelemetryEvent | None) -> dict[str, object]:
    if event is None:
        return {}
    fields = {
        "provider": event.provider,
        "model_name": event.model_name,
        "session_id": event.session_id,
        "total_cost_usd": event.total_cost_usd,
        "num_turns": event.num_turns,
        "duration_ms": event.duration_ms,
        "duration_api_ms": event.duration_api_ms,
    }
    return {key: value for key, value in fields.items() if value is not None}


def _format_timeline_event(event: TelemetryEvent, *, started_ms: int) -> str:
    elapsed = max(0, event.ts_ms - started_ms) / 1000.0
    title = event.event_type
    if event.tool_name:
        title += f": {event.tool_name}"
    lines = [f"### {elapsed:08.3f} {title}", ""]
    if event.event_type == "llm_start":
        if event.provider:
            lines.append(f"Provider: `{event.provider}`")
        if event.model_name:
            lines.append(f"Model: `{event.model_name}`")
    elif event.event_type == "tool_use":
        summary = _tool_input_summary(event.tool_name, event.tool_input or {})
        if summary:
            lines.append(summary)
        if event.tool_input:
            lines.extend(["", "```json", json.dumps(event.tool_input, ensure_ascii=False, indent=2, sort_keys=True), "```"])
    elif event.event_type == "tool_result":
        lines.append("Status: error" if event.is_error else "Status: ok")
        if event.tool_result_excerpt:
            lines.extend(["", "Excerpt:", "", _markdown_text_block(event.tool_result_excerpt)])
    elif event.event_type == "assistant_text" and event.text_excerpt:
        lines.append(_markdown_text_block(event.text_excerpt))
    elif event.event_type == "llm_result":
        for key, value in _summary_from_event(event).items():
            lines.append(f"- {key}: {value}")
    elif event.event_type == "llm_error":
        lines.append(f"Error: `{event.error_type or 'Exception'}`")
        if event.error_message:
            lines.append(_markdown_text_block(event.error_message))
    return "\n".join(lines) + "\n\n"


def _tool_input_summary(tool_name: str | None, tool_input: dict[str, object]) -> str:
    if tool_name in {"Read", "Edit", "Write"} and tool_input.get("file_path"):
        return f"File: `{tool_input['file_path']}`"
    if tool_name == "Grep":
        parts = [f"{key}: `{tool_input[key]}`" for key in ("pattern", "path", "glob") if tool_input.get(key)]
        return "\n".join(parts)
    if tool_name == "Glob":
        parts = [f"{key}: `{tool_input[key]}`" for key in ("pattern", "path") if tool_input.get(key)]
        return "\n".join(parts)
    return ""


def _markdown_text_block(text: str) -> str:
    content = str(text or "")
    return "```text\n" + content + "\n```"