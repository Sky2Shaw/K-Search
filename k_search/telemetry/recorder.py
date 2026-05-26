from __future__ import annotations

from pathlib import Path
from typing import Iterable

from k_search.telemetry.context import (
    TelemetryArtifacts,
    TelemetryContext,
    build_attempt_dir,
    is_telemetry_enabled,
)
from k_search.telemetry.events import TelemetryEvent
from k_search.telemetry.sinks import CostJsonSink, JsonlSink, MarkdownTimelineSink, TelemetrySink


class TelemetryRecorder:
    def __init__(
        self,
        *,
        context: TelemetryContext | None = None,
        sinks: Iterable[TelemetrySink] = (),
        artifacts: TelemetryArtifacts | None = None,
        enabled: bool = True,
    ) -> None:
        self.context = context or TelemetryContext()
        self.sinks = list(sinks)
        self.artifacts = artifacts or TelemetryArtifacts()
        self.enabled = bool(enabled)
        self.events: list[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        if not self.enabled:
            return
        merged_context = self.context.to_dict()
        merged_context.update(event.context or {})
        event.context = merged_context
        self.events.append(event)
        for sink in self.sinks:
            try:
                sink.write_event(event)
            except Exception:
                pass

    def close(self) -> None:
        if not self.enabled:
            return
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                pass


def noop_recorder() -> TelemetryRecorder:
    return TelemetryRecorder(enabled=False)


def build_file_recorder(
    *,
    context: TelemetryContext,
    prompt: str,
    root: Path | None = None,
) -> TelemetryRecorder:
    if not is_telemetry_enabled():
        return noop_recorder()
    try:
        attempt_dir = build_attempt_dir(context, root=root)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "prompt.md").write_text(str(prompt or ""), encoding="utf-8")
        artifacts = TelemetryArtifacts(
            trace_path=str(attempt_dir / "agent_trace.jsonl"),
            timeline_path=str(attempt_dir / "tool_timeline.md"),
            cost_path=str(attempt_dir / "cost.json"),
        )
        return TelemetryRecorder(
            context=context,
            artifacts=artifacts,
            sinks=[
                JsonlSink(attempt_dir / "agent_trace.jsonl"),
                MarkdownTimelineSink(attempt_dir / "tool_timeline.md"),
                CostJsonSink(attempt_dir / "cost.json"),
            ],
        )
    except Exception:
        return noop_recorder()