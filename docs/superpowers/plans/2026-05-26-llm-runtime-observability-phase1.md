# LLM Runtime Observability Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add best-effort, file-based telemetry for Claude Agent SDK project-editing attempts so K-Search can observe tool calls, runtime events, and attempt cost.

**Architecture:** Add a focused `k_search.telemetry` package. `AscendCAgenticCodegenRunner` creates attempt-scoped context and recorder artifacts, `ClaudeAgentProjectEditorClient` emits SDK message events, and telemetry sinks write JSONL, Markdown timeline, and cost summary files without affecting codegen behavior.

**Tech Stack:** Python dataclasses, pathlib, JSON/JSONL, pytest, existing Claude Agent SDK mock helpers, existing agentic AscendC runner/editor modules.

---

## File Structure

- Create: `k_search/telemetry/__init__.py`
- Create: `k_search/telemetry/context.py`
- Create: `k_search/telemetry/events.py`
- Create: `k_search/telemetry/recorder.py`
- Create: `k_search/telemetry/sinks.py`
- Create: `k_search/telemetry/claude_sdk_adapter.py`
- Modify: `k_search/kernel_generators/claude_agent_project_editor.py`
- Modify: `k_search/kernel_generators/ascendc_agentic_codegen.py`
- Modify: `k_search/testing/mock_claude_agent_sdk.py`
- Test: `tests/telemetry/test_claude_sdk_adapter.py`
- Test: `tests/telemetry/test_recorder_sinks.py`
- Modify test: `tests/kernel_generators/test_claude_agent_sdk_mock.py`
- Modify test: `tests/kernel_generators/test_ascendc_agentic_codegen.py`

No OpenAI-compatible telemetry, Claude `query()` telemetry, AttemptRecord, postmortem, dashboard, or budget policy is included in this plan.

---

### Task 1: Telemetry Event And Context Core

**Files:**
- Create: `k_search/telemetry/__init__.py`
- Create: `k_search/telemetry/context.py`
- Create: `k_search/telemetry/events.py`
- Test: `tests/telemetry/test_recorder_sinks.py`

- [ ] **Step 1: Write failing tests for context paths and event serialization**

Create `tests/telemetry/test_recorder_sinks.py` with these initial tests:

```python
import json

from k_search.telemetry.context import (
    TelemetryContext,
    build_attempt_dir,
    is_telemetry_enabled,
    telemetry_root,
)
from k_search.telemetry.events import TelemetryEvent


def test_build_attempt_dir_uses_sanitized_attempt_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_TELEMETRY_DIR", str(tmp_path))
    monkeypatch.setenv("KSEARCH_RUN_ID", "run:alpha")
    context = TelemetryContext(
        task_name="task/name",
        round_index=7,
        attempt_index=2,
        action_node_id="n/12",
    )

    path = build_attempt_dir(context)

    assert path == tmp_path / "task_name" / "run_alpha" / "round_0007" / "action_n_12" / "attempt_0002"


def test_telemetry_root_defaults_to_project_local_output(monkeypatch, tmp_path):
    monkeypatch.delenv("KSEARCH_TELEMETRY_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    assert telemetry_root() == tmp_path / ".ksearch-output-mqa" / "telemetry"


def test_is_telemetry_enabled_honors_falsey_env(monkeypatch):
    monkeypatch.setenv("KSEARCH_TELEMETRY", "off")

    assert is_telemetry_enabled() is False


def test_event_to_dict_omits_none_and_serializes_context():
    event = TelemetryEvent(
        event_type="tool_use",
        context={"round_index": 1},
        tool_name="Read",
        tool_input={"file_path": "kernel/foo.h"},
        total_cost_usd=None,
    )

    payload = event.to_dict()

    assert payload["event_type"] == "tool_use"
    assert payload["context"] == {"round_index": 1}
    assert payload["tool_name"] == "Read"
    assert payload["tool_input"] == {"file_path": "kernel/foo.h"}
    assert "total_cost_usd" not in payload
    json.dumps(payload)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/telemetry/test_recorder_sinks.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'k_search.telemetry'`.

- [ ] **Step 3: Create telemetry package exports**

Create `k_search/telemetry/__init__.py`:

```python
"""Best-effort runtime telemetry for K-Search LLM calls."""

from k_search.telemetry.context import TelemetryContext, TelemetryArtifacts
from k_search.telemetry.events import TelemetryEvent

__all__ = [
    "TelemetryArtifacts",
    "TelemetryContext",
    "TelemetryEvent",
]
```

- [ ] **Step 4: Implement context helpers**

Create `k_search/telemetry/context.py`:

```python
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TelemetryContext:
    run_id: str | None = None
    task_name: str | None = None
    definition: str | None = None
    flow: str | None = None
    stage: str | None = None
    round_index: int | None = None
    attempt_index: int | None = None
    action_node_id: str | None = None
    action_title: str | None = None
    model_name: str | None = None
    provider: str | None = None
    target_gpu: str | None = None
    language: str | None = None
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True)
class TelemetryArtifacts:
    trace_path: str | None = None
    timeline_path: str | None = None
    cost_path: str | None = None


def is_telemetry_enabled() -> bool:
    raw = os.getenv("KSEARCH_TELEMETRY", "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def telemetry_root() -> Path:
    raw = os.getenv("KSEARCH_TELEMETRY_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.cwd() / ".ksearch-output-mqa" / "telemetry").resolve()


def default_run_id() -> str:
    for name in ("KSEARCH_RUN_ID", "KSEARCH_RUN_START"):
        raw = os.getenv(name, "").strip()
        if raw:
            return raw
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def safe_path_component(value: Any, *, default: str, max_len: int = 96) -> str:
    text = str(value if value is not None else "").strip() or default
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text).strip(".")
    return (safe or default)[:max_len]


def _round_component(value: int | None) -> str:
    if value is None:
        return "round_global"
    return f"round_{int(value):04d}"


def _attempt_component(value: int | None) -> str:
    if value is None:
        return "attempt_unknown"
    return f"attempt_{int(value):04d}"


def build_attempt_dir(context: TelemetryContext, *, root: Path | None = None) -> Path:
    task_name = safe_path_component(context.task_name or context.definition, default="__unknown__")
    run_id = safe_path_component(context.run_id or default_run_id(), default="run")
    action = "action_" + safe_path_component(context.action_node_id, default="unknown")
    return (
        (root or telemetry_root())
        / task_name
        / run_id
        / _round_component(context.round_index)
        / action
        / _attempt_component(context.attempt_index)
    )
```

- [ ] **Step 5: Implement event model**

Create `k_search/telemetry/events.py`:

```python
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
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
pytest tests/telemetry/test_recorder_sinks.py -v
```

Expected: PASS for the four core tests.

Commit:

```bash
git add k_search/telemetry/__init__.py k_search/telemetry/context.py k_search/telemetry/events.py tests/telemetry/test_recorder_sinks.py
git commit -m "feat: add telemetry context and events"
```

---

### Task 2: Recorder And File Sinks

**Files:**
- Create: `k_search/telemetry/recorder.py`
- Create: `k_search/telemetry/sinks.py`
- Modify: `tests/telemetry/test_recorder_sinks.py`

- [ ] **Step 1: Append failing recorder and sink tests**

Append to `tests/telemetry/test_recorder_sinks.py`:

```python
from pathlib import Path

from k_search.telemetry.recorder import TelemetryRecorder, build_file_recorder, noop_recorder
from k_search.telemetry.sinks import CostJsonSink, JsonlSink, MarkdownTimelineSink


def test_jsonl_sink_writes_one_event_per_line(tmp_path):
    path = tmp_path / "agent_trace.jsonl"
    sink = JsonlSink(path)
    sink.write_event(TelemetryEvent(event_type="llm_start", model_name="claude"))
    sink.write_event(TelemetryEvent(event_type="tool_use", tool_name="Glob", tool_input={"pattern": "**/*.h"}))
    sink.close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event_type"] for row in rows] == ["llm_start", "tool_use"]
    assert rows[1]["tool_name"] == "Glob"


def test_markdown_timeline_sink_formats_tool_events(tmp_path):
    path = tmp_path / "tool_timeline.md"
    sink = MarkdownTimelineSink(path)
    sink.write_event(TelemetryEvent(event_type="llm_start", provider="claude-agent", model_name="claude"))
    sink.write_event(TelemetryEvent(event_type="tool_use", tool_name="Read", tool_input={"file_path": "kernel/foo.h"}))
    sink.write_event(TelemetryEvent(event_type="tool_result", tool_name="Read", tool_result_excerpt="alpha"))
    sink.close()

    text = path.read_text(encoding="utf-8")
    assert "# Claude Agent Timeline" in text
    assert "tool_use: Read" in text
    assert "kernel/foo.h" in text
    assert "tool_result" in text


def test_cost_sink_writes_latest_result_on_close(tmp_path):
    path = tmp_path / "cost.json"
    sink = CostJsonSink(path)
    sink.write_event(TelemetryEvent(event_type="llm_start", model_name="claude"))
    sink.write_event(
        TelemetryEvent(
            event_type="llm_result",
            provider="claude-agent",
            model_name="claude",
            session_id="sess-1",
            total_cost_usd=0.125,
            duration_ms=1000,
            duration_api_ms=800,
            num_turns=3,
            usage={"input_tokens": 10},
            model_usage={"claude": {"output_tokens": 5}},
        )
    )
    sink.close()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["session_id"] == "sess-1"
    assert payload["summary"]["total_cost_usd"] == 0.125
    assert payload["usage"] == {"input_tokens": 10}


def test_recorder_merges_context_and_swallows_sink_errors(tmp_path):
    class BrokenSink:
        def write_event(self, event):
            raise RuntimeError("disk unhappy")

        def close(self):
            raise RuntimeError("close unhappy")

    path = tmp_path / "trace.jsonl"
    context = TelemetryContext(task_name="x", round_index=4)
    recorder = TelemetryRecorder(context=context, sinks=[BrokenSink(), JsonlSink(path)])

    recorder.emit(TelemetryEvent(event_type="tool_use", tool_name="Grep"))
    recorder.close()

    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["context"]["task_name"] == "x"
    assert row["context"]["round_index"] == 4
    assert row["tool_name"] == "Grep"


def test_build_file_recorder_creates_prompt_and_artifact_paths(tmp_path):
    context = TelemetryContext(task_name="task", run_id="run", round_index=1, attempt_index=1)

    recorder = build_file_recorder(context=context, prompt="hello prompt", root=tmp_path)
    recorder.close()

    assert recorder.artifacts.trace_path is not None
    assert recorder.artifacts.timeline_path is not None
    assert recorder.artifacts.cost_path is not None
    attempt_dir = Path(recorder.artifacts.trace_path).parent
    assert (attempt_dir / "prompt.md").read_text(encoding="utf-8") == "hello prompt"


def test_noop_recorder_has_empty_artifacts():
    recorder = noop_recorder()

    recorder.emit(TelemetryEvent(event_type="llm_start"))
    recorder.close()

    assert recorder.artifacts.trace_path is None
    assert recorder.events == []
```

- [ ] **Step 2: Run tests to verify missing recorder and sink modules**

Run:

```bash
pytest tests/telemetry/test_recorder_sinks.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `k_search.telemetry.recorder` or `k_search.telemetry.sinks`.

- [ ] **Step 3: Implement sinks**

Create `k_search/telemetry/sinks.py`:

```python
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
```

- [ ] **Step 4: Implement recorder**

Create `k_search/telemetry/recorder.py`:

```python
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
```

- [ ] **Step 5: Run recorder/sink tests and commit**

Run:

```bash
pytest tests/telemetry/test_recorder_sinks.py -v
```

Expected: PASS.

Commit:

```bash
git add k_search/telemetry/recorder.py k_search/telemetry/sinks.py tests/telemetry/test_recorder_sinks.py
git commit -m "feat: add telemetry recorder and sinks"
```

---

### Task 3: Claude SDK Adapter

**Files:**
- Create: `k_search/telemetry/claude_sdk_adapter.py`
- Test: `tests/telemetry/test_claude_sdk_adapter.py`

- [ ] **Step 1: Write failing adapter tests**

Create `tests/telemetry/test_claude_sdk_adapter.py`:

```python
from types import SimpleNamespace

from k_search.telemetry.claude_sdk_adapter import event_from_claude_message


def _event_types(message):
    return [event.event_type for event in event_from_claude_message(message)]


def test_text_block_converts_to_assistant_text():
    message = SimpleNamespace(content=[{"type": "text", "text": "edited file"}])

    events = event_from_claude_message(message)

    assert _event_types(message) == ["assistant_text"]
    assert events[0].text_excerpt == "edited file"


def test_tool_use_block_converts_to_tool_use():
    block = SimpleNamespace(type="tool_use", id="toolu_1", name="Read", input={"file_path": "kernel/foo.h"})
    message = SimpleNamespace(content=[block])

    event = event_from_claude_message(message)[0]

    assert event.event_type == "tool_use"
    assert event.tool_use_id == "toolu_1"
    assert event.tool_name == "Read"
    assert event.tool_input == {"file_path": "kernel/foo.h"}


def test_tool_result_block_converts_to_tool_result():
    block = SimpleNamespace(type="tool_result", tool_use_id="toolu_1", content="alpha", is_error=False)
    message = SimpleNamespace(content=[block])

    event = event_from_claude_message(message)[0]

    assert event.event_type == "tool_result"
    assert event.tool_use_id == "toolu_1"
    assert event.tool_result_excerpt == "alpha"
    assert event.is_error is False


def test_result_message_captures_cost_and_session_metadata():
    message = SimpleNamespace(
        result="final summary",
        session_id="sess-1",
        total_cost_usd=0.25,
        duration_ms=1200,
        duration_api_ms=900,
        num_turns=4,
        usage={"input_tokens": 10},
        model_usage={"claude": {"output_tokens": 5}},
        subtype="success",
        is_error=False,
    )

    event = event_from_claude_message(message)[0]

    assert event.event_type == "llm_result"
    assert event.session_id == "sess-1"
    assert event.total_cost_usd == 0.25
    assert event.num_turns == 4
    assert event.text_excerpt == "final summary"


def test_thinking_block_records_only_metadata():
    block = SimpleNamespace(type="thinking", thinking="private reasoning text")
    message = SimpleNamespace(content=[block])

    event = event_from_claude_message(message)[0]

    assert event.event_type == "assistant_thinking_metadata"
    assert "private reasoning text" not in str(event.to_dict())
    assert event.text_excerpt == "thinking_chars=22"


def test_unknown_block_does_not_raise():
    message = SimpleNamespace(content=[SimpleNamespace(type="new_block", value={"x": 1})])

    events = event_from_claude_message(message)

    assert events[0].event_type == "system_message"
    assert "new_block" in events[0].raw_type
```

- [ ] **Step 2: Run tests to verify adapter is missing**

Run:

```bash
pytest tests/telemetry/test_claude_sdk_adapter.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'k_search.telemetry.claude_sdk_adapter'`.

- [ ] **Step 3: Implement Claude SDK adapter**

Create `k_search/telemetry/claude_sdk_adapter.py`:

```python
from __future__ import annotations

import os
from typing import Any

from k_search.telemetry.events import TelemetryEvent


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


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


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
```

- [ ] **Step 4: Run adapter tests and commit**

Run:

```bash
pytest tests/telemetry/test_claude_sdk_adapter.py -v
```

Expected: PASS.

Commit:

```bash
git add k_search/telemetry/claude_sdk_adapter.py tests/telemetry/test_claude_sdk_adapter.py
git commit -m "feat: adapt claude sdk messages to telemetry events"
```

---

### Task 4: Project Editor Telemetry Integration

**Files:**
- Modify: `k_search/kernel_generators/claude_agent_project_editor.py`
- Modify: `k_search/testing/mock_claude_agent_sdk.py`
- Modify test: `tests/kernel_generators/test_claude_agent_sdk_mock.py`

- [ ] **Step 1: Add failing project editor telemetry test**

Append to `tests/kernel_generators/test_claude_agent_sdk_mock.py`:

```python
import json

from k_search.telemetry.context import TelemetryContext
from k_search.telemetry.recorder import build_file_recorder


def test_claude_project_editor_writes_tool_timeline_and_cost(monkeypatch, tmp_path):
    from pathlib import Path
    from types import SimpleNamespace
    from k_search.kernel_generators.claude_agent_project_editor import ClaudeAgentProjectEditorClient

    (tmp_path / "project" / "kernel").mkdir(parents=True)
    (tmp_path / "project" / "kernel" / "foo.h").write_text("alpha\nbeta\n", encoding="utf-8")

    def edit_project(prompt, options, call_index):
        project_dir = Path(options.kwargs["cwd"])
        (project_dir / "kernel" / "foo.h").write_text("alpha\nBETA\n", encoding="utf-8")
        return [
            MockClaudeMessage(content=[SimpleNamespace(type="tool_use", id="toolu_1", name="Glob", input={"pattern": "**/*.h"})]),
            MockClaudeMessage(content=[SimpleNamespace(type="tool_result", tool_use_id="toolu_1", content="kernel/foo.h", is_error=False)]),
            MockClaudeMessage(content=[SimpleNamespace(type="tool_use", id="toolu_2", name="Read", input={"file_path": "kernel/foo.h"})]),
            MockClaudeMessage(content=[{"type": "text", "text": "edited foo.h"}]),
            MockClaudeMessage(
                result="final summary",
                session_id="sess-1",
                total_cost_usd=0.125,
                usage={"input_tokens": 10},
                model_usage={"claude": {"output_tokens": 5}},
                duration_ms=1000,
                duration_api_ms=800,
                num_turns=3,
            ),
        ]

    install_mock_claude_agent_sdk(monkeypatch, responses=[edit_project])
    recorder = build_file_recorder(
        context=TelemetryContext(task_name="task", run_id="run", round_index=1, attempt_index=1, model_name="claude"),
        prompt="Please edit the project.",
        root=tmp_path / "telemetry",
    )
    client = ClaudeAgentProjectEditorClient(model_name="claude", timeout_seconds=30)

    result = client.edit_project(
        project_dir=tmp_path / "project",
        prompt="Please edit the project.",
        telemetry_recorder=recorder,
    )
    recorder.close()

    assert result.text == "final summary"
    assert result.trace_path == recorder.artifacts.trace_path
    assert result.timeline_path == recorder.artifacts.timeline_path
    assert result.cost_path == recorder.artifacts.cost_path
    assert result.session_id == "sess-1"
    assert result.total_cost_usd == 0.125
    rows = [json.loads(line) for line in Path(result.trace_path).read_text(encoding="utf-8").splitlines()]
    assert [row["event_type"] for row in rows if row["event_type"] == "tool_use"] == ["tool_use", "tool_use"]
    assert "tool_use: Glob" in Path(result.timeline_path).read_text(encoding="utf-8")
    assert json.loads(Path(result.cost_path).read_text(encoding="utf-8"))["summary"]["session_id"] == "sess-1"
```

- [ ] **Step 2: Run the new test to verify the editor signature fails**

Run:

```bash
pytest tests/kernel_generators/test_claude_agent_sdk_mock.py::test_claude_project_editor_writes_tool_timeline_and_cost -v
```

Expected: FAIL with `TypeError` because `edit_project()` does not accept `telemetry_recorder`.

- [ ] **Step 3: Extend mock messages with result metadata passthrough**

Modify `k_search/testing/mock_claude_agent_sdk.py`.

Change `MockClaudeMessage.__init__` signature and body to support arbitrary SDK-like attributes:

```python
class MockClaudeMessage:
    """Message object with only explicitly supplied SDK-like attributes."""

    def __init__(
        self,
        *,
        result: Any = _MISSING,
        text: Any = _MISSING,
        content: Any = _MISSING,
        is_error: Any = _MISSING,
        subtype: Any = _MISSING,
        **extra: Any,
    ) -> None:
        if result is not _MISSING:
            self.result = result
        if text is not _MISSING:
            self.text = text
        if content is not _MISSING:
            self.content = content
        if is_error is not _MISSING:
            self.is_error = is_error
        if subtype is not _MISSING:
            self.subtype = subtype
        for key, value in extra.items():
            setattr(self, key, value)
```

- [ ] **Step 4: Extend project edit result and editor signature**

Modify imports in `k_search/kernel_generators/claude_agent_project_editor.py`:

```python
from k_search.telemetry.claude_sdk_adapter import event_from_claude_message
from k_search.telemetry.events import TelemetryEvent
from k_search.telemetry.recorder import TelemetryRecorder, noop_recorder
```

Extend `ClaudeProjectEditResult`:

```python
@dataclass
class ClaudeProjectEditResult:
    text: str
    transcript: str
    prompt: str
    prompt_chars: int
    prompt_lines: int
    trace_path: str | None = None
    timeline_path: str | None = None
    cost_path: str | None = None
    session_id: str | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    model_usage: dict[str, Any] | None = None
    num_turns: int | None = None
    duration_ms: int | None = None
```

Change `edit_project` signature:

```python
def edit_project(
    self,
    *,
    project_dir: str | Path,
    prompt: str,
    telemetry_recorder: TelemetryRecorder | None = None,
) -> ClaudeProjectEditResult:
```

- [ ] **Step 5: Emit telemetry events in the SDK stream**

Inside `edit_project()`, create a recorder variable after SDK import:

```python
        recorder = telemetry_recorder or noop_recorder()
```

Inside `_run_edit()`, before `await client.query(prompt_text)`, emit:

```python
                    recorder.emit(
                        TelemetryEvent(
                            event_type="llm_start",
                            provider="claude-agent",
                            model_name=self.model_name,
                        )
                    )
```

Inside the `receive_response()` loop, before transcript extraction, emit adapter events and capture result metadata:

```python
            result_event: TelemetryEvent | None = None
```

Then in the loop:

```python
                        for event in event_from_claude_message(message):
                            event.provider = event.provider or "claude-agent"
                            event.model_name = event.model_name or self.model_name
                            recorder.emit(event)
                            if event.event_type == "llm_result":
                                result_event = event
```

After the loop succeeds, emit:

```python
                recorder.emit(
                    TelemetryEvent(
                        event_type="llm_end",
                        provider="claude-agent",
                        model_name=self.model_name,
                    )
                )
```

When returning `ClaudeProjectEditResult`, include:

```python
                trace_path=recorder.artifacts.trace_path,
                timeline_path=recorder.artifacts.timeline_path,
                cost_path=recorder.artifacts.cost_path,
                session_id=result_event.session_id if result_event else None,
                total_cost_usd=result_event.total_cost_usd if result_event else None,
                usage=result_event.usage if result_event else None,
                model_usage=result_event.model_usage if result_event else None,
                num_turns=result_event.num_turns if result_event else None,
                duration_ms=result_event.duration_ms if result_event else None,
```

- [ ] **Step 6: Emit llm_error before re-raising provider failures**

In both `except Exception as exc` blocks in `edit_project()`, before raising, emit:

```python
                recorder.emit(
                    TelemetryEvent(
                        event_type="llm_error",
                        provider="claude-agent",
                        model_name=self.model_name,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
```

Use the existing exception conversion and raising behavior exactly as it is today after this emit.

- [ ] **Step 7: Run editor tests and commit**

Run:

```bash
pytest tests/kernel_generators/test_claude_agent_sdk_mock.py::test_claude_project_editor_client_uses_sdk_client_with_cwd_and_file_tools tests/kernel_generators/test_claude_agent_sdk_mock.py::test_claude_project_editor_writes_tool_timeline_and_cost -v
```

Expected: PASS.

Commit:

```bash
git add k_search/kernel_generators/claude_agent_project_editor.py k_search/testing/mock_claude_agent_sdk.py tests/kernel_generators/test_claude_agent_sdk_mock.py
git commit -m "feat: record telemetry for claude project editor"
```

---

### Task 5: AscendC Agentic Runner Telemetry Integration

**Files:**
- Modify: `k_search/kernel_generators/ascendc_agentic_codegen.py`
- Modify test: `tests/kernel_generators/test_ascendc_agentic_codegen.py`

- [ ] **Step 1: Add failing runner telemetry tests**

Append to `tests/kernel_generators/test_ascendc_agentic_codegen.py`:

```python
def test_runner_creates_attempt_telemetry_files(tmp_path, monkeypatch):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "kernel").mkdir()
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    monkeypatch.setenv("KSEARCH_TELEMETRY_DIR", str(tmp_path / "telemetry"))
    monkeypatch.setenv("KSEARCH_RUN_ID", "run-1")
    task = AscendCTask(task_path=task_dir, definition_name="x")

    class TelemetryAwareClient:
        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            root = Path(project_dir)
            (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
            if telemetry_recorder is not None:
                from k_search.telemetry.events import TelemetryEvent

                telemetry_recorder.emit(TelemetryEvent(event_type="llm_start", provider="claude-agent", model_name="claude"))
                telemetry_recorder.emit(
                    TelemetryEvent(
                        event_type="llm_result",
                        provider="claude-agent",
                        model_name="claude",
                        session_id="sess-runner",
                        total_cost_usd=0.5,
                        num_turns=2,
                        duration_ms=100,
                    )
                )
            return ClaudeProjectEditResult(
                text="edited",
                transcript="edited",
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
                trace_path=telemetry_recorder.artifacts.trace_path if telemetry_recorder else None,
                timeline_path=telemetry_recorder.artifacts.timeline_path if telemetry_recorder else None,
                cost_path=telemetry_recorder.artifacts.cost_path if telemetry_recorder else None,
                session_id="sess-runner",
                total_cost_usd=0.5,
                num_turns=2,
                duration_ms=100,
            )

    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=TelemetryAwareClient())

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec",
            action_text="change beta",
            trace_logs="",
            perf_summary="",
            target_gpu="ascend_910b",
            round_num=3,
            attempt_idx=2,
            mode="action",
        ),
        base_solution=None,
    )

    assert result.trace_path is not None
    assert result.timeline_path is not None
    assert result.cost_path is not None
    assert Path(result.trace_path).exists()
    assert Path(result.timeline_path).exists()
    assert Path(result.cost_path).exists()
    assert Path(result.trace_path).parent.name == "attempt_0002"
    assert Path(result.trace_path).parent.parent.name == "action_unknown"
    assert result.session_id == "sess-runner"
    assert result.total_cost_usd == 0.5


def test_runner_disables_telemetry_with_env(tmp_path, monkeypatch):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "kernel").mkdir()
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    monkeypatch.setenv("KSEARCH_TELEMETRY", "0")
    task = AscendCTask(task_path=task_dir, definition_name="x")

    class Client:
        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            Path(project_dir, "kernel", "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
            return ClaudeProjectEditResult(
                text="edited",
                transcript="edited",
                prompt=prompt,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1,
            )

    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=Client())

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec",
            action_text="change beta",
            trace_logs="",
            perf_summary="",
            target_gpu="ascend_910b",
            round_num=1,
            attempt_idx=1,
            mode="action",
        ),
        base_solution=None,
    )

    assert result.trace_path is None
    assert result.timeline_path is None
    assert result.cost_path is None
    assert result.changed_paths == ["kernel/foo.h"]
```

- [ ] **Step 2: Run new runner tests to verify result fields are missing**

Run:

```bash
pytest tests/kernel_generators/test_ascendc_agentic_codegen.py::test_runner_creates_attempt_telemetry_files tests/kernel_generators/test_ascendc_agentic_codegen.py::test_runner_disables_telemetry_with_env -v
```

Expected: FAIL with missing `trace_path` on `AscendCAgenticCodegenResult` or missing `telemetry_recorder` argument in fake clients.

- [ ] **Step 3: Extend AscendC agentic result dataclass**

Modify `AscendCAgenticCodegenResult` in `k_search/kernel_generators/ascendc_agentic_codegen.py`:

```python
@dataclass
class AscendCAgenticCodegenResult:
    solution: Solution
    raw: str
    cleaned: dict[str, str]
    transcript: str
    prompt: str
    prompt_chars: int
    changed_paths: list[str]
    diff_text: str
    project_path: str
    trace_path: str | None = None
    timeline_path: str | None = None
    cost_path: str | None = None
    session_id: str | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    model_usage: dict[str, Any] | None = None
    num_turns: int | None = None
    duration_ms: int | None = None
```

- [ ] **Step 4: Build telemetry context and recorder in runner**

Add imports to `k_search/kernel_generators/ascendc_agentic_codegen.py`:

```python
from k_search.telemetry.context import TelemetryContext
from k_search.telemetry.recorder import build_file_recorder
```

In `run()`, after `prompt = self.prompt_builder.build(request)`, add:

```python
            telemetry_context = TelemetryContext(
                task_name=getattr(task, "definition_name", None),
                definition=getattr(task, "definition_name", None),
                flow="agentic_codegen",
                stage=request.mode,
                round_index=request.round_num,
                attempt_index=request.attempt_idx,
                model_name=self.model_name,
                provider="claude-agent",
                target_gpu=request.target_gpu,
                language="ascendc",
            )
            telemetry_recorder = build_file_recorder(context=telemetry_context, prompt=prompt)
```

Call the editor with recorder:

```python
            try:
                edit_result: ClaudeProjectEditResult = self.editor_client.edit_project(
                    project_dir=session.project_dir,
                    prompt=prompt,
                    telemetry_recorder=telemetry_recorder,
                )
            finally:
                telemetry_recorder.close()
```

- [ ] **Step 5: Keep compatibility with simple fake clients**

Some existing fake clients accept only `project_dir` and `prompt`. Add a helper near the runner:

```python
def _edit_project_with_optional_telemetry(
    editor_client: Any,
    *,
    project_dir: Path,
    prompt: str,
    telemetry_recorder: Any,
) -> ClaudeProjectEditResult:
    try:
        return editor_client.edit_project(
            project_dir=project_dir,
            prompt=prompt,
            telemetry_recorder=telemetry_recorder,
        )
    except TypeError as exc:
        if "telemetry_recorder" not in str(exc):
            raise
        return editor_client.edit_project(project_dir=project_dir, prompt=prompt)
```

Use the helper inside the `try/finally`:

```python
                edit_result = _edit_project_with_optional_telemetry(
                    self.editor_client,
                    project_dir=session.project_dir,
                    prompt=prompt,
                    telemetry_recorder=telemetry_recorder,
                )
```

- [ ] **Step 6: Propagate telemetry fields to result**

Add these fields to the `AscendCAgenticCodegenResult(...)` return:

```python
                trace_path=edit_result.trace_path or telemetry_recorder.artifacts.trace_path,
                timeline_path=edit_result.timeline_path or telemetry_recorder.artifacts.timeline_path,
                cost_path=edit_result.cost_path or telemetry_recorder.artifacts.cost_path,
                session_id=edit_result.session_id,
                total_cost_usd=edit_result.total_cost_usd,
                usage=edit_result.usage,
                model_usage=edit_result.model_usage,
                num_turns=edit_result.num_turns,
                duration_ms=edit_result.duration_ms,
```

- [ ] **Step 7: Run runner tests and commit**

Run:

```bash
pytest tests/kernel_generators/test_ascendc_agentic_codegen.py -v
```

Expected: PASS.

Commit:

```bash
git add k_search/kernel_generators/ascendc_agentic_codegen.py tests/kernel_generators/test_ascendc_agentic_codegen.py
git commit -m "feat: attach telemetry to ascendc agentic attempts"
```

---

### Task 6: Full Regression And Polish

**Files:**
- Review: `k_search/telemetry/*.py`
- Review: `k_search/kernel_generators/claude_agent_project_editor.py`
- Review: `k_search/kernel_generators/ascendc_agentic_codegen.py`
- Review tests: `tests/telemetry/*.py`

- [ ] **Step 1: Run focused telemetry and agentic suites**

Run:

```bash
pytest tests/telemetry tests/kernel_generators/test_claude_agent_sdk_mock.py tests/kernel_generators/test_ascendc_agentic_codegen.py -v
```

Expected: PASS.

- [ ] **Step 2: Run existing LLM client regressions**

Run:

```bash
pytest tests/kernel_generators/test_llm_clients.py -v
```

Expected: PASS.

- [ ] **Step 3: Run whitespace and red-flag checks**

Run:

```bash
git diff --check
rg -n "TO""DO|TB""D|FIX""ME|place""holder" k_search/telemetry k_search/kernel_generators/claude_agent_project_editor.py k_search/kernel_generators/ascendc_agentic_codegen.py tests/telemetry tests/kernel_generators/test_claude_agent_sdk_mock.py tests/kernel_generators/test_ascendc_agentic_codegen.py
```

Expected: `git diff --check` prints nothing. `rg` exits with no matches.

- [ ] **Step 4: Inspect generated sample telemetry from the integration test**

Run:

```bash
pytest tests/kernel_generators/test_claude_agent_sdk_mock.py::test_claude_project_editor_writes_tool_timeline_and_cost -v
```

Expected: PASS. If the test output path is not printed, inspect the temporary directory by adding a local `print(result.timeline_path)` while debugging and remove the print before committing. The committed code must not include debug prints.

- [ ] **Step 5: Commit final polish if files changed**

If Step 1 through Step 4 required polish changes, commit them:

```bash
git add k_search/telemetry k_search/kernel_generators/claude_agent_project_editor.py k_search/kernel_generators/ascendc_agentic_codegen.py tests/telemetry tests/kernel_generators/test_claude_agent_sdk_mock.py tests/kernel_generators/test_ascendc_agentic_codegen.py
git commit -m "test: verify llm telemetry phase 1"
```

If no files changed after the earlier task commits, skip this commit.

---

## Completion Checklist

- [ ] `prompt.md`, `agent_trace.jsonl`, `tool_timeline.md`, and `cost.json` are produced for Claude agentic codegen when telemetry is enabled.
- [ ] `KSEARCH_TELEMETRY=0` disables artifact creation and leaves codegen behavior intact.
- [ ] `tool_timeline.md` is readable without `jq` and includes common tool summaries.
- [ ] `agent_trace.jsonl` is one event per line and each write is flushed.
- [ ] `cost.json` includes `summary`, `usage`, and `model_usage`.
- [ ] No private thinking text is recorded.
- [ ] Existing agentic codegen tests still pass.
- [ ] Telemetry sink failures do not raise from `TelemetryRecorder.emit()`.
