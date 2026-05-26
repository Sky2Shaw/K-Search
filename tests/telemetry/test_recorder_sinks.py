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