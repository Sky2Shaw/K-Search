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


def test_is_telemetry_enabled_defaults_to_true(monkeypatch):
    monkeypatch.delenv("KSEARCH_TELEMETRY", raising=False)

    assert is_telemetry_enabled() is True


def test_is_telemetry_enabled_truthy_values(monkeypatch):
    for value in ("1", "true", "on", "yes"):
        monkeypatch.setenv("KSEARCH_TELEMETRY", value)
        assert is_telemetry_enabled() is True


def test_default_run_id_prefers_ksearch_run_id(monkeypatch):
    monkeypatch.setenv("KSEARCH_RUN_ID", "run-1")
    monkeypatch.setenv("KSEARCH_RUN_START", "run-2")

    from k_search.telemetry.context import default_run_id

    assert default_run_id() == "run-1"


def test_default_run_id_falls_back_to_timestamp(monkeypatch):
    monkeypatch.delenv("KSEARCH_RUN_ID", raising=False)
    monkeypatch.delenv("KSEARCH_RUN_START", raising=False)

    from k_search.telemetry.context import default_run_id

    result = default_run_id()
    assert len(result) == 15  # YYYYMMDD_HHMMSS
    assert "_" in result


def test_safe_path_component_truncates_long_values():
    from k_search.telemetry.context import safe_path_component

    long_value = "a" * 200
    result = safe_path_component(long_value, default="x")
    assert len(result) == 96


def test_safe_path_component_uses_default_for_empty():
    from k_search.telemetry.context import safe_path_component

    assert safe_path_component("", default="fallback") == "fallback"
    assert safe_path_component(None, default="fallback") == "fallback"


def test_build_attempt_dir_with_none_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("KSEARCH_TELEMETRY_DIR", str(tmp_path))
    monkeypatch.setenv("KSEARCH_RUN_ID", "run1")

    context = TelemetryContext(task_name="task")

    path = build_attempt_dir(context)

    assert "round_global" in str(path)
    assert "action_unknown" in str(path)
    assert "attempt_unknown" in str(path)


def test_telemetry_context_to_dict_omits_none():
    ctx = TelemetryContext(task_name="x", round_index=3)

    payload = ctx.to_dict()

    assert payload["task_name"] == "x"
    assert payload["round_index"] == 3
    assert "definition" not in payload
    assert "extra" not in payload