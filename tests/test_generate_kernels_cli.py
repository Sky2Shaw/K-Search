from types import SimpleNamespace

import pytest

from generate_kernels_and_eval import _build_task_from_args, _resolve_llm_config_from_args


def test_resolve_llm_config_defaults_to_openai_and_reads_env_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    args = SimpleNamespace(llm_provider=None, api_key=None)

    llm_provider, api_key = _resolve_llm_config_from_args(args)

    assert llm_provider == "openai"
    assert api_key == "env-key"


def test_resolve_llm_config_openai_requires_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    args = SimpleNamespace(llm_provider="openai", api_key=None)

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        _resolve_llm_config_from_args(args)


def test_resolve_llm_config_claude_agent_does_not_require_llm_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    args = SimpleNamespace(llm_provider="claude-agent", api_key=None)

    llm_provider, api_key = _resolve_llm_config_from_args(args)

    assert llm_provider == "claude-agent"
    assert api_key is None


def test_resolve_llm_config_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    args = SimpleNamespace(llm_provider="other", api_key=None)

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        _resolve_llm_config_from_args(args)


def test_build_task_from_args_constructs_ascendc_task(tmp_path):
    (tmp_path / "spec.md").write_text("AscendC vector add operator.", encoding="utf-8")
    args = SimpleNamespace(
        task_source="ascendc",
        task_path=str(tmp_path),
        local=None,
        definition="vec_add",
        ascendc_build_cmd="echo build",
        ascendc_test_cmd="echo test",
        ascendc_bench_cmd="echo latency_ms=1.0",
        ascendc_timeout_seconds=12,
        ascendc_reference_latency_ms=2.0,
        artifacts_dir=".ksearch-test",
    )

    task = _build_task_from_args(args)

    assert task.name == "vec_add"
    cfg = task.get_config_for_logging()
    assert cfg["task_source"] == "ascendc"
    assert cfg["build_cmd"] == "echo build"
    assert cfg["reference_latency_ms"] == 2.0
