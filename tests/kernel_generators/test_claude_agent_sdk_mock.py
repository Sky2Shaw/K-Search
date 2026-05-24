import shlex
import sys

from k_search.kernel_generators.kernel_generator import KernelGenerator
from k_search.kernel_generators.llm_clients import ClaudeAgentLLMClient
from k_search.tasks.ascendc_task import AscendCTask, format_ascendc_project_files
from k_search.testing import (
    MockClaudeMessage,
    install_mock_claude_agent_sdk,
)


def _py_cmd(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def test_mock_claude_agent_sdk_records_options_and_streams_messages(monkeypatch):
    sdk = install_mock_claude_agent_sdk(
        monkeypatch,
        responses=[
            [
                MockClaudeMessage(content=[{"type": "text", "text": "assistant chunk"}]),
                MockClaudeMessage(result="final result"),
            ]
        ],
    )
    client = ClaudeAgentLLMClient(
        model_name="claude-sonnet-4-6",
        allowed_tools=["Read"],
        disallowed_tools=["Bash"],
    )

    assert client.generate("optimize this") == "final result"

    assert sdk.calls[0].prompt == "optimize this"
    assert sdk.calls[0].options.kwargs["model"] == "claude-sonnet-4-6"
    assert sdk.calls[0].options.kwargs["allowed_tools"] == ["Read"]
    assert sdk.calls[0].options.kwargs["disallowed_tools"] == ["Bash"]


def test_claude_agent_sdk_mock_drives_ascendc_two_round_optimization(
    monkeypatch, tmp_path
):
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (tmp_path / "spec.md").write_text("Optimize a tiny AscendC project.", encoding="utf-8")
    (kernel_dir / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    initial_project = format_ascendc_project_files(
        {"kernel/foo.h": "alpha\nbeta\ngamma\n"}
    )
    optimized_patch = (
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-beta\n"
        "+BETA\n"
        " gamma\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )
    sdk = install_mock_claude_agent_sdk(
        monkeypatch,
        responses=[initial_project, optimized_patch],
    )

    task = AscendCTask(
        task_path=tmp_path,
        definition_name="mock_ascendc",
        codegen_mode="auto",
        build_cmd=_py_cmd(
            "from pathlib import Path; "
            "assert Path('kernel/foo.h').exists(); "
            "print('build ok')"
        ),
        test_cmd=_py_cmd("print('correctness ok')"),
        bench_cmd=_py_cmd(
            "from pathlib import Path; "
            "text = Path('kernel/foo.h').read_text(); "
            "print('latency_ms=0.5' if 'BETA' in text else 'latency_ms=1.0')"
        ),
        reference_latency_ms=2.0,
        timeout_seconds=30,
    )
    generator = KernelGenerator(
        model_name="claude-sonnet-4-6",
        language="ascendc",
        target_gpu="ascend_910b",
        llm_provider="claude-agent",
    )

    solution = generator.generate(task=task, max_opt_rounds=2)

    foo = next(src for src in solution.sources if src.path == "kernel/foo.h")
    assert "BETA" in foo.content
    assert len(sdk.calls) == 2
    assert "<ascendc_project>" in sdk.calls[0].prompt
    assert "<ascendc_patch>" in sdk.calls[1].prompt
