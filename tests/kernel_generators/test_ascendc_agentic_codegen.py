from pathlib import Path

import pytest

from k_search.kernel_generators.ascendc_agentic_codegen import (
    AscendCAgenticCodegenRequest,
    AscendCAgenticCodegenRunner,
    AscendCAgenticPromptBuilder,
)
from k_search.kernel_generators.claude_agent_project_editor import ClaudeProjectEditResult
from k_search.tasks.ascendc_task import AscendCTask


class EditingClient:
    def __init__(self, new_text: str):
        self.new_text = new_text
        self.calls = []

    def edit_project(self, *, project_dir, prompt):
        root = Path(project_dir)
        self.calls.append((root, prompt))
        target = root / "kernel" / "foo.h"
        target.write_text(self.new_text, encoding="utf-8")
        return ClaudeProjectEditResult(
            text="edited kernel/foo.h",
            transcript="located and edited kernel/foo.h",
            prompt=prompt,
            prompt_chars=len(prompt),
            prompt_lines=prompt.count("\n") + 1,
        )


class NoChangeClient:
    def edit_project(self, *, project_dir, prompt):
        return ClaudeProjectEditResult(
            text="no changes",
            transcript="no changes",
            prompt=prompt,
            prompt_chars=len(prompt),
            prompt_lines=prompt.count("\n") + 1,
        )


def test_prompt_builder_omits_full_project_container_and_includes_action():
    builder = AscendCAgenticPromptBuilder(max_chars=20_000)
    request = AscendCAgenticCodegenRequest(
        definition_text="Task: x\nSpecification:\nVector add.",
        action_text="Increase tile length within UB capacity.",
        trace_logs="compile ok",
        perf_summary="- last_attempt_mean_latency_ms: 1.2",
        target_gpu="ascend_910b",
        round_num=2,
        attempt_idx=1,
        mode="action",
    )

    prompt = builder.build(request)

    assert "Increase tile length" in prompt
    assert "ascend_910b" in prompt
    assert "compile ok" in prompt
    assert "<ascendc_project>" not in prompt
    assert "Read/Grep/Glob/Edit/Write" in prompt


def test_prompt_builder_raises_section_aware_error_when_budget_exceeded():
    builder = AscendCAgenticPromptBuilder(max_chars=200)
    request = AscendCAgenticCodegenRequest(
        definition_text="D" * 500,
        action_text="A" * 500,
        trace_logs="T" * 500,
        perf_summary="P" * 500,
        target_gpu="ascend_910b",
        round_num=1,
        attempt_idx=1,
        mode="debug",
    )

    with pytest.raises(ValueError, match="agentic prompt exceeded"):
        builder.build(request)


def test_runner_edits_worktree_and_returns_solution(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "spec.md").write_text("Optimize tiny project.", encoding="utf-8")
    (task_dir / "kernel").mkdir()
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x")
    client = EditingClient("alpha\nBETA\ngamma\n")
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)

    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text=task.get_agentic_definition_text(language="ascendc"),
            action_text="Change beta to BETA.",
            trace_logs="",
            perf_summary="",
            target_gpu="ascend_910b",
            round_num=3,
            attempt_idx=1,
            mode="action",
        ),
        base_solution=None,
    )

    assert "BETA" in next(src.content for src in result.solution.sources if src.path == "kernel/foo.h")
    assert result.changed_paths == ["kernel/foo.h"]
    assert "-beta" in result.diff_text
    assert "+BETA" in result.diff_text
    assert client.calls
    assert "<ascendc_project>" not in client.calls[0][1]


def test_runner_fails_when_agent_makes_no_file_changes(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "kernel.cpp").write_text("void run() {}\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x")
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=NoChangeClient())

    with pytest.raises(RuntimeError, match="did not change any files"):
        runner.run(
            task=task,
            request=AscendCAgenticCodegenRequest(
                definition_text="spec",
                action_text="change code",
                trace_logs="",
                perf_summary="",
                target_gpu="ascend_910b",
                round_num=1,
                attempt_idx=1,
                mode="action",
            ),
            base_solution=None,
        )