from pathlib import Path

from k_search.kernel_generators.agents import AgentRunResult, ProjectAgent
from k_search.kernel_generators.claude_agent_project_editor import ClaudeProjectEditResult


class FakeClient:
    def __init__(self):
        self.calls = []

    def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
        self.calls.append((Path(project_dir), prompt))
        return ClaudeProjectEditResult(
            text="done", transcript="t", prompt=prompt,
            prompt_chars=len(prompt), prompt_lines=prompt.count("\n") + 1,
        )


class _Echo(ProjectAgent):
    allowed_tools = ["Read", "Grep"]

    def build_prompt(self, context) -> str:
        return f"PROMPT::{context}"


def test_project_agent_runs_and_returns_result(tmp_path):
    client = FakeClient()
    agent = _Echo(model_name="claude", editor_client=client)
    result = agent.run(project_dir=tmp_path, context="ctx")
    assert isinstance(result, AgentRunResult)
    assert result.text == "done"
    assert client.calls[0][1] == "PROMPT::ctx"


def test_project_agent_subclass_declares_tools():
    agent = _Echo(model_name="claude", editor_client=FakeClient())
    assert agent.allowed_tools == ["Read", "Grep"]
    assert "Bash" in agent.disallowed_tools


def test_tool_lists_are_instance_isolated():
    a = _Echo(model_name="claude", editor_client=FakeClient())
    b = _Echo(model_name="claude", editor_client=FakeClient())
    a.allowed_tools.append("DANGEROUS")
    assert "DANGEROUS" not in b.allowed_tools
    assert "DANGEROUS" not in _Echo.allowed_tools


def test_code_reader_agent_tools_and_prompt(tmp_path):
    from k_search.kernel_generators.agents import CodeReaderAgent

    agent = CodeReaderAgent(model_name="claude", editor_client=FakeClient(), max_chars=1234)
    assert agent.allowed_tools == ["Read", "Grep", "Glob", "Write"]
    assert "Edit" not in agent.allowed_tools

    prompt = agent.build_prompt({"definition_text": "Vector add operator."})
    assert "CODE_MAP.md" in prompt
    assert "MUST NOT modify" in prompt
    assert "Vector add operator." in prompt
    assert "1234" in prompt  # max_chars injected
    assert "# CODE_MAP" in prompt  # template anchor


def test_code_reader_agent_explicitly_disallows_edit():
    from k_search.kernel_generators.agents import CodeReaderAgent

    agent = CodeReaderAgent(model_name="claude", editor_client=FakeClient())
    assert "Edit" in agent.disallowed_tools
    assert "Bash" in agent.disallowed_tools


def test_codegen_agent_delegates_to_prompt_builder():
    from k_search.kernel_generators.agents import CodegenAgent
    from k_search.kernel_generators.ascendc_agentic_codegen import AscendCAgenticCodegenRequest

    agent = CodegenAgent(model_name="claude", editor_client=FakeClient())
    assert agent.allowed_tools == ["Read", "Grep", "Glob", "Edit", "Write"]

    request = AscendCAgenticCodegenRequest(
        definition_text="Task: x", action_text="opt", trace_logs="", perf_summary="",
        target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
    )
    prompt_no_map = agent.build_prompt({"request": request, "has_code_map": False})
    prompt_map = agent.build_prompt({"request": request, "has_code_map": True})
    assert "First inspect the project" in prompt_no_map
    assert "CODE_MAP.md" in prompt_map
