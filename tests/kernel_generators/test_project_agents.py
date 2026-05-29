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
