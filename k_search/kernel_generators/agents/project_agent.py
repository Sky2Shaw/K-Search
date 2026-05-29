from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from k_search.kernel_generators.ascendc_agentic_codegen import _edit_project_with_optional_telemetry
from k_search.kernel_generators.claude_agent_project_editor import (
    ClaudeAgentProjectEditorClient,
    ClaudeProjectEditResult,
)


@dataclass
class AgentRunResult:
    text: str
    transcript: str
    edit_result: ClaudeProjectEditResult


class ProjectAgent:
    """Base for roles that converse with a project worktree via the Claude Agent SDK.

    Subclasses set `allowed_tools` and implement `build_prompt(context)`.
    """

    allowed_tools: list[str] = ["Read", "Grep", "Glob", "Edit", "Write"]
    disallowed_tools: list[str] = ["Bash"]

    def __init__(self, *, model_name: str, editor_client: Any | None = None) -> None:
        self.model_name = str(model_name)
        self.editor_client = editor_client or ClaudeAgentProjectEditorClient(
            model_name=self.model_name,
            allowed_tools=list(self.allowed_tools),
            disallowed_tools=list(self.disallowed_tools),
        )

    def build_prompt(self, context: Any) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, *, project_dir: str | Path, context: Any, telemetry_recorder: Any = None) -> AgentRunResult:
        prompt = self.build_prompt(context)
        edit_result = _edit_project_with_optional_telemetry(
            self.editor_client,
            project_dir=Path(project_dir),
            prompt=prompt,
            telemetry_recorder=telemetry_recorder,
        )
        return AgentRunResult(
            text=edit_result.text,
            transcript=edit_result.transcript,
            edit_result=edit_result,
        )
