from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from k_search.kernel_generators.agentic_worktree import create_agentic_worktree
from k_search.kernel_generators.claude_agent_project_editor import (
    ClaudeAgentProjectEditorClient,
    ClaudeProjectEditResult,
)
from k_search.tasks.task_base import Solution


AgenticMode = Literal["generate", "action", "debug", "improve"]


@dataclass
class AscendCAgenticCodegenRequest:
    definition_text: str
    action_text: str
    trace_logs: str
    perf_summary: str
    target_gpu: str
    round_num: int
    attempt_idx: int
    mode: AgenticMode


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
    worktree_path: str


def _truncate(text: str, limit: int) -> str:
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 40)].rstrip() + "\n[truncated for agentic prompt budget]"


class AscendCAgenticPromptBuilder:
    def __init__(self, *, max_chars: int | None = None) -> None:
        if max_chars is None:
            raw = os.getenv("KSEARCH_AGENTIC_PROMPT_MAX_CHARS", "").strip()
            max_chars = int(raw) if raw.isdigit() and int(raw) > 0 else 20_000
        self.max_chars = int(max_chars)

    def build(self, request: AscendCAgenticCodegenRequest) -> str:
        sections = {
            "definition": _truncate(request.definition_text, 5000),
            "action": _truncate(request.action_text, 3000),
            "perf_summary": _truncate(request.perf_summary, 2500),
            "trace_logs": _truncate(request.trace_logs, 4000),
        }
        prompt = (
            "You are an AscendC performance optimization agent working inside a candidate project directory.\n"
            f"Target GPU: {request.target_gpu}\n"
            f"Mode: {request.mode}\n"
            f"Round: {int(request.round_num)}\n"
            f"Attempt: {int(request.attempt_idx)}\n\n"
            "Available tools: Read/Grep/Glob/Edit/Write. Bash is disabled.\n"
            "First inspect the project with Glob, Grep, and Read. Then edit only necessary files.\n"
            "Do not read or modify .git, build directories, caches, generated logs, or large artifacts.\n"
            "Preserve operator semantics, public entry points, host tiling contract, correctness harness behavior, and build layout.\n"
            "Do not return a full source container. Modify files in the project directory.\n"
            "End with a concise summary and changed-file list.\n\n"
            "Task specification:\n"
            f"{sections['definition']}\n\n"
            "Chosen action or debug intent:\n"
            f"{sections['action']}\n\n"
            "Performance summary:\n"
            f"{sections['perf_summary'] or '(none)'}\n\n"
            "Recent failure or trace excerpt:\n"
            f"{sections['trace_logs'] or '(none)'}\n"
        )
        if len(prompt) > self.max_chars:
            sizes = ", ".join(f"{name}={len(value)}" for name, value in sorted(sections.items()))
            raise ValueError(
                f"agentic prompt exceeded {self.max_chars} chars: prompt={len(prompt)}, sections: {sizes}"
            )
        return prompt


class AscendCAgenticCodegenRunner:
    def __init__(
        self,
        *,
        model_name: str,
        editor_client: Any | None = None,
        prompt_builder: AscendCAgenticPromptBuilder | None = None,
    ) -> None:
        self.model_name = str(model_name)
        self.editor_client = editor_client or ClaudeAgentProjectEditorClient(model_name=self.model_name)
        self.prompt_builder = prompt_builder or AscendCAgenticPromptBuilder()

    def run(
        self,
        *,
        task: Any,
        request: AscendCAgenticCodegenRequest,
        base_solution: Solution | None,
    ) -> AscendCAgenticCodegenResult:
        session = create_agentic_worktree(task_path=getattr(task, "task_path", None))
        try:
            overlay = getattr(task, "overlay_solution_sources", None)
            if callable(overlay):
                overlay(project_dir=session.project_dir, solution=base_solution)
                session.commit_all("ksearch agentic overlay baseline")

            prompt = self.prompt_builder.build(request)
            edit_result: ClaudeProjectEditResult = self.editor_client.edit_project(
                project_dir=session.project_dir,
                prompt=prompt,
            )
            changed_paths = session.changed_paths()
            if not changed_paths:
                raise RuntimeError(
                    "Claude agentic AscendC codegen did not change any files "
                    f"(round={request.round_num}, attempt={request.attempt_idx})"
                )
            diff_text = session.diff_text()
            solution = task.make_solution_from_project_dir(
                project_dir=session.project_dir,
                changed_paths=changed_paths,
                raw_agent_output=edit_result.text,
                round_num=request.round_num,
                model_name=self.model_name,
                target_gpu=request.target_gpu,
                language="ascendc",
            )
            cleaned = {src.path: src.content for src in solution.sources or []}
            return AscendCAgenticCodegenResult(
                solution=solution,
                raw=task.code_for_world_model_from_raw(raw=cleaned, language="ascendc"),
                cleaned=cleaned,
                transcript=edit_result.transcript,
                prompt=prompt,
                prompt_chars=len(prompt),
                changed_paths=changed_paths,
                diff_text=diff_text,
                worktree_path=str(session.project_dir),
            )
        finally:
            session.cleanup()