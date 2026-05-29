# 记忆管理框架 + 可扩展 Agent 角色 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 AscendC agentic 代码生成引入跨 attempt 的代码理解记忆(code_map),并把"与项目对话"的逻辑统一到可扩展的 `ProjectAgent` 角色体系,使后续 attempt 读记忆而非重扫全项目。

**Architecture:** 三层 —— `memory/`(`MemoryStore`+`MemoryKind` 持久化任意类型记忆)、`agents/`(`ProjectAgent` 基类 + `CodeReaderAgent`/`CodegenAgent`)、orchestrator(`AscendCAgenticCodegenRunner` 编排 worktree/memory/agents/eval)。`world_model` 与 `code_map` 解耦。

**Tech Stack:** Python 3.10+、dataclass、`pytest`、Claude Agent SDK(`ClaudeAgentProjectEditorClient`)。

**设计依据:** `docs/superpowers/specs/2026-05-29-memory-and-agent-roles-design.md`

---

## File Structure

| 文件 | 责任 |
|---|---|
| `k_search/kernel_generators/memory/__init__.py` | 导出 `MemoryStore`、`MemoryKind`、`CODE_MAP` |
| `k_search/kernel_generators/memory/memory_store.py` | 记忆持久化与 worktree materialize(类型无感) |
| `k_search/kernel_generators/agents/__init__.py` | 导出 `ProjectAgent`、`AgentRunResult`、`CodeReaderAgent`、`CodegenAgent` |
| `k_search/kernel_generators/agents/project_agent.py` | agent 基类:tools + build_prompt + run(透传遥测) |
| `k_search/kernel_generators/agents/code_reader_agent.py` | 只读角色,产出 `CODE_MAP.md` |
| `k_search/kernel_generators/agents/codegen_agent.py` | codegen 角色,封装 prompt builder + 编辑 |
| `k_search/kernel_generators/ascendc_agentic_codegen.py` | 改:prompt builder 增 code_map 分支;runner 接入 memory+reader+过滤+result 字段 |
| `k_search/kernel_generators/kernel_generator.py` | 改:baseline 调用点门控回写 |
| `k_search/kernel_generators/kernel_generator_world_model.py` | 改:优化循环采纳时门控回写 |
| `tests/kernel_generators/test_memory_store.py` | 新增单测 |
| `tests/kernel_generators/test_project_agents.py` | 新增单测 |
| `tests/kernel_generators/test_ascendc_agentic_codegen.py` | 扩展:code_map 接入 |

---

## Task 1: MemoryStore + MemoryKind

**Files:**
- Create: `k_search/kernel_generators/memory/__init__.py`
- Create: `k_search/kernel_generators/memory/memory_store.py`
- Test: `tests/kernel_generators/test_memory_store.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/kernel_generators/test_memory_store.py
from pathlib import Path

from k_search.kernel_generators.memory import CODE_MAP, MemoryKind, MemoryStore


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(artifacts_dir=str(tmp_path / "artifacts"), task_name="opx")


def test_load_missing_returns_none(tmp_path):
    assert _store(tmp_path).load(CODE_MAP) is None


def test_save_then_load_roundtrip(tmp_path):
    store = _store(tmp_path)
    store.save(CODE_MAP, "# CODE_MAP\nhello\n")
    assert store.load(CODE_MAP) == "# CODE_MAP\nhello\n"


def test_save_empty_is_noop(tmp_path):
    store = _store(tmp_path)
    store.save(CODE_MAP, "   ")
    assert store.load(CODE_MAP) is None


def test_materialize_and_read_from_worktree(tmp_path):
    store = _store(tmp_path)
    project = tmp_path / "wt"
    project.mkdir()
    assert store.materialize(CODE_MAP, project) is False  # nothing saved yet
    store.save(CODE_MAP, "mapped\n")
    assert store.materialize(CODE_MAP, project) is True
    assert (project / "CODE_MAP.md").read_text(encoding="utf-8") == "mapped\n"
    assert store.read_from_worktree(CODE_MAP, project) == "mapped\n"


def test_kind_is_generic(tmp_path):
    plan = MemoryKind("plan", "PLAN.md", gated_writeback=True)
    store = _store(tmp_path)
    store.save(plan, "step 1\n")
    assert store.load(plan) == "step 1\n"
    assert store.load(CODE_MAP) is None  # kinds are isolated
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/kernel_generators/test_memory_store.py -v`
Expected: FAIL（`ModuleNotFoundError: k_search.kernel_generators.memory`）

- [ ] **Step 3: 实现**

```python
# k_search/kernel_generators/memory/memory_store.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from k_search.utils.paths import get_ksearch_artifacts_dir


@dataclass(frozen=True)
class MemoryKind:
    """Describes one persisted memory type (code_map today; plan/review later)."""

    name: str
    filename: str
    gated_writeback: bool = True


CODE_MAP = MemoryKind(name="code_map", filename="CODE_MAP.md", gated_writeback=True)


class MemoryStore:
    """Persists per-task memory under <artifacts>/<task>/memory/<kind>/<filename>."""

    def __init__(self, *, artifacts_dir: str | Path | None, task_name: str | None) -> None:
        self._artifacts_dir = artifacts_dir
        self._task_name = str(task_name or "") or None

    @classmethod
    def for_task(cls, task: object) -> "MemoryStore":
        artifacts_dir = getattr(task, "artifacts_dir", None)
        task_name = getattr(task, "definition_name", None) or getattr(task, "name", None)
        return cls(artifacts_dir=artifacts_dir, task_name=task_name)

    def _path(self, kind: MemoryKind) -> Path:
        base = get_ksearch_artifacts_dir(base_dir=self._artifacts_dir, task_name=self._task_name)
        return base / "memory" / kind.name / kind.filename

    def load(self, kind: MemoryKind) -> str | None:
        p = self._path(kind)
        if not p.is_file():
            return None
        text = p.read_text(encoding="utf-8", errors="replace")
        return text if text.strip() else None

    def save(self, kind: MemoryKind, text: str | None) -> None:
        if not text or not str(text).strip():
            return
        p = self._path(kind)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(text), encoding="utf-8")

    def materialize(self, kind: MemoryKind, project_dir: str | Path) -> bool:
        """Copy saved memory into the worktree so an agent can Read it. Returns True if written."""
        text = self.load(kind)
        if text is None:
            return False
        dest = Path(project_dir).expanduser().resolve() / kind.filename
        dest.write_text(text, encoding="utf-8")
        return True

    def read_from_worktree(self, kind: MemoryKind, project_dir: str | Path) -> str | None:
        p = Path(project_dir).expanduser().resolve() / kind.filename
        if not p.is_file():
            return None
        text = p.read_text(encoding="utf-8", errors="replace")
        return text if text.strip() else None
```

```python
# k_search/kernel_generators/memory/__init__.py
from k_search.kernel_generators.memory.memory_store import CODE_MAP, MemoryKind, MemoryStore

__all__ = ["CODE_MAP", "MemoryKind", "MemoryStore"]
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/kernel_generators/test_memory_store.py -v`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add k_search/kernel_generators/memory/ tests/kernel_generators/test_memory_store.py
git commit -m "feat: add MemoryStore + MemoryKind for code map memory"
```

---

## Task 2: ProjectAgent 基类 + AgentRunResult

**Files:**
- Create: `k_search/kernel_generators/agents/__init__.py`
- Create: `k_search/kernel_generators/agents/project_agent.py`
- Test: `tests/kernel_generators/test_project_agents.py`

复用现有 `ascendc_agentic_codegen._edit_project_with_optional_telemetry`(对不支持 `telemetry_recorder` 的 client 自动降级)。

- [ ] **Step 1: 写失败测试**

```python
# tests/kernel_generators/test_project_agents.py
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
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/kernel_generators/test_project_agents.py -v`
Expected: FAIL（`ModuleNotFoundError: k_search.kernel_generators.agents`）

- [ ] **Step 3: 实现**

```python
# k_search/kernel_generators/agents/project_agent.py
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
```

```python
# k_search/kernel_generators/agents/__init__.py
from k_search.kernel_generators.agents.project_agent import AgentRunResult, ProjectAgent

__all__ = ["AgentRunResult", "ProjectAgent"]
```

注:`_edit_project_with_optional_telemetry` 当 client 不接受 `telemetry_recorder` 时会自动重试不带该参数(见 `ascendc_agentic_codegen.py`),故 `FakeClient` 接受该 kwarg 即可。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/kernel_generators/test_project_agents.py -v`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add k_search/kernel_generators/agents/__init__.py k_search/kernel_generators/agents/project_agent.py tests/kernel_generators/test_project_agents.py
git commit -m "feat: add ProjectAgent base for extensible agent roles"
```

---

## Task 3: CodeReaderAgent

**Files:**
- Create: `k_search/kernel_generators/agents/code_reader_agent.py`
- Modify: `k_search/kernel_generators/agents/__init__.py`
- Test: `tests/kernel_generators/test_project_agents.py`（追加）

- [ ] **Step 1: 写失败测试（追加到 test_project_agents.py 末尾）**

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/kernel_generators/test_project_agents.py::test_code_reader_agent_tools_and_prompt -v`
Expected: FAIL（`ImportError: cannot import name 'CodeReaderAgent'`）

- [ ] **Step 3: 实现**

```python
# k_search/kernel_generators/agents/code_reader_agent.py
from __future__ import annotations

import os
from typing import Any

from k_search.kernel_generators.agents.project_agent import ProjectAgent

_READER_PROMPT_TEMPLATE = """\
You are a code-understanding agent working inside an AscendC operator project directory.
Your ONLY job is to read and understand the project, then write a concise CODE_MAP.md
at the project root (CWD). You MUST NOT modify, create, or delete any source file —
the only file you may write is CODE_MAP.md.

Available tools: Read, Grep, Glob, Write. Edit and Bash are disabled.
Do not read .git, build directories, caches, generated logs, or large artifacts.
Keep CODE_MAP.md under {max_chars} characters — it is a navigation map, not a copy of the code.
Describe what exists and where; do NOT propose optimizations or changes (that is another agent's job).

Operator task specification (for context only):
{definition_text}

Inspect the project with Glob/Grep/Read, then write CODE_MAP.md using EXACTLY this template:

# CODE_MAP

## Files
<relative/path> — <one-line role> (kernel / host tiling / InferShape / op-def / test harness / build / other)
... one line per source file ...

## Entry Points & Call Chain
- Public entry point(s): <symbol> in <file>
- Host → kernel launch path: <how host tiling reaches the kernel>

## Host <-> Kernel Contract
- Tiling struct / fields passed host->kernel: <...>
- Workspace / sync / block-dim assumptions: <...>

## Kernel Design
- Core split strategy: <...>
- Tiling formula (main/tail, tile sizes): <...>
- Buffer allocation (UB/L1, queue depth, double-buffer): <...>
- Pipeline (CopyIn -> Compute -> CopyOut) stages: <...>

## Invariants & Constraints
- Semantics that MUST be preserved: <...>
- Alignment / dtype / shape constraints: <...>
- Build layout assumptions: <...>

Finish by writing CODE_MAP.md. Output a one-line confirmation; do not paste the file back.
"""


def _default_code_map_max_chars() -> int:
    raw = os.getenv("KSEARCH_CODE_MAP_MAX_CHARS", "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else 8000


class CodeReaderAgent(ProjectAgent):
    """Read-only role that produces the code_map memory (CODE_MAP.md)."""

    allowed_tools = ["Read", "Grep", "Glob", "Write"]

    def __init__(self, *, model_name: str, editor_client: Any | None = None, max_chars: int | None = None) -> None:
        super().__init__(model_name=model_name, editor_client=editor_client)
        self.max_chars = int(max_chars) if max_chars else _default_code_map_max_chars()

    def build_prompt(self, context: Any) -> str:
        definition_text = ""
        if isinstance(context, dict):
            definition_text = str(context.get("definition_text", "") or "")
        return _READER_PROMPT_TEMPLATE.format(
            max_chars=self.max_chars,
            definition_text=definition_text.strip() or "(no specification provided)",
        )
```

更新 `agents/__init__.py`:

```python
# k_search/kernel_generators/agents/__init__.py
from k_search.kernel_generators.agents.code_reader_agent import CodeReaderAgent
from k_search.kernel_generators.agents.project_agent import AgentRunResult, ProjectAgent

__all__ = ["AgentRunResult", "ProjectAgent", "CodeReaderAgent"]
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/kernel_generators/test_project_agents.py -v`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add k_search/kernel_generators/agents/code_reader_agent.py k_search/kernel_generators/agents/__init__.py tests/kernel_generators/test_project_agents.py
git commit -m "feat: add CodeReaderAgent producing code_map memory"
```

---

## Task 4: PromptBuilder 增加 code_map 分支

**Files:**
- Modify: `k_search/kernel_generators/ascendc_agentic_codegen.py:101`（`AscendCAgenticPromptBuilder.build`）
- Test: `tests/kernel_generators/test_ascendc_agentic_codegen.py`（追加）

`build` 增加可选 `has_code_map` 参数(默认 `False`,保持现有调用与测试不变)。

- [ ] **Step 1: 写失败测试（追加）**

```python
def test_prompt_builder_uses_code_map_branch_when_present():
    builder = AscendCAgenticPromptBuilder(max_chars=20_000)
    request = AscendCAgenticCodegenRequest(
        definition_text="Task: x",
        action_text="optimize",
        trace_logs="",
        perf_summary="",
        target_gpu="ascend_910b",
        round_num=1,
        attempt_idx=1,
        mode="action",
    )

    with_map = builder.build(request, has_code_map=True)
    without_map = builder.build(request, has_code_map=False)

    assert "CODE_MAP.md" in with_map
    assert "Read it first" in with_map
    assert "update the affected sections" in with_map
    assert "CODE_MAP.md" not in without_map
    assert "First inspect the project with Glob, Grep, and Read" in without_map
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/kernel_generators/test_ascendc_agentic_codegen.py::test_prompt_builder_uses_code_map_branch_when_present -v`
Expected: FAIL（`TypeError: build() got an unexpected keyword argument 'has_code_map'`）

- [ ] **Step 3: 实现**

把 `ascendc_agentic_codegen.py` 中 `build` 的签名与"inspect"那一行改为分支(其余行不变):

```python
    def build(self, request: AscendCAgenticCodegenRequest, *, has_code_map: bool = False) -> str:
        sections = {
            "definition": _truncate(request.definition_text, 5000),
            "action": _truncate(request.action_text, 3000),
            "perf_summary": _truncate(request.perf_summary, 2500),
            "trace_logs": _truncate(request.trace_logs, 4000),
        }
        if has_code_map:
            inspect_line = (
                "A CODE_MAP.md at the project root describes file roles, kernel structure, "
                "tiling, buffers, and contracts. Read it first instead of grepping the whole project. "
                "After editing code, update the affected sections of CODE_MAP.md to keep it accurate.\n"
            )
        else:
            inspect_line = "First inspect the project with Glob, Grep, and Read. Then edit only necessary files.\n"
        prompt = (
            "You are an AscendC performance optimization agent working inside a candidate project directory.\n"
            "IMPORTANT: You must ONLY edit files inside the current project directory (CWD). Do NOT use absolute paths from any external directories.\n"
            f"Target GPU: {request.target_gpu}\n"
            f"Mode: {request.mode}\n"
            f"Round: {int(request.round_num)}\n"
            f"Attempt: {int(request.attempt_idx)}\n\n"
            "Available tools: Read/Grep/Glob/Edit/Write. Bash is disabled.\n"
            + inspect_line
            + "Do not read or modify .git, build directories, caches, generated logs, or large artifacts.\n"
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
        prompt = sanitize_worktree_paths(prompt)
        if len(prompt) > self.max_chars:
            sizes = ", ".join(f"{name}={len(value)}" for name, value in sorted(sections.items()))
            raise ValueError(
                f"agentic prompt exceeded {self.max_chars} chars: prompt={len(prompt)}, sections: {sizes}"
            )
        return prompt
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/kernel_generators/test_ascendc_agentic_codegen.py -v`
Expected: PASS（含原有 prompt 测试与新测试）

- [ ] **Step 5: 提交**

```bash
git add k_search/kernel_generators/ascendc_agentic_codegen.py tests/kernel_generators/test_ascendc_agentic_codegen.py
git commit -m "feat: add code_map prompt branch to agentic codegen builder"
```

---

## Task 5: CodegenAgent（封装 prompt builder + 编辑）

**Files:**
- Create: `k_search/kernel_generators/agents/codegen_agent.py`
- Modify: `k_search/kernel_generators/agents/__init__.py`
- Test: `tests/kernel_generators/test_project_agents.py`（追加）

`CodegenAgent` 用 `AscendCAgenticCodegenRequest` 作为 context,`build_prompt` 委托 `AscendCAgenticPromptBuilder`,并按是否存在 code_map 选择分支。

- [ ] **Step 1: 写失败测试（追加）**

```python
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
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/kernel_generators/test_project_agents.py::test_codegen_agent_delegates_to_prompt_builder -v`
Expected: FAIL（`ImportError: cannot import name 'CodegenAgent'`）

- [ ] **Step 3: 实现**

```python
# k_search/kernel_generators/agents/codegen_agent.py
from __future__ import annotations

from typing import Any

from k_search.kernel_generators.agents.project_agent import ProjectAgent
from k_search.kernel_generators.ascendc_agentic_codegen import (
    AscendCAgenticCodegenRequest,
    AscendCAgenticPromptBuilder,
)


class CodegenAgent(ProjectAgent):
    """Codegen role: edits the project to implement an action, optionally guided by code_map."""

    allowed_tools = ["Read", "Grep", "Glob", "Edit", "Write"]

    def __init__(self, *, model_name: str, editor_client: Any | None = None,
                 prompt_builder: AscendCAgenticPromptBuilder | None = None) -> None:
        super().__init__(model_name=model_name, editor_client=editor_client)
        self.prompt_builder = prompt_builder or AscendCAgenticPromptBuilder()

    def build_prompt(self, context: Any) -> str:
        request: AscendCAgenticCodegenRequest = context["request"]
        has_code_map = bool(context.get("has_code_map", False))
        return self.prompt_builder.build(request, has_code_map=has_code_map)
```

更新 `agents/__init__.py`:

```python
# k_search/kernel_generators/agents/__init__.py
from k_search.kernel_generators.agents.code_reader_agent import CodeReaderAgent
from k_search.kernel_generators.agents.codegen_agent import CodegenAgent
from k_search.kernel_generators.agents.project_agent import AgentRunResult, ProjectAgent

__all__ = ["AgentRunResult", "ProjectAgent", "CodeReaderAgent", "CodegenAgent"]
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/kernel_generators/test_project_agents.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add k_search/kernel_generators/agents/codegen_agent.py k_search/kernel_generators/agents/__init__.py tests/kernel_generators/test_project_agents.py
git commit -m "feat: add CodegenAgent wrapping agentic prompt builder"
```

---

## Task 6: Runner 接入 memory + CodeReaderAgent + 过滤 + result 字段

**Files:**
- Modify: `k_search/kernel_generators/ascendc_agentic_codegen.py`（`AscendCAgenticCodegenResult` 增字段;`run()` 接入）
- Test: `tests/kernel_generators/test_ascendc_agentic_codegen.py`（追加）

行为：
- `AscendCAgenticCodegenResult` 增 `code_map_text: str | None = None`。
- `run()`：overlay 后 → 若 `KSEARCH_ENABLE_CODE_MAP` 开(默认开) → `store = MemoryStore.for_task(task)`；`store.load(CODE_MAP)` 为空则跑 `CodeReaderAgent` 并 `read_from_worktree`+`save`(首轮即落盘)；`materialize` 进 worktree；`has_code_map` 传给 prompt。
- codegen 编辑后：从 `project_changed_paths` 过滤掉 `CODE_MAP.md` 再做"无改动"判定与 solution 构造；`code_map_text` 读回放入 result。
- reader 失败仅 warn,降级为 `has_code_map=False`。

- [ ] **Step 1: 写失败测试（追加）**

```python
def test_runner_generates_and_persists_code_map_on_first_round(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))

    class ReaderClient:
        """First call (reader) writes CODE_MAP.md; second call (codegen) edits foo.h."""
        def __init__(self):
            self.prompts = []

        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            self.prompts.append(prompt)
            root = Path(project_dir)
            if "CODE_MAP.md using EXACTLY" in prompt:
                (root / "CODE_MAP.md").write_text("# CODE_MAP\nfoo.h is the kernel\n", encoding="utf-8")
                text = "wrote CODE_MAP.md"
            else:
                (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
                text = "edited"
            return ClaudeProjectEditResult(
                text=text, transcript=text, prompt=prompt,
                prompt_chars=len(prompt), prompt_lines=prompt.count("\n") + 1,
            )

    client = ReaderClient()
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)
    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )

    # reader ran first, then codegen with code_map branch
    assert any("CODE_MAP.md using EXACTLY" in p for p in client.prompts)
    assert any("Read it first instead of grepping" in p for p in client.prompts)
    # code_map persisted to artifacts and returned
    from k_search.kernel_generators.memory import CODE_MAP, MemoryStore
    store = MemoryStore.for_task(task)
    assert store.load(CODE_MAP) is not None
    assert result.code_map_text is not None
    # CODE_MAP.md must NOT count as an operator source change
    assert "CODE_MAP.md" not in result.changed_paths
    assert "kernel/foo.h" in result.changed_paths


def test_runner_reuses_existing_code_map_without_reader(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "1")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    # pre-seed code_map
    from k_search.kernel_generators.memory import CODE_MAP, MemoryStore
    MemoryStore.for_task(task).save(CODE_MAP, "# CODE_MAP\npreseeded\n")

    class CodegenOnlyClient:
        def __init__(self):
            self.prompts = []

        def edit_project(self, *, project_dir, prompt, telemetry_recorder=None):
            self.prompts.append(prompt)
            assert "CODE_MAP.md using EXACTLY" not in prompt  # reader must NOT run
            root = Path(project_dir)
            assert (root / "CODE_MAP.md").read_text(encoding="utf-8") == "# CODE_MAP\npreseeded\n"
            (root / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
            return ClaudeProjectEditResult(
                text="edited", transcript="edited", prompt=prompt,
                prompt_chars=len(prompt), prompt_lines=prompt.count("\n") + 1,
            )

    client = CodegenOnlyClient()
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)
    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=2, attempt_idx=1, mode="improve",
        ),
        base_solution=None,
    )
    assert len(client.prompts) == 1  # only codegen, no reader
    assert "Read it first instead of grepping" in client.prompts[0]
    assert "CODE_MAP.md" not in result.changed_paths


def test_runner_code_map_disabled_keeps_legacy_behavior(tmp_path, monkeypatch):
    monkeypatch.setenv("KSEARCH_ENABLE_CODE_MAP", "0")
    task_dir = tmp_path / "task"
    (task_dir / "kernel").mkdir(parents=True)
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=task_dir, definition_name="x", artifacts_dir=str(tmp_path / "artifacts"))
    client = EditingClient("alpha\nBETA\ngamma\n")
    runner = AscendCAgenticCodegenRunner(model_name="claude", editor_client=client)
    result = runner.run(
        task=task,
        request=AscendCAgenticCodegenRequest(
            definition_text="spec", action_text="change beta", trace_logs="", perf_summary="",
            target_gpu="ascend_910b", round_num=1, attempt_idx=1, mode="action",
        ),
        base_solution=None,
    )
    assert "First inspect the project" in client.calls[0][1]
    assert "CODE_MAP.md" not in client.calls[0][1]
    assert result.code_map_text is None
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/kernel_generators/test_ascendc_agentic_codegen.py -k code_map -v`
Expected: FAIL（`AttributeError: ... 'code_map_text'` 或 reader 未运行）

- [ ] **Step 3: 实现**

在 `AscendCAgenticCodegenResult` dataclass 增字段:

```python
    code_map_text: str | None = None
```

在文件顶部增加导入(放到现有 import 区):

```python
from k_search.kernel_generators.memory import CODE_MAP, MemoryStore
```

注:`CodeReaderAgent`/`CodegenAgent` 在 `run()` 内**惰性导入**,避免 `agents` ↔ `ascendc_agentic_codegen` 循环导入(`agents` 依赖本模块的 `_edit_project_with_optional_telemetry` 与 `AscendCAgenticPromptBuilder`)。

`run()` 改造（替换 overlay 之后到 `prompt = self.prompt_builder.build(request)` 这一段，并在求 `changed_paths` 处加过滤）:

```python
            overlay = getattr(task, "overlay_solution_sources", None)
            if callable(overlay):
                overlay(project_dir=session.project_dir, solution=base_solution)
                session.commit_all("ksearch agentic overlay baseline")

            # --- code_map memory (optional) ---
            code_map_enabled = os.getenv("KSEARCH_ENABLE_CODE_MAP", "1").strip().lower() not in {"0", "false", "no", "off"}
            store = MemoryStore.for_task(task) if code_map_enabled else None
            has_code_map = False
            if store is not None:
                from k_search.kernel_generators.agents import CodeReaderAgent  # lazy: avoid import cycle
                if store.load(CODE_MAP) is None:
                    try:
                        reader = CodeReaderAgent(model_name=self.model_name)
                        reader.run(
                            project_dir=session.project_dir,
                            context={"definition_text": request.definition_text},
                        )
                        produced = store.read_from_worktree(CODE_MAP, session.project_dir)
                        if produced:
                            store.save(CODE_MAP, produced)
                    except Exception as exc:  # degrade, never block codegen
                        print(f"[WARN] code_map reader failed; continuing without it: {type(exc).__name__}: {exc}", flush=True)
                has_code_map = store.materialize(CODE_MAP, session.project_dir)

            prompt = self.prompt_builder.build(request, has_code_map=has_code_map)
```

在计算 changed_paths 处过滤 code_map 文件(原 `project_changed_paths = session.project_changed_paths()` 一段改为):

```python
            project_changed_paths = session.project_changed_paths()
            changed_paths = project_changed_paths or session.changed_paths()
            changed_paths = [p for p in changed_paths if p != CODE_MAP.filename]
            if not changed_paths:
                raise RuntimeError(
                    "Claude agentic AscendC codegen did not change any files "
                    f"(round={request.round_num}, attempt={request.attempt_idx})"
                )
```

在构造返回 `AscendCAgenticCodegenResult(...)` 前读回 code_map,并把它加入返回:

```python
            code_map_text = store.read_from_worktree(CODE_MAP, session.project_dir) if store is not None else None
```

在 `return AscendCAgenticCodegenResult(` 的字段里追加：

```python
                code_map_text=code_map_text,
```

注:`materialize` 会在 worktree 写出 `CODE_MAP.md`，它会出现在 `session.changed_paths()` 中（baseline commit 之后新增），故上面的过滤是必需的；`make_solution_from_project_dir` 因 `.md` 不在 `_is_source_candidate` 不会采集它（无需改动 `ascendc_task.py`）。

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/kernel_generators/test_ascendc_agentic_codegen.py -v`
Expected: PASS（原有 + 3 新 code_map 测试）

- [ ] **Step 5: 提交**

```bash
git add k_search/kernel_generators/ascendc_agentic_codegen.py tests/kernel_generators/test_ascendc_agentic_codegen.py
git commit -m "feat: wire code_map memory and CodeReaderAgent into agentic runner"
```

---

## Task 7: 调用点门控回写（采纳才落盘）

**Files:**
- Modify: `k_search/kernel_generators/kernel_generator_world_model.py:734` 附近（优化循环采纳分支）
- Modify: `k_search/kernel_generators/kernel_generator.py:427` 附近（baseline 路径）
- Test: `tests/kernel_generators/test_ascendc_agentic_codegen.py`（追加：门控辅助函数单测）

为避免在两处重复构造 store，增加一个模块级辅助到 `memory_store.py`，并在采纳时调用。

- [ ] **Step 1: 写失败测试（追加到 test_memory_store.py）**

```python
def test_save_code_map_if_adopted_only_on_new_best(tmp_path):
    from k_search.kernel_generators.memory import CODE_MAP, MemoryStore, save_code_map_if_adopted

    class _Task:
        artifacts_dir = str(tmp_path / "artifacts")
        definition_name = "opx"

    task = _Task()
    # not adopted -> no write
    save_code_map_if_adopted(task=task, code_map_text="A\n", adopted=False)
    assert MemoryStore.for_task(task).load(CODE_MAP) is None
    # adopted -> write
    save_code_map_if_adopted(task=task, code_map_text="B\n", adopted=True)
    assert MemoryStore.for_task(task).load(CODE_MAP) == "B\n"
    # adopted but empty text -> no overwrite
    save_code_map_if_adopted(task=task, code_map_text=None, adopted=True)
    assert MemoryStore.for_task(task).load(CODE_MAP) == "B\n"
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/kernel_generators/test_memory_store.py::test_save_code_map_if_adopted_only_on_new_best -v`
Expected: FAIL（`ImportError: cannot import name 'save_code_map_if_adopted'`）

- [ ] **Step 3: 实现辅助 + 接入调用点**

在 `memory_store.py` 末尾追加:

```python
def save_code_map_if_adopted(*, task: object, code_map_text: str | None, adopted: bool) -> None:
    """Write code_map back to artifacts only when the attempt was adopted (new best)."""
    if not adopted or not code_map_text or not str(code_map_text).strip():
        return
    MemoryStore.for_task(task).save(CODE_MAP, code_map_text)
```

更新 `memory/__init__.py` 导出:

```python
from k_search.kernel_generators.memory.memory_store import (
    CODE_MAP,
    MemoryKind,
    MemoryStore,
    save_code_map_if_adopted,
)

__all__ = ["CODE_MAP", "MemoryKind", "MemoryStore", "save_code_map_if_adopted"]
```

在 `kernel_generator_world_model.py` 采纳分支(`if all_passed and round_score > best_score:` 块内,约 `:734`)追加回写:

```python
                        if all_passed and round_score > best_score:
                            best_score = float(round_score)
                            best_eval = round_eval
                            best_solution = solution
                            from k_search.kernel_generators.memory import save_code_map_if_adopted
                            save_code_map_if_adopted(
                                task=task,
                                code_map_text=getattr(result, "code_map_text", None),
                                adopted=True,
                            )
```

在 `kernel_generator.py` 的 `_generate_ascendc_solution_agentically` 返回前(baseline 路径,`:438` 前),baseline 首轮 reader 已落盘,此处对 codegen 的更新同样按"是否产生有效结果"回写(baseline 路径无 best 比较,采用 eval 通过即采纳):

```python
        from k_search.kernel_generators.memory import save_code_map_if_adopted
        save_code_map_if_adopted(
            task=task,
            code_map_text=getattr(result, "code_map_text", None),
            adopted=bool(getattr(result.eval_result, "is_passed", lambda: False)()),
        )
        return result
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/kernel_generators/test_memory_store.py -v`
Expected: PASS

- [ ] **Step 5: 回归 + 提交**

```bash
python -m pytest tests/kernel_generators/ -v
git add k_search/kernel_generators/memory/ k_search/kernel_generators/kernel_generator.py k_search/kernel_generators/kernel_generator_world_model.py tests/kernel_generators/test_memory_store.py
git commit -m "feat: gated code_map writeback at codegen call sites"
```

---

## Task 8: 全量回归

- [ ] **Step 1: 运行全部 kernel_generators 测试**

Run: `python -m pytest tests/kernel_generators/ -v`
Expected: 全 PASS（含原 `test_ascendc_agentic_codegen.py` 既有用例不回归）

- [ ] **Step 2: 运行受影响的相邻测试**

Run: `python -m pytest tests/ -k "agentic or ascendc or memory or world_model" -q`
Expected: PASS（无回归）

- [ ] **Step 3: 提交（若有遗留修复）**

```bash
git add -A && git commit -m "test: full regression for code_map memory + agent roles"
```

---

## 自检结论（spec 覆盖）

- §4.1 MemoryStore/MemoryKind → Task 1 ✓
- §4.2 ProjectAgent → Task 2 ✓
- §4.3 CodeReaderAgent + prompt 模板 → Task 3 ✓
- §4.4 CodegenAgent（封装 builder）→ Task 4+5 ✓
- §4.5 + §5 Runner 编排/数据流 → Task 6 ✓
- §5 回写门控 → Task 7 ✓
- §6 prompt 分支 → Task 4 ✓
- §7 边界（CODE_MAP.md 过滤、reader 降级、总开关）→ Task 6 ✓
- §8 配置开关（`KSEARCH_ENABLE_CODE_MAP`/`KSEARCH_CODE_MAP_MAX_CHARS`）→ Task 3+6 ✓
- §9 测试 → Task 1–8 ✓
- 预留 plan/review:`ProjectAgent` 基类 + `MemoryKind` 通用,本期不实现 ✓（YAGNI）
