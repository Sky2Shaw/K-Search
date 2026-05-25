# AscendC Agentic Worktree Codegen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `--llm-provider claude-agent --language ascendc` use Claude Agent SDK file tools in an isolated git-backed candidate worktree instead of sending full AscendC project source through long prompts.

**Architecture:** Add a narrow agentic codegen path that runs before legacy prompt construction. `AgenticWorktreeSession` owns temporary git isolation, `ClaudeAgentProjectEditorClient` owns SDK transport, `AscendCTask` owns AscendC source scanning and solution creation, and `AscendCAgenticCodegenRunner` coordinates one codegen attempt.

**Tech Stack:** Python 3.10+, stdlib `subprocess`/`tempfile`/`shutil`/`pathlib`, existing Claude Agent SDK dependency, pytest, existing K-Search `Solution`/`EvalResult` types.

---

## Scope Check

This plan implements the approved first stage only:

- Agentic codegen for Claude+AscendC by default.
- Temporary git worktree isolation with copied temp git repo fallback.
- Claude file tools only: `Read`, `Grep`, `Glob`, `Edit`, `Write`.
- No Bash, MCP, subagents, or Claude-driven build/test/bench.
- Existing OpenAI, Triton, CUDA, MLX, and legacy AscendC codegen remain available.

## File Structure

- Create `k_search/kernel_generators/agentic_worktree.py`
  - Owns git-backed candidate worktree lifecycle.
  - Exposes `AgenticWorktreeSession` and `create_agentic_worktree()`.

- Create `tests/kernel_generators/test_agentic_worktree.py`
  - Covers real git worktree, fallback temp repo, baseline commits, diff collection, and cleanup.

- Modify `k_search/tasks/ascendc_task.py`
  - Adds AscendC project-dir hooks: compact definition, source overlay, changed path validation, and final project-dir to `Solution` conversion.

- Modify `tests/test_ascendc_task.py`
  - Covers project-dir scanning, source overlay, forbidden changed paths, and raw-code to solution conversion for agentic routing.

- Create `k_search/kernel_generators/claude_agent_project_editor.py`
  - Owns Agent SDK `ClaudeSDKClient` project editing session.
  - Exposes `ClaudeProjectEditResult` and `ClaudeAgentProjectEditorClient`.

- Modify `k_search/testing/mock_claude_agent_sdk.py`
  - Adds `ClaudeSDKClient` session mock while preserving existing `query()` mock behavior.

- Modify `tests/kernel_generators/test_claude_agent_sdk_mock.py`
  - Covers project-editor client SDK options and mocked cwd file editing.

- Create `k_search/kernel_generators/ascendc_agentic_codegen.py`
  - Owns compact prompt building, prompt budget guard, and one agentic AscendC codegen attempt.

- Create `tests/kernel_generators/test_ascendc_agentic_codegen.py`
  - Covers compact prompt construction, budget failure, no-change failure, changed-file solution creation, and forbidden path failure.

- Modify `k_search/kernel_generators/kernel_generator.py`
  - Routes baseline Claude+AscendC generation through the agentic runner before legacy prompts are built.

- Modify `k_search/kernel_generators/kernel_generator_world_model.py`
  - Routes world-model Claude+AscendC action/debug codegen through the agentic runner before legacy prompts are built.

- Modify `tests/kernel_generators/test_claude_agent_sdk_mock.py`
  - Updates the two-round AscendC mock test so SDK edits files in `cwd` and prompts do not include full source containers.

- Modify `tests/kernel_generators/test_llm_clients.py`
  - Adds regression coverage that non-agentic `ClaudeAgentLLMClient.generate()` still uses the existing `query()` path.

- Modify `README.md`
  - Documents Claude+AscendC agentic default, worktree isolation, disabled Bash, prompt budget, and preservation env.

- Modify `tests/test_launcher_docs.py`
  - Asserts README documents the agentic Claude+AscendC behavior.

---

### Task 1: Git-Backed Agentic Worktree Session

**Files:**
- Create: `tests/kernel_generators/test_agentic_worktree.py`
- Create: `k_search/kernel_generators/agentic_worktree.py`

- [ ] **Step 1: Write failing worktree lifecycle tests**

Create `tests/kernel_generators/test_agentic_worktree.py` with:

```python
import os
import shutil
import subprocess
from pathlib import Path

from k_search.kernel_generators.agentic_worktree import create_agentic_worktree


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def _init_repo(root: Path) -> None:
    _git(root, "init")
    _git(root, "config", "user.email", "ksearch@example.invalid")
    _git(root, "config", "user.name", "K Search Tests")
    (root / "kernel").mkdir()
    (root / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    (root / "spec.md").write_text("Optimize this project.", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "initial")


def test_create_agentic_worktree_from_git_repo_tracks_project_subdir(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    task_path = repo / "kernel"
    session = create_agentic_worktree(task_path=task_path)

    try:
        assert session.project_dir == session.worktree_root / "kernel"
        assert session.project_dir.exists()
        assert session.is_real_worktree is True
        assert session.repo_root == repo.resolve()
        assert (session.project_dir / "foo.h").read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"

        (session.project_dir / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
        assert session.changed_paths() == ["kernel/foo.h"]
        assert "-beta" in session.diff_text()
        assert "+BETA" in session.diff_text()
    finally:
        path = session.worktree_root
        session.cleanup()

    assert not path.exists()


def test_create_agentic_worktree_falls_back_for_non_git_task_path(tmp_path):
    task_path = tmp_path / "plain_task"
    task_path.mkdir()
    (task_path / "kernel.cpp").write_text("int old_value = 1;\n", encoding="utf-8")
    (task_path / "build").mkdir()
    (task_path / "build" / "artifact.o").write_text("ignored\n", encoding="utf-8")

    session = create_agentic_worktree(task_path=task_path)

    try:
        assert session.is_real_worktree is False
        assert session.repo_root is None
        assert session.project_dir == session.worktree_root
        assert (session.project_dir / "kernel.cpp").exists()
        assert (session.project_dir / "build" / "artifact.o").exists()
        assert session.baseline_commit

        (session.project_dir / "kernel.cpp").write_text("int old_value = 2;\n", encoding="utf-8")
        assert session.changed_paths() == ["kernel.cpp"]
    finally:
        path = session.worktree_root
        session.cleanup()

    assert not path.exists()


def test_agentic_worktree_preserve_env_keeps_directory(tmp_path, monkeypatch):
    task_path = tmp_path / "plain_task"
    task_path.mkdir()
    (task_path / "kernel.cpp").write_text("int x = 1;\n", encoding="utf-8")
    monkeypatch.setenv("KSEARCH_KEEP_AGENTIC_WORKTREES", "1")

    session = create_agentic_worktree(task_path=task_path)
    path = session.worktree_root
    session.cleanup()

    try:
        assert path.exists()
    finally:
        shutil.rmtree(path, ignore_errors=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/kernel_generators/test_agentic_worktree.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'k_search.kernel_generators.agentic_worktree'`.

- [ ] **Step 3: Implement `agentic_worktree.py`**

Create `k_search/kernel_generators/agentic_worktree.py` with:

```python
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class AgenticWorktreeError(RuntimeError):
    """Raised when K-Search cannot prepare or inspect an agentic worktree."""


def _run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=check,
    )


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=cwd, check=check)


def _git_stdout(cwd: Path, *args: str) -> str:
    return _git(cwd, *args).stdout.strip()


def _git_config_identity(cwd: Path) -> None:
    _git(cwd, "config", "user.email", "ksearch@example.invalid")
    _git(cwd, "config", "user.name", "K Search Agentic Codegen")


def _find_git_root(path: Path) -> Optional[Path]:
    proc = _git(path, "rev-parse", "--show-toplevel", check=False)
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return Path(out).resolve() if out else None


def _copy_project(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns(".git", ".ksearch", "__pycache__")
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


@dataclass
class AgenticWorktreeSession:
    worktree_root: Path
    project_dir: Path
    repo_root: Optional[Path]
    is_real_worktree: bool
    keep: bool
    baseline_commit: str

    def changed_paths(self) -> list[str]:
        proc = _git(self.worktree_root, "diff", "--name-only", "HEAD", "--", check=True)
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]

    def has_changes(self) -> bool:
        proc = _git(self.worktree_root, "diff", "--quiet", "HEAD", "--", check=False)
        return proc.returncode == 1

    def diff_text(self) -> str:
        return _git_stdout(self.worktree_root, "diff", "HEAD", "--")

    def commit_all(self, message: str) -> str:
        self.baseline_commit = _commit_all(self.worktree_root, message)
        return self.baseline_commit

    def cleanup(self) -> None:
        if self.keep:
            return
        if self.is_real_worktree and self.repo_root is not None:
            _git(self.repo_root, "worktree", "remove", "--force", str(self.worktree_root), check=False)
            return
        shutil.rmtree(self.worktree_root, ignore_errors=True)


def _commit_all(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    proc = _git(root, "diff", "--cached", "--quiet", check=False)
    if proc.returncode == 1:
        _git(root, "commit", "-m", message)
    return _git_stdout(root, "rev-parse", "HEAD")


def create_agentic_worktree(*, task_path: str | Path | None) -> AgenticWorktreeSession:
    if task_path is None:
        raise AgenticWorktreeError("AscendC agentic codegen requires task_path")

    task_root = Path(task_path).expanduser().resolve()
    if not task_root.exists() or not task_root.is_dir():
        raise AgenticWorktreeError(f"task_path is not a directory: {task_root}")

    keep = os.getenv("KSEARCH_KEEP_AGENTIC_WORKTREES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    repo_root = _find_git_root(task_root)
    temp_root = Path(tempfile.mkdtemp(prefix="ksearch_agentic_worktree_")).resolve()

    if repo_root is not None:
        try:
            _git(repo_root, "worktree", "add", "--detach", str(temp_root), "HEAD")
            rel_project = task_root.relative_to(repo_root)
            project_dir = temp_root / rel_project
            _git_config_identity(temp_root)
            baseline_commit = _commit_all(temp_root, "ksearch agentic baseline")
            return AgenticWorktreeSession(
                worktree_root=temp_root,
                project_dir=project_dir,
                repo_root=repo_root,
                is_real_worktree=True,
                keep=keep,
                baseline_commit=baseline_commit,
            )
        except Exception:
            shutil.rmtree(temp_root, ignore_errors=True)

    fallback_root = Path(tempfile.mkdtemp(prefix="ksearch_agentic_temp_repo_")).resolve()
    _copy_project(task_root, fallback_root)
    _git(fallback_root, "init")
    _git_config_identity(fallback_root)
    baseline_commit = _commit_all(fallback_root, "ksearch agentic baseline")
    return AgenticWorktreeSession(
        worktree_root=fallback_root,
        project_dir=fallback_root,
        repo_root=None,
        is_real_worktree=False,
        keep=keep,
        baseline_commit=baseline_commit,
    )
```

- [ ] **Step 4: Run tests to verify worktree behavior passes**

Run:

```bash
pytest tests/kernel_generators/test_agentic_worktree.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add k_search/kernel_generators/agentic_worktree.py tests/kernel_generators/test_agentic_worktree.py
git commit -m "feat(agentic): add git-backed worktree sessions"
```

---

### Task 2: AscendC Project-Directory Adapter Hooks

**Files:**
- Modify: `tests/test_ascendc_task.py`
- Modify: `k_search/tasks/ascendc_task.py`

- [ ] **Step 1: Add failing AscendC adapter tests**

Append these tests to `tests/test_ascendc_task.py`:

```python
def test_ascendc_task_overlays_solution_sources_into_project_dir(tmp_path):
    task = AscendCTask(task_path=tmp_path, definition_name="x")
    solution = Solution(
        name="candidate",
        definition="x",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.ASCENDC,
            target_hardware=["ascend_910b"],
            entry_point="kernel/foo.h::run",
        ),
        sources=[
            SourceFile(path="kernel/foo.h", content="overlaid\n"),
            SourceFile(path="tiling.cpp", content="tiling\n"),
        ],
    )

    task.overlay_solution_sources(project_dir=tmp_path, solution=solution)

    assert (tmp_path / "kernel" / "foo.h").read_text(encoding="utf-8") == "overlaid\n"
    assert (tmp_path / "tiling.cpp").read_text(encoding="utf-8") == "tiling\n"


def test_make_solution_from_project_dir_scans_sources_and_ignores_build_dirs(tmp_path):
    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel" / "foo.h").write_text("int x = 1;\n", encoding="utf-8")
    (tmp_path / "kernel.cpp").write_text("void run() {}\n", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "generated.cpp").write_text("ignored\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("ignored\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x")

    solution = task.make_solution_from_project_dir(
        project_dir=tmp_path,
        changed_paths=["kernel/foo.h", "kernel.cpp"],
        raw_agent_output="changed kernel files",
        round_num=4,
        model_name="claude-sonnet-4-6",
        target_gpu="ascend_910b",
        language="ascendc",
    )

    assert solution.definition == "x"
    assert solution.spec.language == SupportedLanguages.ASCENDC
    assert solution.spec.target_hardware == ["ascend_910b"]
    assert solution.get_entry_path() == "kernel.cpp"
    assert {src.path for src in solution.sources} == {"kernel/foo.h", "kernel.cpp"}
    assert "changed kernel files" in str(solution.description)


def test_make_solution_from_project_dir_rejects_forbidden_changed_path(tmp_path):
    (tmp_path / "kernel.cpp").write_text("void run() {}\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x")

    with pytest.raises(ValueError, match="forbidden agentic changed path"):
        task.make_solution_from_project_dir(
            project_dir=tmp_path,
            changed_paths=["build/generated.cpp"],
            raw_agent_output="changed build output",
            round_num=1,
            model_name="claude",
            target_gpu="ascend_910b",
            language="ascendc",
        )


def test_get_agentic_definition_text_omits_source_containers(tmp_path):
    (tmp_path / "spec.md").write_text("Vector add spec.", encoding="utf-8")
    (tmp_path / "kernel.cpp").write_text("int source_should_not_appear = 1;\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x")

    text = task.get_agentic_definition_text(language="ascendc")

    assert "Vector add spec." in text
    assert "Existing project source excerpts" not in text
    assert "source_should_not_appear" not in text
    assert "<ascendc_project>" not in text


def test_solution_from_raw_code_for_agentic_parses_full_container_without_advancing_patch_state(tmp_path):
    task = AscendCTask(task_path=tmp_path, definition_name="x")
    raw = format_ascendc_project_files({"kernel.cpp": "void run() {}\n"})

    solution = task.solution_from_raw_code_for_agentic(
        raw_code=raw,
        round_num=8,
        model_name="claude",
        target_gpu="ascend_910b",
        language="ascendc",
    )

    assert solution.definition == "x"
    assert {src.path for src in solution.sources} == {"kernel.cpp"}
```

- [ ] **Step 2: Run targeted tests to verify they fail**

Run:

```bash
pytest tests/test_ascendc_task.py -v -k "agentic or project_dir or overlays"
```

Expected: FAIL with missing `AscendCTask` methods.

- [ ] **Step 3: Implement AscendC adapter methods**

Modify `k_search/tasks/ascendc_task.py`.

Add this helper near `_normalize_rel_path`:

```python
def _is_forbidden_agentic_changed_path(path: str) -> bool:
    rel = str(path or "").strip().replace("\\", "/")
    if not rel:
        return True
    parts = tuple(p for p in rel.split("/") if p)
    forbidden_parts = {
        ".git",
        ".ksearch",
        "__pycache__",
        "build",
        "cmake-build-debug",
        "logs",
        "llm_logs",
    }
    return any(part in forbidden_parts for part in parts)
```

Add these methods inside `AscendCTask`:

```python
    def get_agentic_definition_text(self, *, language: str) -> str:
        return self.get_definition_text(
            language=str(language),
            include_sources=False,
            include_format=False,
        )

    def overlay_solution_sources(
        self,
        *,
        project_dir: str | Path,
        solution: Solution | None,
    ) -> None:
        if solution is None:
            return
        root = Path(project_dir).expanduser().resolve()
        for src in solution.sources or []:
            rel = _normalize_rel_path(src.path)
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(str(src.content or ""), encoding="utf-8")

    def _validate_agentic_changed_paths(self, changed_paths: list[str] | None) -> None:
        for path in changed_paths or []:
            rel = _normalize_rel_path(path)
            if _is_forbidden_agentic_changed_path(rel):
                raise ValueError(f"forbidden agentic changed path: {rel}")

    def make_solution_from_project_dir(
        self,
        *,
        project_dir: str | Path,
        changed_paths: list[str] | None,
        raw_agent_output: str,
        round_num: int,
        model_name: str,
        target_gpu: str,
        language: str,
    ) -> Solution:
        self._validate_agentic_changed_paths(changed_paths)
        root = Path(project_dir).expanduser().resolve()
        sources = _collect_project_sources(root)
        if not sources:
            raise ValueError(f"agentic project produced no source files: {root}")
        return Solution(
            name=f"{model_name}_{self.name}_ascendc_agentic_r{int(round_num)}",
            definition=self.name,
            author=str(model_name),
            spec=BuildSpec(
                language=SupportedLanguages.ASCENDC,
                target_hardware=[str(target_gpu or "ascend")],
                entry_point=_default_entry_point(sources),
            ),
            sources=sources,
            description=(
                f"{model_name} agentic AscendC project for {self.name} "
                f"(round {int(round_num)}): {str(raw_agent_output or '').strip()[:500]}"
            ),
        )

    def solution_from_raw_code_for_agentic(
        self,
        *,
        raw_code: str,
        round_num: int,
        model_name: str,
        target_gpu: str,
        language: str,
    ) -> Solution:
        files = parse_ascendc_project_files(raw_code)
        sources = [SourceFile(path=path, content=content) for path, content in sorted(files.items())]
        return Solution(
            name=f"{model_name}_{self.name}_ascendc_agentic_base_r{int(round_num)}",
            definition=self.name,
            author=str(model_name),
            spec=BuildSpec(
                language=SupportedLanguages.ASCENDC,
                target_hardware=[str(target_gpu or "ascend")],
                entry_point=_default_entry_point(sources),
            ),
            sources=sources,
            description=f"{model_name} AscendC agentic base for {self.name} (round {int(round_num)})",
        )
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
pytest tests/test_ascendc_task.py -v -k "agentic or project_dir or overlays"
```

Expected: PASS.

- [ ] **Step 5: Run full AscendC task tests**

Run:

```bash
pytest tests/test_ascendc_task.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add k_search/tasks/ascendc_task.py tests/test_ascendc_task.py
git commit -m "feat(ascendc): add agentic project-dir solution hooks"
```

---

### Task 3: Claude Agent SDK Project Editor Client And Mock Session

**Files:**
- Create: `k_search/kernel_generators/claude_agent_project_editor.py`
- Modify: `k_search/testing/mock_claude_agent_sdk.py`
- Modify: `tests/kernel_generators/test_claude_agent_sdk_mock.py`
- Modify: `tests/kernel_generators/test_llm_clients.py`

- [ ] **Step 1: Add failing project-editor client tests**

Append this test to `tests/kernel_generators/test_claude_agent_sdk_mock.py`:

```python
def test_claude_project_editor_client_uses_sdk_client_with_cwd_and_file_tools(monkeypatch, tmp_path):
    from pathlib import Path

    from k_search.kernel_generators.claude_agent_project_editor import (
        ClaudeAgentProjectEditorClient,
    )

    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    def edit_project(prompt, options, call_index):
        project_dir = Path(options.kwargs["cwd"])
        target = project_dir / "kernel" / "foo.h"
        target.write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
        return [
            MockClaudeMessage(content=[{"type": "text", "text": "edited foo.h"}]),
            MockClaudeMessage(result="final summary"),
        ]

    sdk = install_mock_claude_agent_sdk(monkeypatch, responses=[edit_project])
    client = ClaudeAgentProjectEditorClient(model_name="claude-sonnet-4-6", timeout_seconds=30)

    result = client.edit_project(project_dir=tmp_path, prompt="Please edit the project.")

    assert result.text == "final summary"
    assert result.transcript == "edited foo.h\nfinal summary"
    assert (tmp_path / "kernel" / "foo.h").read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"
    assert len(sdk.client_calls) == 1
    call = sdk.client_calls[0]
    assert call.prompt == "Please edit the project."
    assert call.options.kwargs["cwd"] == str(tmp_path)
    assert call.options.kwargs["allowed_tools"] == ["Read", "Grep", "Glob", "Edit", "Write"]
    assert call.options.kwargs["disallowed_tools"] == ["Bash"]
    assert call.options.kwargs["permission_mode"] == "acceptEdits"
    assert call.options.kwargs["model"] == "claude-sonnet-4-6"
```

Append this regression test to `tests/kernel_generators/test_llm_clients.py`:

```python
def test_claude_agent_llm_client_still_uses_query_not_sdk_client(monkeypatch):
    seen = {"query": 0, "client": 0}

    class FakeClient:
        def __init__(self, options):
            seen["client"] += 1

    async def fake_query(prompt, options):
        seen["query"] += 1
        yield SimpleNamespace(result="query text")

    fake_module = SimpleNamespace(
        ClaudeAgentOptions=lambda **kwargs: SimpleNamespace(kwargs=kwargs),
        ClaudeSDKClient=FakeClient,
        query=fake_query,
    )
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_module)

    client = ClaudeAgentLLMClient(model_name="claude-sonnet-4-6")

    assert client.generate("prompt") == "query text"
    assert seen == {"query": 1, "client": 0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/kernel_generators/test_claude_agent_sdk_mock.py::test_claude_project_editor_client_uses_sdk_client_with_cwd_and_file_tools tests/kernel_generators/test_llm_clients.py::test_claude_agent_llm_client_still_uses_query_not_sdk_client -v
```

Expected: first test FAILS because `claude_agent_project_editor.py` is missing. The second may fail until the mock and module are aligned.

- [ ] **Step 3: Extend the Claude SDK mock with `ClaudeSDKClient`**

Modify `k_search/testing/mock_claude_agent_sdk.py`.

Add a `client_calls` field in `MockClaudeAgentSDK.__init__`:

```python
        self.client_calls: list[MockClaudeCall] = []
```

Replace `as_module()` with a version that preserves `query()` and adds `ClaudeSDKClient`:

```python
    def as_module(self) -> Any:
        sdk = self

        class ClaudeAgentOptions(MockClaudeAgentOptions):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                sdk.options.append(self)

        async def query(prompt: str, options: MockClaudeAgentOptions):
            call_index = len(sdk.calls)
            sdk.calls.append(MockClaudeCall(prompt=str(prompt or ""), options=options))
            response = sdk._next_response(
                prompt=str(prompt or ""),
                options=options,
                call_index=call_index,
            )
            if isinstance(response, BaseException):
                raise response
            for message in sdk._coerce_messages(response):
                yield message

        class ClaudeSDKClient:
            def __init__(self, options: MockClaudeAgentOptions) -> None:
                self.options = options
                self._prompt = ""
                self._messages: list[MockClaudeMessage] = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def query(self, prompt: str) -> None:
                call_index = len(sdk.client_calls)
                self._prompt = str(prompt or "")
                sdk.client_calls.append(MockClaudeCall(prompt=self._prompt, options=self.options))
                response = sdk._next_response(
                    prompt=self._prompt,
                    options=self.options,
                    call_index=call_index,
                )
                if isinstance(response, BaseException):
                    raise response
                self._messages = sdk._coerce_messages(response)

            async def receive_response(self):
                for message in self._messages:
                    yield message

        return SimpleNamespace(
            ClaudeAgentOptions=ClaudeAgentOptions,
            ClaudeSDKClient=ClaudeSDKClient,
            query=query,
        )
```

- [ ] **Step 4: Implement the project editor client**

Create `k_search/kernel_generators/claude_agent_project_editor.py` with:

```python
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from k_search.kernel_generators.llm_clients import (
    ClaudeAgentLLMClient,
    LLMProviderFatalError,
    _as_provider_exception,
    _default_claude_agent_max_turns,
    _default_claude_agent_thinking_enabled,
    _default_claude_agent_timeout_seconds,
    _log_llm_interaction,
)


DEFAULT_PROJECT_EDITOR_TOOLS = ["Read", "Grep", "Glob", "Edit", "Write"]


@dataclass
class ClaudeProjectEditResult:
    text: str
    transcript: str
    prompt: str
    prompt_chars: int
    prompt_lines: int


@dataclass
class ClaudeAgentProjectEditorClient:
    model_name: str
    max_turns: Optional[int] = field(default_factory=_default_claude_agent_max_turns)
    allowed_tools: list[str] = field(default_factory=lambda: list(DEFAULT_PROJECT_EDITOR_TOOLS))
    disallowed_tools: list[str] = field(default_factory=lambda: ["Bash"])
    thinking_enabled: bool = field(default_factory=_default_claude_agent_thinking_enabled)
    timeout_seconds: float = field(default_factory=_default_claude_agent_timeout_seconds)

    def edit_project(self, *, project_dir: str | Path, prompt: str) -> ClaudeProjectEditResult:
        project_root = Path(project_dir).expanduser().resolve()
        prompt_text = str(prompt or "")
        try:
            import claude_agent_sdk  # type: ignore
        except ImportError as exc:
            _log_llm_interaction(
                provider="claude-agent",
                model_name=self.model_name,
                prompt=prompt_text,
                response="",
                error=str(exc),
            )
            raise RuntimeError(
                "Claude Agent SDK provider requires the 'claude-agent-sdk' package. "
                "Install it with: pip install claude-agent-sdk"
            ) from exc

        async def _run_client() -> ClaudeProjectEditResult:
            options_kwargs: dict[str, Any] = {
                "model": self.model_name,
                "cwd": str(project_root),
                "allowed_tools": list(self.allowed_tools),
                "disallowed_tools": list(self.disallowed_tools),
                "permission_mode": "acceptEdits",
            }
            if self.max_turns is not None:
                options_kwargs["max_turns"] = self.max_turns
            if not self.thinking_enabled:
                options_kwargs["thinking"] = {"type": "disabled"}

            options = claude_agent_sdk.ClaudeAgentOptions(**options_kwargs)
            chunks: list[str] = []
            final_text = ""
            try:
                async with claude_agent_sdk.ClaudeSDKClient(options=options) as client:
                    await client.query(prompt_text)
                    async for message in client.receive_response():
                        is_result_message = hasattr(message, "result")
                        if is_result_message:
                            ClaudeAgentLLMClient._ensure_successful_result_message(message)
                        text = ClaudeAgentLLMClient._extract_message_text(message)
                        if not text:
                            continue
                        chunks.append(text)
                        if is_result_message:
                            final_text = text
            except LLMProviderFatalError:
                raise
            except Exception as exc:
                provider_exc = _as_provider_exception(
                    provider="claude-agent",
                    model_name=self.model_name,
                    exc=exc,
                )
                if isinstance(provider_exc, LLMProviderFatalError):
                    raise provider_exc from exc
                raise RuntimeError(f"Claude Agent SDK project editor failed: {exc}") from exc

            transcript = "\n".join(chunks).strip()
            result_text = (final_text or transcript).strip()
            if not result_text:
                raise RuntimeError("Claude Agent SDK project editor returned empty text")
            return ClaudeProjectEditResult(
                text=result_text,
                transcript=transcript,
                prompt=prompt_text,
                prompt_chars=len(prompt_text),
                prompt_lines=(prompt_text.count("\n") + 1 if prompt_text else 0),
            )

        try:
            result = self._run_async(_run_client)
            _log_llm_interaction(
                provider="claude-agent",
                model_name=self.model_name,
                prompt=prompt_text,
                response=result.transcript,
            )
            return result
        except Exception as exc:
            provider_exc = _as_provider_exception(provider="claude-agent", model_name=self.model_name, exc=exc)
            _log_llm_interaction(
                provider="claude-agent",
                model_name=self.model_name,
                prompt=prompt_text,
                response="",
                error=str(provider_exc),
            )
            if provider_exc is exc:
                raise
            raise provider_exc from exc

    def _run_async(self, coro_factory: Any) -> ClaudeProjectEditResult:
        timeout = float(self.timeout_seconds or 0)
        started = time.monotonic()

        async def _timed_run() -> ClaudeProjectEditResult:
            if timeout <= 0:
                return await coro_factory()
            return await asyncio.wait_for(coro_factory(), timeout=timeout)

        def _timeout_error(exc: BaseException) -> TimeoutError:
            return TimeoutError(
                f"Claude Agent SDK project editor timed out after {timeout:g}s. "
                "Set KSEARCH_LLM_TIMEOUT_SECONDS or API_TIMEOUT_MS to adjust this limit."
            )

        def _looks_like_timeout_cancel(exc: BaseException) -> bool:
            if timeout <= 0:
                return False
            elapsed = time.monotonic() - started
            return elapsed >= (timeout * 0.9) and "exit code 143" in str(exc)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                return asyncio.run(_timed_run())
            except TimeoutError as exc:
                raise _timeout_error(exc) from exc
            except RuntimeError as exc:
                if _looks_like_timeout_cancel(exc):
                    raise _timeout_error(exc) from exc
                raise

        def _runner() -> ClaudeProjectEditResult:
            return asyncio.run(_timed_run())

        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                return executor.submit(_runner).result()
            except TimeoutError as exc:
                raise _timeout_error(exc) from exc
            except RuntimeError as exc:
                if _looks_like_timeout_cancel(exc):
                    raise _timeout_error(exc) from exc
                raise
```

- [ ] **Step 5: Run project-editor and regression tests**

Run:

```bash
pytest tests/kernel_generators/test_claude_agent_sdk_mock.py::test_claude_project_editor_client_uses_sdk_client_with_cwd_and_file_tools tests/kernel_generators/test_llm_clients.py::test_claude_agent_llm_client_still_uses_query_not_sdk_client -v
```

Expected: PASS.

- [ ] **Step 6: Run existing Claude client tests**

Run:

```bash
pytest tests/kernel_generators/test_llm_clients.py tests/kernel_generators/test_claude_agent_sdk_mock.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add k_search/kernel_generators/claude_agent_project_editor.py k_search/testing/mock_claude_agent_sdk.py tests/kernel_generators/test_claude_agent_sdk_mock.py tests/kernel_generators/test_llm_clients.py
git commit -m "feat(claude): add project editor sdk client"
```

---

### Task 4: AscendC Agentic Prompt Builder And Runner

**Files:**
- Create: `tests/kernel_generators/test_ascendc_agentic_codegen.py`
- Create: `k_search/kernel_generators/ascendc_agentic_codegen.py`

- [ ] **Step 1: Write failing runner and prompt tests**

Create `tests/kernel_generators/test_ascendc_agentic_codegen.py` with:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/kernel_generators/test_ascendc_agentic_codegen.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'k_search.kernel_generators.ascendc_agentic_codegen'`.

- [ ] **Step 3: Implement prompt builder and runner**

Create `k_search/kernel_generators/ascendc_agentic_codegen.py` with:

```python
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

```

- [ ] **Step 4: Run runner tests**

Run:

```bash
pytest tests/kernel_generators/test_ascendc_agentic_codegen.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add k_search/kernel_generators/ascendc_agentic_codegen.py tests/kernel_generators/test_ascendc_agentic_codegen.py
git commit -m "feat(ascendc): add agentic codegen runner"
```

---

### Task 5: Route Baseline `KernelGenerator` Before Legacy Prompt Construction

**Files:**
- Modify: `tests/kernel_generators/test_claude_agent_sdk_mock.py`
- Modify: `k_search/kernel_generators/kernel_generator.py`

- [ ] **Step 1: Update two-round Claude+AscendC mock test for agentic routing**

Replace `test_claude_agent_sdk_mock_drives_ascendc_two_round_optimization` in `tests/kernel_generators/test_claude_agent_sdk_mock.py` with:

```python
def test_claude_agent_sdk_mock_drives_agentic_ascendc_two_round_optimization(
    monkeypatch, tmp_path
):
    from pathlib import Path

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (tmp_path / "spec.md").write_text("Optimize a tiny AscendC project.", encoding="utf-8")
    (kernel_dir / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    def first_edit(prompt, options, call_index):
        project_dir = Path(options.kwargs["cwd"])
        (project_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n// initial agent edit\n", encoding="utf-8")
        return "kept initial project"

    def second_edit(prompt, options, call_index):
        project_dir = Path(options.kwargs["cwd"])
        (project_dir / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
        return "edited kernel/foo.h"

    sdk = install_mock_claude_agent_sdk(
        monkeypatch,
        responses=[first_edit, second_edit],
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
    assert len(sdk.client_calls) == 2
    assert sdk.calls == []
    assert "<ascendc_project>" not in sdk.client_calls[0].prompt
    assert "<ascendc_project>" not in sdk.client_calls[1].prompt
    assert sdk.client_calls[0].options.kwargs["cwd"]
```

- [ ] **Step 2: Run the updated test to verify it fails**

Run:

```bash
pytest tests/kernel_generators/test_claude_agent_sdk_mock.py::test_claude_agent_sdk_mock_drives_agentic_ascendc_two_round_optimization -v
```

Expected: FAIL because `KernelGenerator.generate()` still calls legacy `ClaudeAgentLLMClient.generate()` with `query()`.

- [ ] **Step 2a: Add fallback-specific baseline regression test**

Append this test to `tests/kernel_generators/test_llm_clients.py`:

```python
def test_baseline_ascendc_agentic_failure_can_fallback_to_legacy(monkeypatch, tmp_path):
    from k_search.kernel_generators.kernel_generator import KernelGenerator
    from k_search.tasks.ascendc_task import AscendCTask, format_ascendc_project_files

    class FailingRunner:
        def run(self, *, task, request, base_solution):
            raise RuntimeError("agentic unavailable")

    class LegacyClient:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt):
            self.prompts.append(prompt)
            return format_ascendc_project_files({"kernel.cpp": "void run() {}\n"})

    (tmp_path / "spec.md").write_text("Optimize tiny project.", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x")
    legacy_client = LegacyClient()
    generator = KernelGenerator(
        model_name="fake",
        language="ascendc",
        target_gpu="ascend_910b",
        llm_provider="claude-agent",
        llm_client=legacy_client,
    )
    generator._ascendc_agentic_runner = FailingRunner()
    monkeypatch.setenv("KSEARCH_ASCENDC_AGENTIC_FALLBACK", "legacy")

    solution = generator.generate(task=task, max_opt_rounds=1)

    assert {src.path for src in solution.sources} == {"kernel.cpp"}
    assert legacy_client.prompts
    assert "<ascendc_project>" in legacy_client.prompts[0]
```

Run:

```bash
pytest tests/kernel_generators/test_llm_clients.py::test_baseline_ascendc_agentic_failure_can_fallback_to_legacy -v
```

Expected before Step 4: FAIL with `RuntimeError: agentic unavailable`. Expected after Step 4: PASS.

- [ ] **Step 3: Add helper methods to `KernelGenerator`**

Modify `k_search/kernel_generators/kernel_generator.py`.

Add imports:

```python
from .ascendc_agentic_codegen import (
    AscendCAgenticCodegenRequest,
    AscendCAgenticCodegenRunner,
)
```

Add methods inside `KernelGenerator`:

```python
    def _should_use_ascendc_agentic_codegen(self, task: Any) -> bool:
        if self.llm_provider != "claude-agent":
            return False
        if str(self.language or "").strip().lower() != "ascendc":
            return False
        if os.getenv("KSEARCH_DISABLE_ASCENDC_AGENTIC_CODEGEN", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return False
        return hasattr(task, "make_solution_from_project_dir")

    def _allow_ascendc_agentic_legacy_fallback(self) -> bool:
        return os.getenv("KSEARCH_ASCENDC_AGENTIC_FALLBACK", "").strip().lower() == "legacy"

    def _agentic_runner(self) -> AscendCAgenticCodegenRunner:
        runner = getattr(self, "_ascendc_agentic_runner", None)
        if runner is None:
            runner = AscendCAgenticCodegenRunner(model_name=str(self.model_name))
            self._ascendc_agentic_runner = runner
        return runner

    def _generate_ascendc_solution_agentically(
        self,
        *,
        task: Any,
        action_text: str,
        trace_logs: str,
        perf_summary: str,
        round_num: int,
        attempt_idx: int,
        mode: str,
        base_solution: Optional[Solution],
    ) -> Solution:
        definition_hook = getattr(task, "get_agentic_definition_text", None)
        if callable(definition_hook):
            definition_text = str(definition_hook(language=str(self.language)) or "").strip()
        else:
            definition_text = str(task.get_definition_text(language=str(self.language)) or "").strip()
        request = AscendCAgenticCodegenRequest(
            definition_text=definition_text,
            action_text=str(action_text or "").strip(),
            trace_logs=str(trace_logs or "").strip(),
            perf_summary=str(perf_summary or "").strip(),
            target_gpu=str(self.target_gpu),
            round_num=int(round_num),
            attempt_idx=int(attempt_idx),
            mode=str(mode),  # type: ignore[arg-type]
        )
        result = self._agentic_runner().run(
            task=task,
            request=request,
            base_solution=base_solution,
        )
        print(
            f"[LLM] agentic ascendc result provider={self.llm_provider} model={self.model_name} "
            f"round={round_num} prompt_chars={result.prompt_chars} "
            f"changed_files={','.join(result.changed_paths)} worktree={result.worktree_path}",
            flush=True,
        )
        return result.solution
```

- [ ] **Step 4: Route initial and optimization generation before legacy prompts**

In `KernelGenerator.generate()`, modify the seed generation branch before `gen_prompt_fn` is used:

```python
            if self._should_use_ascendc_agentic_codegen(task):
                try:
                    solution = self._generate_ascendc_solution_agentically(
                        task=task,
                        action_text=(
                            "Create an optimized AscendC candidate from the current project files. "
                            "Keep public interfaces and harness behavior unchanged."
                        ),
                        trace_logs="",
                        perf_summary="",
                        round_num=0,
                        attempt_idx=1,
                        mode="generate",
                        base_solution=None,
                    )
                    current_code, current_raw_code = code_from_solution(self.language, solution)
                    seed_solution = solution
                except Exception as exc:
                    if not self._allow_ascendc_agentic_legacy_fallback():
                        raise
                    print(
                        f"[WARN] agentic AscendC seed codegen failed; falling back to legacy prompt path: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                else:
                    pass
```

After this block, keep the existing legacy seed prompt construction guarded by `if seed_solution is None and current_raw_code is None:` so fallback can continue into the old path and successful agentic seed skips it.

In the next-round prompt preparation block, before `opt_prompt_fn = getattr(task, "get_optimization_prompt", None)`, add:

```python
                if self._should_use_ascendc_agentic_codegen(task):
                    base_for_agentic = solution
                    action_text = (
                        "Improve the current AscendC candidate. If the last attempt failed, fix compile, "
                        "runtime, or correctness first. If it passed, reduce measured latency while preserving semantics."
                    )
                    try:
                        solution = self._generate_ascendc_solution_agentically(
                            task=task,
                            action_text=action_text,
                            trace_logs=str(trace_logs or ""),
                            perf_summary=str(previous_round_summary_for_prompt or ""),
                            round_num=round_num + 1,
                            attempt_idx=1,
                            mode="improve",
                            base_solution=base_for_agentic,
                        )
                        current_code, current_raw_code = code_from_solution(self.language, solution)
                        continue
                    except Exception as exc:
                        if not self._allow_ascendc_agentic_legacy_fallback():
                            raise
                        print(
                            f"[WARN] agentic AscendC optimization codegen failed; "
                            f"falling back to legacy prompt path: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
```

Keep the existing legacy prompt code below this branch unchanged.

- [ ] **Step 5: Run the updated agentic baseline test**

Run:

```bash
pytest tests/kernel_generators/test_claude_agent_sdk_mock.py::test_claude_agent_sdk_mock_drives_agentic_ascendc_two_round_optimization -v
```

Expected: PASS.

- [ ] **Step 6: Run baseline generator regression tests**

Run:

```bash
pytest tests/kernel_generators/test_claude_agent_sdk_mock.py tests/kernel_generators/test_llm_clients.py tests/kernel_generators/test_kernel_generator_codegen.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add k_search/kernel_generators/kernel_generator.py tests/kernel_generators/test_claude_agent_sdk_mock.py
git commit -m "feat(ascendc): route claude baseline codegen through agentic worktrees"
```

---

### Task 6: Route World-Model AscendC Codegen Before Legacy Prompt Construction

**Files:**
- Modify: `tests/kernel_generators/test_llm_clients.py`
- Modify: `k_search/kernel_generators/kernel_generator_world_model.py`

- [ ] **Step 1: Add a failing world-model agentic routing test**

Append this test to `tests/kernel_generators/test_llm_clients.py`:

```python
def test_world_model_ascendc_codegen_uses_agentic_runner_before_prompt_construction(tmp_path):
    from k_search.kernel_generators.kernel_generator_world_model import (
        WorldModelKernelGeneratorWithBaseline,
    )
    from k_search.tasks.ascendc_task import AscendCTask
    from k_search.tasks.task_base import EvalResult

    (tmp_path / "spec.md").write_text("Optimize tiny project.", encoding="utf-8")
    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    class FakeAgenticRunner:
        def __init__(self):
            self.requests = []

        def run(self, *, task, request, base_solution):
            self.requests.append(request)
            (tmp_path / "kernel" / "foo.h").write_text("alpha\nBETA\ngamma\n", encoding="utf-8")
            solution = task.make_solution_from_project_dir(
                project_dir=tmp_path,
                changed_paths=["kernel/foo.h"],
                raw_agent_output="edited",
                round_num=request.round_num,
                model_name="fake",
                target_gpu=request.target_gpu,
                language="ascendc",
            )
            from k_search.kernel_generators.ascendc_agentic_codegen import AscendCAgenticCodegenResult

            return AscendCAgenticCodegenResult(
                solution=solution,
                raw=task.code_for_world_model_from_raw(raw={src.path: src.content for src in solution.sources}, language="ascendc"),
                cleaned={src.path: src.content for src in solution.sources},
                transcript="edited",
                prompt="compact prompt",
                prompt_chars=14,
                changed_paths=["kernel/foo.h"],
                diff_text="diff",
                worktree_path=str(tmp_path),
            )

    class FakeWorldModel:
        def propose_action_nodes(self, **kwargs):
            return None

        def get_tree_path_text(self, definition_name):
            return ""

        def get(self, definition_name):
            return ""

        def choose_next_action_node_id(self, definition_name):
            return "n1"

        def set_active_leaf_id(self, definition_name, node_id):
            return None

        def get_node_obj(self, definition_name, node_id):
            return {
                "id": "n1",
                "node_id": "n1",
                "parent_id": "root",
                "action": {
                    "title": "capitalize beta",
                    "description": "Change beta to BETA.",
                    "difficulty_1_to_5": 1,
                    "expected_vs_baseline_factor": 1.1,
                },
            }

        def get_solution_ref_for_node(self, definition_name, node_id):
            return None

        def attach_solution_to_active_leaf(self, **kwargs):
            return None

        def refine(self, **kwargs):
            return None

        def note_action_too_hard(self, **kwargs):
            return None

    task = AscendCTask(
        task_path=tmp_path,
        definition_name="x",
        build_cmd="",
        test_cmd="",
        bench_cmd="python -c \"print('latency_ms=1.0')\"",
        timeout_seconds=30,
    )
    generator = WorldModelKernelGeneratorWithBaseline(
        model_name="fake",
        language="ascendc",
        target_gpu="ascend_910b",
        llm_provider="claude-agent",
        llm_client=SimpleNamespace(generate=lambda prompt: "{}"),
    )
    fake_runner = FakeAgenticRunner()
    generator._wm = FakeWorldModel()
    generator._solution_db = None
    generator._ascendc_agentic_runner = fake_runner

    solution = generator._generate_world_model_cycles_v2(
        task=task,
        max_opt_rounds=1,
        wm_stagnation_window=1,
        max_dai=1,
    )

    assert "BETA" in next(src.content for src in solution.sources if src.path == "kernel/foo.h")
    assert len(fake_runner.requests) == 1
    assert fake_runner.requests[0].action_text
    assert "<ascendc_project>" not in fake_runner.requests[0].action_text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/kernel_generators/test_llm_clients.py::test_world_model_ascendc_codegen_uses_agentic_runner_before_prompt_construction -v
```

Expected: FAIL because world-model codegen still builds legacy prompts and calls `_generate_code_from_prompt()`.

- [ ] **Step 3: Import agentic request types in world-model generator**

Modify `k_search/kernel_generators/kernel_generator_world_model.py`.

Add imports:

```python
from k_search.kernel_generators.ascendc_agentic_codegen import AscendCAgenticCodegenRequest
```

- [ ] **Step 4: Insert world-model agentic branch before prompt construction**

In `_generate_world_model_cycles_v2`, inside the inner `while True` loop after `chosen_action_text` is validated and before the `elif attempt_idx == 1:` legacy prompt branch builds large prompts, add:

```python
                if self._should_use_ascendc_agentic_codegen(task):
                    trace_excerpt = str(getattr(task, "get_last_round_trace_logs_for_prompt", lambda: "")() or "")
                    perf_lines: list[str] = []
                    if last_eval is not None:
                        perf_lines.extend(last_eval.perf_summary_lines(prefix="last_attempt"))
                    if base_eval is not None:
                        perf_lines.extend(base_eval.perf_summary_lines(prefix="base"))
                    perf_summary = "\n".join(perf_lines).strip()
                    if attempt_idx == 1:
                        agentic_mode = "action"
                        agentic_action = str(chosen_action_text or "")
                        base_solution_for_agentic = None
                        if isinstance(base_raw_code, str) and base_raw_code.strip():
                            try:
                                base_solution_for_agentic = task.solution_from_raw_code_for_agentic(
                                    raw_code=base_raw_code,
                                    round_num=round_num,
                                    model_name=str(self.model_name),
                                    target_gpu=str(self.target_gpu),
                                    language=str(self.language),
                                )
                            except Exception:
                                base_solution_for_agentic = None
                    else:
                        agentic_mode = "debug" if cycle_best_solution is None else "improve"
                        agentic_action = (
                            str(chosen_action_text or "")
                            + "\n\nContinue the same action. If the previous attempt failed, fix it first. "
                            "If it passed, improve latency without broadening scope."
                        )
                        base_solution_for_agentic = last_solution or cycle_best_solution

                    definition_hook = getattr(task, "get_agentic_definition_text", None)
                    if callable(definition_hook):
                        agentic_definition = str(definition_hook(language=str(self.language)) or "")
                    else:
                        agentic_definition = _definition_text_for_codegen_prompt(
                            task,
                            language=str(self.language),
                            has_explicit_base_code=False,
                        )
                    request = AscendCAgenticCodegenRequest(
                        definition_text=agentic_definition,
                        action_text=agentic_action,
                        trace_logs=trace_excerpt,
                        perf_summary=perf_summary,
                        target_gpu=str(self.target_gpu),
                        round_num=int(round_num),
                        attempt_idx=int(attempt_idx),
                        mode=agentic_mode,  # type: ignore[arg-type]
                    )
                    try:
                        result = self._agentic_runner().run(
                            task=task,
                            request=request,
                            base_solution=base_solution_for_agentic,
                        )
                    except LLMProviderFatalError as exc:
                        _emit(f"[ERROR] fatal LLM provider error during agentic codegen: {exc}")
                        raise
                    except (TimeoutError, ValueError, RuntimeError) as exc:
                        if self._allow_ascendc_agentic_legacy_fallback():
                            _emit(
                                f"[WARN] agentic codegen failed for action_node_id={chosen_leaf} "
                                f"round={round_num}; falling back to legacy prompt path: "
                                f"{type(exc).__name__}: {exc}"
                            )
                        else:
                            msg = (
                                f"agentic codegen failed for action_node_id={chosen_leaf} "
                                f"round={round_num}: {type(exc).__name__}: {exc}"
                            )
                            _emit(f"[WARN] {msg}")
                            round_eval = EvalResult(
                                status="codegen_failed",
                                log_excerpt=msg,
                                metrics={"score_name": "codegen", "score": -1.0},
                            )
                            last_eval = round_eval
                            rounds_consumed = max(rounds_consumed, attempt_idx)
                            break
                    else:
                        solution = result.solution
                        current_code, current_raw_code = code_from_solution(self.language, solution)
                        last_solution = solution
                        current_wm_code = _wm_guardrail(_code_for_wm_from_raw(current_raw_code))
                        _emit(
                            f"[LLM] agentic ascendc result round={round_num} "
                            f"prompt_chars={result.prompt_chars} "
                            f"changed_files={','.join(result.changed_paths)} "
                            f"worktree={result.worktree_path}"
                        )
                        _stage(f"evaluate solution (round {round_num})")
                        round_eval = task.run_benchmark(
                            solution=solution,
                            dump_traces=False,
                            round_num=int(round_num),
                        )
                        all_passed = bool(getattr(round_eval, "is_passed", lambda: False)())
                        round_score = float(getattr(round_eval, "score", lambda: -1.0)())
                        last_eval = round_eval
                        if all_passed and round_score > best_score:
                            best_score = float(round_score)
                            best_eval = round_eval
                            best_solution = solution
                        if all_passed:
                            if round_score > cycle_best_score:
                                cycle_best_score = float(round_score)
                                cycle_best_eval = round_eval
                                cycle_best_solution = solution
                                cycle_best_raw = str(current_raw_code or "")
                                cycle_best_wm_code = str(current_wm_code or "")
                                cycle_best_round = int(round_num)
                                no_improve_streak = 0
                            else:
                                no_improve_streak += 1
                        else:
                            no_improve_streak += 1
                        if cycle_best_solution is not None and base_score > 0:
                            if cycle_best_score > base_score:
                                no_improve_over_base_streak = 0
                            else:
                                no_improve_over_base_streak += 1
                        rounds_consumed += 1
                        if no_improve_streak >= stagnation_window or no_improve_over_base_streak >= stagnation_window:
                            break
                        continue
```

If fallback is enabled, the branch logs a warning and falls through into the existing legacy prompt construction below it.

This branch must happen before any call to `get_generate_code_from_action_prompt_from_text()`, `get_debug_generated_code_prompt_from_text()`, `get_improve_generated_code_prompt_from_text()`, or `render_world_model_section()` for agentic AscendC codegen.

- [ ] **Step 5: Run world-model agentic test**

Run:

```bash
pytest tests/kernel_generators/test_llm_clients.py::test_world_model_ascendc_codegen_uses_agentic_runner_before_prompt_construction -v
```

Expected: PASS.

- [ ] **Step 6: Run world-model and Claude regression tests**

Run:

```bash
pytest tests/kernel_generators/test_llm_clients.py tests/kernel_generators/test_world_model_parsing.py tests/kernel_generators/test_claude_agent_sdk_mock.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add k_search/kernel_generators/kernel_generator_world_model.py tests/kernel_generators/test_llm_clients.py
git commit -m "feat(ascendc): route world-model codegen through agentic worktrees"
```

---

### Task 7: Documentation And Configuration Regression

**Files:**
- Modify: `README.md`
- Modify: `tests/test_launcher_docs.py`

- [ ] **Step 1: Add failing README assertion**

Extend `test_readme_documents_claude_agent_sdk_installation` in `tests/test_launcher_docs.py`:

```python
def test_readme_documents_claude_agent_sdk_installation():
    readme = (REPO_ROOT / "README.md").read_text()
    claude_section = readme.split("### Claude Agent SDK Backend", 1)[1]

    assert "uv pip install claude-agent-sdk" in claude_section
    assert "Claude+AscendC uses agentic worktree codegen by default" in claude_section
    assert "KSEARCH_AGENTIC_PROMPT_MAX_CHARS" in claude_section
    assert "KSEARCH_KEEP_AGENTIC_WORKTREES" in claude_section
    assert "Bash is disabled" in claude_section
```

- [ ] **Step 2: Run docs test to verify it fails**

Run:

```bash
pytest tests/test_launcher_docs.py::test_readme_documents_claude_agent_sdk_installation -v
```

Expected: FAIL because README still says the Claude backend is prompt-to-text only.

- [ ] **Step 3: Update README Claude Agent SDK Backend section**

In `README.md`, replace the sentence:

```markdown
The initial Claude backend is prompt-to-text only. K-Search still owns code parsing, benchmark execution, world-model updates, and artifact persistence.
```

with:

```markdown
Claude+AscendC uses agentic worktree codegen by default. K-Search creates an isolated candidate git worktree, gives Claude a compact optimization request, and lets Claude use `Read`, `Grep`, `Glob`, `Edit`, and `Write` inside that worktree. Bash is disabled. K-Search then scans the edited project into a `Solution` and still owns benchmark execution, world-model updates, and artifact persistence.

Useful environment variables:

| Variable | Description | Default |
| --- | --- | --- |
| `KSEARCH_AGENTIC_PROMPT_MAX_CHARS` | Hard budget for compact agentic codegen prompts | `20000` |
| `KSEARCH_KEEP_AGENTIC_WORKTREES` | Set to `1` to preserve temporary candidate worktrees for inspection | unset |
| `KSEARCH_DISABLE_ASCENDC_AGENTIC_CODEGEN` | Set to `1` to force the legacy prompt-to-text AscendC path | unset |
| `KSEARCH_ASCENDC_AGENTIC_FALLBACK` | Set to `legacy` to allow legacy fallback after an agentic codegen failure | unset |
```

- [ ] **Step 4: Run docs test**

Run:

```bash
pytest tests/test_launcher_docs.py::test_readme_documents_claude_agent_sdk_installation -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git add README.md tests/test_launcher_docs.py
git commit -m "docs: document ascendc agentic claude codegen"
```

---

### Task 8: Full Verification And Cleanup

**Files:**
- Review all files modified by Tasks 1-7.

- [ ] **Step 1: Run focused agentic suite**

Run:

```bash
pytest tests/kernel_generators/test_agentic_worktree.py tests/kernel_generators/test_ascendc_agentic_codegen.py tests/kernel_generators/test_claude_agent_sdk_mock.py tests/test_ascendc_task.py -v
```

Expected: PASS.

- [ ] **Step 2: Run generator regression suite**

Run:

```bash
pytest tests/kernel_generators/test_llm_clients.py tests/kernel_generators/test_kernel_generator_codegen.py tests/kernel_generators/test_world_model_parsing.py -v
```

Expected: PASS.

- [ ] **Step 3: Run CLI and docs tests**

Run:

```bash
pytest tests/test_generate_kernels_cli.py tests/test_launcher_docs.py -v
```

Expected: PASS.

- [ ] **Step 4: Run full tests if runtime is acceptable**

Run:

```bash
pytest -v
```

Expected: PASS. If external environment tests are unavailable, record the exact failing command and failure reason in the handoff.

- [ ] **Step 5: Inspect git status**

Run:

```bash
git status --short
```

Expected: only intended code, tests, and README changes are present. The unrelated `docs/deep-research-report.md` remains untracked and unstaged.

- [ ] **Step 6: Commit final verification fixes if any were needed**

If Step 4 revealed small fixes, commit them:

```bash
git add k_search tests README.md
git commit -m "fix(ascendc): stabilize agentic worktree codegen"
```

If no fixes were needed, skip this commit.

---

## Spec Coverage Checklist

- Agentic default for Claude+AscendC: Task 5 and Task 6.
- Temporary git worktree with fallback temp git repo: Task 1.
- Direct file edits and project-dir scanning into `Solution`: Task 2 and Task 4.
- Claude SDK `cwd`, tools, permissions, timeout, transcript: Task 3.
- Compact prompt and prompt budget guard: Task 4.
- No Bash, MCP, subagents, or agent-run build/test/bench: Task 3 and README in Task 7.
- No automatic fallback to long prompt unless env escape hatch is set: Task 5 and Task 7.
- Non-Claude and non-AscendC behavior unchanged: Task 5, Task 6, and Task 8 regression suites.

## Implementation Notes

- The most important routing rule is: agentic Claude+AscendC must branch before legacy prompt construction. If a branch receives a prebuilt prompt containing `<ascendc_project>`, the root cause is not fixed.
- Keep `ClaudeAgentLLMClient.generate()` unchanged except regression tests. It remains the prompt-to-text backend.
- The project editor client uses `ClaudeSDKClient`; the prompt-to-text client uses `query()`.
- `AscendCAgenticCodegenRunner` is an orchestrator only. Do not move git commands, SDK option construction, or AscendC source scanning into it.
- Any temporary worktree preserved by `KSEARCH_KEEP_AGENTIC_WORKTREES=1` is a debugging artifact. It is not a persisted K-Search solution.
