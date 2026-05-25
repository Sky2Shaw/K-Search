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
    ignore = shutil.ignore_patterns(".git", ".ksearch", "__pycache__", "build", "cmake-build-debug", "logs")
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
