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