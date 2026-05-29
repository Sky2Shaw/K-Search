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
