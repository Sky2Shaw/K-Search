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
