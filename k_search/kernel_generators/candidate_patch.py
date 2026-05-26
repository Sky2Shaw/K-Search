from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidatePatch:
    candidate_id: str
    parent_candidate_id: str | None
    base_ref: str
    project_rel_path: str
    changed_paths: list[str]
    diff_text: str
    prompt_path: str
    transcript_path: str
    eval_path: str
    manifest_path: str
    snapshot_id: str | None
    snapshot_manifest_path: str | None
    round_num: int
    action_node_id: str | None
    model_name: str
    eval_result: dict[str, Any]
