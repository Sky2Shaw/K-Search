# AscendC Diff/Patch Codegen Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce AscendC optimization-round LLM output from full multi-file rewrites (~10K+ tokens, hits Sonnet 4.6's 64000 output cap) to unified-diff patches against the previous round's code (~1–5K tokens).

**Architecture:**
- A new module `k_search/tasks/ascendc_patch.py` owns the `<ascendc_patch>` container format, a custom unified-diff applier, and the `parse_ascendc_project_patch` entry point.
- `AscendCTask` gains a `codegen_mode` field (`auto` | `full` | `patch`), caches `_last_parsed_files`, and routes `get_optimization_prompt` / `make_solution_from_generated_code` / `code_for_world_model_from_raw` through patch-first then full-fallback parsing.
- The world-model code path picks up the format text via `AscendCTask.get_code_format_text`, so the existing `kernel_generator_world_model.py` plumbing needs no changes for prompts.
- The CUDA-only retry framework in `kernel_generator.py:_generate_code_from_prompt` is widened to also retry AscendC parse failures by calling a new `task.preview_parse_generated_code` hook; both `kernel_generator.py` and `kernel_generator_world_model.py` thread `task` into that call. A 3-failure streak flips `codegen_mode` to `full` inside the task so subsequent rounds emit the full container.
- A CLI flag `--ascendc-codegen-mode` + env var `KSEARCH_ASCENDC_CODEGEN_MODE` thread the mode into `AscendCTask` from `generate_kernels_and_eval.py`.

**Tech Stack:** Python 3.10+, regex/string-based unified-diff applier (no new third-party dependency), pytest.

**Touched files:**
- Create: `k_search/tasks/ascendc_patch.py`
- Create: `tests/test_ascendc_patch.py`
- Modify: `k_search/tasks/ascendc_task.py`
- Modify: `tests/test_ascendc_task.py` (regression assertions for full-mode and codegen_mode plumbing)
- Modify: `k_search/kernel_generators/kernel_generator.py` (extend retry framework to AscendC; thread `task`)
- Modify: `k_search/kernel_generators/kernel_generator_world_model.py` (pass `task` into `_generate_code_from_prompt`)
- Modify: `generate_kernels_and_eval.py` (CLI flag wiring)

**Out of scope (do not touch):**
- CUDA / Tilelang / MLX task code or prompts
- The world-model JSON-patch logic in `world_model_manager.py:_apply_patch` (that's WM-tree decision-tree ops, unrelated to code diff)
- LLM tool-calling / agentic mode
- The retry policy itself for CUDA (`max_parse_retries = 5`) — we widen the predicate but reuse the existing budget

---

## Task Decomposition Notes

The unified-diff applier and `<ascendc_patch>` parser live in a dedicated module because:
1. They are testable in isolation (no need to stand up an `AscendCTask`).
2. Keeping `ascendc_task.py` lean preserves the user's `<400` line diff budget.
3. The applier may be reused by future tasks (e.g. CUDA patch mode) without re-importing AscendC-specific symbols.

The mode-selection state lives on `AscendCTask` (not the generator) so the same CLI flag works for both `kernel_generator.py` and `kernel_generator_world_model.py` without touching either.

---

## Task 1: Unified-diff applier — failing tests

**Files:**
- Create: `tests/test_ascendc_patch.py`
- (Module under test: `k_search/tasks/ascendc_patch.py`, not yet created)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ascendc_patch.py
import pytest

from k_search.tasks.ascendc_patch import (
    ASCENDC_PATCH_FORMAT_TEXT,
    apply_unified_diff,
    parse_ascendc_project_patch,
)


def test_apply_unified_diff_modifies_single_hunk():
    base = "alpha\nbeta\ngamma\ndelta\nepsilon\n"
    diff = (
        "@@ -1,5 +1,5 @@\n"
        " alpha\n"
        " beta\n"
        "-gamma\n"
        "+GAMMA\n"
        " delta\n"
        " epsilon\n"
    )
    assert apply_unified_diff(base, diff) == "alpha\nbeta\nGAMMA\ndelta\nepsilon\n"


def test_apply_unified_diff_supports_pure_insertion_and_pure_deletion():
    base = "line1\nline2\nline3\n"
    insert_diff = (
        "@@ -1,3 +1,4 @@\n"
        " line1\n"
        " line2\n"
        "+inserted\n"
        " line3\n"
    )
    assert apply_unified_diff(base, insert_diff) == "line1\nline2\ninserted\nline3\n"

    delete_diff = (
        "@@ -1,3 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        " line3\n"
    )
    assert apply_unified_diff(base, delete_diff) == "line1\nline3\n"


def test_apply_unified_diff_raises_on_context_mismatch():
    base = "alpha\nbeta\ngamma\n"
    diff = (
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-WRONG_CONTEXT\n"
        "+new\n"
        " gamma\n"
    )
    with pytest.raises(ValueError) as exc_info:
        apply_unified_diff(base, diff)
    assert "context mismatch" in str(exc_info.value).lower()


def test_apply_unified_diff_normalizes_crlf():
    base = "alpha\r\nbeta\r\ngamma\r\n"
    diff = (
        "@@ -1,3 +1,3 @@\r\n"
        " alpha\r\n"
        "-beta\r\n"
        "+BETA\r\n"
        " gamma\r\n"
    )
    assert apply_unified_diff(base, diff) == "alpha\nBETA\ngamma\n"


def test_parse_ascendc_project_patch_applies_hunks_against_baseline():
    base_files = {
        "kernel/foo.h": "int a = 1;\nint b = 2;\nint c = 3;\n",
        "kernel/bar.cpp": "void run() {}\n",
    }
    raw = (
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " int a = 1;\n"
        "-int b = 2;\n"
        "+int b = 22;\n"
        " int c = 3;\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )
    files = parse_ascendc_project_patch(raw, base_files=base_files)
    assert files["kernel/foo.h"] == "int a = 1;\nint b = 22;\nint c = 3;\n"
    assert files["kernel/bar.cpp"] == "void run() {}\n"


def test_parse_ascendc_project_patch_supports_op_replace_for_full_rewrite():
    base_files = {"kernel/foo.h": "old\n"}
    raw = (
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h" op="replace">\n'
        "completely\nnew\ncontent\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )
    files = parse_ascendc_project_patch(raw, base_files=base_files)
    assert files["kernel/foo.h"] == "completely\nnew\ncontent\n"


def test_parse_ascendc_project_patch_creates_new_file_when_baseline_missing():
    base_files = {"kernel/foo.h": "existing\n"}
    raw = (
        "<ascendc_patch>\n"
        '<patch path="kernel/new_file.h" op="replace">\n'
        "brand new\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )
    files = parse_ascendc_project_patch(raw, base_files=base_files)
    assert files["kernel/new_file.h"] == "brand new\n"
    assert files["kernel/foo.h"] == "existing\n"


def test_parse_ascendc_project_patch_raises_value_error_when_container_missing():
    with pytest.raises(ValueError):
        parse_ascendc_project_patch("garbage with no patch tags", base_files={})


def test_ascendc_patch_format_text_documents_unified_diff_and_replace_op():
    assert "<ascendc_patch>" in ASCENDC_PATCH_FORMAT_TEXT
    assert "@@" in ASCENDC_PATCH_FORMAT_TEXT
    assert 'op="replace"' in ASCENDC_PATCH_FORMAT_TEXT
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ascendc_patch.py -v
```

Expected: ALL tests FAIL with `ModuleNotFoundError: No module named 'k_search.tasks.ascendc_patch'`.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_ascendc_patch.py
git commit -m "test(ascendc): add failing tests for unified-diff patch applier"
```

---

## Task 2: Implement `ascendc_patch` module

**Files:**
- Create: `k_search/tasks/ascendc_patch.py`

- [ ] **Step 1: Write the module**

```python
# k_search/tasks/ascendc_patch.py
"""Unified-diff patch container support for AscendC codegen.

This module owns the `<ascendc_patch>` response format the LLM is asked to
emit during optimization rounds, plus a small custom applier that turns the
hunks back into full file contents.

The implementation deliberately avoids `unidiff` / `whatthepatch` so we keep
the dependency surface minimal and have precise control over the ValueError
messages the upstream retry logic relies on.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple


ASCENDC_PATCH_FORMAT_TEXT = """Return only this patch container, with no markdown or explanations:
<ascendc_patch>
<patch path="kernel/example.h">
@@ -120,7 +120,7 @@
 unchanged context line
 another context line
-old line to remove
+new line to add
 closing context line
 third context line
</patch>
<patch path="kernel/another_file.cpp" op="replace">
// Use op="replace" only when you are rewriting >70% of a small file.
// The body of a replace patch is the FULL new file content (no diff syntax).
</patch>
</ascendc_patch>

Rules:
- Use unified diff syntax (default). Each hunk must include >=3 context lines around each change so the applier can locate it.
- `path` must be the file's path relative to the kernel project root (the same paths you saw in the current implementation).
- Reference only files that exist in the current implementation, or add a new file via op="replace".
- Do NOT include `a/` / `b/` prefixes, file mode lines, or `diff --git` headers.
- Do NOT escape characters inside the patch body."""


_PATCH_BLOCK_RE = re.compile(
    r'<patch\s+path="([^"]+)"\s*(?:op="([^"]+)"\s*)?>\s*(.*?)\s*</patch>',
    re.DOTALL,
)
_HUNK_HEADER_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@.*$")


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _split_lines_keep_trailing(text: str) -> List[str]:
    """Split into lines preserving information about a trailing newline.

    A trailing newline produces an empty final element so that re-joining with
    "\\n" round-trips. Files with no trailing newline produce no extra element.
    """
    if text == "":
        return [""]
    parts = text.split("\n")
    return parts


def _join_lines(lines: Iterable[str]) -> str:
    return "\n".join(lines)


def _iter_hunks(diff_body: str) -> Iterable[Tuple[int, int, List[str]]]:
    """Yield (orig_start, orig_count, hunk_lines) for each `@@ ... @@` block."""
    lines = diff_body.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        m = _HUNK_HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue
        orig_start = int(m.group(1))
        orig_count = int(m.group(2)) if m.group(2) is not None else 1
        i += 1
        hunk: List[str] = []
        while i < n and not _HUNK_HEADER_RE.match(lines[i]):
            hunk.append(lines[i])
            i += 1
        yield orig_start, orig_count, hunk


def apply_unified_diff(base_text: str, diff_body: str) -> str:
    """Apply a unified-diff body (no file headers) to ``base_text``.

    The diff must NOT include `--- a/...` / `+++ b/...` headers; the patch
    container strips those before calling us. Each hunk is located by its
    declared `orig_start` line, with the context lines verified against the
    base. A context mismatch raises ValueError so the caller can retry.
    """
    base_text = _normalize_newlines(base_text)
    diff_body = _normalize_newlines(diff_body)
    base_lines = base_text.split("\n")
    # Track index in base_lines as we walk hunks; hunks are 1-indexed in the header.
    out: List[str] = []
    cursor = 0  # 0-indexed pointer into base_lines

    hunks = list(_iter_hunks(diff_body))
    if not hunks:
        raise ValueError("patch body contained no @@ hunks")

    for orig_start, _orig_count, hunk_lines in hunks:
        target_idx = orig_start - 1  # convert 1-indexed to 0-indexed
        if target_idx < cursor:
            raise ValueError(
                f"hunks out of order: hunk starts at line {orig_start} but "
                f"cursor already at line {cursor + 1}"
            )
        if target_idx > len(base_lines):
            raise ValueError(
                f"hunk start line {orig_start} exceeds base file length "
                f"{len(base_lines)}"
            )
        # Copy unchanged lines between previous hunk and this one.
        out.extend(base_lines[cursor:target_idx])
        cursor = target_idx
        for hl in hunk_lines:
            if hl.startswith("\\ "):
                # "\\ No newline at end of file" marker — tolerate.
                continue
            if hl == "":
                # Blank physical line in diff body — treat as context blank line.
                tag, body = " ", ""
            else:
                tag, body = hl[0], hl[1:]
            if tag == " ":
                if cursor >= len(base_lines) or base_lines[cursor] != body:
                    actual = base_lines[cursor] if cursor < len(base_lines) else "<EOF>"
                    raise ValueError(
                        f"context mismatch at base line {cursor + 1}: "
                        f"expected {body!r}, got {actual!r}"
                    )
                out.append(body)
                cursor += 1
            elif tag == "-":
                if cursor >= len(base_lines) or base_lines[cursor] != body:
                    actual = base_lines[cursor] if cursor < len(base_lines) else "<EOF>"
                    raise ValueError(
                        f"context mismatch at base line {cursor + 1}: "
                        f"expected to remove {body!r}, got {actual!r}"
                    )
                cursor += 1
            elif tag == "+":
                out.append(body)
            else:
                raise ValueError(f"unknown diff line prefix: {hl!r}")

    # Append remainder of base after the last hunk.
    out.extend(base_lines[cursor:])
    return _join_lines(out)


def parse_ascendc_project_patch(
    raw: str,
    *,
    base_files: Dict[str, str],
) -> Dict[str, str]:
    """Parse an `<ascendc_patch>` container and return the post-patch file map.

    The returned dict is a *copy* of ``base_files`` with each named patch
    applied. Files not mentioned in the patch are passed through unchanged.

    Raises ValueError on:
      - No ``<patch path="...">`` blocks found
      - Patch body fails to apply (context mismatch, malformed hunk, etc.)

    op="replace" replaces the file body verbatim (and creates new files).
    Default op (unified diff) applies hunks to the matching base file.
    """
    text = _normalize_newlines(str(raw or ""))
    if "<ascendc_patch>" not in text and "<patch " not in text:
        raise ValueError("response did not contain an <ascendc_patch> container")

    out: Dict[str, str] = {p: c for p, c in base_files.items()}
    matched = False
    for match in _PATCH_BLOCK_RE.finditer(text):
        matched = True
        path = match.group(1).strip()
        op = (match.group(2) or "").strip().lower()
        body = match.group(3) or ""
        if op == "replace":
            replaced = body if body.endswith("\n") else body + "\n"
            out[path] = replaced
            continue
        if op and op != "diff":
            raise ValueError(f"unsupported patch op={op!r} for path={path!r}")
        if path not in out:
            raise ValueError(
                f"patch references unknown file {path!r}; use op=\"replace\" "
                "to create a new file"
            )
        out[path] = apply_unified_diff(out[path], body)

    if not matched:
        raise ValueError("response contained <ascendc_patch> but no <patch> blocks")
    return out
```

- [ ] **Step 2: Run patch-module tests**

```bash
pytest tests/test_ascendc_patch.py -v
```

Expected: ALL 9 tests PASS.

- [ ] **Step 3: Run the full ascendc test suite to catch unrelated regressions**

```bash
pytest tests/test_ascendc_task.py tests/test_ascendc_patch.py -v
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add k_search/tasks/ascendc_patch.py
git commit -m "feat(ascendc): add unified-diff patch container parser"
```

---

## Task 3: AscendCTask — codegen_mode field + baseline caching

**Files:**
- Modify: `k_search/tasks/ascendc_task.py`

- [ ] **Step 1: Add the failing regression tests first**

Append to `tests/test_ascendc_task.py`:

```python
import os

from k_search.tasks.ascendc_patch import ASCENDC_PATCH_FORMAT_TEXT


def test_ascendc_task_defaults_to_auto_codegen_mode():
    task = AscendCTask(task_path=None, definition_name="x")
    assert task.codegen_mode == "auto"


def test_ascendc_task_reads_codegen_mode_from_env(monkeypatch):
    monkeypatch.setenv("KSEARCH_ASCENDC_CODEGEN_MODE", "full")
    task = AscendCTask(task_path=None, definition_name="x")
    assert task.codegen_mode == "full"


def test_ascendc_task_explicit_codegen_mode_overrides_env(monkeypatch):
    monkeypatch.setenv("KSEARCH_ASCENDC_CODEGEN_MODE", "full")
    task = AscendCTask(task_path=None, definition_name="x", codegen_mode="patch")
    assert task.codegen_mode == "patch"


def test_ascendc_task_invalid_codegen_mode_raises():
    with pytest.raises(ValueError):
        AscendCTask(task_path=None, definition_name="x", codegen_mode="bogus")
```

Add at the top of `tests/test_ascendc_task.py`:
```python
import pytest
```
(skip if already present).

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ascendc_task.py -v -k "codegen_mode"
```

Expected: FAIL — `AscendCTask.__init__() got an unexpected keyword argument 'codegen_mode'`.

- [ ] **Step 3: Modify `AscendCTask.__init__` to accept and validate `codegen_mode`**

In `k_search/tasks/ascendc_task.py`, add this constant near the existing `ASCENDC_CODE_FORMAT_TEXT`:

```python
import os  # add to existing imports if missing

VALID_CODEGEN_MODES = ("auto", "full", "patch")
```

Then update `__init__`:

```python
def __init__(
    self,
    *,
    task_path: str | Path | None,
    definition_name: str | None = None,
    build_cmd: str | None = None,
    test_cmd: str | None = None,
    bench_cmd: str | None = None,
    reference_latency_ms: float | None = None,
    timeout_seconds: int = 600,
    artifacts_dir: str | None = None,
    codegen_mode: str | None = None,
) -> None:
    self.task_path = Path(task_path).expanduser().resolve() if task_path else None
    self._name = str(definition_name or (self.task_path.stem if self.task_path else "ascendc_task")).strip()
    self.build_cmd = str(build_cmd or "").strip()
    self.test_cmd = str(test_cmd or "").strip()
    self.bench_cmd = str(bench_cmd or "").strip()
    self.reference_latency_ms = float(reference_latency_ms) if reference_latency_ms else None
    self.timeout_seconds = int(timeout_seconds or 600)
    self.artifacts_dir = artifacts_dir
    self._last_eval: EvalResult | None = None

    mode = codegen_mode or os.environ.get("KSEARCH_ASCENDC_CODEGEN_MODE") or "auto"
    mode = str(mode).strip().lower()
    if mode not in VALID_CODEGEN_MODES:
        raise ValueError(
            f"invalid codegen_mode={mode!r}; expected one of {VALID_CODEGEN_MODES}"
        )
    self.codegen_mode = mode
    self._last_parsed_files: dict[str, str] | None = None
    self._patch_failure_streak = 0
    self._max_patch_failures = 3
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ascendc_task.py -v -k "codegen_mode"
```

Expected: 4 tests PASS.

- [ ] **Step 5: Run full ascendc tests for regression**

```bash
pytest tests/test_ascendc_task.py tests/test_ascendc_patch.py -v
```

Expected: All tests PASS (no regression on existing 6 tests).

- [ ] **Step 6: Commit**

```bash
git add k_search/tasks/ascendc_task.py tests/test_ascendc_task.py
git commit -m "feat(ascendc): add codegen_mode field with env + CLI support"
```

---

## Task 4: Switch optimization prompt + code-format hook by mode

**Files:**
- Modify: `k_search/tasks/ascendc_task.py`

- [ ] **Step 1: Add a failing test for `get_optimization_prompt` mode switching**

Append to `tests/test_ascendc_task.py`:

```python
def test_get_optimization_prompt_uses_patch_format_when_baseline_available(tmp_path):
    (tmp_path / "spec.md").write_text("Vector add.", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="patch")
    prompt = task.get_optimization_prompt(
        language="ascendc",
        target_gpu="ascend_910b",
        trace_logs="ok",
        current_code='<ascendc_project><file path="kernel.cpp">int a=1;</file></ascendc_project>',
    )
    assert "<ascendc_patch>" in prompt
    assert "@@" in prompt
    # We must NOT instruct the model to emit the full container in patch mode.
    assert "Return only the full AscendC multi-file container" not in prompt


def test_get_optimization_prompt_falls_back_to_full_when_current_code_empty(tmp_path):
    (tmp_path / "spec.md").write_text("Vector add.", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="patch")
    prompt = task.get_optimization_prompt(
        language="ascendc",
        target_gpu="ascend_910b",
        trace_logs="ok",
        current_code="",
    )
    assert "<ascendc_project>" in prompt
    assert "<ascendc_patch>" not in prompt


def test_get_optimization_prompt_in_full_mode_never_emits_patch_format(tmp_path):
    (tmp_path / "spec.md").write_text("Vector add.", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="full")
    prompt = task.get_optimization_prompt(
        language="ascendc",
        target_gpu="ascend_910b",
        trace_logs="ok",
        current_code="<ascendc_project></ascendc_project>",
    )
    assert "<ascendc_patch>" not in prompt
    assert "<ascendc_project>" in prompt


def test_get_code_format_text_in_patch_mode_returns_patch_format(tmp_path):
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="patch")
    fmt = task.get_code_format_text(language="ascendc", target_gpu="ascend_910b")
    assert "<ascendc_patch>" in fmt
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ascendc_task.py -v -k "optimization_prompt or code_format_text"
```

Expected: FAIL — current `get_optimization_prompt` always emits the full container; `get_code_format_text` doesn't exist yet.

- [ ] **Step 3: Update `get_optimization_prompt` and add `get_code_format_text`**

In `k_search/tasks/ascendc_task.py`, change the import to include the new module:

```python
from k_search.tasks.ascendc_patch import (
    ASCENDC_PATCH_FORMAT_TEXT,
    parse_ascendc_project_patch,
)
```

Add a helper inside `AscendCTask`:

```python
def _resolve_codegen_mode(self, *, has_baseline: bool) -> str:
    """Resolve effective mode for one call.

    - "full" -> always full
    - "patch" -> patch when there is a baseline, else full (cold start safety)
    - "auto" -> patch when there is a baseline, else full
    """
    if self.codegen_mode == "full":
        return "full"
    return "patch" if has_baseline else "full"

def _format_text_for_mode(self, mode: str) -> str:
    return ASCENDC_PATCH_FORMAT_TEXT if mode == "patch" else ASCENDC_CODE_FORMAT_TEXT
```

Update `get_optimization_prompt`:

```python
def get_optimization_prompt(
    self,
    *,
    language: str,
    target_gpu: str,
    trace_logs: str,
    current_code: str,
    current_best: str | None = None,
    previous_round_summary: str | None = None,
) -> str:
    extra = []
    if previous_round_summary:
        extra.append("Previous Round Summary:\n" + previous_round_summary)
    if current_best:
        extra.append("Current Best Solution So Far:\n" + current_best)
    extra_text = "\n\n".join(extra)

    has_baseline = bool(str(current_code or "").strip())
    mode = self._resolve_codegen_mode(has_baseline=has_baseline)
    format_text = self._format_text_for_mode(mode)

    if mode == "patch":
        response_rule = "- Return only the <ascendc_patch> container (unified diff)."
    else:
        response_rule = "- Return only the full AscendC multi-file container."

    return f"""You are optimizing an AscendC multi-file operator project for {target_gpu}.

Original Specification:
{self.get_definition_text(language=language)}

Current Implementation Status:
{trace_logs or "(no logs)"}

Current Implementation:
{current_code}

{extra_text}

Rules:
- If compilation or correctness failed, fix that first.
- If it passed, improve measured latency while preserving semantics.
- Keep changes small enough for one K-Search round.
{response_rule}

Response format:
{format_text}

Generate the corrected and optimized implementation:"""
```

Add a new method on `AscendCTask`:

```python
def get_code_format_text(self, *, language: str, target_gpu: str) -> str:
    """Hook used by the world-model generator to embed a code-format reminder."""
    # When the world-model generator builds prompts it always passes a
    # `base_code` argument; the prompt builders use this format text purely
    # as a reminder. Returning the patch format is safe because cold-start
    # prompts (no base) will be parsed leniently in code_for_world_model_from_raw.
    return self._format_text_for_mode(
        self._resolve_codegen_mode(has_baseline=True)
    )
```

Update the `get_per_task_requirement_text` to remain backward compatible (return the full-format text when world-model isn't involved):

```python
def get_per_task_requirement_text(self, *, language: str, target_gpu: str, phase: str) -> str:
    if phase == "optimize" and self.codegen_mode != "full":
        return ASCENDC_PATCH_FORMAT_TEXT
    return ASCENDC_CODE_FORMAT_TEXT
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ascendc_task.py -v -k "optimization_prompt or code_format_text"
```

Expected: 4 tests PASS.

- [ ] **Step 5: Run all ascendc tests**

```bash
pytest tests/test_ascendc_task.py tests/test_ascendc_patch.py -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add k_search/tasks/ascendc_task.py tests/test_ascendc_task.py
git commit -m "feat(ascendc): switch optimization prompt to patch format when baseline present"
```

---

## Task 5: Patch-first parsing in `make_solution_from_generated_code`

**Files:**
- Modify: `k_search/tasks/ascendc_task.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_ascendc_task.py`:

```python
def test_make_solution_accepts_patch_response_against_disk_baseline(tmp_path):
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "foo.h").write_text("int a = 1;\nint b = 2;\nint c = 3;\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="patch")

    raw_patch = (
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " int a = 1;\n"
        "-int b = 2;\n"
        "+int b = 22;\n"
        " int c = 3;\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )
    solution = task.make_solution_from_generated_code(
        cleaned_code=raw_patch,
        raw_code=raw_patch,
        round_num=2,
        model_name="m",
        target_gpu="ascend_910b",
        language="ascendc",
    )
    foo = next(s for s in solution.sources if s.path == "kernel/foo.h")
    assert "int b = 22;" in foo.content


def test_make_solution_falls_back_to_full_container_when_response_is_not_a_patch(tmp_path):
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "foo.h").write_text("int a = 1;\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="auto")

    raw_full = format_ascendc_project_files({"kernel/foo.h": "int a = 999;\n"})
    solution = task.make_solution_from_generated_code(
        cleaned_code=raw_full,
        raw_code=raw_full,
        round_num=2,
        model_name="m",
        target_gpu="ascend_910b",
        language="ascendc",
    )
    foo = next(s for s in solution.sources if s.path == "kernel/foo.h")
    assert foo.content == "int a = 999;"


def test_make_solution_records_patch_failure_streak_and_auto_falls_back(tmp_path):
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="auto")

    bogus_patch = (
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-WRONG_CONTEXT\n"
        "+gamma\n"
        " beta\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )
    # Three failures should trigger fallback to full mode.
    for _ in range(3):
        with pytest.raises(ValueError):
            task.make_solution_from_generated_code(
                cleaned_code=bogus_patch,
                raw_code=bogus_patch,
                round_num=2,
                model_name="m",
                target_gpu="ascend_910b",
                language="ascendc",
            )
    assert task.codegen_mode == "full"
    assert task._patch_failure_streak >= 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ascendc_task.py -v -k "make_solution"
```

Expected: FAIL — patch responses currently get rejected by `parse_ascendc_project_files`.

- [ ] **Step 3: Update `make_solution_from_generated_code`**

Add a helper to load baseline files lazily:

```python
def _load_baseline_files_from_disk(self) -> dict[str, str]:
    if self.task_path is None or not self.task_path.is_dir():
        return {}
    sources = _collect_project_sources(self.task_path)
    return {s.path: s.content for s in sources}

def _resolve_patch_base_files(self) -> dict[str, str]:
    if self._last_parsed_files is not None:
        return dict(self._last_parsed_files)
    return self._load_baseline_files_from_disk()

def _parse_codegen_response(self, raw: Any) -> dict[str, str]:
    """Try patch first (when allowed), fall back to full container.

    Idempotent on identical input: if we already parsed this exact `raw`
    payload (typical when the retry framework calls preview_parse and then
    make_solution_from_generated_code with the same string), return the cached
    result instead of re-applying the patch (which would now mismatch because
    `_last_parsed_files` has already advanced).
    """
    text = str(raw or "")
    if self._last_parsed_raw == text and self._last_parsed_files is not None:
        return dict(self._last_parsed_files)

    looks_like_patch = "<ascendc_patch>" in text or "<patch " in text
    if self.codegen_mode != "full" and looks_like_patch:
        try:
            base_files = self._resolve_patch_base_files()
            files = parse_ascendc_project_patch(text, base_files=base_files)
            self._patch_failure_streak = 0
            self._last_parsed_files = dict(files)
            self._last_parsed_raw = text
            return files
        except ValueError as exc:
            self._patch_failure_streak += 1
            if (
                self.codegen_mode == "auto"
                and self._patch_failure_streak >= self._max_patch_failures
            ):
                print(
                    f"[WARN] ascendc patch parse failed {self._patch_failure_streak}"
                    f" times in a row; falling back to full codegen mode."
                )
                self.codegen_mode = "full"
            raise

    files = parse_ascendc_project_files(text)
    self._patch_failure_streak = 0
    self._last_parsed_files = dict(files)
    self._last_parsed_raw = text
    return files
```

Also extend `__init__` (revisit Task 3's snippet to add the new cache field):

```python
self._last_parsed_files: dict[str, str] | None = None
self._last_parsed_raw: str | None = None
self._patch_failure_streak = 0
self._max_patch_failures = 3
```

Update `make_solution_from_generated_code`:

```python
def make_solution_from_generated_code(
    self,
    *,
    cleaned_code: Any,
    raw_code: Any,
    round_num: int,
    model_name: str,
    target_gpu: str,
    language: str,
) -> Solution:
    files = self._parse_codegen_response(raw_code if raw_code is not None else cleaned_code)
    sources = [SourceFile(path=path, content=content) for path, content in sorted(files.items())]
    return Solution(
        name=f"{model_name}_{self.name}_ascendc_optimized_r{int(round_num)}",
        definition=self.name,
        author=str(model_name),
        spec=BuildSpec(
            language=SupportedLanguages.ASCENDC,
            target_hardware=[str(target_gpu or "ascend")],
            entry_point=_default_entry_point(sources),
        ),
        sources=sources,
        description=f"{model_name} optimized AscendC project for {self.name} (round {int(round_num)})",
    )
```

Add the preview hook used by the retry framework in Task 6:

```python
def preview_parse_generated_code(self, *, raw_code: str) -> None:
    """Validate that `raw_code` will parse successfully.

    Called by `KernelGenerator._generate_code_from_prompt` immediately after
    `_clean_generated_code` returns. Raises ValueError on bad patch / bad full
    container so the retry framework can re-prompt the LLM. State updates
    (last-parsed-files cache, failure streak) happen inside `_parse_codegen_response`.
    Safe to call multiple times with the same raw_code: the second call will
    succeed-trivially because `_last_parsed_files` was updated by the first.
    """
    self._parse_codegen_response(raw_code)
```

Also update `code_for_world_model_from_raw` to handle patch responses gracefully:

```python
def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str:
    if isinstance(raw, dict):
        return format_ascendc_project_files({str(k): str(v or "") for k, v in raw.items()})
    text = str(raw or "")
    if "<ascendc_patch>" in text or "<patch " in text:
        try:
            base_files = self._resolve_patch_base_files()
            files = parse_ascendc_project_patch(text, base_files=base_files)
            return format_ascendc_project_files(files)
        except ValueError:
            # Bad patch; let the WM see the raw response (truncated upstream).
            return text
    return text
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ascendc_task.py -v -k "make_solution"
```

Expected: 3 tests PASS.

- [ ] **Step 5: Run full ascendc tests**

```bash
pytest tests/test_ascendc_task.py tests/test_ascendc_patch.py -v
```

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add k_search/tasks/ascendc_task.py tests/test_ascendc_task.py
git commit -m "feat(ascendc): parse patch responses against cached baseline with auto-fallback"
```

---

## Task 6: Extend retry framework to AscendC

The patch parser raises `ValueError` on context-mismatched hunks. The retry loop in `_generate_code_from_prompt` (at `kernel_generator.py:148-177`) already does the right thing for CUDA — re-prompt up to 5 times on parse failure — but it's gated on `is_cuda`. We widen the predicate and call the new `preview_parse_generated_code` task hook.

**Files:**
- Modify: `k_search/kernel_generators/kernel_generator.py`
- Modify: `k_search/kernel_generators/kernel_generator_world_model.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_ascendc_task.py`:

```python
def test_kernel_generator_retries_ascendc_on_bad_patch_then_succeeds(tmp_path, monkeypatch):
    """A flaky LLM that returns a bad patch on attempt 1 and a good one on attempt 2
    should produce a parseable solution thanks to the widened retry framework.
    """
    from k_search.kernel_generators.kernel_generator import KernelGenerator

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="x", codegen_mode="patch")

    bad_patch = (
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-WRONG\n"
        "+BETA\n"
        " gamma\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )
    good_patch = (
        "<ascendc_patch>\n"
        '<patch path="kernel/foo.h">\n'
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-beta\n"
        "+BETA\n"
        " gamma\n"
        "</patch>\n"
        "</ascendc_patch>\n"
    )

    class FlakyClient:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            return bad_patch if self.calls == 1 else good_patch

    gen = KernelGenerator(
        model_name="fake",
        language="ascendc",
        target_gpu="ascend_910b",
        llm_client=FlakyClient(),
    )
    result = gen._generate_code_from_prompt("ignored prompt", task=task)
    assert "BETA" in result["raw"]
    # Calling make_solution_from_generated_code with the same raw must NOT re-fail
    # (idempotency cache).
    sol = task.make_solution_from_generated_code(
        cleaned_code=result["cleaned"],
        raw_code=result["raw"],
        round_num=2,
        model_name="fake",
        target_gpu="ascend_910b",
        language="ascendc",
    )
    foo = next(s for s in sol.sources if s.path == "kernel/foo.h")
    assert "BETA" in foo.content
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
pytest tests/test_ascendc_task.py -v -k "retries_ascendc"
```

Expected: FAIL — either `_generate_code_from_prompt` doesn't accept `task=`, or the retry loop is still CUDA-only and propagates the bad-patch ValueError.

- [ ] **Step 3: Widen the retry predicate and add the preview hook call**

In `k_search/kernel_generators/kernel_generator.py:147-181`, replace:

```python
def _generate_code_from_prompt(self, prompt: str):
    # If we fail to parse CUDA XML (missing kernel.h/kernel.cu/main.cpp), retry generation.
    max_parse_retries = 5
    is_cuda = (self.language or "").lower() == "cuda"

    last_err: Exception | None = None
    for attempt in range(1, (max_parse_retries if is_cuda else 1) + 1):
        try:
            effective_prompt = prompt
            generated_code = str(self.llm_client.generate(effective_prompt) or "").strip()

            cleaned_code = self._clean_generated_code(generated_code)

            if is_cuda:
                # cleaned_code should be a dict of required files for CUDA.
                if not isinstance(cleaned_code, dict):
                    raise ValueError("CUDA generation did not return a parsed file dict")
                required = ("kernel.h", "kernel.cu", "main.cpp")
                missing = [k for k in required if (k not in cleaned_code) or (not str(cleaned_code.get(k, "")).strip())]
                if missing:
                    raise ValueError(f"missing required XML files: {missing}")

            return {"raw": generated_code, "cleaned": cleaned_code}

        except Exception as e:
            last_err = e
            if is_cuda and attempt < max_parse_retries:
                print(f"[WARN] CUDA XML parse failed ({e}); retrying generation ({attempt}/{max_parse_retries})...")
                continue
            print(f"Error while generating code: {e}")
            raise

    # Unreachable, but keeps type-checkers happy.
    assert last_err is not None
    raise last_err
```

with:

```python
def _generate_code_from_prompt(self, prompt: str, task: Optional[Task] = None):
    # Retry parse failures up to 5 times for languages with structured multi-file responses.
    max_parse_retries = 5
    lang = (self.language or "").lower()
    is_cuda = lang == "cuda"
    is_ascendc = lang == "ascendc"
    should_retry = is_cuda or is_ascendc

    last_err: Exception | None = None
    for attempt in range(1, (max_parse_retries if should_retry else 1) + 1):
        try:
            effective_prompt = prompt
            generated_code = str(self.llm_client.generate(effective_prompt) or "").strip()

            cleaned_code = self._clean_generated_code(generated_code)

            if is_cuda:
                if not isinstance(cleaned_code, dict):
                    raise ValueError("CUDA generation did not return a parsed file dict")
                required = ("kernel.h", "kernel.cu", "main.cpp")
                missing = [k for k in required if (k not in cleaned_code) or (not str(cleaned_code.get(k, "")).strip())]
                if missing:
                    raise ValueError(f"missing required XML files: {missing}")
            elif is_ascendc and task is not None:
                preview = getattr(task, "preview_parse_generated_code", None)
                if callable(preview):
                    preview(raw_code=generated_code)

            return {"raw": generated_code, "cleaned": cleaned_code}

        except Exception as e:
            last_err = e
            if should_retry and attempt < max_parse_retries:
                tag = "CUDA XML" if is_cuda else "AscendC"
                print(f"[WARN] {tag} parse failed ({e}); retrying generation ({attempt}/{max_parse_retries})...")
                continue
            print(f"Error while generating code: {e}")
            raise

    assert last_err is not None
    raise last_err
```

- [ ] **Step 4: Thread `task` into both call sites in `kernel_generator.py`**

At `kernel_generator.py:331` (initial generation):

```python
code_result = self._generate_code_from_prompt(prompt, task=task)
```

At `kernel_generator.py:554` (optimization round):

```python
code_result = self._generate_code_from_prompt(opt_prompt, task=task)
```

- [ ] **Step 5: Thread `task` into the world-model generator call site**

At `kernel_generator_world_model.py:674`:

```python
code_result = self._generate_code_from_prompt(prompt, task=task)
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
pytest tests/test_ascendc_task.py -v -k "retries_ascendc"
```

Expected: PASS — flaky LLM gets retried, final solution parses cleanly.

- [ ] **Step 7: Run the full pytest suite for the modified surface**

```bash
pytest tests/test_ascendc_task.py tests/test_ascendc_patch.py -v
```

Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add k_search/kernel_generators/kernel_generator.py k_search/kernel_generators/kernel_generator_world_model.py tests/test_ascendc_task.py
git commit -m "feat(generator): widen parse-retry loop to AscendC via task preview hook"
```

---

## Task 7: CLI wiring — `--ascendc-codegen-mode`

**Files:**
- Modify: `generate_kernels_and_eval.py`

- [ ] **Step 1: Read current ascendc arg block to understand its style**

Run `grep -n -A1 -B1 "ascendc-reference-latency-ms" generate_kernels_and_eval.py` to confirm placement context.

- [ ] **Step 2: Add the CLI flag and pipe it into the task constructor**

In `generate_kernels_and_eval.py` near line 506 (where the other `--ascendc-*` args live), add:

```python
parser.add_argument(
    "--ascendc-codegen-mode",
    choices=["auto", "full", "patch"],
    default=None,
    help=(
        "AscendC codegen response format. 'auto' (default) emits patches when a baseline "
        "is available and full containers otherwise. 'full' forces the legacy full "
        "multi-file container every round (regression-safe). 'patch' forces unified-diff "
        "responses. Falls back to env var KSEARCH_ASCENDC_CODEGEN_MODE."
    ),
)
```

In the AscendC task construction block near line 376, pass it through:

```python
task = AscendCTask(
    task_path=str(args.task_path),
    definition_name=str(args.definition or Path(args.task_path).stem),
    build_cmd=getattr(args, "ascendc_build_cmd", None),
    test_cmd=getattr(args, "ascendc_test_cmd", None),
    bench_cmd=getattr(args, "ascendc_bench_cmd", None),
    timeout_seconds=int(getattr(args, "ascendc_timeout_seconds", 600) or 600),
    reference_latency_ms=getattr(args, "ascendc_reference_latency_ms", None),
    artifacts_dir=str(getattr(args, "artifacts_dir", "") or "") or None,
    codegen_mode=getattr(args, "ascendc_codegen_mode", None),
)
```

(Confirm via `Read` whether `artifacts_dir` was already being passed; preserve current behavior — only add `codegen_mode=`.)

- [ ] **Step 3: Smoke-check CLI parse**

```bash
python generate_kernels_and_eval.py --help 2>&1 | grep -A2 "ascendc-codegen-mode"
```

Expected: shows the new flag and its `auto/full/patch` choices.

- [ ] **Step 4: Confirm existing CLI tests still pass**

```bash
pytest tests/test_generate_kernels_cli.py -v
```

Expected: All tests PASS (no new test required here — the unit tests in Task 3 already verify env+constructor wiring).

- [ ] **Step 5: Commit**

```bash
git add generate_kernels_and_eval.py
git commit -m "feat(cli): add --ascendc-codegen-mode flag for diff vs full codegen"
```

---

## Task 8: Regression sweep + smoke run

**Files:**
- (No file changes; verification only.)

- [ ] **Step 1: Verify the diff stays under the 400-line budget**

```bash
git diff --stat main...HEAD
```

Expected: `total insertions + deletions < 400` excluding the test files. If over budget, look for incidental refactors and remove them.

- [ ] **Step 2: Run the full pytest suite for the modified surface**

```bash
pytest tests/test_ascendc_task.py tests/test_ascendc_patch.py tests/test_generate_kernels_cli.py -v
```

Expected: All tests PASS.

- [ ] **Step 3: Sanity-check full-mode regression by env override**

```bash
KSEARCH_ASCENDC_CODEGEN_MODE=full python -c "
from k_search.tasks.ascendc_task import AscendCTask
t = AscendCTask(task_path=None, definition_name='x')
print('mode:', t.codegen_mode)
p = t.get_optimization_prompt(
    language='ascendc', target_gpu='ascend_910b',
    trace_logs='', current_code='<ascendc_project></ascendc_project>',
)
assert '<ascendc_patch>' not in p, 'full mode should not emit patch format'
assert '<ascendc_project>' in p
print('full mode regression OK')
"
```

Expected: prints `mode: full` and `full mode regression OK`.

- [ ] **Step 4: End-to-end smoke run (requires Ascend NPU + env)**

This step verifies the acceptance criterion that Round 1 codegen single response < 30K tokens.

```bash
bash /mnt/workspace/K-Search/scripts/ascendc_mqa_wm.sh 2>&1 | tee /tmp/ksearch_smoke.log
```

Expected: Round 1 evaluation completes without `Claude's response exceeded the 64000 output token maximum`. Grep `/tmp/ksearch_smoke.log` for `<ascendc_patch>` to confirm the LLM was prompted in patch mode and responded with the diff container.

If the Ascend env is not available, document in the PR description that this step was skipped and call it out for the reviewer to run locally.

- [ ] **Step 5: Final commit (only if Step 4 surfaced doc fixes)**

If the smoke run reveals a copy-paste error in prompt text or a missing import path, fix it and commit:

```bash
git add <file>
git commit -m "fix(ascendc): <specific fix from smoke run>"
```

---

## Self-Review Checklist

- [ ] Every spec edge case has a matching task or test:
  - Round 1 attempt 1 with on-disk baseline → covered by `test_make_solution_accepts_patch_response_against_disk_baseline`.
  - Truly cold start (no baseline anywhere) → `_resolve_codegen_mode(has_baseline=False)` forces `full`.
  - Patch references new file → `test_parse_ascendc_project_patch_creates_new_file_when_baseline_missing`.
  - Empty file after patch → `apply_unified_diff` returns empty string; no special handling needed (compile error will surface naturally).
  - CRLF normalization → `test_apply_unified_diff_normalizes_crlf`.
- [ ] `KSEARCH_ASCENDC_CODEGEN_MODE=full` → behavior identical to pre-change (verified via Task 8 Step 3).
- [ ] No changes to CUDA/Tilelang/MLX paths.
- [ ] No new top-level dependency.
- [ ] Production diff (non-test) < 400 lines; test additions tracked separately and may exceed if needed for coverage.
- [ ] Retry framework extension is the **only** change to the generators; no new subclasses, no agentic logic.
