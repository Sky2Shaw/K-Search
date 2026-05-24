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
        # Strip trailing empty strings that are artifacts of split("\n") on
        # a diff body ending with a newline; they are not real blank context lines.
        while hunk and hunk[-1] == "":
            hunk.pop()
        yield orig_start, orig_count, hunk


def _locate_hunk_by_context(
    base_lines: List[str],
    hunk_lines: List[str],
    hint_idx: int,
    *,
    min_cursor: int,
) -> int:
    """Find where the hunk's context (and removal) lines actually live in base.

    LLM-generated unified diffs frequently carry wrong `orig_start` line numbers
    while the surrounding context content is correct. We rebuild the expected
    "left side" of the hunk (context + removal lines, in declared order, with
    the leading tag stripped) and search ``base_lines`` for that exact subsequence.

    Returns the 0-indexed line where the hunk should be applied. If the hunk is
    pure-insertion (no context/removal lines), falls back to ``hint_idx``.
    Raises ValueError if nothing matches or only matches lie before ``min_cursor``.
    """
    expected: List[str] = []
    for hl in hunk_lines:
        if hl.startswith("\\ "):
            continue
        if hl == "":
            expected.append("")
            continue
        tag, body = hl[0], hl[1:]
        if tag in (" ", "-"):
            expected.append(body)

    if not expected:
        # Pure insertion. Trust hint but clamp to bounds.
        return max(min_cursor, min(hint_idx, len(base_lines)))

    n = len(expected)
    candidates: List[int] = []
    for start in range(min_cursor, len(base_lines) - n + 1):
        if base_lines[start : start + n] == expected:
            candidates.append(start)
    if not candidates:
        # One last try: fuzzy on whitespace (strip both sides).
        expected_stripped = [s.strip() for s in expected]
        for start in range(min_cursor, len(base_lines) - n + 1):
            window = [s.strip() for s in base_lines[start : start + n]]
            if window == expected_stripped:
                candidates.append(start)
    if not candidates:
        raise ValueError(
            "context mismatch: could not locate hunk context in base file "
            f"(hint line {hint_idx + 1}, {n} context/removal lines)"
        )
    # Prefer the candidate closest to the hint.
    return min(candidates, key=lambda c: abs(c - hint_idx))


def apply_unified_diff(base_text: str, diff_body: str) -> str:
    """Apply a unified-diff body (no file headers) to ``base_text``.

    The diff must NOT include `--- a/...` / `+++ b/...` headers; the patch
    container strips those before calling us. Each hunk is located by
    searching the base file for its declared context (not by trusting the
    `@@ -A,B @@` line number, which LLM-generated diffs frequently get wrong).
    A failed context search raises ValueError so the caller can retry.
    """
    base_text = _normalize_newlines(base_text)
    diff_body = _normalize_newlines(diff_body)
    base_lines = base_text.split("\n")
    out: List[str] = []
    cursor = 0  # 0-indexed pointer into base_lines

    hunks = list(_iter_hunks(diff_body))
    if not hunks:
        raise ValueError("patch body contained no @@ hunks")

    for orig_start, _orig_count, hunk_lines in hunks:
        hint_idx = max(0, orig_start - 1)  # convert 1-indexed to 0-indexed
        target_idx = _locate_hunk_by_context(
            base_lines, hunk_lines, hint_idx, min_cursor=cursor
        )
        # Copy unchanged lines between previous hunk end and this hunk start.
        out.extend(base_lines[cursor:target_idx])
        cursor = target_idx
        for hl in hunk_lines:
            if hl.startswith("\\ "):
                continue
            if hl == "":
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
        try:
            out[path] = apply_unified_diff(out[path], body)
        except ValueError as exc:
            raise ValueError(f"patch for {path!r}: {exc}") from exc

    if not matched:
        raise ValueError("response contained <ascendc_patch> but no <patch> blocks")
    return out
