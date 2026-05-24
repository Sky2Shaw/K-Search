"""AscendC task adapter.

This backend is intentionally command-driven. K-Search owns generation, solution
packing, scoring, and logs; the project-specific Ascend/CANN environment owns
the actual build, correctness, and benchmark commands.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from k_search.tasks.task_base import (
    BuildSpec,
    EvalResult,
    Solution,
    SourceFile,
    SupportedLanguages,
    load_ksearch_solution_json,
    solution_from_json_dict,
)


_FILE_BLOCK_RE = re.compile(
    r"<file\s+path=\"([^\"]+)\"\s*>\s*(?:<!\[CDATA\[(.*?)\]\]>|(.*?))\s*</file>",
    re.DOTALL,
)


ASCENDC_CODE_FORMAT_TEXT = """Return only this multi-file container, with no markdown or explanations:
<ascendc_project>
<file path="kernel.cpp"><![CDATA[
// AscendC device code
]]></file>
<file path="tiling.cpp"><![CDATA[
// Host tiling code if needed
]]></file>
<file path="CMakeLists.txt"><![CDATA[
# Build file if the task requires changing it
]]></file>
</ascendc_project>
Include every file needed by the candidate. Keep paths relative to the project root."""

VALID_CODEGEN_MODES = ("auto", "full", "patch")


def _normalize_rel_path(path: str) -> str:
    raw = str(path or "").strip().replace("\\", "/")
    if not raw:
        raise ValueError("empty source path")
    p = Path(raw)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe source path: {path!r}")
    return raw


def parse_ascendc_project_files(raw: Any) -> dict[str, str]:
    """Parse K-Search's AscendC multi-file container into path -> content."""
    if isinstance(raw, dict):
        return {_normalize_rel_path(str(k)): str(v or "") for k, v in raw.items()}

    text = str(raw or "")
    files: dict[str, str] = {}
    for match in _FILE_BLOCK_RE.finditer(text):
        path = _normalize_rel_path(match.group(1))
        content = match.group(2) if match.group(2) is not None else match.group(3)
        files[path] = str(content or "").strip("\n")
    if not files:
        raise ValueError("AscendC response did not contain any <file path=\"...\"> blocks")
    return files


def format_ascendc_project_files(files: dict[str, str]) -> str:
    """Format path -> content as the container expected from AscendC prompts."""
    parts = ["<ascendc_project>"]
    for path in sorted(files):
        rel = _normalize_rel_path(path)
        content = str(files[path] or "")
        if "]]>" in content:
            content = content.replace("]]>", "]] >")
        parts.append(f'<file path="{rel}"><![CDATA[\n{content}\n]]></file>')
    parts.append("</ascendc_project>")
    return "\n".join(parts)


def _read_first_existing_text(root: Path, names: list[str]) -> tuple[str, Path | None]:
    for name in names:
        p = root / name
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8", errors="replace"), p
    return "", None


def _is_source_candidate(path: Path) -> bool:
    if path.name == "CMakeLists.txt":
        return True
    return path.suffix.lower() in {
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hh",
        ".hpp",
        ".json",
        ".yaml",
        ".yml",
        ".txt",
        ".cmake",
    }


def _collect_project_sources(root: Path, *, max_files: int = 80, max_bytes_per_file: int = 200_000) -> list[SourceFile]:
    if not root.exists() or not root.is_dir():
        return []
    out: list[SourceFile] = []
    skip_dirs = {".git", ".ksearch", "build", "cmake-build-debug", "__pycache__"}
    for p in sorted(root.rglob("*")):
        if any(part in skip_dirs for part in p.relative_to(root).parts):
            continue
        if not p.is_file() or not _is_source_candidate(p):
            continue
        try:
            if p.stat().st_size > max_bytes_per_file:
                continue
            rel = str(p.relative_to(root)).replace("\\", "/")
            out.append(SourceFile(path=rel, content=p.read_text(encoding="utf-8", errors="replace")))
        except Exception:
            continue
        if len(out) >= max_files:
            break
    return out


def _default_entry_point(sources: list[SourceFile]) -> str:
    preferred = ("kernel.cpp", "op_kernel.cpp", "main.cpp")
    paths = {s.path for s in sources}
    for p in preferred:
        if p in paths:
            return f"{p}::run"
    if sources:
        return f"{sources[0].path}::run"
    return "kernel.cpp::run"


def _parse_latency_ms(output: str) -> float | None:
    text = str(output or "")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("latency_ms", "mean_latency_ms", "avg_latency_ms"):
                val = obj.get(key)
                if isinstance(val, (int, float)) and float(val) > 0:
                    return float(val)
    except Exception:
        pass
    patterns = (
        r"\b(?:mean_|avg_)?latency_ms\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"\b(?:mean|avg)?\s*latency\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*ms\b",
        r"\btime_ms\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    )
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        try:
            val = float(m.group(1))
            if val > 0:
                return val
        except Exception:
            continue
    return None


class AscendCTask:
    """Direct AscendC backend using external build/correctness/benchmark commands."""

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
        self._last_parsed_raw: str | None = None
        self._patch_failure_streak = 0
        self._max_patch_failures = 3

    @property
    def name(self) -> str:
        return self._name

    def get_definition_text(self, language: str | None = None) -> str:
        task_path = self.task_path
        spec_text = ""
        spec_path: Path | None = None
        if task_path and task_path.is_file():
            spec_text = task_path.read_text(encoding="utf-8", errors="replace")
            spec_path = task_path
        elif task_path and task_path.is_dir():
            spec_text, spec_path = _read_first_existing_text(
                task_path,
                ["ksearch_task.md", "spec.md", "README.md", "task.yml", "task.yaml"],
            )

        lines = [
            f"Task: {self.name}",
            "Target language: AscendC",
            "Target platform hint: use the CLI --target-gpu value, usually ascend_910b/ascend_310p.",
        ]
        if spec_path is not None:
            lines.append(f"Specification source: {spec_path}")
        if spec_text.strip():
            lines.extend(["", "Specification:", spec_text.strip()])
        else:
            lines.extend(
                [
                    "",
                    "Specification:",
                    "Optimize the provided AscendC operator project while preserving its public inputs, outputs, tiling contract, and build/test harness behavior.",
                ]
            )

        if task_path and task_path.is_dir():
            sources = _collect_project_sources(task_path, max_files=20, max_bytes_per_file=40_000)
            if sources:
                lines.append("\nExisting project source excerpts:")
                for src in sources:
                    content = src.content
                    if len(content) > 4000:
                        content = content[:4000] + "\n...<truncated>..."
                    lines.append(f"\n--- {src.path} ---\n{content}")

        lines.extend(["", ASCENDC_CODE_FORMAT_TEXT])
        return "\n".join(lines).strip()

    def get_generation_prompt(self, *, language: str, target_gpu: str) -> str:
        return f"""You are generating an AscendC multi-file operator project optimized for {target_gpu}.

Original Specification:
{self.get_definition_text(language=language)}

Rules:
- Preserve operator semantics, public entry points, host tiling contract, and correctness harness expectations.
- Prefer small, reviewable AscendC changes over broad rewrites.
- Treat Host tiling, TilingData, blockDim, workspace, TPipe/TQue/TBuf, DataCopy/DataCopyPad, UB/L1/L0, AIC/AIV, Matmul API, and tail/alignment handling as first-class performance surfaces.
- Return only the AscendC multi-file container.

Generate the implementation:"""

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
- Return only the full AscendC multi-file container.

Generate the corrected and optimized implementation:"""

    def get_per_task_requirement_text(self, *, language: str, target_gpu: str, phase: str) -> str:
        return ASCENDC_CODE_FORMAT_TEXT

    def get_baseline_targets_text(self) -> str:
        if self.reference_latency_ms and self.reference_latency_ms > 0:
            return f"- reference_latency_ms: {self.reference_latency_ms:.6f}"
        return ""

    def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str:
        if isinstance(raw, dict):
            return format_ascendc_project_files({str(k): str(v or "") for k, v in raw.items()})
        return str(raw or "")

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
        files = parse_ascendc_project_files(raw_code if raw_code is not None else cleaned_code)
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

    def get_solution(self, solution_name: str) -> Solution | None:
        ref = str(solution_name or "").strip()
        if not ref:
            return None
        if ref.lower() in {"base", "baseline", "current"}:
            sources = _collect_project_sources(self.task_path) if self.task_path and self.task_path.is_dir() else []
            if not sources:
                return None
            return Solution(
                name=ref,
                definition=self.name,
                author="baseline",
                spec=BuildSpec(
                    language=SupportedLanguages.ASCENDC,
                    target_hardware=["ascend"],
                    entry_point=_default_entry_point(sources),
                ),
                sources=sources,
                description="Baseline AscendC project loaded from task_path",
            )
        try:
            d = load_ksearch_solution_json(
                solution_ref=ref,
                definition_name=self.name,
                artifacts_dir=self.artifacts_dir,
            )
            return solution_from_json_dict(d)
        except Exception:
            return None

    def _prepare_workdir(self, solution: Solution) -> Path:
        root = Path(tempfile.mkdtemp(prefix=f"ksearch_ascendc_{self.name}_"))
        if self.task_path and self.task_path.is_dir():
            ignore = shutil.ignore_patterns(".git", ".ksearch", "build", "cmake-build-debug", "__pycache__")
            shutil.copytree(self.task_path, root, dirs_exist_ok=True, ignore=ignore)
        for src in solution.sources or []:
            rel = _normalize_rel_path(src.path)
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(str(src.content or ""), encoding="utf-8")
        return root

    def _run_shell(self, cmd: str, *, cwd: Path) -> subprocess.CompletedProcess[str] | None:
        c = str(cmd or "").strip()
        if not c:
            return None
        return subprocess.run(
            c,
            cwd=str(cwd),
            shell=True,
            capture_output=True,
            text=True,
            timeout=max(1, int(self.timeout_seconds)),
            check=False,
        )

    @staticmethod
    def _append_command_log(logs: list[str], label: str, proc: subprocess.CompletedProcess[str] | None) -> None:
        if proc is None:
            logs.append(f"[{label}] skipped")
            return
        logs.append(f"[{label}] exit_code={proc.returncode}")
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if out:
            logs.append(f"[{label} stdout]\n{out}")
        if err:
            logs.append(f"[{label} stderr]\n{err}")

    def run_benchmark(
        self,
        *,
        solution: Solution,
        config: Any = None,
        dump_traces: bool = False,
        round_num: int | None = None,
    ) -> EvalResult:
        logs: list[str] = []
        workdir = self._prepare_workdir(solution)
        logs.append(f"[workdir] {workdir}")

        try:
            build = self._run_shell(self.build_cmd, cwd=workdir)
            self._append_command_log(logs, "build", build)
            if build is not None and build.returncode != 0:
                return self._record_eval(
                    EvalResult(
                        status="compile_failed",
                        log_excerpt=self._truncate_log(logs),
                        metrics={"workdir": str(workdir), "round": round_num},
                    )
                )

            test = self._run_shell(self.test_cmd, cwd=workdir)
            self._append_command_log(logs, "correctness", test)
            if test is not None and test.returncode != 0:
                return self._record_eval(
                    EvalResult(
                        status="failed",
                        log_excerpt=self._truncate_log(logs),
                        metrics={"workdir": str(workdir), "round": round_num},
                    )
                )

            bench = self._run_shell(self.bench_cmd, cwd=workdir)
            self._append_command_log(logs, "benchmark", bench)
            if bench is not None and bench.returncode != 0:
                return self._record_eval(
                    EvalResult(
                        status="benchmark_failed",
                        log_excerpt=self._truncate_log(logs),
                        metrics={"workdir": str(workdir), "round": round_num},
                    )
                )

            latency_ms = _parse_latency_ms(((bench.stdout or "") + "\n" + (bench.stderr or "")) if bench else "")
            if bench is not None and latency_ms is None:
                return self._record_eval(
                    EvalResult(
                        status="benchmark_failed",
                        log_excerpt=self._truncate_log(logs + ["[benchmark] missing latency_ms in output"]),
                        metrics={"workdir": str(workdir), "round": round_num},
                    )
                )

            score = 0.0
            speedup = None
            score_name = "score"
            if latency_ms and latency_ms > 0:
                if self.reference_latency_ms and self.reference_latency_ms > 0:
                    speedup = float(self.reference_latency_ms) / float(latency_ms)
                    score = speedup
                    score_name = "vs_baseline"
                else:
                    score = 1.0 / float(latency_ms)
                    score_name = "inv_latency"

            return self._record_eval(
                EvalResult(
                    status="passed",
                    latency_ms=latency_ms,
                    reference_latency_ms=self.reference_latency_ms,
                    mean_vs_baseline_factor=speedup,
                    speedup_factor=speedup,
                    log_excerpt=self._truncate_log(logs),
                    metrics={
                        "score": score,
                        "score_name": score_name,
                        "workdir": str(workdir),
                        "round": round_num,
                    },
                )
            )
        except subprocess.TimeoutExpired as e:
            logs.append(f"[timeout] command exceeded {self.timeout_seconds}s: {e}")
            return self._record_eval(
                EvalResult(
                    status="timeout",
                    log_excerpt=self._truncate_log(logs),
                    metrics={"workdir": str(workdir), "round": round_num},
                )
            )
        except Exception as e:
            logs.append(f"[error] {type(e).__name__}: {e}")
            return self._record_eval(
                EvalResult(
                    status="failed",
                    log_excerpt=self._truncate_log(logs),
                    metrics={"workdir": str(workdir), "round": round_num},
                )
            )

    def _record_eval(self, result: EvalResult) -> EvalResult:
        self._last_eval = result
        return result

    @staticmethod
    def _truncate_log(logs: list[str], *, max_chars: int = 8000) -> str:
        text = "\n\n".join(str(x) for x in logs)
        if len(text) > max_chars:
            return text[:max_chars] + "\n...<truncated>..."
        return text

    def seed_eval_for_base_solution(self, *, base_solution: Solution, config: Any = None) -> EvalResult:
        return self.run_benchmark(solution=base_solution, config=config, round_num=0)

    def get_last_round_trace_logs_for_prompt(self) -> str:
        return self._last_eval.log_excerpt if self._last_eval is not None else ""

    def get_last_round_passed_count(self) -> int:
        return 1 if self._last_eval is not None and self._last_eval.is_passed() else 0

    def get_last_round_total_workloads(self) -> int:
        return 1 if self._last_eval is not None else 0

    def get_config_for_logging(self) -> dict[str, Any]:
        return {
            "task_source": "ascendc",
            "name": self.name,
            "task_path": str(self.task_path) if self.task_path else None,
            "build_cmd": self.build_cmd,
            "test_cmd": self.test_cmd,
            "bench_cmd": self.bench_cmd,
            "reference_latency_ms": self.reference_latency_ms,
            "timeout_seconds": self.timeout_seconds,
        }

    def run_final_evaluation(
        self,
        *,
        solutions: list[Solution],
        config: Any = None,
        dump_traces: bool = False,
        workload_limit: int | None = None,
    ) -> dict[str, Any]:
        results = []
        for idx, sol in enumerate(solutions or [], start=1):
            result = self.run_benchmark(solution=sol, config=config, dump_traces=dump_traces, round_num=idx)
            results.append(
                {
                    "solution": sol.name,
                    "result": result.to_dict(include_log_excerpt=True, max_log_chars=8000),
                }
            )
        best = None
        for item in results:
            score = item["result"].get("metrics", {}).get("score")
            if not isinstance(score, (int, float)):
                continue
            if best is None or score > best.get("score", -1):
                best = {"solution": item["solution"], "score": score}
        return {
            "task": self.name,
            "task_source": "ascendc",
            "results": results,
            "best": best,
        }
