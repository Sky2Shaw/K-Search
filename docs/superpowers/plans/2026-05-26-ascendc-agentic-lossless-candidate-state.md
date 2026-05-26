# AscendC Agentic Lossless Candidate State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix A1 by making the AscendC agentic evaluation target exactly match the Claude-edited worktree, then persist enough candidate artifacts to reproduce and inherit the same state.

**Architecture:** Phase 1 keeps `Solution` as a compatibility object, but removes it from the critical evaluation path for AscendC agentic runs. `AscendCTask` gets a shared benchmark implementation that can evaluate either a prepared `Solution` workdir or an existing project directory. `AscendCAgenticCodegenRunner` evaluates before cleanup, returns `eval_result`, and writes attempt artifacts. Later phases introduce `CandidatePatch` and then `ProjectSnapshot` as the real engineering-level candidate state.

**Tech Stack:** Python dataclasses, pytest, git worktrees, existing `EvalResult` / `Solution` task abstractions.

---

## Current Hot Spots

- Modify `k_search/tasks/ascendc_task.py`
  - Existing lossy scan: `_collect_project_sources()` at lines 140-159.
  - Existing destructive overlay: `overlay_solution_sources()` at lines 523-537.
  - Existing lossy evaluation path: `_prepare_workdir()` and `run_benchmark()` at lines 643-730.
- Modify `k_search/kernel_generators/ascendc_agentic_codegen.py`
  - Existing runner edits `session.project_dir`, scans into `Solution`, then cleans up at lines 136-211.
- Modify `k_search/kernel_generators/kernel_generator_world_model.py`
  - Existing agentic branch re-evaluates `result.solution` at lines 724-729.
- Modify `k_search/kernel_generators/kernel_generator.py`
  - Existing non-world-model agentic path returns only `Solution`, then the main loop evaluates that `Solution`.
- Add or modify tests:
  - `tests/test_ascendc_task.py`
  - `tests/kernel_generators/test_ascendc_agentic_codegen.py`
  - `tests/kernel_generators/test_llm_clients.py`

---

## Phase 1: Evaluate The Edited Worktree Directly

### Task 1: Add `run_benchmark_in_project_dir()`

**Files:**
- Modify: `k_search/tasks/ascendc_task.py`
- Test: `tests/test_ascendc_task.py`

- [ ] **Step 1: Write failing test for direct project-dir evaluation**

Add this test near the existing `test_ascendc_task_run_benchmark_executes_build_test_bench_and_scores_latency` test:

```python
def test_ascendc_task_run_benchmark_in_project_dir_uses_existing_project(tmp_path):
    project = tmp_path / "candidate"
    project.mkdir()
    (project / "kernel.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")

    task = AscendCTask(
        task_path=tmp_path / "baseline",
        definition_name="vec_add",
        build_cmd=_py_cmd(
            "from pathlib import Path; "
            "assert Path('kernel.cpp').read_text() == 'int main() { return 0; }\\n'; "
            "print('build ok')"
        ),
        test_cmd=_py_cmd("print('correctness passed')"),
        bench_cmd=_py_cmd("print('latency_ms=2.5')"),
        reference_latency_ms=5.0,
        timeout_seconds=30,
    )

    result = task.run_benchmark_in_project_dir(project_dir=project, round_num=7)

    assert result.status == "passed"
    assert result.latency_ms == 2.5
    assert result.metrics["score"] == 2.0
    assert result.metrics["workdir"] == str(project.resolve())
    assert result.metrics["round"] == 7
    assert "build ok" in result.log_excerpt
```

- [ ] **Step 2: Run the focused failing test**

Run:

```bash
pytest tests/test_ascendc_task.py::test_ascendc_task_run_benchmark_in_project_dir_uses_existing_project -q
```

Expected: `AttributeError: 'AscendCTask' object has no attribute 'run_benchmark_in_project_dir'`.

- [ ] **Step 3: Refactor benchmark logic to a shared workdir helper**

In `AscendCTask`, add a private helper and make both public methods call it:

```python
    def _run_benchmark_in_workdir(
        self,
        *,
        workdir: Path,
        round_num: int | None,
    ) -> EvalResult:
        logs: list[str] = []
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
                        metrics={"workdir": str(workdir), "round": round_num, "score": -1.0},
                    )
                )

            if bench is None:
                return self._record_eval(
                    EvalResult(
                        status="benchmark_missing",
                        log_excerpt=self._truncate_log(logs + ["[benchmark] bench_cmd is empty"]),
                        metrics={"workdir": str(workdir), "round": round_num, "score": -1.0},
                    )
                )

            if self.reference_latency_ms and self.reference_latency_ms > 0:
                speedup = float(self.reference_latency_ms) / float(latency_ms)
                score = speedup
                score_name = "vs_baseline"
            else:
                speedup = None
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
                    metrics={"workdir": str(workdir), "round": round_num, "score": -1.0},
                )
            )
```

Then add:

```python
    def run_benchmark_in_project_dir(
        self,
        *,
        project_dir: str | Path,
        round_num: int | None = None,
        dump_traces: bool = False,
    ) -> EvalResult:
        del dump_traces
        workdir = Path(project_dir).expanduser().resolve()
        return self._run_benchmark_in_workdir(workdir=workdir, round_num=round_num)
```

Change `run_benchmark()` to:

```python
        workdir = self._prepare_workdir(solution)
        return self._run_benchmark_in_workdir(workdir=workdir, round_num=round_num)
```

- [ ] **Step 4: Run benchmark tests**

Run:

```bash
pytest tests/test_ascendc_task.py::test_ascendc_task_run_benchmark_executes_build_test_bench_and_scores_latency tests/test_ascendc_task.py::test_ascendc_task_run_benchmark_in_project_dir_uses_existing_project tests/test_ascendc_task.py::test_ascendc_task_run_benchmark_reports_compile_failure -q
```

Expected: all selected tests pass.

### Task 2: Return `eval_result` from agentic runner

**Files:**
- Modify: `k_search/kernel_generators/ascendc_agentic_codegen.py`
- Test: `tests/kernel_generators/test_ascendc_agentic_codegen.py`

- [ ] **Step 1: Write failing runner test**

Add this test after `test_runner_edits_worktree_and_returns_solution`:

```python
def test_runner_evaluates_edited_worktree_before_cleanup(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "kernel").mkdir()
    (task_dir / "kernel" / "foo.h").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    task = AscendCTask(
        task_path=task_dir,
        definition_name="x",
        build_cmd=_py_cmd(
            "from pathlib import Path; "
            "assert Path('kernel/foo.h').read_text() == 'alpha\\nBETA\\ngamma\\n'; "
            "print('build saw edited worktree')"
        ),
        test_cmd=_py_cmd("print('ok')"),
        bench_cmd=_py_cmd("print('latency_ms=4.0')"),
        reference_latency_ms=8.0,
    )
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

    assert result.eval_result.status == "passed"
    assert result.eval_result.metrics["score"] == 2.0
    assert "build saw edited worktree" in result.eval_result.log_excerpt
    assert result.eval_result.metrics["workdir"] == result.project_path
```

- [ ] **Step 2: Run the focused failing test**

Run:

```bash
pytest tests/kernel_generators/test_ascendc_agentic_codegen.py::test_runner_evaluates_edited_worktree_before_cleanup -q
```

Expected: failure because `AscendCAgenticCodegenResult` has no `eval_result`.

- [ ] **Step 3: Add result fields**

Modify imports:

```python
from k_search.tasks.task_base import EvalResult, Solution
```

Modify `AscendCAgenticCodegenResult`:

```python
@dataclass
class AscendCAgenticCodegenResult:
    solution: Solution
    eval_result: EvalResult
    raw: str
    cleaned: dict[str, str]
    transcript: str
    prompt: str
    prompt_chars: int
    changed_paths: list[str]
    diff_text: str
    project_path: str
    diff_after_eval: str | None = None
    evaluator_mutated_project: bool = False
    trace_path: str | None = None
    timeline_path: str | None = None
    cost_path: str | None = None
    session_id: str | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    model_usage: dict[str, Any] | None = None
    num_turns: int | None = None
    duration_ms: int | None = None
```

- [ ] **Step 4: Evaluate before `make_solution_from_project_dir()` and cleanup**

In `AscendCAgenticCodegenRunner.run()`, replace the post-edit section with:

```python
            changed_paths = session.changed_paths()
            if not changed_paths:
                raise RuntimeError(
                    "Claude agentic AscendC codegen did not change any files "
                    f"(round={request.round_num}, attempt={request.attempt_idx})"
                )
            diff_text = session.diff_text()
            run_in_project_dir = getattr(task, "run_benchmark_in_project_dir", None)
            if not callable(run_in_project_dir):
                raise RuntimeError("AscendC agentic task does not support run_benchmark_in_project_dir")
            eval_result = run_in_project_dir(project_dir=session.project_dir, round_num=request.round_num)
            diff_after_eval = session.diff_text()
            evaluator_mutated_project = diff_after_eval != diff_text
            solution = task.make_solution_from_project_dir(
                project_dir=session.project_dir,
                changed_paths=changed_paths,
                raw_agent_output=edit_result.text,
                round_num=request.round_num,
                model_name=self.model_name,
                target_gpu=request.target_gpu,
                language="ascendc",
            )
```

Add these fields to the returned dataclass:

```python
                eval_result=eval_result,
                diff_after_eval=diff_after_eval,
                evaluator_mutated_project=evaluator_mutated_project,
```

- [ ] **Step 5: Run runner tests**

Run:

```bash
pytest tests/kernel_generators/test_ascendc_agentic_codegen.py -q
```

Expected: all runner tests pass after updating any test fixture construction of `AscendCAgenticCodegenResult` to include `eval_result`.

### Task 3: Stop world-model agentic branch from re-evaluating `Solution.sources`

**Files:**
- Modify: `k_search/kernel_generators/kernel_generator_world_model.py`
- Test: `tests/kernel_generators/test_llm_clients.py`

- [ ] **Step 1: Update fake result construction in tests**

In `test_world_model_ascendc_codegen_uses_agentic_runner_before_prompt_construction`, pass an `EvalResult` into `AscendCAgenticCodegenResult`:

```python
                eval_result=EvalResult(
                    status="passed",
                    latency_ms=1.0,
                    metrics={"score": 1.0, "score_name": "inv_latency", "workdir": str(tmp_path)},
                ),
```

- [ ] **Step 2: Add assertion that world-model branch uses runner eval**

Change the fake task `bench_cmd` to a command that would fail if the old re-evaluation path still runs:

```python
        bench_cmd="python -c \"raise SystemExit('should not re-evaluate Solution.sources')\"",
```

The test should still pass because `round_eval = result.eval_result`.

- [ ] **Step 3: Modify the world-model branch**

In `k_search/kernel_generators/kernel_generator_world_model.py`, replace:

```python
                        _stage(f"evaluate solution (round {round_num})")
                        round_eval = task.run_benchmark(
                            solution=solution,
                            dump_traces=False,
                            round_num=int(round_num),
                        )
```

with:

```python
                        _stage(f"use agentic worktree eval (round {round_num})")
                        round_eval = result.eval_result
```

- [ ] **Step 4: Run the focused world-model test**

Run:

```bash
pytest tests/kernel_generators/test_llm_clients.py::test_world_model_ascendc_codegen_uses_agentic_runner_before_prompt_construction -q
```

Expected: pass; no second benchmark is invoked.

### Task 4: Stop non-world-model agentic path from losing eval state

**Files:**
- Modify: `k_search/kernel_generators/kernel_generator.py`
- Test: `tests/kernel_generators/test_llm_clients.py`

- [ ] **Step 1: Change `_generate_ascendc_solution_agentically()` to return the full result**

Rename the method to `_generate_ascendc_agentic_result()` or keep the old name and update its return type. Recommended minimal change:

```python
    def _generate_ascendc_solution_agentically(...) -> AscendCAgenticCodegenResult:
        ...
        result = self._agentic_runner().run(...)
        print(...)
        return result
```

Then update call sites:

```python
agentic_result = self._generate_ascendc_solution_agentically(...)
solution = agentic_result.solution
seed_solution = solution
```

For the main optimization loop, store:

```python
pending_agentic_eval: EvalResult | None = None
```

After agentic generation:

```python
pending_agentic_eval = agentic_result.eval_result
```

Before evaluating a solution in the loop:

```python
if pending_agentic_eval is not None:
    eval_result = pending_agentic_eval
    pending_agentic_eval = None
else:
    eval_result = task.run_benchmark(solution=solution, dump_traces=False, round_num=int(round_num))
```

- [ ] **Step 2: Add a regression test**

Add a test where a fake agentic runner returns a passed `eval_result`, while `task.run_benchmark()` would fail if called. Assert generation succeeds and consumes the fake agentic eval.

Use the same construction style as `test_world_model_ascendc_codegen_uses_agentic_runner_before_prompt_construction`, but target `KernelGenerator.generate()`.

- [ ] **Step 3: Run non-world-model tests**

Run:

```bash
pytest tests/kernel_generators/test_llm_clients.py -q
```

Expected: all tests pass.

### Task 5: Persist minimal candidate artifacts

**Files:**
- Create: `k_search/kernel_generators/agentic_candidate_artifacts.py`
- Modify: `k_search/kernel_generators/ascendc_agentic_codegen.py`
- Test: `tests/kernel_generators/test_ascendc_agentic_codegen.py`

- [ ] **Step 1: Add artifact writer**

Create `k_search/kernel_generators/agentic_candidate_artifacts.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_agentic_candidate_artifacts(
    *,
    artifacts_dir: str | Path | None,
    task_name: str,
    round_num: int,
    attempt_idx: int,
    prompt: str,
    transcript: str,
    changed_paths: list[str],
    diff_text: str,
    eval_result: Any,
    metadata: dict[str, Any],
) -> dict[str, str]:
    if artifacts_dir is None:
        root = Path.cwd() / ".ksearch"
    else:
        root = Path(artifacts_dir).expanduser().resolve()
    candidate_id = f"round_{int(round_num):04d}_attempt_{int(attempt_idx):02d}"
    out_dir = root / str(task_name or "ascendc") / "candidates" / candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "prompt_path": out_dir / "prompt.md",
        "transcript_path": out_dir / "transcript.md",
        "changed_paths_path": out_dir / "changed_paths.txt",
        "diff_path": out_dir / "diff.patch",
        "eval_path": out_dir / "eval.json",
        "manifest_path": out_dir / "manifest.json",
    }
    files["prompt_path"].write_text(prompt, encoding="utf-8")
    files["transcript_path"].write_text(transcript, encoding="utf-8")
    files["changed_paths_path"].write_text("\n".join(changed_paths) + "\n", encoding="utf-8")
    files["diff_path"].write_text(diff_text, encoding="utf-8")
    files["eval_path"].write_text(json.dumps(_jsonable(eval_result), indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "candidate_id": candidate_id,
        "round_num": int(round_num),
        "attempt_idx": int(attempt_idx),
        "changed_paths": changed_paths,
        **metadata,
        "diff_path": "diff.patch",
        "eval_path": "eval.json",
    }
    files["manifest_path"].write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return {key: str(path) for key, path in files.items()}
```

- [ ] **Step 2: Call artifact writer from runner**

In `AscendCAgenticCodegenRunner.run()`, after `eval_result` and before returning, call:

```python
            artifact_paths = write_agentic_candidate_artifacts(
                artifacts_dir=getattr(task, "artifacts_dir", None),
                task_name=getattr(task, "definition_name", None) or getattr(task, "name", "ascendc"),
                round_num=request.round_num,
                attempt_idx=request.attempt_idx,
                prompt=prompt,
                transcript=edit_result.transcript,
                changed_paths=changed_paths,
                diff_text=diff_text,
                eval_result=eval_result,
                metadata={
                    "model_name": self.model_name,
                    "target_gpu": request.target_gpu,
                    "mode": request.mode,
                    "project_path": str(session.project_dir),
                    "baseline_commit": session.baseline_commit,
                    "evaluator_mutated_project": evaluator_mutated_project,
                },
            )
```

Add `artifact_paths: dict[str, str] | None = None` to `AscendCAgenticCodegenResult`.

- [ ] **Step 3: Test artifact files**

Extend `test_runner_evaluates_edited_worktree_before_cleanup` by constructing `AscendCTask(..., artifacts_dir=str(tmp_path / "artifacts"))` and asserting:

```python
    assert result.artifact_paths is not None
    assert Path(result.artifact_paths["diff_path"]).read_text(encoding="utf-8") == result.diff_text
    assert "kernel/foo.h" in Path(result.artifact_paths["changed_paths_path"]).read_text(encoding="utf-8")
    assert json.loads(Path(result.artifact_paths["eval_path"]).read_text(encoding="utf-8"))["status"] == "passed"
```

Add imports:

```python
import json
from pathlib import Path
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/kernel_generators/test_ascendc_agentic_codegen.py tests/test_ascendc_task.py -q
```

Expected: all selected tests pass.

---

## Phase 2: Make Inheritance Patch-Based

### Task 6: Stop destructive source overlay for agentic inheritance

**Files:**
- Modify: `k_search/tasks/ascendc_task.py`
- Modify: `k_search/kernel_generators/ascendc_agentic_codegen.py`
- Test: `tests/test_ascendc_task.py`

- [ ] **Step 1: Add regression test for large files**

Write a test showing current `overlay_solution_sources()` deletes a large header that is not in `Solution.sources`. Expected behavior after the fix: the header remains.

```python
def test_overlay_solution_sources_does_not_delete_unlisted_project_files(tmp_path):
    (tmp_path / "kernel").mkdir()
    (tmp_path / "kernel" / "large_header.hpp").write_text("x" * 210_000, encoding="utf-8")
    solution = Solution(
        name="candidate",
        definition="x",
        author="test",
        spec=BuildSpec(language=SupportedLanguages.ASCENDC, target_hardware=["ascend"], entry_point="kernel/foo.h::run"),
        sources=[SourceFile(path="kernel/foo.h", content="changed\n")],
    )
    task = AscendCTask(task_path=tmp_path, definition_name="x")

    task.overlay_solution_sources(project_dir=tmp_path, solution=solution)

    assert (tmp_path / "kernel" / "large_header.hpp").exists()
    assert (tmp_path / "kernel" / "foo.h").read_text(encoding="utf-8") == "changed\n"
```

- [ ] **Step 2: Change overlay behavior**

For Phase 2, make `overlay_solution_sources()` additive/overwrite-only:

```python
        for src in solution.sources or []:
            rel = _normalize_rel_path(src.path)
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(str(src.content or ""), encoding="utf-8")
```

Remove `_remove_project_source_candidates(root)` from this path. Do not delete `_remove_project_source_candidates()` yet; keep it until all legacy tests are reviewed.

- [ ] **Step 3: Run overlay tests**

Run:

```bash
pytest tests/test_ascendc_task.py -q
```

Expected: tests pass after updating any old expectation that depended on deleting unlisted files.

### Task 7: Introduce `CandidatePatch`

**Files:**
- Create: `k_search/kernel_generators/candidate_patch.py`
- Modify: `k_search/kernel_generators/agentic_candidate_artifacts.py`
- Test: `tests/kernel_generators/test_ascendc_agentic_codegen.py`

- [ ] **Step 1: Add dataclass**

Create:

```python
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
    round_num: int
    action_node_id: str | None
    model_name: str
    eval_result: dict[str, Any]
```

- [ ] **Step 2: Add result field**

Add to `AscendCAgenticCodegenResult`:

```python
    candidate_patch: CandidatePatch | None = None
```

Populate it from artifact paths and `session.baseline_commit`.

- [ ] **Step 3: Preserve parent candidate id**

Add optional field to `AscendCAgenticCodegenRequest`:

```python
    parent_candidate_id: str | None = None
    action_node_id: str | None = None
```

Pass these into the manifest and `CandidatePatch`.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/kernel_generators/test_ascendc_agentic_codegen.py -q
```

Expected: runner tests pass and the manifest contains `candidate_id`, `parent_candidate_id`, `base_ref`, `changed_paths`, and `eval_path`.

---

## Phase 3: Connect Candidate Identity To World Model

### Task 8: Attach `candidate_id` to world-model nodes

**Files:**
- Modify: `k_search/kernel_generators/kernel_generator_world_model.py`
- Modify: `k_search/kernel_generators/world_model_manager.py`
- Test: `tests/kernel_generators/test_llm_clients.py`
- Test: `tests/kernel_generators/test_world_model_parsing.py`

- [ ] **Step 1: Extend world-model attach call**

Where the world-model branch attaches a solution/eval to the active leaf, pass:

```python
candidate_id=getattr(result.candidate_patch, "candidate_id", None),
candidate_manifest_path=(result.artifact_paths or {}).get("manifest_path"),
changed_paths=result.changed_paths,
diff_summary=result.diff_text[:4000],
```

- [ ] **Step 2: Store candidate metadata**

In the world model manager, persist these keys on the node:

```python
node["candidate_id"] = candidate_id
node["candidate_manifest_path"] = candidate_manifest_path
node["changed_paths"] = changed_paths or []
node["diff_summary"] = diff_summary or ""
```

- [ ] **Step 3: Run world-model tests**

Run:

```bash
pytest tests/kernel_generators/test_world_model_parsing.py tests/kernel_generators/test_llm_clients.py::test_world_model_ascendc_codegen_uses_agentic_runner_before_prompt_construction -q
```

Expected: node records include candidate identity without requiring a full code excerpt.

---

## Phase 4: ProjectSnapshot Long-Term Track

### Task 9: Add snapshot manifest model

**Files:**
- Create: `k_search/kernel_generators/project_snapshot.py`
- Test: `tests/kernel_generators/test_project_snapshot.py`

- [ ] **Step 1: Add dataclasses**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FileMeta:
    path: str
    sha256: str
    size: int
    mode: str
    kind: Literal["file", "symlink"]


@dataclass(frozen=True)
class ProjectSnapshot:
    snapshot_id: str
    parent_snapshot_id: str | None
    base_commit: str | None
    project_root: str
    manifest: dict[str, FileMeta]
    archive_path: str | None
    diff_from_parent: str | None
    created_by_round: int
    eval_result: dict | None
```

- [ ] **Step 2: Add materialize/create helpers**

Implement helpers that:

- Walk all files under a project directory excluding `.git`, `.ksearch`, `build`, `cmake-build-debug`, `__pycache__`, `logs`, and `llm_logs`.
- Record sha256, size, mode, and file/symlink kind.
- Optionally archive the snapshot directory.

- [ ] **Step 3: Use snapshots only after Phase 1 and Phase 2 are stable**

Do not replace Phase 1 direct worktree evaluation with snapshots. Snapshot materialization becomes the inheritance and reproduction layer after `CandidatePatch` is proven.

---

## Verification Matrix

- [ ] Direct worktree evaluation:

```bash
pytest tests/test_ascendc_task.py::test_ascendc_task_run_benchmark_in_project_dir_uses_existing_project -q
```

- [ ] Agentic runner evaluates before cleanup:

```bash
pytest tests/kernel_generators/test_ascendc_agentic_codegen.py::test_runner_evaluates_edited_worktree_before_cleanup -q
```

- [ ] World-model branch does not re-evaluate lossy `Solution.sources`:

```bash
pytest tests/kernel_generators/test_llm_clients.py::test_world_model_ascendc_codegen_uses_agentic_runner_before_prompt_construction -q
```

- [ ] AscendC task regression suite:

```bash
pytest tests/test_ascendc_task.py -q
```

- [ ] Agentic generator regression suite:

```bash
pytest tests/kernel_generators/test_ascendc_agentic_codegen.py tests/kernel_generators/test_llm_clients.py -q
```

---

## Acceptance Criteria

- `AscendCAgenticCodegenRunner.run()` calls `task.run_benchmark_in_project_dir(project_dir=session.project_dir, ...)` before `session.cleanup()`.
- For AscendC agentic world-model runs, `round_eval is result.eval_result`; no second `task.run_benchmark(solution=...)` happens.
- Each agentic attempt writes `prompt.md`, `transcript.md`, `changed_paths.txt`, `diff.patch`, `eval.json`, and `manifest.json`.
- `result.diff_text` is the pre-evaluation candidate diff. `result.diff_after_eval` is recorded separately.
- `eval_result.metrics["workdir"]` equals `result.project_path` for agentic AscendC runs.
- `overlay_solution_sources()` no longer deletes project files that are absent from lossy `Solution.sources`.
- World-model nodes can store `candidate_id` and artifact manifest path.

---

## Rollout Order

1. Merge Phase 1 first. This fixes the highest-risk state mismatch without redesigning all candidate storage.
2. Merge Phase 2 next. This prevents the next generation from inheriting a destructively reduced project.
3. Merge Phase 3 after candidate artifacts are stable. This aligns world-model memory with candidate identity.
4. Treat Phase 4 as a separate architecture migration. It is the long-term answer for long runs, branching, concurrency, and exact reproduction.
