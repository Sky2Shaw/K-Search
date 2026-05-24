import os
import shlex
import sys
from pathlib import Path

import pytest

from k_search.tasks.task_base import BuildSpec, Solution, SourceFile, SupportedLanguages, code_from_solution
from k_search.kernel_generators.kernel_generator_prompts import get_prompt_from_definition_text
from k_search.kernel_generators.world_model_prompts import get_generate_code_from_action_prompt_from_text
from k_search.tasks.ascendc_task import (
    ASCENDC_CODE_FORMAT_TEXT,
    AscendCTask,
    format_ascendc_project_files,
    parse_ascendc_project_files,
)


def _py_cmd(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def test_parse_and_format_ascendc_project_files_round_trips_cpp_templates():
    raw = """
Before text is ignored.
<ascendc_project>
<file path="kernel.cpp">
#include <type_traits>
template <typename T>
__aicore__ inline T identity(T x) { return x; }
</file>
<file path="CMakeLists.txt">
cmake_minimum_required(VERSION 3.16)
</file>
</ascendc_project>
"""

    files = parse_ascendc_project_files(raw)

    assert files["kernel.cpp"].startswith("#include <type_traits>")
    assert "__aicore__ inline T identity" in files["kernel.cpp"]
    assert files["CMakeLists.txt"] == "cmake_minimum_required(VERSION 3.16)"

    formatted = format_ascendc_project_files(files)
    reparsed = parse_ascendc_project_files(formatted)

    assert reparsed == files


def test_code_from_solution_reconstructs_ascendc_multifile_container():
    solution = Solution(
        name="candidate",
        definition="vec_add",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.ASCENDC,
            target_hardware=["ascend_910b"],
            entry_point="kernel.cpp::AddCustom",
        ),
        sources=[
            SourceFile(path="kernel.cpp", content="void AddCustom() {}"),
            SourceFile(path="tiling.cpp", content="void TilingFunc() {}"),
        ],
    )

    code_dict, raw = code_from_solution("ascendc", solution)

    assert code_dict == {
        "kernel.cpp": "void AddCustom() {}",
        "tiling.cpp": "void TilingFunc() {}",
    }
    assert parse_ascendc_project_files(raw) == code_dict


def test_ascendc_task_makes_solution_from_generated_multifile_project(tmp_path):
    (tmp_path / "spec.md").write_text("Vector add AscendC operator.", encoding="utf-8")
    task = AscendCTask(task_path=tmp_path, definition_name="vec_add")
    raw = format_ascendc_project_files(
        {
            "kernel.cpp": "extern \"C\" __global__ __aicore__ void AddCustom() {}",
            "tiling.cpp": "int TilingFunc() { return 0; }",
        }
    )

    solution = task.make_solution_from_generated_code(
        cleaned_code=raw,
        raw_code=raw,
        round_num=3,
        model_name="model",
        target_gpu="ascend_910b",
        language="ascendc",
    )

    assert solution.definition == "vec_add"
    assert solution.spec.language == SupportedLanguages.ASCENDC
    assert solution.spec.target_hardware == ["ascend_910b"]
    assert solution.get_entry_path() == "kernel.cpp"
    assert {src.path for src in solution.sources} == {"kernel.cpp", "tiling.cpp"}


def test_ascendc_task_run_benchmark_executes_build_test_bench_and_scores_latency(tmp_path):
    solution = Solution(
        name="candidate",
        definition="vec_add",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.ASCENDC,
            target_hardware=["ascend_910b"],
            entry_point="kernel.cpp::AddCustom",
        ),
        sources=[SourceFile(path="kernel.cpp", content="int main() { return 0; }\n")],
    )
    task = AscendCTask(
        task_path=tmp_path,
        definition_name="vec_add",
        build_cmd=_py_cmd("from pathlib import Path; assert Path('kernel.cpp').exists(); print('build ok')"),
        test_cmd=_py_cmd("print('correctness passed')"),
        bench_cmd=_py_cmd("print('latency_ms=2.5')"),
        reference_latency_ms=5.0,
        timeout_seconds=30,
    )

    result = task.run_benchmark(solution=solution, round_num=1)

    assert result.status == "passed"
    assert result.latency_ms == 2.5
    assert result.reference_latency_ms == 5.0
    assert result.mean_vs_baseline_factor == 2.0
    assert result.metrics["score"] == 2.0
    assert result.metrics["workdir"]
    assert "build ok" in result.log_excerpt
    assert "correctness passed" in result.log_excerpt


def test_ascendc_task_run_benchmark_reports_compile_failure(tmp_path):
    solution = Solution(
        name="candidate",
        definition="vec_add",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.ASCENDC,
            target_hardware=["ascend_910b"],
            entry_point="kernel.cpp::AddCustom",
        ),
        sources=[SourceFile(path="kernel.cpp", content="broken\n")],
    )
    task = AscendCTask(
        task_path=tmp_path,
        definition_name="vec_add",
        build_cmd=_py_cmd("import sys; print('compile failed'); sys.exit(2)"),
        test_cmd=_py_cmd("raise SystemExit('should not run')"),
        bench_cmd=_py_cmd("raise SystemExit('should not run')"),
        timeout_seconds=30,
    )

    result = task.run_benchmark(solution=solution, round_num=1)

    assert result.status == "compile_failed"
    assert result.score() == -1.0
    assert "compile failed" in result.log_excerpt


def test_ascendc_prompt_builders_accept_ascendc_language():
    prompt = get_prompt_from_definition_text(
        "ascendc",
        "Implement vector add.",
        "ascend_910b",
        per_task_requirement=ASCENDC_CODE_FORMAT_TEXT,
    )
    action_prompt = get_generate_code_from_action_prompt_from_text(
        "ascendc",
        definition_text="Implement vector add.",
        base_code="<ascendc_project></ascendc_project>",
        action_text="Increase tile length within UB capacity.",
        code_format=ASCENDC_CODE_FORMAT_TEXT,
        target_gpu="ascend_910b",
    )

    assert "AscendC" in prompt
    assert "ascend_910b" in prompt
    assert "<ascendc_project>" in prompt
    assert "Increase tile length" in action_prompt


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
