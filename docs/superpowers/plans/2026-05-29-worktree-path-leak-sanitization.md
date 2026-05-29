# Worktree 路径泄露根治 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让送达 LLM 的所有文本不再携带物理 worktree 路径,根治跨轮路径泄露。

**Architecture:** 新增一个通配正则净化函数,在两层调用——源头(`_truncate_log`,所有 `log_excerpt` 的唯一收口)与边界(`PromptBuilder.build`,送达 LLM 的最后防线);同时移除现有"把 original_task_path 翻译成物理 worktree 路径"的反向消毒。

**Tech Stack:** Python 3.10+、pytest、re

设计依据: `docs/superpowers/specs/2026-05-29-worktree-path-leak-sanitization-design.md`

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `k_search/utils/path_sanitize.py`(新建) | 提供 `sanitize_worktree_paths(text)` 单一纯函数 |
| `k_search/tasks/ascendc_task.py`(改) | `_truncate_log` 注入源头净化(A) |
| `k_search/kernel_generators/ascendc_agentic_codegen.py`(改) | 删反向替换 + 加边界净化(B) |
| `tests/utils/test_path_sanitize.py`(新建) | 净化函数单测 |
| `tests/utils/__init__.py`(新建,若需要) | 使 tests/utils 成为包 |
| `tests/test_ascendc_task.py`(改) | `_truncate_log` 净化测试 |
| `tests/kernel_generators/test_ascendc_agentic_codegen.py`(改) | 改写反向替换测试为占位符断言 |

---

### Task 1: 共享净化函数 `sanitize_worktree_paths`

**Files:**
- Create: `k_search/utils/path_sanitize.py`
- Test: `tests/utils/test_path_sanitize.py`
- Create (if missing): `tests/utils/__init__.py`

- [ ] **Step 1: 创建 tests/utils 包标记(若不存在)**

Run: `test -f tests/utils/__init__.py || (mkdir -p tests/utils && touch tests/utils/__init__.py)`

- [ ] **Step 2: 写失败的测试**

Create `tests/utils/test_path_sanitize.py`:

```python
from k_search.utils.path_sanitize import sanitize_worktree_paths


def test_replaces_single_worktree_path():
    text = "[workdir] /tmp/ksearch_agentic_worktree_02ut4r9r/tile2asc/mqa/kernel/x.cpp"
    out = sanitize_worktree_paths(text)
    assert "ksearch_agentic_worktree_02ut4r9r" not in out
    assert out == "[workdir] <PROJECT_ROOT>/tile2asc/mqa/kernel/x.cpp"


def test_replaces_temp_repo_fallback_path():
    text = "see /tmp/ksearch_agentic_temp_repo_abc123/kernel/foo.h"
    out = sanitize_worktree_paths(text)
    assert "ksearch_agentic_temp_repo_abc123" not in out
    assert out == "see <PROJECT_ROOT>/kernel/foo.h"


def test_replaces_multiple_distinct_random_suffixes():
    text = (
        "old /tmp/ksearch_agentic_worktree_02ut4r9r/a.cpp "
        "new /tmp/ksearch_agentic_worktree_14595g7x/b.cpp"
    )
    out = sanitize_worktree_paths(text)
    assert "02ut4r9r" not in out
    assert "14595g7x" not in out
    assert out.count("<PROJECT_ROOT>") == 2


def test_bare_worktree_root_replaced():
    text = "/tmp/ksearch_agentic_worktree_gvc"
    out = sanitize_worktree_paths(text)
    assert out == "<PROJECT_ROOT>"


def test_no_match_returns_unchanged():
    text = "compile ok, no paths here"
    assert sanitize_worktree_paths(text) == text


def test_idempotent():
    text = "/tmp/ksearch_agentic_worktree_xyz/kernel/x.cpp"
    once = sanitize_worktree_paths(text)
    assert sanitize_worktree_paths(once) == once


def test_custom_placeholder():
    text = "/tmp/ksearch_agentic_worktree_xyz/x.cpp"
    out = sanitize_worktree_paths(text, placeholder="<ROOT>")
    assert out == "<ROOT>/x.cpp"
```

- [ ] **Step 3: 运行测试确认失败**

Run: `python -m pytest tests/utils/test_path_sanitize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'k_search.utils.path_sanitize'`

- [ ] **Step 4: 写最小实现**

Create `k_search/utils/path_sanitize.py`:

```python
from __future__ import annotations

import re

# 匹配 ksearch 临时 worktree / fallback 临时 repo 的绝对路径根:
#   /<任意前缀>/ksearch_agentic_worktree_<rand>
#   /<任意前缀>/ksearch_agentic_temp_repo_<rand>
# 捕获到随机根目录为止,其后的相对子路径保留不动。
_WORKTREE_ROOT_RE = re.compile(
    r"/[^\s]*?/(?:ksearch_agentic_worktree|ksearch_agentic_temp_repo)_[A-Za-z0-9]+"
)


def sanitize_worktree_paths(text: str, *, placeholder: str = "<PROJECT_ROOT>") -> str:
    """把任意 ksearch 临时 worktree / 临时 repo 的绝对路径前缀替换为语义占位符。

    用通配正则而非精确字符串,故任意历史轮次的残留路径都会被替换,无需知道
    "当前 worktree 是谁";天然幂等。
    """
    if not text:
        return text
    return _WORKTREE_ROOT_RE.sub(placeholder, text)
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/utils/test_path_sanitize.py -v`
Expected: PASS(7 passed)

- [ ] **Step 6: 提交**

```bash
git add k_search/utils/path_sanitize.py tests/utils/test_path_sanitize.py tests/utils/__init__.py
git commit -m "feat: add sanitize_worktree_paths shared utility"
```

---

### Task 2: 源头净化 —— `_truncate_log`(方案 A)

**Files:**
- Modify: `k_search/tasks/ascendc_task.py:808-813`(`_truncate_log` 静态方法)
- Test: `tests/test_ascendc_task.py`

- [ ] **Step 1: 写失败的测试**

在 `tests/test_ascendc_task.py` 末尾追加:

```python
def test_truncate_log_sanitizes_worktree_paths():
    from k_search.tasks.ascendc_task import AscendCTask

    logs = [
        "[workdir] /tmp/ksearch_agentic_worktree_02ut4r9r/tile2asc/mqa",
        "-- Build files: /tmp/ksearch_agentic_worktree_02ut4r9r/kernel/build",
    ]
    out = AscendCTask._truncate_log(logs)
    assert "ksearch_agentic_worktree_02ut4r9r" not in out
    assert "<PROJECT_ROOT>" in out


def test_truncate_log_sanitizes_before_truncation():
    from k_search.tasks.ascendc_task import AscendCTask

    # 占位符必须在截断前替换,不能被 max_chars 拦腰截断破坏。
    logs = ["/tmp/ksearch_agentic_worktree_02ut4r9r/" + "x" * 50]
    out = AscendCTask._truncate_log(logs, max_chars=20)
    assert "ksearch_agentic_worktree_02ut4r9r" not in out
    assert out.startswith("<PROJECT_ROOT>")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_ascendc_task.py::test_truncate_log_sanitizes_worktree_paths tests/test_ascendc_task.py::test_truncate_log_sanitizes_before_truncation -v`
Expected: FAIL — 输出仍含 `ksearch_agentic_worktree_02ut4r9r`

- [ ] **Step 3: 修改实现**

在 `k_search/tasks/ascendc_task.py` 顶部 import 区加入(与其它 `from k_search...` import 同组):

```python
from k_search.utils.path_sanitize import sanitize_worktree_paths
```

把 `_truncate_log`(当前 808-813 行)改为先净化再截断:

```python
    @staticmethod
    def _truncate_log(logs: list[str], *, max_chars: int = 8000) -> str:
        text = sanitize_worktree_paths("\n\n".join(str(x) for x in logs))
        if len(text) > max_chars:
            return text[:max_chars] + "\n...<truncated>..."
        return text
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_ascendc_task.py -v`
Expected: PASS(含两个新测试,且原有测试不回归)

- [ ] **Step 5: 提交**

```bash
git add k_search/tasks/ascendc_task.py tests/test_ascendc_task.py
git commit -m "feat: sanitize worktree paths at log_excerpt source (_truncate_log)"
```

---

### Task 3: 边界净化 + 移除反向替换 —— `PromptBuilder.build`(方案 B)

**Files:**
- Modify: `k_search/kernel_generators/ascendc_agentic_codegen.py:135-146`(`build` 方法尾部)
- Test: `tests/kernel_generators/test_ascendc_agentic_codegen.py`

- [ ] **Step 1: 改写现有反向替换测试为占位符断言**

把 `tests/kernel_generators/test_ascendc_agentic_codegen.py` 中 `test_prompt_builder_sanitizes_original_task_paths`(当前 377-403 行)整体替换为:

```python
def test_prompt_builder_replaces_paths_with_placeholder(tmp_path):
    """Prompt must NOT inject any physical path; original dirs become a placeholder."""
    original_dir = tmp_path / "original_project"
    original_dir.mkdir()
    worktree_dir = tmp_path / "worktree_project"

    builder = AscendCAgenticPromptBuilder(max_chars=20_000)
    request = AscendCAgenticCodegenRequest(
        definition_text=(
            f"Task: x\nSpecification source: {original_dir}/ksearch_task.md\n"
            f"Specification:\nSee {original_dir}/kernel/foo.h for details."
        ),
        action_text="Optimize the kernel.",
        trace_logs="[workdir] /tmp/ksearch_agentic_worktree_02ut4r9r/kernel/x.cpp",
        perf_summary="",
        target_gpu="ascend_910b",
        round_num=1,
        attempt_idx=1,
        mode="action",
    )

    prompt = builder.build(
        request,
        project_dir=worktree_dir,
        original_task_path=original_dir,
    )

    # 既不注入旧 worktree 物理路径,也不注入当前 worktree 物理路径。
    assert "ksearch_agentic_worktree_02ut4r9r" not in prompt
    assert str(worktree_dir) not in prompt
    assert "<PROJECT_ROOT>" in prompt
    assert "Specification source:" in prompt
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest "tests/kernel_generators/test_ascendc_agentic_codegen.py::test_prompt_builder_replaces_paths_with_placeholder" -v`
Expected: FAIL — 当前实现把 `worktree_dir` 写进 prompt 且 trace 旧路径未净化,断言不成立

- [ ] **Step 3: 修改实现**

在 `k_search/kernel_generators/ascendc_agentic_codegen.py` 顶部 import 区加入:

```python
from k_search.utils.path_sanitize import sanitize_worktree_paths
```

把 `build` 方法当前的 135-146 行(从 `# Sanitize absolute paths` 注释到 `return prompt`)替换为:

```python
        # 不变量:送达 LLM 的文本不得携带物理 worktree 路径,统一抹成语义占位符。
        prompt = sanitize_worktree_paths(prompt)
        if len(prompt) > self.max_chars:
            sizes = ", ".join(f"{name}={len(value)}" for name, value in sorted(sections.items()))
            raise ValueError(
                f"agentic prompt exceeded {self.max_chars} chars: prompt={len(prompt)}, sections: {sizes}"
            )
        return prompt
```

注:`build` 签名仍保留 `project_dir` / `original_task_path` 两个参数(调用方仍在传),
但不再用于路径翻译——保持向后兼容,避免改调用方。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/kernel_generators/test_ascendc_agentic_codegen.py -v`
Expected: PASS(改写后的测试通过,其余 PromptBuilder 测试不回归)

- [ ] **Step 5: 提交**

```bash
git add k_search/kernel_generators/ascendc_agentic_codegen.py tests/kernel_generators/test_ascendc_agentic_codegen.py
git commit -m "fix: sanitize prompt paths to placeholder, drop reverse path translation"
```

---

### Task 4: 全量回归与端到端验证

**Files:** 无改动,仅验证

- [ ] **Step 1: 跑相关测试全集**

Run: `python -m pytest tests/utils/test_path_sanitize.py tests/test_ascendc_task.py tests/kernel_generators/test_ascendc_agentic_codegen.py -v`
Expected: 全部 PASS

- [ ] **Step 2: 对历史 prompt 复现场景做断言式验证**

Run:
```bash
python -c "
from k_search.utils.path_sanitize import sanitize_worktree_paths
p = open('.ksearch-output-mqa/telemetry/__unknown__/20260526_125227/round_0002/action_unknown/attempt_0002/prompt.md').read()
out = sanitize_worktree_paths(p)
assert 'ksearch_agentic_worktree_02ut4r9r' not in out, 'old path still leaks'
assert 'ksearch_agentic_worktree_14595g7x' not in out, 'current path still present'
print('OK: round_0002 prompt fully sanitized')
"
```
Expected: 打印 `OK: round_0002 prompt fully sanitized`

> 注:此为只读验证,证明净化函数能清掉真实历史日志中的所有 worktree 路径。

- [ ] **Step 3: 最终提交(若 step 2 脚本被保留为一次性验证则无需提交)**

无新增文件则跳过。

---

## 自审结果

- **Spec 覆盖**:§4.1 共享函数→Task1;§4.2 源头→Task2;§4.3 边界+删反向替换→Task3;§6 测试散落各 Task;§8 验收→Task4。无遗漏。
- **占位符扫描**:无 TBD/TODO;每个代码步骤均给出完整代码。
- **类型/命名一致**:函数全程为 `sanitize_worktree_paths`,占位符全程 `<PROJECT_ROOT>`,正则常量 `_WORKTREE_ROOT_RE`,前后一致。
- **已知风险处理**:现有 `test_prompt_builder_sanitizes_original_task_paths` 会与新行为冲突,Task3 Step1 已明确将其改写,而非新增导致双重断言矛盾。
