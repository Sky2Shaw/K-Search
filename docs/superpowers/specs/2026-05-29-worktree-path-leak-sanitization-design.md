# Worktree 路径泄露根治设计

日期: 2026-05-29
状态: 待评审

## 1. 问题陈述

K-Search 的 AscendC agentic 迭代搜索中,每一轮在一个**随机命名的临时 worktree**
(`tempfile.mkdtemp(prefix="ksearch_agentic_worktree_")`)里执行 build/benchmark,用完即删。
上一轮评测产生的 `log_excerpt` 天然包含该轮 worktree 的绝对路径
(`[workdir] /tmp/ksearch_agentic_worktree_02ut4r9r/...`、CMake 构建路径等)。

这份日志通过 `trace_logs` 被原样拼进**下一轮**的 prompt
(`Recent failure or trace excerpt:` section)送给 LLM。但下一轮的 worktree 是新的随机路径
(`14595g7x`),于是 LLM 在新环境里看到一堆指向**已被删除的旧目录**的绝对路径。

### 证据

实测 20 轮日志中,每轮 prompt 都出现"错位相邻"的两个路径
(round N 的路径泄露进 round N+1 的 trace section):

| Round | trace section 中(泄露) | task section 中(当前) |
|---|---|---|
| 1 | — | `02ut4r9r` |
| 2 | `02ut4r9r` | `14595g7x` |
| 3 | `14595g7x` | `gvc...` |
| … | … | … |

定位到 `round_0002/prompt.md`:第19行(task section)是当前路径,被现有消毒逻辑正确替换;
第156行起(trace section)是 `[workdir] /tmp/ksearch_agentic_worktree_02ut4r9r/...`,
现有消毒逻辑完全没碰。

## 2. 第一性原理分析

剥到最底层,有三条不可去除的事实:

1. LLM 需要看到上一轮执行反馈(编译错误、性能数据)才能迭代 —— agentic 搜索的必要条件。
2. 反馈在某个物理工作目录里产生,build 工具(CMake)固有地会把绝对路径写进日志 —— 不可去除。
3. 每轮工作目录是临时、随机、用完即删的 —— worktree 隔离设计的结果。

矛盾:反馈是"过去时"(在已销毁的 round N 目录产生),消费它的 LLM 在"现在时"(round N+1 新目录)。

**根因的根因**:系统把**物理路径**当作信息传给了 LLM。但 LLM 工作在 CWD,用相对路径即可,
**物理绝对路径对它是 100% 噪声且有害**(诱导它访问不存在的路径,违反 prompt 第115行
"禁止使用外部绝对路径"的约束)。

由此得到要建立的**不变量(invariant)**:

> 任何跨越"轮次边界"或"agent 边界"送达 LLM 的文本,都不得携带物理环境地址。
> 物理路径是环境实现细节,不是语义信息。

现有 `ascendc_agentic_codegen.py:135-140` 的消毒之所以错,不只是漏了 trace_logs,
更是**方向反了**——它把 `original_task_path` 翻译成**另一个物理路径**(当前 worktree),
而正确做法是把所有物理路径**抹除成语义占位符**。

## 3. 方案选型

考虑过三层切入点:
- A 源头(`log_excerpt` 落地时)、B 边界(`PromptBuilder` 拼 prompt 时)、C 根源(消除随机路径)。

C(稳定 worktree 路径)被否决:即便路径稳定,LLM 看到绝对路径仍是噪声,且引入并发/复用冲突,
治标层级最低,收益最不本质。

**采纳 A+B 双层**:
- **A 源头净化**是唯一能"一次根治、处处可信"的层。`_truncate_log` 是所有 `log_excerpt` 的
  唯一收口(`@staticmethod`),在此净化后,下游所有消费者(agentic prompt、世界模型 prompt、
  snapshot、artifacts)自动获得干净数据。
- **B 边界净化**作为深度防御。送达 LLM 是最关键边界,值得一道不依赖上游正确性的最后防线。

## 4. 详细设计

### 4.1 共享净化函数(新增)

新增模块 `k_search/utils/path_sanitize.py`,提供:

```python
def sanitize_worktree_paths(text: str, *, placeholder: str = "<PROJECT_ROOT>") -> str:
    """把任意 ksearch 临时 worktree / 临时 repo 的绝对路径前缀替换为语义占位符。"""
```

**关键设计点 —— 用通配正则而非精确字符串:**

匹配模式覆盖两类临时根:
- `ksearch_agentic_worktree_<rand>`
- `ksearch_agentic_temp_repo_<rand>`(fallback 无 git 时的临时 repo)

正则形如(匹配到临时根目录为止,后续相对路径保留):

```
(/.*?/(?:ksearch_agentic_worktree|ksearch_agentic_temp_repo)_[A-Za-z0-9]+)
```

匹配到的整段绝对前缀 → 替换为 `<PROJECT_ROOT>`。这样:
- **任意历史轮次**的残留路径(不只是上一轮)都能被扫掉;
- `/tmp/ksearch_agentic_worktree_02ut4r9r/tile2asc/mqa/kernel/x.cpp`
  → `<PROJECT_ROOT>/tile2asc/mqa/kernel/x.cpp`;
- 不依赖"知道当前 worktree 是谁",纯模式驱动,天然幂等。

### 4.2 方案 A:源头净化

修改 `k_search/tasks/ascendc_task.py` 的 `_truncate_log`:
在 join 之后、**截断之前**调用 `sanitize_worktree_paths`(先净化再截断,
避免占位符被 `max_chars` 拦腰截断)。

```python
@staticmethod
def _truncate_log(logs: list[str], *, max_chars: int = 8000) -> str:
    text = sanitize_worktree_paths("\n\n".join(str(x) for x in logs))
    if len(text) > max_chars:
        return text[:max_chars] + "\n...<truncated>..."
    return text
```

此处一改,所有 `log_excerpt`(7 处调用点)、`get_last_round_trace_logs_for_prompt()`、
以及存进 snapshot / artifacts 的 eval 日志全部自动干净。

### 4.3 方案 B:边界净化 + 修正反向消毒

修改 `k_search/kernel_generators/ascendc_agentic_codegen.py` 的
`AscendCAgenticPromptBuilder.build`:

1. **删除** 135-140 行现有的"`original_task_path` → 当前 worktree"反向替换
   (它往 prompt 里塞物理路径,方向错误)。
2. 在返回 prompt 前(长度校验之前),对整个 prompt 调用一次
   `sanitize_worktree_paths`,作为最后防线。

> 注:净化在长度校验前执行,因为占位符通常短于原绝对路径,净化只会减小 prompt 体积,
> 不会突破 `max_chars`。

### 4.4 不改动的部分

- `agentic_worktree.py` 的 `mkdtemp` 随机路径机制**不动**(隔离正确,随机无害,
  问题不在它而在日志传递)。
- 世界模型 prompt 各出口(`kernel_generator_world_model.py` / `world_model_prompts.py`)
  **无需**逐个加 sanitize —— 它们消费的 `trace_logs` 已在源头(A)净化。
  (若要更保守,可在世界模型的 prompt 组装收口处加一道 B 同款防线,本设计暂不强制,YAGNI。)

## 5. 数据流(修复后)

```
[round N 执行] -> logs -> _truncate_log()
                              |-- sanitize_worktree_paths()  ← A 源头净化
                              v
                         log_excerpt (已含 <PROJECT_ROOT>) -> _last_eval
                              | (跨轮边界, 已干净)
                              v
[round N+1] get_last_round_trace_logs_for_prompt() -> trace_logs (干净)
                              v
                    PromptBuilder.build()
                              |-- sanitize_worktree_paths(prompt)  ← B 兜底防线
                              v
                          prompt (无物理路径) -> LLM
```

## 6. 测试策略

新增 `tests/utils/test_path_sanitize.py`:
- 单条 worktree 路径 → 占位符;
- temp_repo fallback 路径 → 占位符;
- **多个不同随机后缀**(模拟跨轮残留)同时出现 → 全部替换;
- 路径后接相对子路径时,子路径保留;
- 无匹配文本原样返回;幂等(二次调用结果不变)。

新增/扩展 `tests/kernel_generators/`:
- `AscendCAgenticPromptBuilder.build` 给定含旧 worktree 路径的 trace_logs,
  断言输出 prompt 中**不含** `ksearch_agentic_worktree_` 字样,含 `<PROJECT_ROOT>`;
- 断言已删除的反向替换不再把 `original_task_path` 翻成物理 worktree 路径。

扩展 `tests/.../ascendc_task` 相关:
- 构造含 worktree 绝对路径的 logs,断言 `_truncate_log` 输出已净化。

回归验证:对现有 `.ksearch-output-mqa` 的 round_0002 prompt 复现场景,
确认修复后旧路径不再出现。

## 7. 影响范围小结

| 文件 | 改动 |
|---|---|
| `k_search/utils/path_sanitize.py` | 新增共享净化函数 |
| `k_search/tasks/ascendc_task.py` | `_truncate_log` 注入净化(A) |
| `k_search/kernel_generators/ascendc_agentic_codegen.py` | 删反向替换 + 加兜底净化(B) |
| `tests/utils/test_path_sanitize.py` | 新增单测 |
| `tests/kernel_generators/...` | 扩展 PromptBuilder 测试 |

## 8. 验收标准

1. 任意一轮 prompt 的 trace section 不再出现 `/tmp/ksearch_agentic_worktree_*` 绝对路径。
2. 跨多轮残留的历史路径同样被净化(不止上一轮)。
3. 现有反向"翻译成物理路径"的逻辑被移除,prompt 中不再主动注入物理绝对路径。
4. 全部新增/现有测试通过。
