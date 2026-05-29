# 设计:记忆管理框架 + 可扩展 Agent 角色框架(AscendC Agentic Codegen)

- 日期:2026-05-29
- 范围:仅 AscendC 多文件 agentic 代码生成路径
- 状态:设计待评审

## 1. 背景与问题

K-Search 的 AscendC agentic 流程中,每个优化 attempt 都会:

1. `AscendCAgenticCodegenRunner.run()` 调用 `create_agentic_worktree()` 从 `task_path` **整份复制**项目到一个临时 worktree;
2. prompt 里写死 *"First inspect the project with Glob, Grep, and Read"*,Claude agent 因此**从零 grep/read 整个多文件算子项目**,理解一遍后才动手;
3. attempt 结束 worktree 被 `cleanup()` 销毁,这份"对代码的理解"随之丢弃。

下一个 attempt 重复上述全过程。已有的 `world_model` 跨轮持久化的是**优化策略**(假设、bottleneck、`tiling_policy` 等决策树、下一步 action),并不记录**代码本身的结构**(文件职责、调用链、kernel 设计、契约)。

由此带来四个痛点(均为本设计要解决的目标):省 token/成本、提升跨轮理解一致性、减少冗余探索提速、沉淀可复用知识。

对照其他 backend:CUDA/KernelBench 等是单 `kernel.cu`,通过 `current_code` 直接注入 prompt,不存在"重新读项目"问题。**痛点集中在 AscendC 多文件 agentic 场景**,故本设计仅覆盖该路径。

## 2. 目标与非目标

**目标**
- 建立**可扩展的记忆管理框架**:跨 attempt 持久化"代码理解"等记忆,可平滑新增记忆类型。
- 建立**可扩展的 agent 角色框架**:把"与项目对话"的角色统一抽象,可平滑新增角色。
- 本期落地:`code_map` 记忆 + `CodeReaderAgent` + 将现有 codegen 重构为 `CodegenAgent`。

**非目标(本期不做)**
- 不覆盖 CUDA / KernelBench / 其他 backend。
- 不改造 non-agentic(OpenAI 直出文本)路径。
- 不实现 `plan_agent` / `review_agent`,仅预留基类与目录。
- 不改动 `world_model` 的存储与逻辑(两者解耦)。

## 3. 架构总览(三层)

```
                ┌──────────────────────────────────────────────────────────┐
   memory/      │  MemoryStore(<artifacts>/<task>/memory/<kind>/<filename>)  │
   (持久化)      │  内置 MemoryKind: CODE_MAP("code_map","CODE_MAP.md",gated) │
                └───────────────▲───────────────────────┬───────────────────┘
                         save()  │ (采纳门控)             │ materialize()/read()
                                 │                       ▼
   orchestrator   AscendCAgenticCodegenRunner.run()  ── 编排 worktree / memory / eval / snapshot / artifacts
   (编排)                         │                       │
                                 │ 调用                  │ 调用
                                 ▼                       ▼
   agents/        CodeReaderAgent(ProjectAgent)     CodegenAgent(ProjectAgent)
   (与项目对话)     tools=[Read,Grep,Glob,Write]      tools=[Read,Grep,Glob,Edit,Write]
                  产出 code_map 记忆                  改代码 + 顺手更新 code_map
                  (预留 plan_agent / review_agent)
```

- **agents/**:每个角色只负责"构造 prompt → 让 Claude 在 worktree 内交互 → 返回结果"。不碰 worktree 生命周期、eval、artifacts。
- **memory/**:跨 attempt 持久化任意类型记忆,对类型无感(通过 `MemoryKind` 描述符)。
- **orchestrator**:`AscendCAgenticCodegenRunner` 瘦身为编排器,串起 worktree、memory、agents、eval、snapshot。

`world_model`(要做什么)与 `code_map`(代码是什么)职责正交,互不读写对方存储。

## 4. 组件详述

### 4.1 `k_search/kernel_generators/memory/memory_store.py`(新增)

```python
@dataclass(frozen=True)
class MemoryKind:
    name: str            # 例 "code_map"
    filename: str        # worktree 内文件名,例 "CODE_MAP.md"
    gated_writeback: bool # 是否需"采纳门控"才回写

CODE_MAP = MemoryKind("code_map", "CODE_MAP.md", gated_writeback=True)

class MemoryStore:
    def __init__(self, *, artifacts_dir, task_name): ...
    def load(self, kind: MemoryKind) -> str | None: ...
    def save(self, kind: MemoryKind, text: str) -> None: ...
    def materialize(self, kind: MemoryKind, project_dir) -> bool: ...      # 写入 worktree 根
    def read_from_worktree(self, kind: MemoryKind, project_dir) -> str | None: ...
```

- 底层路径:`<artifacts>/<task>/memory/<kind.name>/<kind.filename>`,`artifacts` 复用 `get_ksearch_artifacts_dir`。
- 类型无感:新增 plan/review 记忆只需再注册一个 `MemoryKind`,`MemoryStore` 不改。
- 不存在/空内容返回 `None`,不抛异常。

### 4.2 `k_search/kernel_generators/agents/project_agent.py`(新增)

`ProjectAgent` 基类,封装现有 `ClaudeAgentProjectEditorClient` 的一次"对项目的交互":

```python
@dataclass
class AgentRunResult:
    text: str
    transcript: str
    edit_result: ClaudeProjectEditResult   # 透传遥测/usage/session 等

class ProjectAgent:
    allowed_tools: list[str]
    disallowed_tools: list[str] = ["Bash"]
    def __init__(self, *, model_name, editor_client=None): ...
    def build_prompt(self, context) -> str: ...        # 子类实现
    def run(self, *, project_dir, context, telemetry_recorder=None) -> AgentRunResult: ...
```

- `run()` 负责:`build_prompt` → 绝对路径消毒(沿用现有 `_truncate`/路径替换逻辑)→ `editor_client.edit_project(...)`(带可选遥测,沿用 `_edit_project_with_optional_telemetry` 的容错)→ 包装 `AgentRunResult`。
- 不同角色仅通过 `allowed_tools` 与 `build_prompt` 区分。

### 4.3 `k_search/kernel_generators/agents/code_reader_agent.py`(新增)

`CodeReaderAgent(ProjectAgent)`:
- `allowed_tools = [Read, Grep, Glob, Write]`(**禁 Edit/Bash**,不得改源码;只写 `CODE_MAP.md`)。
- `build_prompt`:指示其只读理解项目,按固定模板产出 `CODE_MAP.md`,内容包括:
  - 文件清单 + 每个文件职责(kernel / host tiling / InferShape / 测试 harness / build)
  - 算子入口与调用链、host→kernel 契约
  - kernel 关键设计:分核策略、tiling 公式、buffer 分配、流水线(CopyIn/Compute/CopyOut)
  - 不变量/约束(语义、对齐、build 布局)
  - 篇幅上限(`KSEARCH_CODE_MAP_MAX_CHARS`,默认 ~8000),避免记忆反噬 prompt 预算。
- orchestrator 在其运行后用 `store.read_from_worktree(CODE_MAP, project_dir)` 取回文本。

### 4.4 `k_search/kernel_generators/agents/codegen_agent.py`(重构)

由现有 `AscendCAgenticPromptBuilder` + `editor_client.edit_project` 调用重构而成的 `CodegenAgent(ProjectAgent)`:
- `allowed_tools = [Read, Grep, Glob, Edit, Write]`(同现状 `DEFAULT_PROJECT_EDITOR_TOOLS`)。
- `build_prompt`:迁移 `AscendCAgenticPromptBuilder.build()` 全部逻辑(definition/action/perf/trace 分段、`max_chars` 校验、路径消毒),并按 §6 增加 code_map 分支。
- 现有 `AscendCAgenticCodegenRequest` 作为其 `context`,保持字段不变,降低改动面。

### 4.5 `AscendCAgenticCodegenRunner`(瘦身为编排器)

`run()` 编排顺序(详见 §5):
1. `create_agentic_worktree` + overlay baseline(不变)。
2. `store = MemoryStore(artifacts, task)`;`text = store.load(CODE_MAP)`。
3. 若 `text is None`:运行 `CodeReaderAgent` → `store.read_from_worktree` 取回 → `store.save(CODE_MAP, text)`(首次基线,无需门控)。
4. `store.materialize(CODE_MAP, project_dir)`(把记忆写进 worktree 供 codegen 读)。
5. 运行 `CodegenAgent`(prompt 含 code_map 提示)。
6. eval、snapshot、artifacts(不变)。
7. `result.code_map_text = store.read_from_worktree(CODE_MAP, project_dir)`(透传给调用方做门控回写)。

**预留**:`plan_agent.py` / `review_agent.py` 不在本期实现;新增角色 = 一个 `ProjectAgent` 子类 + 一个 `MemoryKind`,orchestrator 增加一处调用点即可。

## 5. 数据流与回写门控

**baseline 首轮**
```
overlay baseline → load(CODE_MAP)=None
  → CodeReaderAgent 扫项目产出 CODE_MAP.md → save(CODE_MAP)   # 基线即真相,立即落盘
  → materialize 进 worktree
  → CodegenAgent: 读 CODE_MAP.md + 改代码 + 更新 CODE_MAP.md
  → eval → result.code_map_text 透传
```

**优化轮 N**
```
overlay parent-best → load(CODE_MAP)=<已有>
  → materialize 进 worktree                                    # 跳过全项目扫描
  → CodegenAgent: 读 CODE_MAP.md + 改代码 + 更新 CODE_MAP.md
  → eval → result.code_map_text 透传
```

**回写门控**(防坏改污染记忆):
- `run()` 只把更新后的记忆放进 `AscendCAgenticCodegenResult.code_map_text`,**不直接回写**。
- 调用方决定回写,复用现有 best 判定(`kernel_generator_world_model.py:734` 的 `all_passed and round_score > best_score`):成为新 best 才 `store.save(CODE_MAP, result.code_map_text)`。
- baseline 路径:reader 产出后已即时落盘(§5 首轮),后续 codegen 的更新同样走门控。
- 失败/劣化 attempt 的 code_map 改动被丢弃。

## 6. Prompt 设计

`CodegenAgent.build_prompt` 增加开关分支:
- **有 code_map**:把 *"First inspect the project with Glob, Grep, and Read"* 替换为:
  > "A `CODE_MAP.md` at the project root describes file roles, kernel structure, tiling, buffers, and contracts. Read it first instead of grepping the whole project. After editing code, update the affected sections of `CODE_MAP.md` to keep it accurate."
- **无 code_map(降级)**:保留原措辞,行为与现状一致。

`CodeReaderAgent.build_prompt`:只读理解 + 固定模板产出 `CODE_MAP.md`(见 §4.3),不得修改任何源码。

## 7. 边界处理与错误降级

1. **CODE_MAP.md 不污染算子改动**:
   - `.md` 不在 `_is_source_candidate` 集合内 → 不会被 `_collect_project_sources` 采集进 `Solution`(已确认,无需改 `make_solution_from_project_dir`)。
   - 但 agent 改它会进 `changed_paths`:`run()` 在计算 `project_changed_paths` 与 "did not change any files" 检测、以及 `project_diff_text` 时**显式过滤 `CODE_MAP.md`**,避免"只改了记忆没改代码"被误判为有效改动,也避免污染 diff/snapshot 比较。
2. **CodeReaderAgent 失败/超时**:不阻塞主流程,降级为"无 code_map"(走原 inspect 分支),打 warn。
3. **记忆读回为空/异常**:不回写,保留旧版本。
4. **总开关** `KSEARCH_ENABLE_CODE_MAP`(默认开):关闭后完全回到现状,行为零变化。
5. **作用域**:仅 `AscendCAgenticCodegenRunner` 内生效;non-agentic / 非 AscendC 路径不受影响。

## 8. 配置开关

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `KSEARCH_ENABLE_CODE_MAP` | on | 总开关,关闭则行为同现状 |
| `KSEARCH_CODE_MAP_MAX_CHARS` | 8000 | code_map 篇幅上限 |
| `KSEARCH_CODE_MAP_FILENAME` | `CODE_MAP.md` | worktree 内记忆文件名(覆盖默认) |

## 9. 测试计划

- `memory_store.py`:load/save/materialize/read_from_worktree(tmp dir;含不存在、空、覆盖、任意 kind 通用性)。
- `project_agent.py` / `code_reader_agent.py` / `codegen_agent.py`:mock `editor_client`,验证 tools 集合、prompt 分支(有/无 code_map)、reader 禁 Edit。
- `AscendCAgenticCodegenRunner`:mock 两个 agent + editor_client,验证
  - (a) 首轮触发 reader 且 save;
  - (b) 后续轮 materialize、不再触发 reader;
  - (c) 采纳才回写、劣化不回写;
  - (d) reader 失败时降级;
  - (e) 仅改 CODE_MAP.md 不计入有效 changed_paths。
- 更新现有 `tests/kernel_generators/test_ascendc_agentic_codegen.py`(prompt builder 已迁移至 `CodegenAgent`)。

## 10. 落地步骤(渐进、保持可运行)

1. 新增 `memory/memory_store.py` + 单测(独立,无侵入)。
2. 新增 `agents/project_agent.py` 基类 + 单测。
3. 把 `AscendCAgenticPromptBuilder` 迁移为 `agents/codegen_agent.py:CodegenAgent`;`ascendc_agentic_codegen.py` 改为引用它(保留旧名薄封装或更新导入)。回归现有测试。
4. 新增 `agents/code_reader_agent.py` + 单测。
5. 在 `AscendCAgenticCodegenRunner.run()` 接入 memory + reader(总开关默认开)。
6. 在两处调用方(`kernel_generator.py:427`、`kernel_generator_world_model.py:689` 上下文)接入门控回写。
7. 更新/补齐测试,全量回归。

## 11. 风险

- **记忆漂移**:codegen agent 未如实更新 CODE_MAP.md 导致与代码不符。缓解:门控只在采纳时回写 + 篇幅上限 + prompt 明确要求更新;后续可引入 `review_agent` 校验。
- **首轮额外成本**:baseline 多一次 reader LLM 调用。权衡:一次性投入换后续多轮省去全项目扫描,净省。可用总开关关闭。
- **记忆质量依赖 reader**:reader 模板需迭代;先用固定模板,后续按实测调整字段。
