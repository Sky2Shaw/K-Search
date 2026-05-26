设计文档：K-Search LLM Runtime Observability & Cost Accounting

1. 背景

K-Search 当前已经具备较清晰的主架构：

Task
  → Generator
  → LLMClient / Claude Agent SDK
  → Solution
  → EvalResult
  → World Model / SolutionDB / Artifacts

当前仓库已有基础 LLM 日志能力：llm_clients.py 会记录 prompt、response、error，并通过 llm_log_context() 按 operator、flow、round、stage 等上下文组织日志。 ￼

但在 Claude Agent SDK 的 agentic codegen 场景中，K-Search 当前主要保存：

prompt
transcript
changed_paths
diff_text
project_path
final solution

ClaudeAgentProjectEditorClient 现在只是从 client.receive_response() 中提取文本，拼接成 transcript，没有结构化记录 tool use / tool result / session_id / usage / cost。 ￼

这导致实际使用时只能看到：

Claude 最终改了哪些代码

但看不到：

Claude 读了哪些文件
grep 了哪些关键字
先后调用了哪些工具
每次工具调用是否成功
花了多少 token / cost / turns
哪个 action 消耗了多少预算

Claude Agent SDK 本身支持这些信息：Python SDK 的 query() 返回 message stream；ClaudeSDKClient 适合连续会话和 response-driven logic；ToolUseBlock 包含 tool id、tool name、input；ToolResultBlock 包含 tool_use_id、content、is_error；ResultMessage 包含 duration、num_turns、session_id、total_cost_usd、usage、model_usage 等字段。 ￼

因此，本设计目标是：在不破坏当前架构的前提下，新增独立、可扩展的 LLM runtime observability 和 cost accounting 能力，并为后续 action token budget 分配打基础。

⸻

2. 设计目标

2.1 功能目标

新增一套独立 telemetry 模块，支持：

1. 记录每次 LLM 调用的 runtime trace
2. 记录 Claude Agent SDK 的 tool timeline
3. 记录 token / cost / latency / turns / session_id
4. 将 cost 归因到 action / round / stage / model / task
5. 输出可读 Markdown timeline 和机器可读 JSONL
6. 为后续 action token budget / cost budget 分配预留接口

2.2 架构目标

必须满足：

1. 不改 Task / Solution / EvalResult 的核心语义
2. 不让 Generator 直接依赖 Claude SDK 的具体 message 类型
3. 不把 telemetry 写死在 ClaudeAgentProjectEditorClient 里
4. 新模块职责单一，可替换，可扩展
5. OpenAI-compatible 和 Claude Agent SDK 都能复用 cost accounting
6. 失败时 telemetry 不影响主流程

2.3 非目标

本期不做：

1. 完整 dashboard
2. 分布式 tracing backend
3. 多用户权限系统
4. 自动 token budget 调度策略
5. 复杂成本预测模型

本期只把数据采集、落盘、归因、摘要做好。

⸻

3. 总体架构

新增模块建议放在：

k_search/telemetry/
  ├── __init__.py
  ├── context.py
  ├── events.py
  ├── recorder.py
  ├── cost.py
  ├── sinks.py
  ├── claude_sdk_adapter.py
  ├── openai_adapter.py
  ├── reports.py
  └── budget.py

整体关系：

KernelGenerator / WorldModelGenerator
        │
        │ llm_log_context / telemetry_context
        ▼
LLMClient / ClaudeAgentProjectEditorClient
        │
        │ emit runtime events
        ▼
TelemetryRecorder
        │
        ├─ JsonlSink
        ├─ MarkdownTimelineSink
        ├─ CostLedgerSink
        └─ Future: SQLite / W&B / OTEL

核心原则：

业务模块只 emit event
telemetry 模块负责结构化、聚合、落盘、报告

⸻

4. 核心抽象

4.1 TelemetryContext

文件：k_search/telemetry/context.py

用于描述一次 LLM 调用属于哪个任务、哪个 action、哪个 round。

from dataclasses import dataclass
from typing import Any
@dataclass(frozen=True)
class TelemetryContext:
    run_id: str | None = None
    task_name: str | None = None
    definition: str | None = None
    flow: str | None = None          # world_model / baseline / agentic_codegen
    stage: str | None = None         # world_model_init / codegen / debug / improve
    round_index: int | None = None
    attempt_index: int | None = None
    action_node_id: str | None = None
    action_title: str | None = None
    parent_solution_id: str | None = None
    solution_id: str | None = None
    model_name: str | None = None
    provider: str | None = None
    target_gpu: str | None = None
    language: str | None = None
    extra: dict[str, Any] | None = None

设计原因：

当前 llm_log_context() 已经有 operator、flow、round、stage 等概念。新模块可以兼容它，但扩展出 action 归因字段。 ￼

未来 action token budget 要依赖：

action_node_id
round_index
attempt_index
model_name
usage
cost
score improvement

⸻

4.2 TelemetryEvent

文件：k_search/telemetry/events.py

统一事件模型。

from dataclasses import dataclass, field
from typing import Any
import time
import uuid
@dataclass
class TelemetryEvent:
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    event_type: str = ""             # llm_start / assistant_text / tool_use / tool_result / llm_result / llm_error
    context: dict[str, Any] = field(default_factory=dict)
    provider: str | None = None
    model_name: str | None = None
    message_id: str | None = None
    session_id: str | None = None
    tool_use_id: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_result_excerpt: str | None = None
    is_error: bool | None = None
    text_excerpt: str | None = None
    raw_type: str | None = None
    usage: dict[str, Any] | None = None
    model_usage: dict[str, Any] | None = None
    total_cost_usd: float | None = None
    duration_ms: int | None = None
    duration_api_ms: int | None = None
    num_turns: int | None = None
    stop_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None

事件类型建议：

llm_start
llm_prompt
assistant_text
assistant_thinking_metadata
tool_use
tool_result
system_message
rate_limit
llm_result
llm_error
llm_end

注意：不要记录模型私有 chain-of-thought。ThinkingBlock 可以记录为 metadata：

{
  "event_type": "assistant_thinking_metadata",
  "has_thinking": true,
  "thinking_chars": 1234
}

不要落盘 thinking 原文。

⸻

4.3 CostRecord

文件：k_search/telemetry/cost.py

用于 cost accounting。

from dataclasses import dataclass, field
from typing import Any
@dataclass
class CostRecord:
    run_id: str | None
    task_name: str | None
    action_node_id: str | None
    stage: str | None
    round_index: int | None
    attempt_index: int | None
    provider: str
    model_name: str
    prompt_chars: int | None = None
    response_chars: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    duration_ms: int | None = None
    duration_api_ms: int | None = None
    num_turns: int | None = None
    raw_usage: dict[str, Any] | None = None
    raw_model_usage: dict[str, Any] | None = None

Claude SDK 的 ResultMessage 已经提供 total_cost_usd、usage、model_usage、duration_ms、num_turns 等字段。 ￼

OpenAI-compatible 的 usage 来源不同，后续由 openai_adapter.py 转成统一 CostRecord。

⸻

5. Recorder 与 Sink 设计

5.1 TelemetryRecorder

文件：k_search/telemetry/recorder.py

职责：

1. 接收事件
2. 追加内存 buffer
3. 转发给 sinks
4. 失败不影响主流程
5. 提供 summary

接口：

class TelemetryRecorder:
    def __init__(self, sinks: list[TelemetrySink], context: TelemetryContext):
        self.sinks = sinks
        self.context = context
        self.events: list[TelemetryEvent] = []
    def emit(self, event: TelemetryEvent) -> None:
        event.context = {**self.context_dict(), **(event.context or {})}
        self.events.append(event)
        for sink in self.sinks:
            try:
                sink.write_event(event)
            except Exception:
                pass
    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception:
                pass

5.2 Sink 接口

文件：k_search/telemetry/sinks.py

class TelemetrySink(Protocol):
    def write_event(self, event: TelemetryEvent) -> None: ...
    def close(self) -> None: ...

第一期实现三个 sink：

JsonlSink:
  写 agent_trace.jsonl
MarkdownTimelineSink:
  写 tool_timeline.md
CostLedgerSink:
  写 cost_ledger.jsonl

后续扩展：

SQLiteSink
WandBSink
OpenTelemetrySink
PrometheusSink

⸻

6. 落盘目录设计

建议所有 telemetry 写到 artifacts 下：

<artifacts>/<task_name>/telemetry/
  <run_id>/
    round_0001/
      action_<node_id>/
        attempt_0001/
          prompt.md
          response.md
          agent_trace.jsonl
          tool_timeline.md
          cost.json
          diff.patch
          changed_files.json

对非 agentic LLM 调用：

<artifacts>/<task_name>/telemetry/
  <run_id>/
    world_model/
      round_0000_world_model_init/
        prompt.md
        response.md
        cost.json

这样和当前 llm_log_context() 的组织方式保持一致，但更细粒度。当前 LLM 日志已经按 run/operator/flow/round/stage 组织，新设计只是把它正式化和扩展化。 ￼

⸻

7. Claude Agent SDK Adapter

文件：k_search/telemetry/claude_sdk_adapter.py

职责：

把 Claude SDK message/block/result 转成 TelemetryEvent 和 CostRecord。

核心函数：

def event_from_claude_message(message: Any) -> list[TelemetryEvent]:
    ...

伪代码：

def event_from_claude_message(message: Any) -> list[TelemetryEvent]:
    events = []
    if is_assistant_message(message):
        for block in message.content:
            if is_tool_use_block(block):
                events.append(TelemetryEvent(
                    event_type="tool_use",
                    raw_type=type(block).__name__,
                    tool_use_id=block.id,
                    tool_name=block.name,
                    tool_input=sanitize_tool_input(block.name, block.input),
                ))
            elif is_tool_result_block(block):
                events.append(TelemetryEvent(
                    event_type="tool_result",
                    raw_type=type(block).__name__,
                    tool_use_id=block.tool_use_id,
                    tool_result_excerpt=truncate(str(block.content), 4000),
                    is_error=block.is_error,
                ))
            elif is_text_block(block):
                events.append(TelemetryEvent(
                    event_type="assistant_text",
                    raw_type=type(block).__name__,
                    text_excerpt=truncate(block.text, 4000),
                ))
            elif is_thinking_block(block):
                events.append(TelemetryEvent(
                    event_type="assistant_thinking_metadata",
                    raw_type=type(block).__name__,
                    text_excerpt=f"thinking_chars={len(block.thinking or '')}",
                ))
    elif is_system_message(message):
        events.append(TelemetryEvent(
            event_type="system_message",
            raw_type=type(message).__name__,
            text_excerpt=truncate(str(message.data), 4000),
        ))
    elif is_result_message(message):
        events.append(TelemetryEvent(
            event_type="llm_result",
            raw_type=type(message).__name__,
            session_id=message.session_id,
            duration_ms=message.duration_ms,
            duration_api_ms=message.duration_api_ms,
            num_turns=message.num_turns,
            total_cost_usd=message.total_cost_usd,
            usage=message.usage,
            model_usage=message.model_usage,
            stop_reason=message.stop_reason,
            is_error=message.is_error,
            text_excerpt=truncate(str(message.result or ""), 4000),
        ))
    return events

Claude 官方文档里 ToolUseBlock / ToolResultBlock 的字段正好支持这个转换，且官方示例也展示了在 receive_response() 中实时识别 tool use / tool result。 ￼

⸻

8. 与现有 ClaudeAgentProjectEditorClient 的集成

当前 edit_project() 中已有：

async with claude_agent_sdk.ClaudeSDKClient(options=options) as client:
    await client.query(prompt_text)
    async for message in client.receive_response():
        ...

它现在只提取文本。 ￼

改造后：

async with claude_agent_sdk.ClaudeSDKClient(options=options) as client:
    recorder.emit(llm_start_event(...))
    await client.query(prompt_text)
    async for message in client.receive_response():
        for event in event_from_claude_message(message):
            recorder.emit(event)
        text = ClaudeAgentLLMClient._extract_message_text(message)
        if text:
            chunks.append(text)
            if hasattr(message, "result"):
                final_text = text
    recorder.emit(llm_end_event(...))

ClaudeProjectEditResult 扩展：

@dataclass
class ClaudeProjectEditResult:
    text: str
    transcript: str
    prompt: str
    prompt_chars: int
    prompt_lines: int
    # new
    trace_path: str | None = None
    timeline_path: str | None = None
    cost_path: str | None = None
    session_id: str | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    model_usage: dict[str, Any] | None = None
    num_turns: int | None = None
    duration_ms: int | None = None

兼容性：

原有字段不变
新增字段均为 optional
调用方不改也能运行

⸻

9. 与 AscendCAgenticCodegenResult 的集成

当前 result 包含：

solution
raw
cleaned
transcript
prompt
prompt_chars
changed_paths
diff_text
project_path

AscendCAgenticCodegenRunner 负责创建 worktree、调用 editor、收集 changed paths、diff、构造 solution。 ￼

扩展为：

@dataclass
class AscendCAgenticCodegenResult:
    solution: Solution
    raw: str
    cleaned: dict[str, str]
    transcript: str
    prompt: str
    prompt_chars: int
    changed_paths: list[str]
    diff_text: str
    project_path: str
    # new
    trace_path: str | None = None
    timeline_path: str | None = None
    cost_path: str | None = None
    session_id: str | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    num_turns: int | None = None
    duration_ms: int | None = None

并在 runner 输出日志中增加：

session_id
total_cost_usd
num_turns
trace_path

⸻

10. 与 llm_clients.py 的集成

OpenAICompatibleLLMClient.generate() 和 ClaudeAgentLLMClient.generate() 都应该接入同一个 telemetry recorder。

当前 llm_clients.py 已经有 _log_llm_interaction() 和 llm_log_context()。 ￼

为了不破坏现有设计，建议：

保留 _log_llm_interaction()
新增 telemetry_recorder 可选旁路

最小改动：

with telemetry_span(context=..., provider="openai", model_name=self.model_name) as rec:
    rec.emit_prompt(prompt)
    response = self.client.responses.create(...)
    rec.emit_openai_response(response)

如果没有启用 telemetry：

telemetry_span 返回 NoopRecorder

⸻

11. OpenAI-Compatible Cost Adapter

文件：k_search/telemetry/openai_adapter.py

OpenAI-compatible 响应格式不完全一致，所以 adapter 要宽容：

def cost_from_openai_response(response: Any) -> CostRecord:
    usage = getattr(response, "usage", None)
    if usage is None and hasattr(response, "model_dump"):
        usage = response.model_dump().get("usage")
    ...

兼容字段：

input_tokens
output_tokens
total_tokens
prompt_tokens
completion_tokens
cached_tokens
reasoning_tokens

注意：第三方 OpenAI-compatible provider 的 usage 字段不稳定，所以 adapter 必须 best-effort，不可影响主流程。

⸻

12. Cost Ledger

文件：k_search/telemetry/cost.py

每次 LLM 调用结束后写一条：

{
  "run_id": "20260526_xxx",
  "task_name": "sfa_gqa",
  "action_node_id": "n12",
  "stage": "agentic_codegen",
  "round_index": 7,
  "attempt_index": 2,
  "provider": "claude-agent",
  "model_name": "claude-sonnet-4-6",
  "input_tokens": 18234,
  "output_tokens": 2401,
  "total_tokens": 20635,
  "total_cost_usd": 0.084,
  "duration_ms": 128000,
  "num_turns": 6,
  "session_id": "..."
}

并维护聚合文件：

cost_summary.json

示例：

{
  "total_cost_usd": 3.42,
  "total_input_tokens": 820000,
  "total_output_tokens": 93000,
  "by_action": {
    "n12": {
      "cost_usd": 0.71,
      "tokens": 120000,
      "attempts": 5,
      "best_score": 1.08
    }
  },
  "by_stage": {
    "world_model_init": {"cost_usd": 0.12},
    "agentic_codegen": {"cost_usd": 2.81},
    "world_model_refine": {"cost_usd": 0.49}
  }
}

⸻

13. 为 action token budget 做准备

本期只记录，不做策略。但要预留 budget.py。

13.1 Budget 输入

未来为 action 分配预算需要：

action_node_id
action_score
difficulty
expected_vs_baseline_factor
parent_score
historical_success_rate
historical_token_cost
historical_cost_per_success
remaining_global_budget
round_index

13.2 Budget 接口

文件：k_search/telemetry/budget.py

@dataclass
class BudgetRequest:
    action_node_id: str
    stage: str
    round_index: int
    attempt_index: int
    model_name: str
    difficulty_1_to_5: int | None = None
    action_score_0_to_1: float | None = None
    expected_vs_baseline_factor: float | None = None
    historical_cost_usd: float | None = None
    historical_tokens: int | None = None
    remaining_run_budget_usd: float | None = None
    remaining_run_tokens: int | None = None
@dataclass
class BudgetDecision:
    max_prompt_chars: int | None = None
    max_turns: int | None = None
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    reason: str = ""
class BudgetPolicy(Protocol):
    def decide(self, request: BudgetRequest) -> BudgetDecision: ...

第一期实现：

NoopBudgetPolicy
StaticBudgetPolicy

例如：

StaticBudgetPolicy(
    default_max_prompt_chars=20000,
    default_max_turns=8,
    default_max_cost_usd=0.25,
)

当前 AscendCAgenticPromptBuilder 已有 KSEARCH_AGENTIC_PROMPT_MAX_CHARS，后续可以从 BudgetDecision 传入，而不是只靠环境变量。 ￼

⸻

14. Hook 机制设计

Claude SDK 支持 hooks，例如 PreToolUse、PostToolUse，并支持 HookMatcher 匹配特定工具或工具模式。官方文档示例展示了可以在 ClaudeAgentOptions 中配置 PreToolUse、PostToolUse、UserPromptSubmit 等 hooks。 ￼

本设计建议第二阶段引入 hooks，第一阶段先从 message stream 采集。

14.1 第一阶段：message stream 观测

优点：

实现简单
不改变 Claude 行为
风险低

缺点：

只能观测，不能拦截

14.2 第二阶段：hook-based policy gate

新增：

k_search/telemetry/claude_hooks.py

支持：

PreToolUse:
  记录工具调用
  检查 Edit/Write 路径是否允许
  检查 Read 文件大小/路径
PostToolUse:
  记录工具结果
  统计错误
PostToolUseFailure:
  记录失败原因

这可以和未来的 ksearch_project.yaml 编辑策略联动。

⸻

15. Markdown Timeline 报告

文件：k_search/telemetry/reports.py

每次 agentic codegen 输出：

# LLM Runtime Trace
## Summary
- task: sfa_gqa
- action_node_id: n12
- round: 7
- attempt: 2
- provider: claude-agent
- model: claude-sonnet-4-6
- session_id: xxx
- duration_ms: 128000
- num_turns: 6
- total_cost_usd: 0.084
## Changed Files
- op_kernel/sfa.h
- op_host/tiling.cpp
## Tool Timeline
| Step | Event | Tool | Target | Result |
|---:|---|---|---|---|
| 1 | tool_use | Glob | `**/*.{h,cpp}` | - |
| 2 | tool_use | Grep | `DataCopy` | - |
| 3 | tool_use | Read | `op_kernel/sfa.h` | ok |
| 4 | tool_use | Edit | `op_kernel/sfa.h` | ok |
| 5 | tool_use | Read | `op_host/tiling.cpp` | ok |
| 6 | tool_use | Edit | `op_host/tiling.cpp` | ok |
## Assistant Text
...

这个报告解决实际痛点：

不只知道 Claude 改了什么，
还能知道它怎么定位、读了哪些文件、为什么可能这样改。

⸻

16. 与 World Model 的关系

本期不改 world model schema，但 telemetry 会记录：

action_node_id
chosen_action_text hash
round_index
attempt_index
solution_id
eval score
cost
tokens
duration

未来可以把这些聚合成 action 统计：

{
  "action_node_id": "n12",
  "attempts": 5,
  "passed_attempts": 1,
  "total_cost_usd": 0.71,
  "total_tokens": 120000,
  "best_score": 1.08,
  "cost_per_passed_solution": 0.71,
  "tokens_per_score_gain": 60000
}

然后用于 action selection：

高收益低成本 action 优先
高难度高成本 action 降权
连续高消耗无收益 action 提前停止

这与当前 WorldModelSelectionPolicy 并不冲突。当前 selection policy 已有 score、difficulty、depth、parent_quality 等权重；未来可以新增 cost-aware weight。 ￼

⸻

17. 与 SolutionDB 的关系

当前 SolutionDB 保存 solution_id、parent_solution_id、eval_result、code、code_excerpt。 ￼

不要把大 trace 塞进 SolutionDB。建议只保存引用：

@dataclass
class SolutionRecord:
    ...
    telemetry_ref: str | None = None
    cost_ref: str | None = None

或者更轻量：

{
  "solution_id": "...",
  "telemetry": {
    "trace_path": "...",
    "cost_path": "...",
    "total_cost_usd": 0.084,
    "total_tokens": 20635
  }
}

这样 SolutionDB 仍保持轻量，telemetry 独立扩展。

⸻

18. 配置设计

新增环境变量：

KSEARCH_TELEMETRY=1
KSEARCH_TELEMETRY_DIR=<path>
KSEARCH_TELEMETRY_MAX_TEXT_CHARS=4000
KSEARCH_TELEMETRY_RECORD_TOOL_RESULTS=1
KSEARCH_TELEMETRY_RECORD_ASSISTANT_TEXT=1
KSEARCH_TELEMETRY_RECORD_THINKING=0
KSEARCH_COST_LEDGER=1
KSEARCH_RUN_BUDGET_USD=10.0
KSEARCH_RUN_BUDGET_TOKENS=2000000

新增 CLI 参数：

--telemetry
--telemetry-dir
--cost-ledger
--run-budget-usd
--run-budget-tokens

默认建议：

telemetry 默认开启 lightweight 模式
tool_result 默认截断
thinking 原文永不记录
cost ledger 默认开启

⸻

19. 安全与隐私

必须做 sanitize：

1. tool_result 截断
2. 大文件内容不完整落盘
3. 环境变量值不落盘
4. API key / token / secret 脱敏
5. thinking block 不保存原文
6. benchmark/test 输出可配置脱敏

脱敏函数：

SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9_-]+",
    r"ANTHROPIC_API_KEY=[^\\s]+",
    r"LLM_API_KEY=[^\\s]+",
    r"AKIA[0-9A-Z]{16}",
]
def redact(text: str) -> str:
    ...

⸻

20. 实施计划

Phase 1：最小可用观测

新增文件：

k_search/telemetry/context.py
k_search/telemetry/events.py
k_search/telemetry/recorder.py
k_search/telemetry/sinks.py
k_search/telemetry/claude_sdk_adapter.py
k_search/telemetry/cost.py
k_search/telemetry/reports.py

改动点：

1. claude_agent_project_editor.py 接入 recorder
2. AscendCAgenticCodegenResult 增加 trace_path/cost/session 字段
3. 每次 agentic codegen 输出 agent_trace.jsonl + tool_timeline.md + cost.json

验收标准：

能看到 Claude 调用 Read/Grep/Glob/Edit/Write 的顺序
能看到每次工具 input
能看到 result session_id / duration / cost / usage
不影响现有 agentic codegen 成功率

⸻

Phase 2：普通 LLM 调用 cost 统一

改动：

1. OpenAICompatibleLLMClient 接入 cost adapter
2. ClaudeAgentLLMClient prompt-to-text 路径接入 telemetry
3. world model init/propose/refine 记录 cost
4. 输出 run-level cost_summary.json

验收标准：

一次完整 K-Search run 能统计：
- world model 花了多少钱
- codegen 花了多少钱
- 每个 action 花了多少钱
- 每轮 token/cost 明细

⸻

Phase 3：Budget 预留接口

改动：

1. 新增 budget.py
2. AscendCAgenticPromptBuilder 支持外部传入 max_chars
3. ClaudeAgentOptions 支持 budget decision 的 max_turns
4. 先实现 StaticBudgetPolicy

验收标准：

可以通过配置限制：
- 单 action 最大 prompt chars
- 单 action 最大 turns
- 单 run 最大 cost

⸻

Phase 4：Hook-based policy gate

改动：

1. claude_hooks.py
2. PreToolUse/PostToolUse 记录
3. Edit/Write path whitelist
4. 大文件 Read 拦截

验收标准：

Claude 尝试修改 forbidden path 时被阻止
timeline 中记录 block 原因
不影响合法 Edit/Write

⸻

21. 关键代码示例

21.1 在 ClaudeAgentProjectEditorClient 中接入

recorder = build_telemetry_recorder(
    context=TelemetryContext(
        task_name=task_name,
        flow="agentic_codegen",
        stage=mode,
        round_index=round_num,
        attempt_index=attempt_idx,
        action_node_id=action_node_id,
        provider="claude-agent",
        model_name=self.model_name,
    )
)
recorder.emit(TelemetryEvent(
    event_type="llm_start",
    provider="claude-agent",
    model_name=self.model_name,
    text_excerpt=f"prompt_chars={len(prompt_text)}",
))
async with claude_agent_sdk.ClaudeSDKClient(options=options) as client:
    await client.query(prompt_text)
    async for message in client.receive_response():
        for event in event_from_claude_message(message):
            recorder.emit(event)
        text = ClaudeAgentLLMClient._extract_message_text(message)
        if text:
            chunks.append(text)
            if hasattr(message, "result"):
                final_text = text
recorder.close()

⸻

21.2 tool_timeline.md 示例

# Agentic Codegen Timeline
- task: sfa_gqa
- action_node_id: n12
- round: 3
- attempt: 2
- session_id: 550e8400-e29b-41d4-a716-446655440000
- total_cost_usd: 0.084
- num_turns: 6
| Step | Type | Tool | Target | Status |
|---:|---|---|---|---|
| 1 | tool_use | Glob | `**/*.cpp` | - |
| 2 | tool_use | Grep | `CalcAccumOffset` | - |
| 3 | tool_use | Read | `op_kernel/sparse_flash_attention.h` | - |
| 4 | tool_result | Read | - | ok |
| 5 | tool_use | Edit | `op_kernel/sparse_flash_attention.h` | - |
| 6 | tool_result | Edit | - | ok |

⸻

22. 设计取舍

为什么不直接扩展 _log_llm_interaction()？

因为 _log_llm_interaction() 是 prompt/response 日志，不适合承载 runtime event stream。

它适合：

一次调用一条 prompt/response

不适合：

一次调用内有几十个 tool_use/tool_result

所以应保留它，同时新增 telemetry event stream。

⸻

为什么不把 telemetry 写进 EvalResult？

因为 EvalResult 表示 benchmark 结果，不应该承载 LLM 运行时轨迹。

正确关系是：

Solution:
  代码快照
EvalResult:
  评测结果
TelemetryTrace:
  LLM 生成这个 Solution 的过程
CostRecord:
  生成这个 Solution 的成本

四者通过 solution_id / action_node_id / round_index 关联。

⸻

为什么第一阶段不用 hooks？

因为 hooks 会改变执行路径，并可能引入阻断逻辑。第一阶段只做 message stream 观测，风险最低。

第二阶段再上 hooks，实现 policy gate。

⸻

23. 最终效果

改造后，每次 action attempt 都能回答：

这个 action 花了多少钱？
用了多少 token？
Claude 读了哪些文件？
grep 了哪些关键词？
改了哪些文件？
工具调用是否失败？
用了几轮？
哪个 session_id？
最后 diff 是什么？
这次 cost 和 score improvement 是否划算？

这为后续两个能力打基础：

1. 可观测 agent runtime
2. cost-aware action scheduling / token budget allocation

⸻

24. 一句话总结

本设计的核心是：

不要把观测逻辑塞进 Generator 或 Task；
新增独立 telemetry 层，以 event stream 的方式旁路采集 LLM 运行轨迹和成本；
通过 action_node_id / round / attempt / solution_id 做归因；
先观测，再治理，最后做 budget policy。

这样既不破坏 K-Search 当前架构，又能把 Claude agentic codegen 从“黑盒改代码”升级为“可审计、可归因、可预算”的工业级 agent runtime。



对，这个诉求比“实时观测”更重要。实时观测解决的是“我现在知道 agent 在干什么”；你真正要的是：

LLM 运行记录成为 K-Search 的长期资产
→ 可复盘
→ 可诊断
→ 可沉淀经验
→ 可反哺 action 选择、prompt、world model、budget policy

所以设计上不能只做 agent_trace.jsonl，还要把 trace 转成可查询、可统计、可学习的实验事实库。

我建议把上一版设计升级为：

LLM Runtime Observability
  → Runtime Trace
LLM Experience Mining
  → Replay / Diagnosis / Learning / Policy Improvement

⸻

1. 设计目标升级

原来的目标是：

看见 LLM 运行轨迹
记录 token/cost
为 action 分配 token budget 准备数据

现在要升级为：

1. 每次 LLM 运行都可复盘
2. 每次失败都能定位到失败阶段
3. 每次 action 尝试都能评价投入产出比
4. 每次成功都能沉淀为优化经验
5. 每类失败都能反哺 prompt / world model / policy
6. 长期运行后，K-Search 能越来越懂哪些 action 值得做

换句话说，trace 不只是日志，而是训练 K-Search 自身的经验数据。

⸻

2. 新增一个核心概念：Attempt Record

当前你有：

Action node
Solution
EvalResult
LLM trace
Cost record
Diff

但它们是分散的。

建议新增一个逻辑实体：

Attempt Record = 一次 action attempt 的完整事实记录

它把一次 LLM 改代码尝试完整串起来：

action_node_id
→ prompt
→ LLM runtime trace
→ tool timeline
→ changed files
→ diff
→ generated solution
→ build/test/bench result
→ cost
→ diagnosis
→ learning summary

这是后续复盘、根因定位、经验挖掘的最小单位。

⸻

3. Attempt Record Schema

建议落盘为：

attempt_record.json

示例：

{
  "schema_version": "ksearch.attempt_record.v1",
  "identity": {
    "run_id": "20260526_001",
    "task_name": "sfa_gqa",
    "definition": "sparse_flash_attention",
    "round_index": 7,
    "attempt_index": 2,
    "action_node_id": "n12",
    "parent_solution_id": "abc123",
    "solution_id": "def456"
  },
  "action": {
    "title": "Reduce repeated GM reads in K/V loop",
    "description": "...",
    "difficulty_1_to_5": 3,
    "expected_vs_baseline_factor": 1.08,
    "chosen_reason": "high memory bandwidth impact"
  },
  "llm": {
    "provider": "claude-agent",
    "model_name": "claude-sonnet-4-6",
    "session_id": "...",
    "prompt_path": "prompt.md",
    "trace_path": "agent_trace.jsonl",
    "timeline_path": "tool_timeline.md",
    "duration_ms": 128000,
    "num_turns": 6,
    "input_tokens": 18234,
    "output_tokens": 2401,
    "total_cost_usd": 0.084
  },
  "code_change": {
    "changed_files": [
      "op_kernel/sfa.h",
      "op_host/tiling.cpp"
    ],
    "diff_path": "diff.patch",
    "diff_stats": {
      "files_changed": 2,
      "lines_added": 38,
      "lines_deleted": 12
    },
    "change_risk": {
      "touched_host_tiling": true,
      "touched_kernel": true,
      "touched_test_or_bench": false,
      "touched_build": false
    }
  },
  "evaluation": {
    "status": "failed",
    "failure_type": "precision_mismatch",
    "latency_ms": null,
    "score": -1.0,
    "workdir": "...",
    "eval_report_path": "eval.json"
  },
  "diagnosis": {
    "stage": "correctness",
    "suspected_surface": "tail/alignment",
    "root_cause_hypothesis": "Changed CopyOut tail path without preserving valid mask.",
    "confidence": 0.62,
    "evidence": [
      "test failed only on tail-heavy workload",
      "diff touched CopyOut boundary branch",
      "max_abs_diff appears near final block"
    ]
  },
  "learning": {
    "outcome": "failed",
    "lesson_type": "negative",
    "reusable_lesson": "When optimizing DataCopy/CopyOut tail path, preserve original mask logic before changing alignment.",
    "suggested_prompt_rule": "For AscendC tail/alignment changes, require explicit reasoning about valid element mask.",
    "suggested_world_model_update": "Lower confidence for aggressive tail-copy rewrites unless test coverage includes tail-heavy cases."
  }
}

这个 record 才是后续复盘和学习的核心。

⸻

4. 从 Runtime Trace 到 Postmortem Trace

实时 trace 关注“发生了什么”：

Claude Read 了哪个文件
Grep 了什么
Edit 了哪里
花了多少钱

复盘 trace 还要回答“为什么失败/成功”：

它是否读了正确文件？
它是否理解了 action？
它是否改动过大？
它是否跳过了关键上下文？
它是否修改了测试/bench？
它是否没有看到失败日志？
失败更可能是 action 方向错，还是实现错？
这个经验能不能沉淀？

所以需要新增一个模块：

k_search/analysis/
  ├── attempt_record.py
  ├── attempt_diagnoser.py
  ├── trace_summarizer.py
  ├── failure_classifier.py
  ├── lesson_miner.py
  ├── action_metrics.py
  └── replay.py

⸻

5. 核心模块设计

5.1 AttemptRecordBuilder

职责：把分散 artifact 合成一个 attempt record。

输入：

TelemetryTrace
CostRecord
Diff
Solution
EvalResult
WorldModel action node

输出：

attempt_record.json

接口：

class AttemptRecordBuilder:
    def build(
        self,
        *,
        context: TelemetryContext,
        action_node: dict,
        solution: Solution | None,
        eval_result: EvalResult | None,
        telemetry_refs: dict,
        diff_text: str | None,
        changed_paths: list[str],
    ) -> AttemptRecord:
        ...

它不负责诊断，只负责组装事实。

⸻

5.2 FailureClassifier

职责：把粗糙的 EvalResult.status 细化。

当前只有：

compile_failed
failed
benchmark_failed
timeout
passed

建议分类为：

codegen_failed
compile_failed
link_failed
runtime_failed
precision_mismatch
nan_or_inf
timeout
benchmark_failed
benchmark_unstable
latency_regression
no_code_change
policy_blocked
tool_error
provider_error

接口：

class FailureClassifier:
    def classify(
        self,
        *,
        eval_result: EvalResult | None,
        trace_events: list[TelemetryEvent],
        diff_text: str | None,
    ) -> FailureClassification:
        ...

输出：

{
  "failure_type": "precision_mismatch",
  "stage": "correctness",
  "confidence": 0.75,
  "evidence": [
    "status=failed",
    "correctness stdout contains max_abs_diff",
    "diff touched CopyOut"
  ]
}

⸻

5.3 TraceSummarizer

职责：把长 trace 变成短摘要，方便人读，也方便 world model 用。

输入：

agent_trace.jsonl
diff.patch
eval logs

输出：

## Agent Behavior Summary
Claude inspected:
- op_kernel/sfa.h
- op_host/tiling.cpp
Claude searched:
- DataCopy
- CopyOut
- blockIdx
Claude edited:
- CopyOut tail branch
- host tiling block size calculation
Risk:
- touched both host tiling and kernel CopyOut
- changed tail path
- did not inspect test oracle

这个摘要可以进入：

attempt_record.learning
HTML report
world model refine prompt
failure analysis report

⸻

5.4 AttemptDiagnoser

职责：对一次 attempt 做根因定位。

这可以先规则驱动，后面再 LLM 辅助。

输入：

AttemptRecord
trace summary
diff stats
failure classification

输出：

{
  "root_cause_category": "implementation_bug",
  "suspected_surface": "tail/alignment",
  "hypothesis": "The action direction is reasonable, but the implementation changed tail CopyOut mask incorrectly.",
  "recommended_next_step": "Continue same action, revert tail branch or add mask guard.",
  "should_continue_action": true,
  "should_lower_action_priority": false
}

这个非常关键，因为它能区分：

action 方向错了
vs
action 方向对，但实现失败

这直接影响 K-Search 的 action selection。

⸻

5.5 LessonMiner

职责：从成功/失败 attempt 中挖经验。

经验分三类：

Positive Lesson:
  某类 action 在某类 shape / operator 上有效
Negative Lesson:
  某类改法高风险，容易失败
Prompt Lesson:
  prompt 缺少某条约束，导致 LLM 经常犯错

输出：

{
  "lesson_id": "ascendc_tail_copyout_mask_001",
  "lesson_type": "negative",
  "scope": {
    "language": "ascendc",
    "surface": "tail/alignment",
    "operator_family": "attention"
  },
  "lesson": "Do not rewrite CopyOut tail branch unless valid mask logic is preserved.",
  "evidence_attempts": ["run1:n12:attempt2", "run3:n8:attempt4"],
  "suggested_prompt_rule": "Before changing tail/alignment code, explicitly preserve or re-derive valid element mask."
}

这些 lesson 后续可以进入：

knowledge base
prompt rules
world model priors
action ranking policy

⸻

6. 复盘视图设计

光有 JSON 不够，需要面向人的复盘报告。

建议每个 run 生成：

run_postmortem.md
run_postmortem.html

6.1 Run Summary

# K-Search Run Postmortem
## Overview
- task: sfa_gqa
- total rounds: 20
- total actions tried: 6
- passed solutions: 3
- best score: 1.12x
- total cost: $4.81
- total tokens: 1.92M
## Outcome
Best action:
- n7: Reduce repeated K/V GM reads
- score: 1.12x
- cost: $0.73
- attempts: 4
Worst cost sink:
- n12: Tail CopyOut rewrite
- cost: $1.21
- attempts: 6
- outcome: no passed solution

6.2 Action-level Report

## Action n12: Tail CopyOut rewrite
Outcome:
- failed
- attempts: 6
- total cost: $1.21
- failure type: precision_mismatch
Behavior:
- Claude repeatedly edited CopyOut tail branch
- Did not inspect golden comparison code
- Read op_kernel/sfa.h 4 times
- Changed host tiling and kernel in same attempt
Diagnosis:
- Action direction too broad
- Implementation likely broke valid mask handling
Recommendation:
- Split into two actions:
  1. Add tail-specific test diagnostics
  2. Optimize CopyOut only after preserving mask invariant

6.3 Failure Gallery

## Repeated Failure Patterns
1. Tail/alignment precision mismatch
   - attempts: 9
   - affected actions: n4, n12
   - suggested prompt rule: require explicit mask invariant
2. No code change
   - attempts: 3
   - likely cause: prompt too vague or project search failed
3. Compile error after CMake edit
   - attempts: 2
   - suggested policy: forbid CMake edits by default

⸻

7. 数据如何反哺 K-Search 能力

你最终要形成闭环：

Trace
→ AttemptRecord
→ Diagnosis
→ Lesson
→ Policy / Prompt / World Model update
→ Better next run

具体可以反哺 5 个地方。

⸻

7.1 反哺 action selection

当前 action selection 主要看：

score
difficulty
depth
parent_quality
overall_rating
confidence

未来可以加：

historical_success_rate
historical_cost_per_pass
historical_tokens_per_score_gain
failure_rate_by_surface

例如：

tail/alignment action 最近 10 次：
  success_rate = 10%
  avg_cost = $0.8
  common_failure = precision_mismatch
则后续降低直接优化 tail 的优先级，
或者要求先生成更细粒度 diagnostic action。

⸻

7.2 反哺 prompt

如果复盘发现 Claude 经常：

不读 host tiling
跳过 test failure detail
大范围重写
修改 benchmark
tail mask 犯错

则自动生成 prompt rule：

For AscendC tail/alignment changes:
- Do not change CopyOut tail branch unless preserving valid element mask.
- If correctness fails only on tail-heavy cases, inspect tail handling before any performance rewrite.

这些规则进入：

get_agentic_definition_text()
AscendCAgenticPromptBuilder
world_model action prompt

⸻

7.3 反哺 world model

失败不是简单 “too hard”。

应该区分：

action_bad:
  方向本身不值得继续
implementation_failed:
  方向可能对，但当前实现错
prompt_insufficient:
  LLM 没看到必要上下文
context_missing:
  项目摘要没包含关键文件
budget_insufficient:
  max turns/prompt 太小导致没读够

然后 world model 更新策略不同：

action_bad:
  降低 action rating
implementation_failed:
  保留 action，但生成更小子 action
context_missing:
  下一轮补充关键文件
budget_insufficient:
  提高该 action token budget
prompt_insufficient:
  加 prompt rule

⸻

7.4 反哺 budget policy

通过 cost ledger，可以统计：

不同 action 类型平均消耗
不同模型成功率
不同 difficulty 的 cost curve
不同失败类型的继续修复价值

例如：

Action Type: Host tiling rewrite
avg_tokens: 80k
success_rate: 35%
avg_score_gain_if_success: 1.08x
Action Type: Tail CopyOut rewrite
avg_tokens: 120k
success_rate: 8%
common_failure: precision_mismatch

Budget policy 可以做：

高成功率高收益 action:
  给更多 turns/token
低成功率高成本 action:
  限制尝试次数
  要求先做 diagnostic action

⸻

7.5 反哺知识库

把 LessonMiner 输出变成长期知识库：

k_search_knowledge/
  ascendc/
    tail_alignment_lessons.jsonl
    datacopy_lessons.jsonl
    tiling_contract_lessons.jsonl

每条 lesson 都要有 evidence attempts，避免胡编经验。

⸻

8. 新增目录结构

建议在原 telemetry 设计上增加：

k_search/
  telemetry/
    context.py
    events.py
    recorder.py
    cost.py
    sinks.py
    reports.py
  analysis/
    attempt_record.py
    attempt_record_builder.py
    failure_classifier.py
    trace_summarizer.py
    attempt_diagnoser.py
    lesson_miner.py
    action_metrics.py
    run_postmortem.py
    replay.py
  knowledge/
    lesson_store.py
    lesson_schema.py

职责边界：

telemetry:
  记录事实，不做判断
analysis:
  对事实做诊断、归类、总结
knowledge:
  存储跨 run 的可复用经验
policy:
  未来基于知识和成本做 action/token 策略

这个边界很重要，避免 telemetry 模块变成大杂烩。

⸻

9. Artifact 布局

建议最终 artifact 组织如下：

.ksearch/<task>/
  telemetry/
    <run_id>/
      round_0007/
        action_n12/
          attempt_0002/
            prompt.md
            agent_trace.jsonl
            tool_timeline.md
            diff.patch
            changed_files.json
            cost.json
            eval.json
            attempt_record.json
            diagnosis.json
            lesson_candidates.json
  analysis/
    <run_id>/
      run_postmortem.md
      run_postmortem.html
      action_metrics.json
      failure_patterns.json
      cost_summary.json
      lessons.jsonl
  knowledge/
    ascendc_lessons.jsonl

⸻

10. 问题根因定位：推荐分类体系

为了支持 RCA，建议分层分类。

10.1 失败发生在哪一层

provider:
  LLM 调用失败、认证失败、超时
agent_runtime:
  工具调用失败、无代码改动、编辑非法路径
codegen:
  输出格式错误、patch 应用失败
build:
  编译错误、链接错误
runtime:
  CANN runtime 报错、越界、workspace 错
correctness:
  精度错误、NaN、shape case 失败
performance:
  latency regression、方差过大、benchmark 不稳定
search:
  action 选择差、重复尝试、过早停止
context:
  prompt 缺少关键文件、失败日志不足、world model 误导

10.2 根因类型

action_design_issue
implementation_bug
context_missing
prompt_rule_missing
budget_insufficient
evaluation_noise
tool_policy_issue
project_manifest_issue
world_model_misclassification

这样后续复盘就不是“失败了”，而是：

失败发生在 correctness 层；
根因类型是 implementation_bug；
具体 surface 是 tail/alignment；
建议继续当前 action，但缩小 scope。

⸻

11. Replay 能力

为了复盘和问题定位，建议支持：

python -m k_search.analysis.replay \
  --attempt-record .ksearch/.../attempt_record.json

输出：

1. 打印当时 prompt
2. 打印 Claude tool timeline
3. 打印 diff
4. 打印 eval logs
5. 打印 diagnosis
6. 可选：重建 candidate project

更进一步：

python -m k_search.analysis.replay \
  --attempt-record ... \
  --materialize /tmp/replay_attempt_n12

它可以恢复：

parent project
+ diff
+ generated solution
+ eval command

这对修复 K-Search 本身的问题特别重要。

⸻

12. 最小落地版本

第一期不要做太大，建议只做 5 件事：

1. AttemptRecord schema
2. agent_trace.jsonl + cost.json + diff.patch 统一引用
3. FailureClassifier 规则版
4. AttemptDiagnoser 规则版
5. run_postmortem.md

先不做 LLM 自动诊断，避免复杂度过高。

规则版已经能解决 60% 复盘问题。

⸻

13. 规则版诊断示例

def diagnose_precision_failure(record):
    diff = record.diff_text
    logs = record.eval.log_excerpt
    evidence = []
    if "max_abs_diff" in logs or "mismatch" in logs.lower():
        evidence.append("correctness log reports precision mismatch")
    if any(x in diff for x in ["CopyOut", "DataCopy", "DataCopyPad", "tail", "mask"]):
        surface = "tail/alignment or copyout"
        evidence.append("diff touched CopyOut/DataCopy/tail/mask code")
    elif any(x in diff for x in ["TilingData", "blockDim", "tiling"]):
        surface = "host tiling contract"
        evidence.append("diff touched host tiling or blockDim")
    else:
        surface = "unknown"
    return {
        "stage": "correctness",
        "root_cause_category": "implementation_bug",
        "suspected_surface": surface,
        "evidence": evidence,
        "should_continue_action": True,
        "recommended_next_step": "Debug current action with stricter correctness preservation."
    }

⸻

14. 长期价值

如果这个体系跑起来，K-Search 会从：

LLM 搜索代码

升级为：

LLM 搜索代码
+ 记录每次搜索行为
+ 分析每次失败
+ 总结每次经验
+ 用经验改进后续搜索

这才是工业级 agent 的关键。

因为成熟工具最重要的不是“某一次成功”，而是：

失败之后系统能不能变聪明

⸻

15. 总结

你的新要求应该把设计文档中的 telemetry 模块升级成三层：

第一层：Runtime Trace
  记录 LLM 做了什么
第二层：Attempt Record
  把 LLM 行为、代码 diff、eval 结果、cost 串成一次完整尝试
第三层：Experience Mining
  从 attempt 中提取 failure pattern、root cause、lesson、budget 信号

最终目标不是“看日志”，而是形成闭环：

运行记录
→ 复盘
→ 根因定位
→ lesson 沉淀
→ prompt 改进
→ world model 改进
→ action selection 改进
→ token budget 改进

这样 K-Search 才会从一个会搜索的工具，逐步进化成一个会从自己搜索历史中学习的工业级算子优化系统。