# LLM Runtime Observability Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify cost accounting for non-agentic LLM calls so a full K-Search run can attribute token usage and cost by stage, round, action, and model.

**Phase Boundary:** This phase extends Phase 1 telemetry from Claude Agentic Codegen attempts to ordinary prompt-to-text LLM calls. It does not add AttemptRecord, failure diagnosis, postmortem reports, hooks, or budget policy.

**Architecture:** Add a run-level cost ledger under `k_search.telemetry`. `OpenAICompatibleLLMClient.generate()` and `ClaudeAgentLLMClient.generate()` extract usage/cost when available and append `LLMCallRecord` rows. World-model call sites already wrap LLM calls with `llm_log_context`; Phase 2 reuses that context to attribute each call to `world_model_init`, `world_model_propose_actions`, `world_model_refine`, or `world_model_mark_too_hard`.

**Tech Stack:** Python dataclasses, contextvars, pathlib, JSON/JSONL, pytest, existing fake OpenAI/Claude test helpers.

---

## File Structure

- Create or extend: `k_search/telemetry/cost.py`
- Create: `k_search/telemetry/pricing.py`
- Create: `k_search/telemetry/llm_call.py`
- Create: `k_search/telemetry/run_cost_summary.py`
- Modify: `k_search/telemetry/__init__.py`
- Modify: `k_search/kernel_generators/llm_clients.py`
- Modify: `tests/kernel_generators/test_llm_clients.py`
- Test: `tests/telemetry/test_cost.py`
- Test: `tests/telemetry/test_run_cost_summary.py`
- Test: `tests/telemetry/test_llm_call_telemetry.py`

Depends on Phase 1:

- `k_search.telemetry.context.TelemetryContext`
- `k_search.telemetry.context.is_telemetry_enabled`
- `k_search.telemetry.context.telemetry_root`
- `k_search.telemetry.context.safe_path_component`
- Phase 1 attempt-level artifacts may already emit `cost.json`, but Phase 2 owns run-level `llm_calls.jsonl` and `cost_summary.json`.

If Phase 1 has not landed yet, implement only the minimum compatible context helpers first instead of duplicating path logic inside Phase 2 modules.

---

## Output Layout

Run-level files:

```text
.ksearch-output-mqa/telemetry/
  <task_name>/
    <run_id>/
      llm_calls.jsonl
      cost_summary.json
```

Attempt-level Phase 1 files remain unchanged:

```text
.ksearch-output-mqa/telemetry/
  <task_name>/
    <run_id>/
      round_0007/
        action_n12/
          attempt_0002/
            agent_trace.jsonl
            tool_timeline.md
            cost.json
```

`llm_calls.jsonl` is append-only. `cost_summary.json` is best-effort regenerated after each successful append so interrupted runs still have a useful partial summary.

---

## Data Model

Use small dataclasses that serialize to JSON without leaking prompts or responses:

```python
@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMCost:
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    total_cost_usd: float | None = None
    currency: str = "USD"
    estimated: bool = False
    pricing_source: str | None = None


@dataclass(frozen=True)
class LLMCallRecord:
    call_id: str
    ts_ms: int
    duration_ms: int | None
    run_id: str | None
    task_name: str | None
    provider: str
    model_name: str
    flow: str | None = None
    stage: str | None = None
    round_index: int | None = None
    action_node_id: str | None = None
    attempt_index: int | None = None
    usage: LLMUsage | None = None
    cost: LLMCost | None = None
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None
```

Do not store full prompt or response in the cost ledger. Existing `KSEARCH_LLM_LOG_DIR` prompt/response logs can remain separate.

---

### Task 1: Cost Model And Pricing Adapter

**Files:**

- Create or extend: `k_search/telemetry/cost.py`
- Create: `k_search/telemetry/pricing.py`
- Test: `tests/telemetry/test_cost.py`

- [ ] **Step 1: Write failing tests for cost calculation**

Cover:

- Known model with input/output token prices computes total cost.
- Unknown model returns `total_cost_usd=None` and does not raise.
- Missing usage returns no cost and does not raise.
- `total_tokens` is backfilled from input/output when absent.
- Cost output rounds only at serialization boundaries, not during internal arithmetic.

- [ ] **Step 2: Implement `LLMUsage`, `LLMCost`, and pricing table**

Start with a conservative static map for models used in this repo. Include aliases where practical:

- `gpt-5.2`
- `gpt-5.4`
- `gpt-5.4-mini`
- `gpt-5.3-codex`
- `claude-sonnet-4-6`

Use per-1M-token pricing fields:

```python
@dataclass(frozen=True)
class ModelPricing:
    input_usd_per_1m_tokens: float
    output_usd_per_1m_tokens: float
    source: str
```

Important: if prices are uncertain or not maintained in repo, mark records as estimated and keep the pricing table easy to update. Do not fetch pricing over the network at runtime.

- [ ] **Step 3: Implement `estimate_llm_cost()`**

Signature:

```python
def estimate_llm_cost(*, provider: str, model_name: str, usage: LLMUsage | None) -> LLMCost | None:
    ...
```

Behavior:

- Return `None` if usage is missing.
- Return `LLMCost(total_cost_usd=None, estimated=True, pricing_source=None)` for unknown model with known usage.
- Compute input/output/total when model pricing is known.
- Treat missing input or output tokens as zero only if `total_tokens` is still preserved in usage; do not invent output/input splits.

- [ ] **Step 4: Run unit tests**

Run:

```bash
pytest tests/telemetry/test_cost.py -v
```

---

### Task 2: Run-Level LLM Call Ledger

**Files:**

- Create: `k_search/telemetry/llm_call.py`
- Create: `k_search/telemetry/run_cost_summary.py`
- Modify: `k_search/telemetry/__init__.py`
- Test: `tests/telemetry/test_run_cost_summary.py`

- [ ] **Step 1: Write failing tests for run directory layout**

Given env:

- `KSEARCH_TELEMETRY_DIR=<tmp>`
- `KSEARCH_RUN_ID=run:alpha`

And context:

- `task_name="task/name"`

The run dir should be:

```text
<tmp>/task_name/run_alpha/
```

The files should be:

- `llm_calls.jsonl`
- `cost_summary.json`

- [ ] **Step 2: Implement `build_run_dir()`**

Add helper in `llm_call.py` or Phase 1 `context.py`:

```python
def build_run_dir(context: TelemetryContext, *, root: Path | None = None) -> Path:
    ...
```

Use the same sanitization behavior as Phase 1 `build_attempt_dir()`.

- [ ] **Step 3: Implement `RunLLMCostRecorder`**

Responsibilities:

- Append one JSON row to `llm_calls.jsonl`.
- Never raise to caller unless tests call strict internals directly.
- Regenerate `cost_summary.json` after append.
- Use atomic write for the summary file via temporary file + replace.

Suggested API:

```python
class RunLLMCostRecorder:
    def __init__(self, context: TelemetryContext, *, root: Path | None = None): ...
    @property
    def calls_path(self) -> Path: ...
    @property
    def summary_path(self) -> Path: ...
    def record(self, record: LLMCallRecord) -> None: ...
```

- [ ] **Step 4: Implement summary aggregation**

`cost_summary.json` shape:

```json
{
  "run_id": "run_alpha",
  "task_name": "task_name",
  "records_count": 4,
  "success_count": 3,
  "failure_count": 1,
  "total_tokens": 12345,
  "total_cost_usd": 0.1234,
  "unknown_cost_records": 1,
  "by_stage": {},
  "by_round": {},
  "by_action": {},
  "by_model": {}
}
```

Aggregation rules:

- Sum only numeric `total_cost_usd`.
- Count unknown-cost records separately.
- Use stable string keys for round, e.g. `round_0007`, `round_global`.
- Use `unknown` for missing stage/action/model.
- Preserve both `input_tokens` and `output_tokens` totals where available.

- [ ] **Step 5: Run unit tests**

Run:

```bash
pytest tests/telemetry/test_run_cost_summary.py -v
```

---

### Task 3: Context Bridge From Existing LLM Logging

**Files:**

- Modify: `k_search/kernel_generators/llm_clients.py`
- Create or modify: `k_search/telemetry/llm_call.py`
- Test: `tests/telemetry/test_llm_call_telemetry.py`

- [ ] **Step 1: Add helper to convert `llm_log_context` metadata to `TelemetryContext`**

Existing `llm_log_context()` already carries:

- `operator`
- `task_name`
- `definition_name`
- `definition`
- `flow`
- `stage`
- `round_index`
- `action_node_id`
- `language`
- `target_gpu`

Implement a bridge:

```python
def telemetry_context_from_llm_log_context(
    metadata: dict[str, Any],
    *,
    provider: str,
    model_name: str,
) -> TelemetryContext:
    ...
```

Mapping:

- `task_name = metadata["task_name"] or metadata["definition_name"] or metadata["definition"] or metadata["operator"]`
- `flow = metadata["flow"]`
- `stage = metadata["stage"]`
- `round_index = metadata["round_index"]`
- `action_node_id = metadata["action_node_id"]`
- `attempt_index = metadata["attempt_index"] or metadata["debug_attempt"]`
- `provider = provider`
- `model_name = model_name`

- [ ] **Step 2: Add best-effort recording helper**

Suggested function:

```python
def record_llm_call_best_effort(
    *,
    provider: str,
    model_name: str,
    started_monotonic: float,
    response: Any | None,
    success: bool,
    error: BaseException | None = None,
) -> None:
    ...
```

This helper should:

- Check `is_telemetry_enabled()`.
- Read the current `_llm_log_context` metadata from `llm_clients.py`.
- Extract usage from response.
- Estimate cost.
- Build an `LLMCallRecord`.
- Append to `RunLLMCostRecorder`.
- Swallow all telemetry errors.

Avoid importing `llm_clients.py` from telemetry modules if it creates cycles. Prefer passing metadata into the helper from `llm_clients.py`.

- [ ] **Step 3: Keep prompt/response logging independent**

Do not remove `_log_llm_interaction()`. Phase 2 cost telemetry and existing prompt logs should both run, and either may fail without breaking the LLM call.

- [ ] **Step 4: Run bridge tests**

Run:

```bash
pytest tests/telemetry/test_llm_call_telemetry.py -v
```

---

### Task 4: OpenAI-Compatible `generate()` Telemetry

**Files:**

- Modify: `k_search/kernel_generators/llm_clients.py`
- Modify: `tests/kernel_generators/test_llm_clients.py`
- Test: `tests/telemetry/test_llm_call_telemetry.py`

- [ ] **Step 1: Extend fake OpenAI responses with usage**

For Responses API fake:

```python
SimpleNamespace(
    output_text="responses text",
    usage=SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120),
)
```

For Chat Completions fake:

```python
SimpleNamespace(
    choices=[...],
    usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10, total_tokens=60),
)
```

- [ ] **Step 2: Implement usage extraction**

Support both styles:

- Responses API: `input_tokens`, `output_tokens`, `total_tokens`
- Chat Completions: `prompt_tokens`, `completion_tokens`, `total_tokens`

Handle dict-like and object-like responses.

- [ ] **Step 3: Record success telemetry**

In `OpenAICompatibleLLMClient.generate()`:

- Capture `started = time.monotonic()` before the API call.
- Keep the raw `response` object until after result extraction.
- Call `record_llm_call_best_effort(...)` before returning.

The return value must remain exactly `str`.

- [ ] **Step 4: Record failure telemetry**

In the exception path:

- Record `success=False`.
- Include `error_type=type(provider_exc).__name__`.
- Include bounded `error_message`.
- Preserve the existing exception behavior exactly.

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest tests/kernel_generators/test_llm_clients.py tests/telemetry/test_llm_call_telemetry.py -v
```

---

### Task 5: Claude `query()` Prompt-To-Text Telemetry

**Files:**

- Modify: `k_search/kernel_generators/llm_clients.py`
- Modify: `k_search/testing/mock_claude_agent_sdk.py`
- Modify: `tests/kernel_generators/test_llm_clients.py`

- [ ] **Step 1: Extend Claude mock result messages with usage/cost fields**

The real SDK result messages may expose fields like:

- `usage`
- `total_cost_usd`
- `duration_ms`
- `duration_api_ms`
- `num_turns`
- `session_id`

Mocks should support at least:

```python
MockResponse(
    result="final text",
    usage={"input_tokens": 80, "output_tokens": 30},
    total_cost_usd=0.001,
    num_turns=1,
)
```

- [ ] **Step 2: Preserve the final result message**

In `ClaudeAgentLLMClient.generate()`:

- Track the last result message.
- Extract final text as today.
- Pass the result message to telemetry extraction after `_run_query()` completes.

If returning from async helper makes this awkward, return a small internal object:

```python
@dataclass
class _ClaudeTextResult:
    text: str
    result_message: Any | None = None
```

Keep the public `generate()` return type as `str`.

- [ ] **Step 3: Extract Claude usage and direct cost**

If result has `total_cost_usd`, use it directly. If result has usage but no direct cost, use pricing adapter. If neither exists, still write a record with `usage=None` and `cost=None`.

- [ ] **Step 4: Record failure telemetry**

Record failed Claude calls when:

- SDK import fails
- SDK query raises
- SDK result message indicates error
- SDK returns empty text
- timeout occurs

Do not change current exception types or messages.

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest tests/kernel_generators/test_llm_clients.py tests/kernel_generators/test_claude_agent_sdk_mock.py -v
```

---

### Task 6: World Model Attribution

**Files:**

- Modify: `k_search/kernel_generators/kernel_generator_world_model.py`
- Modify: `k_search/kernel_generators/world_model_manager.py` only if necessary
- Modify: `tests/kernel_generators/test_llm_clients.py`

- [ ] **Step 1: Audit all world model LLM call sites**

Ensure every world-model call is wrapped in `llm_log_context()` with:

- `operator` or `task_name`
- `flow="world_model"`
- `stage`
- `round_index`
- `action_node_id` when an action has been selected
- `language`
- `target_gpu`

Known stages:

- `world_model_seed_init`
- `world_model_init`
- `world_model_propose_actions`
- `world_model_refine`
- `world_model_mark_too_hard`

- [ ] **Step 2: Add missing `action_node_id` where available**

`world_model_refine` and `world_model_mark_too_hard` already have the chosen leaf. `world_model_propose_actions` normally runs before choosing a leaf, so action may be absent and should aggregate under `unknown`.

- [ ] **Step 3: Avoid changing `WorldModelManager` signatures unless needed**

Prefer context wrapping at caller sites. Only modify `WorldModelManager` if a call path invokes `_llm_call()` outside a caller-provided context.

- [ ] **Step 4: Add an integration-style test**

Use a fake LLM client that returns valid world-model JSON and emits usage. Run a minimal world model flow and assert:

- `cost_summary.json` exists.
- `by_stage.world_model_init.records_count >= 1`.
- `by_stage.world_model_propose_actions.records_count >= 1` if action proposal path runs.
- `by_round.round_0000` or `round_0001` is present.

---

### Task 7: Run-Level Acceptance Test

**Files:**

- Test: `tests/telemetry/test_llm_call_telemetry.py`
- Modify: `tests/kernel_generators/test_llm_clients.py`

- [ ] **Step 1: Add full ledger acceptance test**

Simulate these calls under one `KSEARCH_RUN_ID`:

- OpenAI call with `stage="world_model_init"`, round 0.
- OpenAI call with `stage="world_model_propose_actions"`, round 1.
- Claude call with `stage="world_model_refine"`, round 2, `action_node_id="n1"`.
- One failed call.

Assert:

- `llm_calls.jsonl` has four rows.
- `cost_summary.json.records_count == 4`.
- `success_count == 3`.
- `failure_count == 1`.
- `by_stage` contains all three stages.
- `by_action.n1` contains the Claude refine call.
- `by_model` contains both model names.
- Unknown cost records are counted but do not break totals.

- [ ] **Step 2: Verify telemetry disabled behavior**

With `KSEARCH_TELEMETRY=off`:

- LLM calls still return normally.
- No `llm_calls.jsonl` is created.
- No `cost_summary.json` is created.

- [ ] **Step 3: Run full focused suite**

Run:

```bash
pytest tests/telemetry tests/kernel_generators/test_llm_clients.py tests/kernel_generators/test_claude_agent_sdk_mock.py -v
```

---

## Acceptance Criteria

Phase 2 is complete when one full K-Search run can produce a run-level `cost_summary.json` that answers:

- How much did world model init/propose/refine cost?
- How much did each round cost?
- How much did each action cost when an action id is known?
- How much did each model cost?
- Which records had token usage but unknown pricing?
- Which records failed before cost could be computed?

The implementation must preserve existing LLM behavior:

- No telemetry failure may break a generation call.
- Public return values remain compatible.
- Prompt/response logging remains independent.
- Unknown model pricing is visible but non-fatal.

---

## Non-Goals

- No `attempt_record.json`.
- No `FailureClassifier`.
- No `run_postmortem.md`.
- No hooks or policy gates.
- No prompt/response persistence inside `llm_calls.jsonl`.
- No dynamic pricing fetch at runtime.
- No budget enforcement.

