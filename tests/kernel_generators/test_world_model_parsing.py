import json

from k_search.kernel_generators.world_model import (
    load_world_model_obj,
    try_parse_decision_tree_edit_ops,
    try_parse_world_model_json,
)
from k_search.kernel_generators.world_model_manager import WorldModelManager
from k_search.tasks.task_base import EvalResult


def test_world_model_json_accepts_dict_style_nodes_with_id_fields():
    raw = json.dumps(
        {
            "kernel_summary": "MQA",
            "decision_tree": {
                "root_id": "root",
                "active_leaf_id": "root",
                "nodes": {
                    "root": {"id": "root", "parent_id": None},
                    "n1": {
                        "id": "n1",
                        "parent_id": "root",
                        "decision": "First optimization family",
                        "choice": "Reduce K/V reloads",
                        "overall_rating_0_to_10": 7.5,
                        "confidence_0_to_1": 0.6,
                        "action": {
                            "title": "Optimize K/V loads",
                            "description": "Reuse K/V tiles across query rows.",
                            "score_0_to_1": 0.8,
                            "difficulty_1_to_5": 3,
                        },
                    },
                },
            },
        }
    )

    parsed = try_parse_world_model_json(raw)
    assert parsed is not None

    obj = load_world_model_obj(parsed)
    nodes = obj["decision_tree"]["nodes"]

    assert [node["node_id"] for node in nodes] == ["root", "n1"]
    assert nodes[1]["action"]["title"] == "Optimize K/V loads"


def test_world_model_json_parser_skips_unrelated_json_before_real_world_model():
    real_world_model = json.dumps(
        {
            "kernel_summary": "MQA",
            "decision_tree": {
                "root_id": "root",
                "active_leaf_id": "root",
                "nodes": {
                    "root": {"id": "root", "parent_id": None},
                    "n1": {
                        "id": "n1",
                        "parent_id": "root",
                        "action": {
                            "title": "Optimize K/V loads",
                            "description": "Reuse K/V tiles across query rows.",
                            "score_0_to_1": 0.8,
                            "difficulty_1_to_5": 3,
                        },
                    },
                },
            },
        }
    )
    raw = '{"example": true}\n' + real_world_model

    parsed = try_parse_world_model_json(raw)

    assert parsed is not None
    obj = load_world_model_obj(parsed)
    nodes = obj["decision_tree"]["nodes"]
    assert [node["node_id"] for node in nodes] == ["root", "n1"]


def test_decision_tree_edit_ops_parser_skips_example_json_before_real_ops():
    raw = """
The model may explain itself first.

```json
{"example": true, "ops": "this is not the edit script"}
```

Final edit script:
{"active_leaf_id":"root","ops":[{"op":"insert_node","parent_id":"root","node":{"node_id":"n1","action":{"title":"Optimize K/V loads"}}}]}
"""

    edits = try_parse_decision_tree_edit_ops(raw)

    assert edits is not None
    assert edits.active_leaf_id == "root"
    assert len(edits.ops) == 1
    assert edits.ops[0]["op"] == "insert_node"


def test_world_model_manager_recovers_agent_written_world_model_file(tmp_path):
    wm_path = tmp_path / "world_model.json"
    wm_path.write_text(
        json.dumps(
            {
                "kernel_summary": "MQA",
                "decision_tree": {
                    "root_id": "root",
                    "active_leaf_id": "root",
                    "nodes": {
                        "root": {"id": "root", "parent_id": None},
                        "n1": {
                            "id": "n1",
                            "parent_id": "root",
                            "action": {
                                "title": "Optimize K/V loads",
                                "description": "Reuse K/V tiles across query rows.",
                                "score_0_to_1": 0.8,
                                "difficulty_1_to_5": 3,
                            },
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    manager = WorldModelManager(
        llm_call=lambda prompt: f"The initial world model has been written to `{wm_path}`.",
        target_gpu="Ascend910B3",
        language="ascendc",
    )

    wm = manager.ensure_initialized(definition_name="multi_query_attention", definition_text="spec")

    assert wm is not None
    obj = load_world_model_obj(wm)
    nodes = obj["decision_tree"]["nodes"]
    assert [node["node_id"] for node in nodes] == ["root", "n1"]
    assert manager.choose_next_action_node_id(definition_name="multi_query_attention") == "n1"


def test_world_model_manager_falls_back_to_executable_seed_when_init_response_is_not_json():
    manager = WorldModelManager(
        llm_call=lambda prompt: "I analyzed the kernel and wrote notes elsewhere.",
        target_gpu="Ascend910B3",
        language="ascendc",
    )

    wm = manager.ensure_initialized(
        definition_name="multi_query_attention",
        definition_text="Task: multi_query_attention\nReference Implementation:\ncode",
    )

    assert wm is not None
    obj = load_world_model_obj(wm)
    nodes = obj["decision_tree"]["nodes"]
    open_actions = [
        node
        for node in nodes
        if node["parent_id"] == "root" and node["action"]["title"] and not node["solution_ref"]["solution_id"]
    ]
    assert len(open_actions) >= 3
    assert manager.choose_next_action_node_id(definition_name="multi_query_attention") is not None


def test_world_model_manager_falls_back_when_init_llm_call_times_out():
    def timed_out_llm_call(prompt):
        raise TimeoutError("Claude Agent SDK provider timed out after 1200s")

    manager = WorldModelManager(
        llm_call=timed_out_llm_call,
        target_gpu="Ascend910B3",
        language="ascendc",
    )

    wm = manager.ensure_initialized(
        definition_name="multi_query_attention",
        definition_text="Task: multi_query_attention\nReference Implementation:\ncode",
    )

    assert wm is not None
    obj = load_world_model_obj(wm)
    root = obj["decision_tree"]["nodes"][0]
    assert "LLM init failed" in root["notes"]
    assert "timed out" in root["notes"]
    assert manager.choose_next_action_node_id(definition_name="multi_query_attention") is not None


def test_note_action_too_hard_deterministically_closes_active_action_when_edits_fail():
    manager = WorldModelManager(
        llm_call=lambda prompt: "not valid edit ops",
        target_gpu="Ascend910B3",
        language="ascendc",
    )
    manager.set(
        "multi_query_attention",
        json.dumps(
            {
                "kernel_summary": "MQA",
                "decision_tree": {
                    "root_id": "root",
                    "active_leaf_id": "root",
                    "nodes": [
                        {"node_id": "root", "parent_id": None, "solution_ref": {"solution_id": None}},
                        {
                            "node_id": "n1",
                            "parent_id": "root",
                            "overall_rating_0_to_10": 9,
                            "action": {
                                "title": "too hard action",
                                "score_0_to_1": 0.9,
                                "difficulty_1_to_5": 3,
                            },
                            "solution_ref": {"solution_id": None},
                        },
                        {
                            "node_id": "n2",
                            "parent_id": "root",
                            "overall_rating_0_to_10": 8,
                            "action": {
                                "title": "next action",
                                "score_0_to_1": 0.8,
                                "difficulty_1_to_5": 3,
                            },
                            "solution_ref": {"solution_id": None},
                        },
                    ],
                },
            }
        ),
    )
    manager.set_active_leaf_id(definition_name="multi_query_attention", node_id="n1")

    assert manager.choose_next_action_node_id(definition_name="multi_query_attention") == "n1"

    manager.note_action_too_hard(
        definition_name="multi_query_attention",
        definition_text="spec",
        chosen_action_text="- node_id: n1\n- title: too hard action",
        current_code_excerpt="code",
        current_tree_path="root -> n1",
        eval_result=EvalResult(status="codegen_failed", log_excerpt="timeout"),
        debug_and_improve_round=5,
        debug_and_improve_max_rounds=5,
        round_index=3,
    )

    assert manager.choose_next_action_node_id(definition_name="multi_query_attention") == "n2"
    obj = load_world_model_obj(manager.get("multi_query_attention") or "")
    n1 = next(n for n in obj["decision_tree"]["nodes"] if n["node_id"] == "n1")
    assert n1["action"]["status"] == "too_hard"


def test_note_action_too_hard_keeps_fallback_when_llm_call_raises():
    manager = WorldModelManager(
        llm_call=lambda prompt: (_ for _ in ()).throw(TimeoutError("too-hard edit timed out")),
        target_gpu="Ascend910B3",
        language="ascendc",
    )
    manager.set(
        "multi_query_attention",
        json.dumps(
            {
                "kernel_summary": "MQA",
                "decision_tree": {
                    "root_id": "root",
                    "active_leaf_id": "root",
                    "nodes": [
                        {"node_id": "root", "parent_id": None, "solution_ref": {"solution_id": None}},
                        {
                            "node_id": "n1",
                            "parent_id": "root",
                            "action": {"title": "too hard", "score_0_to_1": 0.9},
                            "solution_ref": {"solution_id": None},
                        },
                        {
                            "node_id": "n2",
                            "parent_id": "root",
                            "action": {"title": "next", "score_0_to_1": 0.8},
                            "solution_ref": {"solution_id": None},
                        },
                    ],
                },
            }
        ),
    )
    manager.set_active_leaf_id(definition_name="multi_query_attention", node_id="n1")

    updated = manager.note_action_too_hard(
        definition_name="multi_query_attention",
        definition_text="spec",
        chosen_action_text="- node_id: n1\n- title: too hard",
        current_code_excerpt="code",
        current_tree_path="root -> n1",
        eval_result=EvalResult(status="codegen_failed", log_excerpt="timeout"),
        debug_and_improve_round=5,
        debug_and_improve_max_rounds=5,
        round_index=3,
    )

    assert updated is not None
    assert manager.choose_next_action_node_id(definition_name="multi_query_attention") == "n2"
