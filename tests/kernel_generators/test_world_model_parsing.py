import json

from k_search.kernel_generators.world_model import (
    load_world_model_obj,
    try_parse_decision_tree_edit_ops,
    try_parse_world_model_json,
)


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
