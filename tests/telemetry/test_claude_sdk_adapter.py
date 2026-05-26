from types import SimpleNamespace

from k_search.telemetry.claude_sdk_adapter import event_from_claude_message


def _event_types(message):
    return [event.event_type for event in event_from_claude_message(message)]


def test_text_block_converts_to_assistant_text():
    message = SimpleNamespace(content=[{"type": "text", "text": "edited file"}])

    events = event_from_claude_message(message)

    assert _event_types(message) == ["assistant_text"]
    assert events[0].text_excerpt == "edited file"


def test_tool_use_block_converts_to_tool_use():
    block = SimpleNamespace(type="tool_use", id="toolu_1", name="Read", input={"file_path": "kernel/foo.h"})
    message = SimpleNamespace(content=[block])

    event = event_from_claude_message(message)[0]

    assert event.event_type == "tool_use"
    assert event.tool_use_id == "toolu_1"
    assert event.tool_name == "Read"
    assert event.tool_input == {"file_path": "kernel/foo.h"}


def test_tool_result_block_converts_to_tool_result():
    block = SimpleNamespace(type="tool_result", tool_use_id="toolu_1", content="alpha", is_error=False)
    message = SimpleNamespace(content=[block])

    event = event_from_claude_message(message)[0]

    assert event.event_type == "tool_result"
    assert event.tool_use_id == "toolu_1"
    assert event.tool_result_excerpt == "alpha"
    assert event.is_error is False


def test_result_message_captures_cost_and_session_metadata():
    message = SimpleNamespace(
        result="final summary",
        session_id="sess-1",
        total_cost_usd=0.25,
        duration_ms=1200,
        duration_api_ms=900,
        num_turns=4,
        usage={"input_tokens": 10},
        model_usage={"claude": {"output_tokens": 5}},
        subtype="success",
        is_error=False,
    )

    event = event_from_claude_message(message)[0]

    assert event.event_type == "llm_result"
    assert event.session_id == "sess-1"
    assert event.total_cost_usd == 0.25
    assert event.num_turns == 4
    assert event.text_excerpt == "final summary"


def test_thinking_block_records_only_metadata():
    block = SimpleNamespace(type="thinking", thinking="private reasoning text")
    message = SimpleNamespace(content=[block])

    event = event_from_claude_message(message)[0]

    assert event.event_type == "assistant_thinking_metadata"
    assert "private reasoning text" not in str(event.to_dict())
    assert event.text_excerpt == "thinking_chars=22"


def test_unknown_block_does_not_raise():
    message = SimpleNamespace(content=[SimpleNamespace(type="new_block", value={"x": 1})])

    events = event_from_claude_message(message)

    assert events[0].event_type == "system_message"
    assert "new_block" in events[0].raw_type