from harness.context import ContextLimits, ContextManager, SYSTEM_PROMPT
from harness.providers.types import ModelResponse
from harness.tools.base import ToolCall, ToolResult


def test_system_contract_and_workspace_metadata_are_always_present(workspace):
    context = ContextManager("repair", workspace, "pytest -q")
    messages = context.messages()
    assert messages[0]["role"] == "system" and "untrusted observations" in SYSTEM_PROMPT
    assert "pytest -q" in messages[1]["content"] and str(workspace) in messages[1]["content"]


def test_context_compaction_keeps_complete_recent_batches(workspace):
    context = ContextManager("repair", workspace, "pytest -q", ContextLimits(max_chars=500, max_turn_batches=2))
    for index in range(4):
        context.add_text_only(ModelResponse(text=f"response-{index}-" + "x" * 100))
    messages = context.messages()
    assert any("older turn batch" in message.get("content", "") for message in messages)
    assert any("response-3" in message.get("content", "") for message in messages)
    assert not any("response-0" in message.get("content", "") for message in messages)


def test_oversized_file_observation_respects_hard_limit_and_keeps_exchange(workspace):
    limits = ContextLimits(max_chars=2400, max_turn_batches=2)
    context = ContextManager("repair", workspace, "pytest -q", limits)
    response = ModelResponse(tool_calls=[ToolCall("read-1", "read_file", {"path": "huge.py"})])
    context.add_exchange(response, [ToolResult("read-1", "read_file", True, {"content": "x" * 20_000})])
    messages = context.messages()
    assert sum(len(__import__("json").dumps(message, ensure_ascii=False, sort_keys=True)) for message in messages) <= limits.max_chars
    assistant = next(message for message in messages if message.get("tool_calls"))
    tool = next(message for message in messages if message["role"] == "tool")
    assert assistant["tool_calls"][0]["id"] == tool["tool_call_id"] == "read-1"
    assert "observation_truncated" in tool["content"]


def test_oversized_command_observation_respects_hard_limit(workspace):
    limits = ContextLimits(max_chars=2400, max_turn_batches=2)
    context = ContextManager("repair", workspace, "pytest -q", limits)
    response = ModelResponse(tool_calls=[ToolCall("cmd-1", "run_command", {"command": "pytest -q"})])
    context.add_exchange(response, [ToolResult("cmd-1", "run_command", True, {"stdout": "failure\n" * 5000, "exit_code": 1})])
    messages = context.messages()
    assert sum(len(__import__("json").dumps(message, ensure_ascii=False, sort_keys=True)) for message in messages) <= limits.max_chars
    assert any(message.get("tool_call_id") == "cmd-1" and "observation_truncated" in message["content"] for message in messages)
