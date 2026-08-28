from __future__ import annotations

from harness.tools.base import ToolCall, ToolContext
from harness.tools.files import ReadFileTool
from harness.tools.registry import ToolRegistry


def test_unknown_tool(context: ToolContext):
    result = ToolRegistry().dispatch(ToolCall("x", "invented", {}), context)
    assert result.error_code == "UNKNOWN_TOOL"


def test_malformed_arguments(context: ToolContext):
    result = ToolRegistry([ReadFileTool()]).dispatch(ToolCall("x", "read_file", {"path": 123}), context)
    assert result.error_code == "INVALID_TOOL_ARGUMENTS"


def test_unknown_argument_is_rejected(context: ToolContext):
    result = ToolRegistry([ReadFileTool()]).dispatch(ToolCall("x", "read_file", {"path": "src/app.py", "surprise": True}), context)
    assert result.error_code == "INVALID_TOOL_ARGUMENTS"


def test_repeated_actions_are_counted(context: ToolContext):
    registry = ToolRegistry([ReadFileTool()])
    call = ToolCall("x", "read_file", {"path": "src/app.py"})
    registry.dispatch(call, context)
    registry.dispatch(call, context)
    assert next(iter(context.state.repeated_actions.values())) == 2

