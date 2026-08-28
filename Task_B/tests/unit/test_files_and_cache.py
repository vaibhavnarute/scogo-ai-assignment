from __future__ import annotations

from pathlib import Path

from harness.config import HarnessConfig
from harness.tools.base import ToolCall, ToolContext
from harness.tools.files import ListFilesTool, ReadFileTool


def test_list_files_is_bounded_and_cached(context: ToolContext):
    tool = ListFilesTool()
    first = tool.execute(ToolCall("1", "list_files", {"path": ".", "depth": 2}), context)
    second = tool.execute(ToolCall("2", "list_files", {"path": ".", "depth": 2}), context)
    assert first.ok and not first.cached
    assert any(entry["path"] == "src/app.py" for entry in first.data["entries"])
    assert second.cached and second.data["cached"] is True


def test_list_files_treats_blank_optional_path_as_workspace_root(context: ToolContext):
    result = ListFilesTool().execute(ToolCall("1", "list_files", {"path": "", "depth": 1}), context)
    assert result.ok
    assert result.data["path"] == "."
    assert any(entry["path"] == "src" for entry in result.data["entries"])


def test_read_file_range_and_cache(context: ToolContext):
    path = context.config.workspace / "src" / "app.py"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    tool = ReadFileTool()
    args = {"path": "src/app.py", "start_line": 2, "end_line": 3}
    first = tool.execute(ToolCall("1", "read_file", args), context)
    second = tool.execute(ToolCall("2", "read_file", args), context)
    assert first.data["content"] == "two\nthree"
    assert first.data["start_line"] == 2 and first.data["end_line"] == 3
    assert second.cached


def test_read_cache_detects_external_change(context: ToolContext):
    tool = ReadFileTool()
    call = ToolCall("1", "read_file", {"path": "src/app.py"})
    assert tool.execute(call, context).data["content"] == "value = 1"
    (context.config.workspace / "src" / "app.py").write_text("value = 2\n", encoding="utf-8")
    result = tool.execute(call, context)
    assert not result.cached and result.data["content"] == "value = 2"


def test_read_missing_file_returns_normalized_error_via_registry(context: ToolContext):
    from harness.tools.registry import ToolRegistry

    result = ToolRegistry([ReadFileTool()]).dispatch(ToolCall("1", "read_file", {"path": "missing.py"}), context)
    assert not result.ok and result.error_code == "NOT_FOUND"


def test_rejects_large_file(workspace: Path):
    from harness.cache import ObservationCache
    from harness.policy import WorkspacePolicy
    from harness.state import RunState

    large = workspace / "large.txt"
    large.write_text("123456", encoding="utf-8")
    config = HarnessConfig(workspace, max_file_bytes=5)
    context = ToolContext(config, WorkspacePolicy(config), RunState("x", workspace), ObservationCache())
    from harness.tools.registry import ToolRegistry
    result = ToolRegistry([ReadFileTool()]).dispatch(ToolCall("1", "read_file", {"path": "large.txt"}), context)
    assert result.error_code == "FILE_TOO_LARGE"


def test_rejects_excessive_line_range(context: ToolContext):
    from harness.tools.registry import ToolRegistry
    result = ToolRegistry([ReadFileTool()]).dispatch(ToolCall("1", "read_file", {"path": "src/app.py", "start_line": 1, "end_line": 9999}), context)
    assert result.error_code == "INVALID_LINE_RANGE"

