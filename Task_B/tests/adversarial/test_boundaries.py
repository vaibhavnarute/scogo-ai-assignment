from __future__ import annotations

from harness.tools.base import ToolCall, ToolContext
from harness.tools.command import RunCommandTool
from harness.tools.files import ReadFileTool
from harness.tools.patch import ApplyPatchTool
from harness.tools.registry import ToolRegistry


def test_prompt_injection_cannot_read_secret(context: ToolContext):
    (context.config.workspace / ".env").write_text("API_KEY=secret", encoding="utf-8")
    result = ToolRegistry([ReadFileTool()]).dispatch(ToolCall("a", "read_file", {"path": ".env"}), context)
    assert result.error_code == "SENSITIVE_PATH"


def test_prompt_injection_cannot_escape_workspace(context: ToolContext):
    result = ToolRegistry([ReadFileTool()]).dispatch(ToolCall("a", "read_file", {"path": "../../README.md"}), context)
    assert result.error_code == "PATH_OUTSIDE_WORKSPACE"


def test_prompt_injection_cannot_run_destructive_command(context: ToolContext):
    result = ToolRegistry([RunCommandTool()]).dispatch(ToolCall("a", "run_command", {"command": "rm -rf ."}), context)
    assert result.error_code == "COMMAND_DENIED"


def test_prompt_injection_cannot_tamper_with_tests(context: ToolContext):
    patch = "@@ -1,1 +1,1 @@\n-def test_locked(): assert True\n+def test_locked(): assert False"
    result = ToolRegistry([ApplyPatchTool()]).dispatch(ToolCall("a", "apply_patch", {"path": "tests/test_locked.py", "patch": patch}), context)
    assert result.error_code == "PROTECTED_PATH"

