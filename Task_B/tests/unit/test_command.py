from __future__ import annotations

from pathlib import Path

from harness.cache import ObservationCache
from harness.config import HarnessConfig
from harness.policy import WorkspacePolicy
from harness.state import RunState
from harness.tools.base import ToolCall, ToolContext
from harness.tools.command import RunCommandTool
from harness.tools.registry import ToolRegistry


def make_context(workspace: Path, **overrides) -> ToolContext:
    config = HarnessConfig(workspace, approval_mode="yes", **overrides)
    return ToolContext(config, WorkspacePolicy(config), RunState("test", workspace), ObservationCache())


def test_successful_command(workspace: Path):
    result = RunCommandTool().execute(ToolCall("c", "run_command", {"command": "python --version"}), make_context(workspace))
    assert result.ok and result.data["exit_code"] == 0
    assert "Python" in result.data["stdout"]


def test_nonzero_exit_is_successful_tool_observation(workspace: Path):
    context = make_context(workspace)
    result = RunCommandTool().execute(ToolCall("c", "run_command", {"command": "python -m pytest missing_test.py -q"}), context)
    assert result.ok and result.data["exit_code"] != 0


def test_timeout_terminates_process(workspace: Path):
    (workspace / "wait.py").write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    context = make_context(workspace, command_timeout_seconds=0.1)
    result = RunCommandTool().execute(ToolCall("c", "run_command", {"command": "python wait.py"}), context)
    assert not result.ok and result.error_code == "COMMAND_TIMEOUT"
    assert result.data["timed_out"] is True


def test_command_output_is_truncated(workspace: Path):
    (workspace / "output.py").write_text("print('x' * 1000)\n", encoding="utf-8")
    context = make_context(workspace, max_command_output_bytes=20)
    result = RunCommandTool().execute(ToolCall("c", "run_command", {"command": "python output.py"}), context)
    assert result.ok and result.data["truncated"] is True
    assert len(result.data["stdout"].encode()) <= 20


def test_denied_command_is_normalized(workspace: Path):
    context = make_context(workspace)
    result = ToolRegistry([RunCommandTool()]).dispatch(ToolCall("c", "run_command", {"command": "curl example.com"}), context)
    assert not result.ok and result.error_code == "COMMAND_DENIED"
    assert context.state.policy_denials == 1


def test_cwd_outside_workspace(workspace: Path):
    context = make_context(workspace)
    result = ToolRegistry([RunCommandTool()]).dispatch(ToolCall("c", "run_command", {"command": "python --version", "cwd": ".."}), context)
    assert result.error_code == "CWD_OUTSIDE_WORKSPACE"

