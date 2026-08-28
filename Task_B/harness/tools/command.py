"""Policy-checked subprocess execution with bounded observations."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from ..command_line import process_group_options, split_command, terminate_process_tree
from ..errors import ErrorCategory, HarnessError
from ..env import repository_subprocess_environment
from ..integrity import diff_snapshots, snapshot_workspace
from ..policy import PathIntent
from .base import BaseTool, RiskCategory, ToolCall, ToolContext, ToolResult


def _truncate(value: str, limit: int) -> tuple[str, bool, int]:
    encoded = value.encode("utf-8", errors="replace")
    size = len(encoded)
    if size <= limit:
        return value, False, size
    return encoded[:limit].decode("utf-8", errors="ignore"), True, size


def _record_command_mutations(context: ToolContext, before: dict[str, str], command: str, tool_call_id: str) -> list[str]:
    after = snapshot_workspace(context.config)
    changes = diff_snapshots(before, after)
    context.state.record_command_mutations(changes, command)
    context.state.workspace_hashes = after
    paths = [change.path for change in changes]
    if paths and context.tracer:
        context.tracer.record("integrity.violation", tool_call_id=tool_call_id, code="COMMAND_MUTATION", paths=paths)
    return paths


class RunCommandTool(BaseTool):
    name = "run_command"
    description = "Run one policy-approved command in the workspace and capture its result."
    risk = RiskCategory.COMMAND
    schema = {
        "type": "object",
        "properties": {"command": {"type": "string"}, "cwd": {"type": "string"}, "timeout_seconds": {"type": "integer", "minimum": 1}},
        "required": ["command"],
        "additionalProperties": False,
    }

    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        command = call.arguments["command"]
        try:
            cwd = context.policy.resolve_path(call.arguments.get("cwd", "."), PathIntent.CWD)
        except HarnessError as exc:
            if exc.code == "PATH_OUTSIDE_WORKSPACE":
                raise HarnessError("CWD_OUTSIDE_WORKSPACE", ErrorCategory.POLICY, "command cwd is outside the workspace", True, exc.details) from exc
            raise
        if not cwd.is_dir():
            raise HarnessError("CWD_NOT_DIRECTORY", ErrorCategory.FILESYSTEM, "command cwd is not a directory", True)
        decision = context.policy.check_command(command, cwd)
        if context.tracer:
            context.tracer.record("policy.allowed" if decision.allowed else "policy.denied", tool_call_id=call.id, tool=self.name, reason=decision.reason, requires_approval=decision.requires_approval)
        if not decision.allowed:
            context.state.policy_denials += 1
            raise HarnessError("COMMAND_DENIED", ErrorCategory.POLICY, decision.reason, True)
        requested_timeout = float(call.arguments.get("timeout_seconds", context.config.command_timeout_seconds))
        timeout = min(requested_timeout, context.config.command_timeout_seconds)
        argv = split_command(command)
        started = time.monotonic()
        is_verification = command.strip() == context.config.verification_command.strip() and cwd == context.config.workspace
        before = snapshot_workspace(context.config)
        if context.state.workspace_hashes:
            preexisting_changes = diff_snapshots(context.state.workspace_hashes, before)
            if preexisting_changes:
                context.state.record_command_mutations(preexisting_changes, "<external-before-command>")
        else:
            context.state.workspace_hashes = before
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=repository_subprocess_environment(),
                shell=False,
                **process_group_options(),
            )
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            assert process is not None
            terminate_process_tree(process)
            stdout, stderr = process.communicate()
            duration_ms = round((time.monotonic() - started) * 1000, 3)
            changed_paths = _record_command_mutations(context, before, command, call.id)
            context.state.record_command(command, cwd, None, True, is_verification)
            out, out_truncated, out_bytes = _truncate(stdout, context.config.max_command_output_bytes)
            err, err_truncated, err_bytes = _truncate(stderr, context.config.max_command_output_bytes)
            return ToolResult(call.id, self.name, False, {"stdout": out, "stderr": err, "exit_code": None, "duration_ms": duration_ms, "timed_out": True, "truncated": out_truncated or err_truncated, "stdout_bytes": out_bytes, "stderr_bytes": err_bytes, "repository_changes": changed_paths}, "COMMAND_TIMEOUT", "command exceeded its timeout", True)
        except OSError as exc:
            raise HarnessError("SPAWN_ERROR", ErrorCategory.COMMAND, "could not start command", True, {"error_type": type(exc).__name__}) from exc
        duration_ms = round((time.monotonic() - started) * 1000, 3)
        exit_code = process.returncode
        changed_paths = _record_command_mutations(context, before, command, call.id)
        stdout, stdout_truncated, stdout_bytes = _truncate(stdout, context.config.max_command_output_bytes)
        stderr, stderr_truncated, stderr_bytes = _truncate(stderr, context.config.max_command_output_bytes)
        command_record = context.state.record_command(command, cwd, exit_code, False, is_verification)
        data = {"command": command, "cwd": cwd.relative_to(context.config.workspace).as_posix() or ".", "exit_code": exit_code, "stdout": stdout, "stderr": stderr, "duration_ms": duration_ms, "timed_out": False, "truncated": stdout_truncated or stderr_truncated, "stdout_bytes": stdout_bytes, "stderr_bytes": stderr_bytes, "command_sequence": command_record.sequence, "is_verification": is_verification, "repository_changes": changed_paths}
        return ToolResult(call.id, self.name, True, data)
