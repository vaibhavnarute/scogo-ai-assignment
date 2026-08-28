"""Explicit provider-independent observe/decide/act/verify loop."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Callable

from .cache import ObservationCache
from .config import HarnessConfig
from .context import ContextLimits, ContextManager
from .errors import ErrorCategory, HarnessError
from .policy import WorkspacePolicy
from .providers.base import ModelProvider
from .providers.types import Usage
from .state import RunState
from .tools.base import ToolCall, ToolContext, ToolResult
from .tools.command import RunCommandTool
from .tools.files import ListFilesTool, ReadFileTool
from .tools.finish import FinishTool
from .tools.patch import ApplyPatchTool
from .tools.registry import ToolRegistry, action_fingerprint
from .trace import EventTracer
from .verify import capture_integrity_baseline


class Lifecycle(StrEnum):
    INIT = "INIT"
    ORIENT = "ORIENT"
    DECIDE = "DECIDE"
    ACT = "ACT"
    OBSERVE = "OBSERVE"
    VERIFY = "VERIFY"
    SUCCESS = "SUCCESS"
    FAILED_SAFE = "FAILED_SAFE"


class RunOutcome(StrEnum):
    VERIFIED_SUCCESS = "VERIFIED_SUCCESS"
    INCOMPLETE = "INCOMPLETE"
    POLICY_DENIED = "POLICY_DENIED"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    FAILED_SAFE = "FAILED_SAFE"


@dataclass(frozen=True, slots=True)
class AgentSettings:
    repeated_action_limit: int = 3
    context_limits: ContextLimits = field(default_factory=ContextLimits)


@dataclass(slots=True)
class AgentResult:
    outcome: RunOutcome
    exit_code: int
    reason: str
    state: RunState
    trace_path: Path
    duration_ms: float
    model_requests: int
    usage: Usage

    def summary(self) -> dict[str, object]:
        return {
            "run_id": self.state.run_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "turns": self.state.turn,
            "model_requests": self.model_requests,
            "tool_calls": self.state.tool_calls_requested,
            "files_read": sorted(self.state.files_read),
            "files_modified": sorted(self.state.files_modified),
            "commands": len(self.state.commands_run),
            "verification_attempts": len(self.state.verification_runs),
            "policy_denials": self.state.policy_denials,
            "duration_ms": self.duration_ms,
            "token_usage": self.usage.as_dict(),
            "trace_path": str(self.trace_path),
        }


ApprovalCallback = Callable[[ToolCall], bool]
Reporter = Callable[[str, str], None]


class Agent:
    def __init__(
        self,
        task: str,
        config: HarnessConfig,
        provider: ModelProvider,
        *,
        settings: AgentSettings | None = None,
        approval_callback: ApprovalCallback | None = None,
        reporter: Reporter | None = None,
    ) -> None:
        self.task = task
        self.config = config
        self.provider = provider
        self.settings = settings or AgentSettings()
        self.approval_callback = approval_callback
        self.reporter = reporter or (lambda _event, _message: None)
        self._reported_phases: set[str] = set()
        self.registry = ToolRegistry([ListFilesTool(), ReadFileTool(), ApplyPatchTool(), RunCommandTool(), FinishTool()])

    def run(self) -> AgentResult:
        started = time.monotonic()
        state = RunState(self.task, self.config.workspace)
        capture_integrity_baseline(state, self.config)
        trace_path = self.config.trace_dir / f"{state.run_id}.jsonl"
        usage = Usage()
        model_requests = 0
        context = ContextManager(self.task, self.config.workspace, self.config.verification_command, self.settings.context_limits)
        cache = ObservationCache()
        policy = WorkspacePolicy(self.config)
        with EventTracer(trace_path, state.run_id) as tracer:
            tool_context = ToolContext(self.config, policy, state, cache, tracer)
            tracer.record("run.started", task=self.task, workspace=str(self.config.workspace), provider=self.provider.provider_name, model=self.provider.model)
            self.reporter("run", f"Started run {state.run_id}")
            self._transition(tracer, Lifecycle.INIT, Lifecycle.ORIENT)
            for turn in range(1, self.config.max_turns + 1):
                state.turn = turn
                self._transition(tracer, Lifecycle.ORIENT if turn == 1 else Lifecycle.OBSERVE, Lifecycle.DECIDE, turn)
                messages = context.messages()
                tracer.record("model.request.started", turn=turn, provider=self.provider.provider_name, model=self.provider.model, message_count=len(messages))
                request_started = time.monotonic()
                model_requests += 1
                try:
                    response = self.provider.generate(messages, self.registry.definitions())
                except HarnessError as exc:
                    tracer.record("model.request.failed", turn=turn, duration_ms=round((time.monotonic() - request_started) * 1000, 3), error=exc.as_dict())
                    self.reporter("warning", f"{exc.code}: {exc.message}" + (" (recoverable)" if exc.recoverable else ""))
                    outcome = RunOutcome.PROVIDER_FAILURE if exc.category in {ErrorCategory.PROVIDER, ErrorCategory.CONFIGURATION} else RunOutcome.FAILED_SAFE
                    return self._finish_result(outcome, 4, exc.message, state, trace_path, started, model_requests, usage, tracer)
                if response.usage:
                    usage = Usage(usage.input_tokens + response.usage.input_tokens, usage.output_tokens + response.usage.output_tokens, usage.cached_tokens + response.usage.cached_tokens)
                tracer.record("model.request.completed", turn=turn, duration_ms=round((time.monotonic() - request_started) * 1000, 3), provider_request_id=response.provider_request_id, finish_reason=response.finish_reason, tool_calls=len(response.tool_calls), retry_count=response.retry_count, token_usage=response.usage.as_dict() if response.usage else None)
                for repair in response.raw_metadata.get("tool_name_repairs", []):
                    tracer.record("provider.tool_call.normalized", turn=turn, **repair)
                    self.reporter(
                        "warning",
                        f"Normalized provider tool name {repair['raw_tool_name']} -> {repair['normalized_tool_name']}",
                    )
                if not response.tool_calls:
                    fingerprint = "model_text:" + hashlib.sha256((response.text or "").encode()).hexdigest()
                    state.repeated_actions[fingerprint] = state.repeated_actions.get(fingerprint, 0) + 1
                    context.add_text_only(response)
                    self.reporter("warning", "Model returned text without a tool call; hidden reasoning/content was not displayed")
                    self._transition(tracer, Lifecycle.DECIDE, Lifecycle.OBSERVE, turn)
                    if state.repeated_actions[fingerprint] >= self.settings.repeated_action_limit:
                        return self._finish_result(RunOutcome.FAILED_SAFE, 2, "REPEATED_ACTION_LOOP", state, trace_path, started, model_requests, usage, tracer)
                    continue
                self._transition(tracer, Lifecycle.DECIDE, Lifecycle.ACT, turn)
                results: list[ToolResult] = []
                for call in response.tool_calls:
                    state.record_tool_call_request()
                    tracer.record("model.tool_call.requested", turn=turn, tool_call_id=call.id, tool=call.name, arguments=call.arguments)
                    self._report_tool_start(call, state)
                    approval = self._approve(call, state, tracer)
                    if approval is False:
                        self.reporter("policy", f"Approval denied for {call.name}")
                        result = ToolResult(call.id, call.name, False, error_code="APPROVAL_DENIED", error_message="user denied the requested action", recoverable=False)
                        results.append(result)
                        context.add_exchange(response, results)
                        return self._finish_result(RunOutcome.POLICY_DENIED, 3, "APPROVAL_DENIED", state, trace_path, started, model_requests, usage, tracer)
                    result = self.registry.dispatch(call, tool_context)
                    results.append(result)
                    self._report_tool_result(call, result, state)
                    if call.name == "finish":
                        self._transition(tracer, Lifecycle.ACT, Lifecycle.VERIFY, turn)
                        if result.ok and result.data and result.data.get("accepted"):
                            return self._finish_result(RunOutcome.VERIFIED_SUCCESS, 0, "completion independently verified", state, trace_path, started, model_requests, usage, tracer)
                    fingerprint = action_fingerprint(call)
                    if state.repeated_actions.get(fingerprint, 0) >= self.settings.repeated_action_limit:
                        context.add_exchange(response, results)
                        return self._finish_result(RunOutcome.FAILED_SAFE, 2, "REPEATED_ACTION_LOOP", state, trace_path, started, model_requests, usage, tracer)
                context.add_exchange(response, results)
                self._transition(tracer, Lifecycle.ACT, Lifecycle.OBSERVE, turn)
            return self._finish_result(RunOutcome.INCOMPLETE, 2, "MAX_TURNS_EXCEEDED", state, trace_path, started, model_requests, usage, tracer)

    def _announce(self, phase: str, message: str) -> None:
        if phase not in self._reported_phases:
            self._reported_phases.add(phase)
            self.reporter("agent", message)

    def _report_tool_start(self, call: ToolCall, state: RunState) -> None:
        arguments = call.arguments if isinstance(call.arguments, dict) else {}
        label = self._tool_label(call)
        if call.name == "list_files":
            self._announce("repository", "Inspecting repository...")
            self.reporter("tool", label)
        elif call.name == "read_file":
            self._announce("implementation", "Inspecting failing implementation...")
            self.reporter("tool", label)
        elif call.name == "apply_patch":
            self._announce("repair", "Applying minimal repair...")
            self.reporter("tool", label)
        elif call.name == "run_command":
            if state.files_modified:
                self.reporter("verify", label.removeprefix("run_command: "))
            else:
                self._announce("repository", "Inspecting repository...")
                self.reporter("tool", label)
        else:
            self.reporter("tool", label)

    @staticmethod
    def _tool_label(call: ToolCall) -> str:
        arguments = call.arguments if isinstance(call.arguments, dict) else {}
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "?", call.name)[:80] or "unknown_tool"
        if call.name == "list_files":
            path = arguments.get("path", ".")
            depth = arguments.get("depth")
            return f"list_files: {path}" + (f" (depth {depth})" if depth is not None else "")
        if call.name == "read_file":
            path = arguments.get("path", "")
            start = arguments.get("start_line")
            end = arguments.get("end_line")
            lines = f" (lines {start or 1}-{end})" if end is not None else ""
            return f"read_file: {path}{lines}"
        if call.name == "apply_patch":
            return f"apply_patch: {arguments.get('path', '')}"
        if call.name == "run_command":
            command = arguments.get("command", "")
            cwd = arguments.get("cwd")
            return f"run_command: {command}" + (f" (cwd {cwd})" if cwd not in {None, "."} else "")
        if call.name == "finish":
            return "finish"
        return safe_name

    def _report_tool_result(self, call: ToolCall, result: ToolResult, state: RunState) -> None:
        data = result.data or {}
        if not result.ok:
            code = result.error_code or "TOOL_FAILED"
            detail = code + (f": {result.error_message}" if result.error_message else "")
            policy_codes = {
                "APPROVAL_DENIED",
                "COMMAND_DENIED",
                "CWD_OUTSIDE_WORKSPACE",
                "PATH_OUTSIDE_WORKSPACE",
                "PATCH_PATH_MISMATCH",
                "PROTECTED_PATH",
                "SENSITIVE_PATH",
            }
            if call.name == "finish":
                self.reporter("verify", f"rejected: {detail}")
            elif code in policy_codes:
                self.reporter("policy", f"denied: {detail}")
            else:
                suffix = " (recoverable)" if result.recoverable else ""
                self.reporter("warning", detail + suffix)
            return
        if call.name == "run_command":
            exit_code = data.get("exit_code")
            summary = self._test_summary(str(data.get("stdout", "")), exit_code) or f"exit {exit_code}"
            detail = f"{summary} (exit {exit_code})"
            if data.get("is_verification") and state.files_modified:
                status = "passed" if exit_code == 0 else "failed"
                self.reporter("verify", f"{status}: {detail}")
            else:
                self.reporter("result", detail)
        elif call.name == "apply_patch":
            path = data.get("path") or (call.arguments.get("path", "") if isinstance(call.arguments, dict) else "")
            changed = data.get("changed_lines")
            detail = f"patched {path}"
            if changed is not None:
                detail += f" ({changed} changed lines)"
            self.reporter("result", detail)
        elif call.name == "list_files":
            self.reporter("result", f"{len(data.get('entries') or [])} entries")
        elif call.name == "read_file":
            self.reporter("result", f"read {data.get('path', '')} lines {data.get('start_line', 1)}-{data.get('end_line', '?')}")
        elif call.name == "finish":
            self.reporter("verify", "completion accepted")
        else:
            self.reporter("result", f"{call.name} completed")
    @staticmethod
    def _test_summary(output: str, exit_code: int | None = None) -> str | None:
        for line in reversed(output.splitlines()):
            matches = re.findall(r"\b\d+\s+(?:failed|passed|errors?|skipped|xfailed|xpassed)\b", line)
            if matches:
                return ", ".join(matches)
        failed = len(re.findall(r"(?m)^FAILED\s+", output))
        if failed:
            return f"{failed} failed"
        if exit_code == 0:
            for line in output.splitlines():
                if "[100%]" not in line:
                    continue
                progress = line.split("[", 1)[0]
                passed = progress.count(".")
                if passed:
                    return f"{passed} passed"
        return None

    def _approve(self, call: ToolCall, state: RunState, tracer: EventTracer) -> bool | None:
        risky = call.name in {"apply_patch", "run_command"}
        if self.config.approval_mode != "interactive" or not risky:
            return None
        tracer.record("approval.requested", turn=state.turn, tool_call_id=call.id, tool=call.name)
        granted = bool(self.approval_callback and self.approval_callback(call))
        state.approvals.append({"tool_call_id": call.id, "tool": call.name, "granted": granted})
        tracer.record("approval.granted" if granted else "approval.denied", turn=state.turn, tool_call_id=call.id, tool=call.name)
        return granted

    @staticmethod
    def _transition(tracer: EventTracer, source: Lifecycle, destination: Lifecycle, turn: int | None = None) -> None:
        tracer.record("state.transition", turn=turn, source=source.value, destination=destination.value)

    def _finish_result(self, outcome: RunOutcome, exit_code: int, reason: str, state: RunState, trace_path: Path, started: float, model_requests: int, usage: Usage, tracer: EventTracer) -> AgentResult:
        terminal = Lifecycle.SUCCESS if outcome == RunOutcome.VERIFIED_SUCCESS else Lifecycle.FAILED_SAFE
        tracer.record("run.completed" if terminal == Lifecycle.SUCCESS else "run.failed", outcome=outcome.value, reason=reason, turns=state.turn, model_requests=model_requests, token_usage=usage.as_dict())
        message = outcome.value if outcome == RunOutcome.VERIFIED_SUCCESS else f"{outcome.value}: {reason}"
        self.reporter("done", message)
        return AgentResult(outcome, exit_code, reason, state, trace_path, round((time.monotonic() - started) * 1000, 3), model_requests, usage)
