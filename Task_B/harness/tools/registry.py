"""Minimal tool lookup and dispatch boundary."""

from __future__ import annotations

import json
from typing import Any

from ..errors import ErrorCategory, HarnessError
from .base import BaseTool, ToolCall, ToolContext, ToolResult
from .tracing import ToolTraceEmitter
from .validation import ToolValidator


def action_fingerprint(call: ToolCall) -> str:
    arguments = json.dumps(call.arguments, sort_keys=True, separators=(",", ":"))
    return f"{call.name}:{arguments}"


class ToolRegistry:
    def __init__(self, tools: list[BaseTool] | None = None, *, validator: ToolValidator | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._validator = validator or ToolValidator()
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def definitions(self) -> list[dict[str, Any]]:
        return [{"name": tool.name, "description": tool.description, "parameters": tool.schema} for tool in self._tools.values()]

    def dispatch(self, call: ToolCall, context: ToolContext) -> ToolResult:
        trace = ToolTraceEmitter(context.tracer)
        tool = self._tools.get(call.name)
        if tool is None:
            return self._failure(call, HarnessError("UNKNOWN_TOOL", ErrorCategory.PROTOCOL, f"unknown tool: {call.name}", True), trace)
        try:
            self._validator.validate(tool.schema, call.arguments)
            trace.validation_passed(call)
            fingerprint = action_fingerprint(call)
            repeats = context.state.repeated_actions.get(fingerprint, 0) + 1
            context.state.repeated_actions[fingerprint] = repeats
            trace.repeated_action(call, repeats)
            trace.started(call)
            result = tool.execute(call, context)
            trace.completed(call, result)
            return result
        except HarnessError as exc:
            return self._failure(call, exc, trace)
        except Exception as exc:  # nearest subsystem boundary; never leak raw details
            error = HarnessError("INTERNAL_TOOL_ERROR", ErrorCategory.INTERNAL, "unexpected tool failure", False, {"error_type": type(exc).__name__})
            return self._failure(call, error, trace)

    @staticmethod
    def _failure(call: ToolCall, error: HarnessError, trace: ToolTraceEmitter) -> ToolResult:
        trace.failed(call, error)
        return ToolResult(
            call.id,
            call.name,
            False,
            error_code=error.code,
            error_message=error.message,
            recoverable=error.recoverable,
            error_details=error.details or None,
        )