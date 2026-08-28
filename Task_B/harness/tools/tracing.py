"""Trace emission for tool dispatch, isolated from registry lookup and execution."""

from __future__ import annotations

from ..errors import HarnessError
from ..trace import EventTracer
from .base import ToolCall, ToolResult


class ToolTraceEmitter:
    def __init__(self, tracer: EventTracer | None) -> None:
        self.tracer = tracer

    def validation_passed(self, call: ToolCall) -> None:
        if self.tracer:
            self.tracer.record("tool.validation.passed", tool_call_id=call.id, tool=call.name, arguments=call.arguments)

    def repeated_action(self, call: ToolCall, repeats: int) -> None:
        if self.tracer and repeats > 1:
            self.tracer.record("agent.repeated_action", tool_call_id=call.id, tool=call.name, duplicate_count=repeats - 1)

    def started(self, call: ToolCall) -> None:
        if self.tracer:
            self.tracer.record("tool.started", tool_call_id=call.id, tool=call.name)

    def completed(self, call: ToolCall, result: ToolResult) -> None:
        if self.tracer:
            self.tracer.record(
                "tool.completed" if result.ok else "tool.failed",
                tool_call_id=call.id,
                tool=call.name,
                ok=result.ok,
                cached=result.cached,
                error_code=result.error_code,
                result=result.data,
            )

    def failed(self, call: ToolCall, error: HarnessError) -> None:
        if self.tracer:
            self.tracer.record("tool.failed", tool_call_id=call.id, tool=call.name, error=error.as_dict())