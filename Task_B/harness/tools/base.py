"""Provider-neutral tool contracts and execution context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..cache import ObservationCache
from ..config import HarnessConfig
from ..policy import WorkspacePolicy
from ..state import RunState
from ..trace import EventTracer


class RiskCategory(StrEnum):
    READ_ONLY = "READ_ONLY"
    MUTATION = "MUTATION"
    COMMAND = "COMMAND"
    COMPLETION = "COMPLETION"


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    tool_call_id: str
    tool: str
    ok: bool
    data: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    recoverable: bool | None = None
    error_details: dict[str, Any] | None = None
    cached: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool": self.tool,
            "ok": self.ok,
            "data": self.data,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "recoverable": self.recoverable,
            "error_details": self.error_details,
            "cached": self.cached,
        }


@dataclass(slots=True)
class ToolContext:
    config: HarnessConfig
    policy: WorkspacePolicy
    state: RunState
    cache: ObservationCache
    tracer: EventTracer | None = None


class BaseTool(ABC):
    name: str
    description: str
    schema: dict[str, Any]
    risk: RiskCategory
    cacheable: bool = False

    @abstractmethod
    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        raise NotImplementedError

