"""Normalized provider response and usage objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..tools.base import ToolCall


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, int]:
        return {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "cached_tokens": self.cached_tokens, "total_tokens": self.total_tokens}


@dataclass(slots=True)
class ModelResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None
    finish_reason: str | None = None
    provider_request_id: str | None = None
    retry_count: int = 0
    raw_metadata: dict[str, Any] = field(default_factory=dict)

