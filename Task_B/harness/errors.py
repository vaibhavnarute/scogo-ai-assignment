"""Stable, secret-safe error types used across harness boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ErrorCategory(StrEnum):
    CONFIGURATION = "CONFIGURATION"
    PROVIDER = "PROVIDER"
    AGENT_LOOP = "AGENT_LOOP"
    PROTOCOL = "PROTOCOL"
    POLICY = "POLICY"
    FILESYSTEM = "FILESYSTEM"
    PATCH = "PATCH"
    COMMAND = "COMMAND"
    TIMEOUT = "TIMEOUT"
    CACHE = "CACHE"
    VERIFICATION = "VERIFICATION"
    INTEGRITY = "INTEGRITY"
    INTERNAL = "INTERNAL"


@dataclass(slots=True)
class HarnessError(Exception):
    code: str
    category: ErrorCategory
    message: str
    recoverable: bool
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "category": self.category.value,
            "message": self.message,
            "recoverable": self.recoverable,
            "details": self.details,
        }

