"""Secret-safe human rendering for the live CLI."""

from __future__ import annotations

import re
import sys
from typing import TextIO

from .trace import sanitize_for_log

EVENT_CATEGORIES = frozenset({"run", "agent", "tool", "result", "policy", "verify", "warning", "done"})
_SECRET_FLAG = re.compile(
    r"(?i)(--?(?:api[-_]?key|access[-_]?token|refresh[-_]?token|token|password|secret|authorization)\s+)(?:\"[^\"]*\"|'[^']*'|\S+)"
)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")


def safe_console_text(value: object, *, limit: int = 1000) -> str:
    """Return a single-line, bounded, centrally redacted console value."""
    sanitized = str(sanitize_for_log(value))
    sanitized = _SECRET_FLAG.sub(r"\1[REDACTED]", sanitized)
    sanitized = _CONTROL.sub(" ", sanitized).strip()
    if len(sanitized) > limit:
        return sanitized[: limit - 1] + "…"
    return sanitized


class ConsoleReporter:
    """Render only the supported concise lifecycle categories."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream

    def __call__(self, event: str, message: str) -> None:
        category = event if event in EVENT_CATEGORIES else "warning"
        rendered = safe_console_text(message)
        print(f"[{category}] {rendered}", file=self.stream or sys.stdout, flush=True)