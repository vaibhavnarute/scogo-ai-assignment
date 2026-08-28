"""Append-only JSONL event tracing with centralized redaction."""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, TextIO

from .errors import ErrorCategory, HarnessError
from .env import configured_secret_values

_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "password",
    "secret",
    "access_token",
    "refresh_token",
    "credential",
    "credentials",
    "token",
}
_SENSITIVE_SUFFIXES = ("_api_key", "_password", "_secret", "_access_token", "_refresh_token", "_credential", "_credentials")
_VALUE_PATTERNS = (
    re.compile(r"(?i)bearer\s+[a-z0-9._~+/-]+"),
    re.compile(r"(?i)(api[_-]?key|password|token)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)(--?(?:api[-_]?key|access[-_]?token|refresh[-_]?token|token|password|secret|authorization)\s+)(?:\"[^\"]*\"|'[^']*'|\S+)"),
)


def sanitize_for_log(value: Any, key: str = "") -> Any:
    folded_key = key.casefold()
    if folded_key in _SENSITIVE_KEYS or folded_key.endswith(_SENSITIVE_SUFFIXES):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): sanitize_for_log(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_log(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        sanitized = value
        for secret in configured_secret_values():
            sanitized = sanitized.replace(secret, "[REDACTED]")
        for pattern in _VALUE_PATTERNS:
            sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized
    return value


class EventTracer:
    def __init__(self, trace_path: Path, run_id: str) -> None:
        self.path = trace_path
        self.run_id = run_id
        self._lock = Lock()
        self._started = time.monotonic()
        self._file: TextIO | None = None
        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = trace_path.open("a", encoding="utf-8", newline="\n")
        except OSError as exc:
            raise HarnessError(
                "TRACE_INITIALIZATION_FAILED",
                ErrorCategory.INTERNAL,
                "could not initialize the required run trace",
                False,
                {"error_type": type(exc).__name__},
            ) from exc

    def record(self, event: str, **fields: Any) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "elapsed_ms": round((time.monotonic() - self._started) * 1000, 3),
            "event": event,
            "run_id": self.run_id,
            **fields,
        }
        line = json.dumps(sanitize_for_log(payload), ensure_ascii=False, sort_keys=True)
        try:
            with self._lock:
                if self._file is None:
                    raise OSError("trace is closed")
                self._file.write(line + "\n")
                self._file.flush()
        except OSError as exc:
            raise HarnessError(
                "TRACE_WRITE_FAILED",
                ErrorCategory.INTERNAL,
                "could not persist a required trace event",
                False,
                {"event": event, "error_type": type(exc).__name__},
            ) from exc

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self) -> "EventTracer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

