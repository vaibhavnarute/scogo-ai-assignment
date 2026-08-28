"""Minimal, strict environment-file loading for provider credentials."""

from __future__ import annotations

import os
import re
from pathlib import Path

from .errors import ErrorCategory, HarnessError

DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_KEY = re.compile(r"(?i)(?:api[_-]?key|token|secret|password|authorization|credential)")


def is_secret_environment_key(key: str) -> bool:
    """Return whether an environment variable can carry authentication material."""
    return bool(_SECRET_KEY.search(key))


def configured_secret_values() -> tuple[str, ...]:
    """Return configured secret values for redaction, longest first."""
    values = {value for key, value in os.environ.items() if is_secret_environment_key(key) and len(value) >= 4}
    return tuple(sorted(values, key=len, reverse=True))


def repository_subprocess_environment() -> dict[str, str]:
    """Build a child environment that cannot inherit harness/provider credentials."""
    return {key: value for key, value in os.environ.items() if not is_secret_environment_key(key)}


def load_dotenv(path: Path | str | None = None, *, override: bool = False) -> set[str]:
    """Load non-empty values without logging or returning secret contents."""
    env_path = Path(path).expanduser().resolve() if path is not None else DEFAULT_ENV_FILE
    if not env_path.exists():
        return set()
    if not env_path.is_file():
        raise HarnessError(
            "CONFIG_INVALID_ENV_FILE",
            ErrorCategory.CONFIGURATION,
            "env path is not a regular file",
            False,
            {"path": str(env_path)},
        )
    try:
        lines = env_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise HarnessError(
            "CONFIG_ENV_READ_FAILED",
            ErrorCategory.CONFIGURATION,
            "could not read env file",
            False,
            {"path": str(env_path), "error_type": type(exc).__name__},
        ) from exc
    loaded: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise HarnessError(
                "CONFIG_INVALID_ENV_FILE",
                ErrorCategory.CONFIGURATION,
                "env file contains an invalid assignment",
                False,
                {"path": str(env_path), "line": line_number},
            )
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not _KEY.fullmatch(key):
            raise HarnessError(
                "CONFIG_INVALID_ENV_FILE",
                ErrorCategory.CONFIGURATION,
                "env file contains an invalid variable name",
                False,
                {"path": str(env_path), "line": line_number},
            )
        if value and value[0] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise HarnessError(
                    "CONFIG_INVALID_ENV_FILE",
                    ErrorCategory.CONFIGURATION,
                    "env file contains an unterminated quoted value",
                    False,
                    {"path": str(env_path), "line": line_number},
                )
            value = value[1:-1]
        if not value or (key in os.environ and not override):
            continue
        os.environ[key] = value
        loaded.add(key)
    return loaded
