"""Visible, reproducible configuration for one harness session."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ErrorCategory, HarnessError


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    workspace: Path
    verification_command: str = "python -m pytest -q"
    max_turns: int = 20
    max_list_depth: int = 4
    max_list_entries: int = 500
    max_file_bytes: int = 256_000
    max_read_lines: int = 500
    max_command_output_bytes: int = 64_000
    command_timeout_seconds: float = 30.0
    approval_mode: str = "safe"
    protected_paths: tuple[str, ...] = ("tests", "fixture.json")
    excluded_paths: tuple[str, ...] = (
        ".git",
        ".harness_runs",
        ".pytest_cache",
        ".coverage",
        "__pycache__",
        "htmlcov",
        "venv",
        ".venv",
    )
    sensitive_patterns: tuple[str, ...] = (
        ".env",
        ".env.*",
        "*.pem",
        "*.key",
        "credentials*",
        "secrets*",
    )
    trace_dir: Path | None = field(default=None)

    def __post_init__(self) -> None:
        workspace = self.workspace.expanduser().resolve()
        if not workspace.is_dir():
            raise HarnessError("WORKSPACE_INVALID", ErrorCategory.CONFIGURATION, f"workspace is not an existing directory: {workspace}", False)
        if self.approval_mode not in {"safe", "interactive", "yes"}:
            raise HarnessError("CONFIG_INVALID_APPROVAL_MODE", ErrorCategory.CONFIGURATION, "approval_mode must be 'safe', 'interactive', or 'yes'", False)
        for name, value in (
            ("max_turns", self.max_turns),
            ("max_list_depth", self.max_list_depth),
            ("max_list_entries", self.max_list_entries),
            ("max_file_bytes", self.max_file_bytes),
            ("max_read_lines", self.max_read_lines),
            ("max_command_output_bytes", self.max_command_output_bytes),
        ):
            if value <= 0:
                raise HarnessError("CONFIG_INVALID_LIMIT", ErrorCategory.CONFIGURATION, f"{name} must be positive", False)
        if self.command_timeout_seconds <= 0:
            raise HarnessError("CONFIG_INVALID_LIMIT", ErrorCategory.CONFIGURATION, "command_timeout_seconds must be positive", False)
        object.__setattr__(self, "workspace", workspace)
        trace_dir = self.trace_dir or Path(__file__).resolve().parent.parent / ".harness_runs"
        object.__setattr__(self, "trace_dir", trace_dir.expanduser().resolve())

    @classmethod
    def from_fixture(cls, workspace: Path, **overrides: object) -> "HarnessConfig":
        """Load the evaluator contract and always protect its metadata file."""
        workspace = workspace.expanduser().resolve()
        metadata_path = workspace / "fixture.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError("CONFIG_INVALID_FIXTURE", ErrorCategory.CONFIGURATION, "fixture.json is missing or invalid", False, {"error_type": type(exc).__name__}) from exc
        verification_command = metadata.get("verification_command")
        protected_paths = metadata.get("protected_paths", [])
        if not isinstance(verification_command, str) or not verification_command.strip():
            raise HarnessError("CONFIG_INVALID_FIXTURE", ErrorCategory.CONFIGURATION, "fixture verification_command is invalid", False)
        if not isinstance(protected_paths, list) or not all(isinstance(path, str) and path for path in protected_paths):
            raise HarnessError("CONFIG_INVALID_FIXTURE", ErrorCategory.CONFIGURATION, "fixture protected_paths is invalid", False)
        protected = tuple(dict.fromkeys([*protected_paths, "fixture.json"]))
        return cls(workspace=workspace, verification_command=verification_command, protected_paths=protected, **overrides)

