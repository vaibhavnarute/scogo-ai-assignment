"""Deterministic filesystem and command policy."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .config import HarnessConfig
from .command_line import split_command
from .errors import ErrorCategory, HarnessError


class PathIntent(StrEnum):
    READ = "READ"
    WRITE = "WRITE"
    CWD = "CWD"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    reason: str
    requires_approval: bool = False


class WorkspacePolicy:
    _shell_metacharacters = re.compile(r"[|&;<>()`\r\n]")
    _denied_programs = {
        "curl",
        "wget",
        "ssh",
        "scp",
        "ftp",
        "powershell",
        "pwsh",
        "cmd",
        "bash",
        "sh",
        "rm",
        "rmdir",
        "del",
        "format",
        "shutdown",
        "reboot",
    }

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.workspace = config.workspace

    def resolve_path(
        self,
        requested: str,
        intent: PathIntent,
        *,
        must_exist: bool = True,
    ) -> Path:
        if not isinstance(requested, str) or not requested.strip() or "\x00" in requested:
            raise HarnessError(
                "INVALID_PATH",
                ErrorCategory.FILESYSTEM,
                "path must be a non-empty string",
                True,
            )
        raw = Path(requested)
        candidate = raw if raw.is_absolute() else self.workspace / raw
        try:
            resolved = candidate.resolve(strict=False)
            relative = resolved.relative_to(self.workspace)
        except (OSError, ValueError):
            raise HarnessError(
                "PATH_OUTSIDE_WORKSPACE",
                ErrorCategory.POLICY,
                "requested path is outside the workspace",
                True,
                {"path": requested},
            ) from None
        if self._is_sensitive(relative):
            raise HarnessError(
                "SENSITIVE_PATH",
                ErrorCategory.POLICY,
                "access to sensitive paths is blocked",
                True,
                {"path": relative.as_posix()},
            )
        if intent == PathIntent.WRITE and self._is_protected(relative):
            raise HarnessError(
                "PROTECTED_PATH",
                ErrorCategory.POLICY,
                "the path is protected from mutation",
                True,
                {"path": relative.as_posix()},
            )
        if must_exist and not resolved.exists():
            raise HarnessError(
                "NOT_FOUND",
                ErrorCategory.FILESYSTEM,
                "requested path does not exist",
                True,
                {"path": relative.as_posix()},
            )
        return resolved

    def is_excluded(self, path: Path) -> bool:
        try:
            relative = path.absolute().relative_to(self.workspace)
        except ValueError:
            return True
        excluded = {item.casefold() for item in self.config.excluded_paths}
        return any(part.casefold() in excluded for part in relative.parts)

    def check_command(self, command: str, cwd: Path) -> PolicyDecision:
        if not isinstance(command, str) or not command.strip():
            return PolicyDecision(False, "command must be a non-empty string")
        if self._shell_metacharacters.search(command):
            return PolicyDecision(False, "shell metacharacters are not allowed")
        try:
            argv = split_command(command)
        except ValueError:
            return PolicyDecision(False, "command quoting is invalid")
        if not argv:
            return PolicyDecision(False, "command is empty")
        executable = Path(argv[0]).name.casefold()
        if executable.endswith(".exe"):
            executable = executable[:-4]
        if executable in self._denied_programs:
            return PolicyDecision(False, f"program {executable!r} is denied")
        if executable in {"python", "python3", "py"} and "-c" in argv[1:]:
            return PolicyDecision(False, "inline Python execution is denied")
        if command.strip() == self.config.verification_command.strip() and cwd == self.workspace:
            return PolicyDecision(True, "configured verification command")
        if self.config.approval_mode == "yes":
            return PolicyDecision(True, "non-interactive approval mode", True)
        if self.config.approval_mode == "interactive":
            return PolicyDecision(True, "interactive approval required", True)
        return PolicyDecision(False, "command requires approval in safe mode", True)

    def _is_sensitive(self, relative: Path) -> bool:
        for part in relative.parts:
            lowered = part.casefold()
            if any(fnmatch.fnmatch(lowered, pattern.casefold()) for pattern in self.config.sensitive_patterns):
                return True
        return False

    def _is_protected(self, relative: Path) -> bool:
        folded = tuple(part.casefold() for part in relative.parts)
        for protected in self.config.protected_paths:
            parts = tuple(part.casefold() for part in Path(protected).parts)
            if folded[: len(parts)] == parts:
                return True
        return False

