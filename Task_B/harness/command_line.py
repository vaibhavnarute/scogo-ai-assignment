"""Cross-platform command parsing and process-tree termination helpers."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
from typing import Any


def split_command(command: str, *, windows: bool | None = None) -> list[str]:
    """Split without a shell while preserving Windows backslashes and quoted paths."""
    use_windows_rules = os.name == "nt" if windows is None else windows
    if not use_windows_rules:
        return shlex.split(command, posix=True)
    parts = shlex.split(command, posix=False)
    normalized: list[str] = []
    for part in parts:
        if len(part) >= 2 and part[0] == part[-1] and part[0] in {'"', "'"}:
            part = part[1:-1]
        normalized.append(part)
    return normalized


def process_group_options() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Best-effort termination of the direct process and its descendants."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            shell=False,
        )
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
    if process.poll() is None:
        process.kill()

