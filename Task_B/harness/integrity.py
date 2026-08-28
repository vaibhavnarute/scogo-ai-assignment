"""Workspace snapshots used to detect mutations outside the patch tool."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from .config import HarnessConfig

MISSING = "<MISSING>"


@dataclass(frozen=True, slots=True)
class RepositoryChange:
    path: str
    before_hash: str
    after_hash: str


def snapshot_workspace(config: HarnessConfig) -> dict[str, str]:
    """Hash relevant files without following directory symlinks or excluded trees."""
    snapshot: dict[str, str] = {}
    excluded = {name.casefold() for name in config.excluded_paths}
    stack = [config.workspace]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name.casefold())
        except OSError:
            continue
        for child in children:
            relative = child.relative_to(config.workspace)
            if any(part.casefold() in excluded for part in relative.parts):
                continue
            try:
                if child.is_symlink():
                    target = child.resolve(strict=False)
                    try:
                        target_value = target.relative_to(config.workspace).as_posix()
                    except ValueError:
                        target_value = "<OUTSIDE_WORKSPACE>"
                    snapshot[relative.as_posix()] = f"SYMLINK:{target_value}"
                elif child.is_dir():
                    stack.append(child)
                elif child.is_file():
                    digest = hashlib.sha256()
                    with child.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(128 * 1024), b""):
                            digest.update(chunk)
                    snapshot[relative.as_posix()] = digest.hexdigest()
            except OSError:
                snapshot[relative.as_posix()] = "<UNREADABLE>"
    return snapshot


def diff_snapshots(before: dict[str, str], after: dict[str, str]) -> list[RepositoryChange]:
    return [
        RepositoryChange(path, before.get(path, MISSING), after.get(path, MISSING))
        for path in sorted(before.keys() | after.keys())
        if before.get(path) != after.get(path)
    ]

