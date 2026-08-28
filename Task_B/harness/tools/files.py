"""Bounded repository orientation and file-read tools."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..errors import ErrorCategory, HarnessError
from ..policy import PathIntent
from .base import BaseTool, RiskCategory, ToolCall, ToolContext, ToolResult


def _file_fingerprint(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def _walk_bounded(root: Path, depth: int, context: ToolContext) -> list[tuple[Path, str]]:
    """Walk only included directories and never follow directory symlinks."""
    candidates: list[tuple[Path, str]] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, current_depth = stack.pop()
        if current_depth >= depth:
            continue
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name.casefold(), reverse=True)
        except OSError as exc:
            raise HarnessError("LIST_ERROR", ErrorCategory.FILESYSTEM, "could not list directory", True, {"error_type": type(exc).__name__}) from exc
        for child in children:
            if context.policy.is_excluded(child):
                continue
            if child.is_symlink():
                try:
                    child.resolve(strict=False).relative_to(context.config.workspace)
                    entry_type = "symlink"
                except (OSError, ValueError):
                    entry_type = "blocked_symlink"
                candidates.append((child, entry_type))
                continue
            if child.is_dir():
                candidates.append((child, "directory"))
                stack.append((child, current_depth + 1))
            elif child.is_file():
                candidates.append((child, "file"))
    return sorted(candidates, key=lambda item: item[0].as_posix().casefold())


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "List a bounded portion of the workspace tree."
    risk = RiskCategory.READ_ONLY
    cacheable = True
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative directory; use '.' for the workspace root."},
            "depth": {"type": "integer", "minimum": 0},
        },
        "additionalProperties": False,
    }

    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        requested = call.arguments.get("path", ".")
        # Some OpenAI-compatible models serialize the optional root path as an
        # empty string. Root listing is read-only and already defaults to '.',
        # so normalize only this tool without weakening general path policy.
        if isinstance(requested, str) and not requested.strip():
            requested = "."
        depth = call.arguments.get("depth", 2)
        if depth > context.config.max_list_depth:
            raise HarnessError("OUTPUT_LIMIT", ErrorCategory.FILESYSTEM, "requested depth exceeds the configured maximum", True)
        root = context.policy.resolve_path(requested, PathIntent.READ)
        if not root.is_dir():
            raise HarnessError("NOT_A_DIRECTORY", ErrorCategory.FILESYSTEM, "path is not a directory", True)
        candidates = _walk_bounded(root, depth, context)
        fingerprint = tuple(
            (path.relative_to(root).as_posix(), entry_type, path.lstat().st_mtime_ns, path.lstat().st_size)
            for path, entry_type in candidates
        )
        key = (self.name, root, depth)
        cached = context.cache.get(key, fingerprint)
        if cached is not None:
            cached["cached"] = True
            return ToolResult(call.id, self.name, True, cached, cached=True)
        truncated = len(candidates) > context.config.max_list_entries
        entries = [
            {"path": path.relative_to(context.config.workspace).as_posix(), "type": entry_type}
            for path, entry_type in candidates[: context.config.max_list_entries]
        ]
        data = {"path": root.relative_to(context.config.workspace).as_posix() or ".", "entries": entries, "truncated": truncated, "cached": False}
        context.cache.put(key, data, fingerprint, {root})
        return ToolResult(call.id, self.name, True, data)


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a bounded UTF-8 text-file range from the workspace."
    risk = RiskCategory.READ_ONLY
    cacheable = True
    schema = {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer", "minimum": 1}, "end_line": {"type": "integer", "minimum": 1}}, "required": ["path"], "additionalProperties": False}

    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        path = context.policy.resolve_path(call.arguments["path"], PathIntent.READ)
        if not path.is_file():
            raise HarnessError("NOT_A_FILE", ErrorCategory.FILESYSTEM, "path is not a file", True)
        size = path.stat().st_size
        if size > context.config.max_file_bytes:
            raise HarnessError("FILE_TOO_LARGE", ErrorCategory.FILESYSTEM, "file exceeds the configured size limit", True, {"bytes": size})
        start = call.arguments.get("start_line", 1)
        end = call.arguments.get("end_line", start + context.config.max_read_lines - 1)
        if end < start or end - start + 1 > context.config.max_read_lines:
            raise HarnessError("INVALID_LINE_RANGE", ErrorCategory.FILESYSTEM, "line range is invalid or too large", True)
        fingerprint = _file_fingerprint(path)
        key = (self.name, path, start, end)
        cached = context.cache.get(key, fingerprint)
        if cached is not None:
            cached["cached"] = True
            context.state.record_file_read(path)
            return ToolResult(call.id, self.name, True, cached, cached=True)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise HarnessError("BINARY_FILE", ErrorCategory.FILESYSTEM, "file is not valid UTF-8 text", True) from exc
        selected = lines[start - 1 : end]
        actual_end = start + len(selected) - 1 if selected else start - 1
        data: dict[str, Any] = {"path": path.relative_to(context.config.workspace).as_posix(), "content": "\n".join(selected), "start_line": start, "end_line": actual_end, "total_lines": len(lines), "truncated": end < len(lines), "cached": False}
        context.cache.put(key, data, fingerprint, {path})
        context.state.record_file_read(path)
        return ToolResult(call.id, self.name, True, data)

