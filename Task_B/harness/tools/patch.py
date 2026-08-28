"""Localized unified-diff patch application without shell delegation."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..errors import ErrorCategory, HarnessError
from ..integrity import diff_snapshots, snapshot_workspace
from ..policy import PathIntent
from .base import BaseTool, RiskCategory, ToolCall, ToolContext, ToolResult
from .patch_formats import normalize_model_patch

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")
_METADATA_PREFIXES = (
    "diff --git ",
    "index ",
    "--- ",
    "+++ ",
    "new file mode ",
    "deleted file mode ",
    "old mode ",
    "new mode ",
)


@dataclass(frozen=True, slots=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[str, ...]
    old_no_newline: bool = False
    new_no_newline: bool = False


def parse_unified_diff(patch: str) -> list[Hunk]:
    if not isinstance(patch, str) or not patch.strip():
        raise HarnessError("PATCH_INVALID", ErrorCategory.PATCH, "patch must be non-empty", True)
    lines = patch.splitlines()
    hunks: list[Hunk] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(_METADATA_PREFIXES):
            index += 1
            continue
        match = _HUNK.match(line)
        if not match:
            raise HarnessError("PATCH_INVALID", ErrorCategory.PATCH, "expected a unified-diff hunk header", True, {"line": index + 1})
        old_start, old_count, new_start, new_count = match.groups()
        body: list[str] = []
        old_no_newline = False
        new_no_newline = False
        index += 1
        while index < len(lines) and not lines[index].startswith("@@ "):
            current = lines[index]
            if current == "\\ No newline at end of file":
                if body and body[-1].startswith("-"):
                    old_no_newline = True
                elif body and body[-1].startswith("+"):
                    new_no_newline = True
                index += 1
                continue
            if not current or current[0] not in {" ", "+", "-"}:
                raise HarnessError("PATCH_INVALID", ErrorCategory.PATCH, "invalid hunk line prefix", True, {"line": index + 1})
            body.append(current)
            index += 1
        hunk = Hunk(
            int(old_start),
            int(old_count) if old_count is not None else 1,
            int(new_start),
            int(new_count) if new_count is not None else 1,
            tuple(body),
            old_no_newline,
            new_no_newline,
        )
        observed_old = sum(line[0] in {" ", "-"} for line in body)
        observed_new = sum(line[0] in {" ", "+"} for line in body)
        if observed_old != hunk.old_count or observed_new != hunk.new_count:
            raise HarnessError("PATCH_INVALID", ErrorCategory.PATCH, "hunk counts do not match its body", True)
        hunks.append(hunk)
    if not hunks:
        raise HarnessError("PATCH_INVALID", ErrorCategory.PATCH, "patch contains no hunks", True)
    return hunks


def apply_hunks(original: list[str], hunks: list[Hunk]) -> tuple[list[str], int]:
    result = list(original)
    offset = 0
    changed = 0
    previous_old_end = 0
    for hunk in hunks:
        if hunk.old_start < previous_old_end:
            raise HarnessError("PATCH_INVALID", ErrorCategory.PATCH, "patch hunks overlap or are out of order", True)
        position = (hunk.old_start if hunk.old_count == 0 else hunk.old_start - 1) + offset
        old_lines = [line[1:] for line in hunk.lines if line[0] in {" ", "-"}]
        new_lines = [line[1:] for line in hunk.lines if line[0] in {" ", "+"}]
        if result[position : position + len(old_lines)] != old_lines:
            raise HarnessError("PATCH_CONFLICT", ErrorCategory.PATCH, "patch context does not match the current file", True, {"old_start": hunk.old_start})
        result[position : position + len(old_lines)] = new_lines
        offset += len(new_lines) - len(old_lines)
        changed += sum(line[0] in {"+", "-"} for line in hunk.lines)
        previous_old_end = hunk.old_start + hunk.old_count
    return result, changed


class ApplyPatchTool(BaseTool):
    name = "apply_patch"
    description = "Apply a localized unified diff or safe Begin/Update patch envelope to one existing workspace file."
    risk = RiskCategory.MUTATION
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "patch": {"type": "string"}},
        "required": ["path", "patch"],
        "additionalProperties": False,
    }

    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        current_snapshot = snapshot_workspace(context.config)
        if context.state.workspace_hashes:
            drift = diff_snapshots(context.state.workspace_hashes, current_snapshot)
            if drift:
                paths = [change.path for change in drift]
                context.state.record_integrity_violation("EXTERNAL_MUTATION", "<external-before-patch>", paths)
                if context.tracer:
                    context.tracer.record("integrity.violation", tool_call_id=call.id, code="EXTERNAL_MUTATION", paths=paths)
                raise HarnessError(
                    "INTEGRITY_VIOLATION",
                    ErrorCategory.INTEGRITY,
                    "workspace changed outside the harness before patch application",
                    False,
                    {"paths": paths},
                )
        else:
            context.state.workspace_hashes = current_snapshot
        path = context.policy.resolve_path(call.arguments["path"], PathIntent.WRITE)
        if not path.is_file():
            raise HarnessError("NOT_A_FILE", ErrorCategory.FILESYSTEM, "patch target is not a file", True)
        if path.stat().st_size > context.config.max_file_bytes:
            raise HarnessError("FILE_TOO_LARGE", ErrorCategory.FILESYSTEM, "patch target exceeds the configured size limit", True)
        try:
            original_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise HarnessError("BINARY_FILE", ErrorCategory.FILESYSTEM, "patch target is not UTF-8 text", True) from exc
        newline = "\r\n" if "\r\n" in original_text else "\n"
        had_final_newline = original_text.endswith(("\n", "\r"))
        original_lines = original_text.splitlines()
        relative_path = path.relative_to(context.config.workspace).as_posix()
        patch_text = normalize_model_patch(call.arguments["patch"], relative_path, original_lines)
        hunks = parse_unified_diff(patch_text)
        updated_lines, changed_lines = apply_hunks(original_lines, hunks)
        final_newline = had_final_newline
        for hunk in hunks:
            if hunk.old_start + hunk.old_count - 1 >= len(original_lines):
                if hunk.new_no_newline:
                    final_newline = False
                elif hunk.old_no_newline:
                    final_newline = True
        updated_text = newline.join(updated_lines) + (newline if final_newline else "")
        before_hash = hashlib.sha256(original_text.encode("utf-8")).hexdigest()
        after_hash = hashlib.sha256(updated_text.encode("utf-8")).hexdigest()
        if before_hash == after_hash:
            raise HarnessError("PATCH_NO_CHANGES", ErrorCategory.PATCH, "patch produced no content change", True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as temporary:
                temporary.write(updated_text)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            os.replace(temporary_name, path)
        except OSError as exc:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)
            raise HarnessError("PATCH_WRITE_ERROR", ErrorCategory.PATCH, "could not persist patched content", True, {"error_type": type(exc).__name__}) from exc
        mutation = context.state.record_mutation(path, before_hash, after_hash)
        context.state.workspace_hashes = snapshot_workspace(context.config)
        invalidated = context.cache.invalidate_path(path)
        data = {"path": mutation.path, "changed_lines": changed_lines, "before_hash": before_hash, "after_hash": after_hash, "mutation_sequence": mutation.sequence, "cache_entries_invalidated": invalidated}
        if context.tracer:
            context.tracer.record("file.modified", tool_call_id=call.id, **data)
            if invalidated:
                context.tracer.record("cache.invalidated", tool_call_id=call.id, path=mutation.path, entries=invalidated)
        return ToolResult(call.id, self.name, True, data)
