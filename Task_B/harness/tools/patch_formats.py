"""Safe normalization of common model-generated patch envelopes."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from ..errors import ErrorCategory, HarnessError

_NUMBERED_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(?: .*)?$")


def _normalized_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/").removeprefix("./")
    if normalized.startswith(("a/", "b/")):
        normalized = normalized[2:]
    return PurePosixPath(normalized).as_posix()


def _validate_metadata_path(value: str, expected_path: str) -> None:
    declared_path = _normalized_path(value)
    normalized_expected = _normalized_path(expected_path)
    if declared_path != normalized_expected:
        raise HarnessError(
            "PATCH_PATH_MISMATCH",
            ErrorCategory.POLICY,
            "patch metadata path does not match the requested target",
            True,
            {"declared_path": declared_path, "expected_path": normalized_expected},
        )


def normalize_model_patch(patch: str, expected_path: str, original: list[str]) -> str:
    """Convert safe existing-file model patches into numbered unified hunks."""
    if not isinstance(patch, str) or not patch.strip():
        return patch
    lines = patch.strip().splitlines()
    if lines[0] == "*** Begin Patch":
        while lines and lines[-1] == "*** End Patch":
            lines.pop()
        if len(lines) < 3:
            raise HarnessError(
                "PATCH_UNSUPPORTED_OPERATION",
                ErrorCategory.PATCH,
                "model patch envelope must update one existing file",
                True,
            )
        operation = lines[1]
        if operation.startswith("*** Update File: "):
            _validate_metadata_path(operation.split(":", 1)[1], expected_path)
        elif operation != "*** Update File":
            raise HarnessError(
                "PATCH_UNSUPPORTED_OPERATION",
                ErrorCategory.PATCH,
                "model patch envelope must update one existing file",
                True,
            )
        body = lines[2:]
        if any(line.startswith("*** ") for line in body):
            raise HarnessError(
                "PATCH_UNSUPPORTED_OPERATION",
                ErrorCategory.PATCH,
                "model patch envelope contains multiple or unsupported file operations",
                True,
            )
    else:
        body = list(lines)
        metadata_paths: list[str] = []
        while body and body[0].startswith(("diff --git ", "index ", "--- ", "+++ ")):
            metadata = body.pop(0)
            if metadata.startswith(("--- ", "+++ ")):
                metadata_paths.append(metadata[4:].split("\t", 1)[0])
        for metadata_path in metadata_paths:
            _validate_metadata_path(metadata_path, expected_path)
    if body and all(line != "@@" for line in body):
        return "\n".join(body)

    normalized: list[str] = []
    cursor = 0
    index = 0
    while index < len(body):
        header = body[index]
        if _NUMBERED_HUNK.match(header):
            normalized.extend(body[index:])
            break
        if header != "@@":
            raise HarnessError(
                "PATCH_INVALID",
                ErrorCategory.PATCH,
                "expected a patch-envelope hunk marker",
                True,
                {"line": index + 3},
            )
        index += 1
        hunk_lines: list[str] = []
        while index < len(body) and body[index] != "@@" and not _NUMBERED_HUNK.match(body[index]):
            line = body[index]
            if not line or line[0] not in {" ", "+", "-"}:
                raise HarnessError(
                    "PATCH_INVALID",
                    ErrorCategory.PATCH,
                    "invalid patch-envelope hunk line",
                    True,
                    {"line": index + 3},
                )
            hunk_lines.append(line)
            index += 1
        old_lines = [line[1:] for line in hunk_lines if line[0] in {" ", "-"}]
        new_lines = [line[1:] for line in hunk_lines if line[0] in {" ", "+"}]
        if not old_lines:
            raise HarnessError(
                "PATCH_INVALID",
                ErrorCategory.PATCH,
                "unnumbered insertion requires existing context",
                True,
            )
        matches = [
            start
            for start in range(cursor, len(original) - len(old_lines) + 1)
            if original[start : start + len(old_lines)] == old_lines
        ]
        if len(matches) != 1:
            raise HarnessError(
                "PATCH_CONFLICT",
                ErrorCategory.PATCH,
                "patch-envelope context must match exactly one location",
                True,
                {"matches": len(matches)},
            )
        start = matches[0]
        normalized.append(f"@@ -{start + 1},{len(old_lines)} +{start + 1},{len(new_lines)} @@")
        normalized.extend(hunk_lines)
        cursor = start + len(old_lines)
    return "\n".join(normalized)
