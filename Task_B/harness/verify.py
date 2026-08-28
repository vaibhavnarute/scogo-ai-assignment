"""Independent completion and repository-integrity checks."""

from __future__ import annotations

from dataclasses import dataclass

from .config import HarnessConfig
from .integrity import snapshot_workspace
from .state import RunState


def _protected_snapshot(config: HarnessConfig, workspace_snapshot: dict[str, str] | None = None) -> dict[str, str]:
    workspace_snapshot = workspace_snapshot or snapshot_workspace(config)
    snapshot: dict[str, str] = {}
    for protected in config.protected_paths:
        prefix = protected.strip("/\\").replace("\\", "/")
        matches = {
            path: digest
            for path, digest in workspace_snapshot.items()
            if path == prefix or path.startswith(prefix + "/")
        }
        snapshot.update(matches or {f"{prefix}/<missing>": "MISSING"})
    return snapshot


def capture_integrity_baseline(state: RunState, config: HarnessConfig) -> None:
    state.workspace_hashes = snapshot_workspace(config)
    state.protected_hashes = _protected_snapshot(config, state.workspace_hashes)


@dataclass(frozen=True, slots=True)
class VerificationDecision:
    accepted: bool
    error_code: str | None
    reason: str
    evidence: dict[str, object]


class Verifier:
    def __init__(self, config: HarnessConfig) -> None:
        self.config = config

    def evaluate_finish(self, state: RunState) -> VerificationDecision:
        current_workspace = snapshot_workspace(self.config)
        if state.integrity_violations:
            return VerificationDecision(False, "INTEGRITY_VIOLATION", "repository mutations occurred outside apply_patch", {"violations": state.integrity_violations})
        if current_workspace != state.workspace_hashes:
            changed = sorted(
                set(current_workspace) ^ set(state.workspace_hashes)
                | {
                    path
                    for path in current_workspace.keys() & state.workspace_hashes.keys()
                    if current_workspace[path] != state.workspace_hashes[path]
                }
            )
            return VerificationDecision(False, "INTEGRITY_VIOLATION", "repository changed outside a recorded tool mutation", {"paths": changed})
        current_protected = _protected_snapshot(self.config, current_workspace)
        if current_protected != state.protected_hashes:
            return VerificationDecision(False, "INTEGRITY_VIOLATION", "protected repository paths changed", {"expected": state.protected_hashes, "actual": current_protected})
        verification = state.latest_verification
        if verification is None:
            return VerificationDecision(False, "VERIFICATION_MISSING", "the configured verification command has not run", {})
        mutation = state.latest_mutation
        if mutation is not None and verification.command_sequence <= mutation.sequence:
            return VerificationDecision(False, "VERIFICATION_MISSING", "verification did not run after the latest mutation", {"mutation_sequence": mutation.sequence, "verification_command_sequence": verification.command_sequence})
        if not verification.passed:
            return VerificationDecision(False, "VERIFICATION_FAILED", "the latest configured verification command did not pass", {"exit_code": verification.exit_code})
        return VerificationDecision(True, None, "completion independently verified", {"verification_sequence": verification.sequence, "verification_command_sequence": verification.command_sequence, "exit_code": verification.exit_code})

