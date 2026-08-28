"""Compatibility facade over focused session, activity, execution, and integrity state."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .state_components import ExecutionLog, IntegrityTracker, SessionState, WorkspaceActivity
from .state_records import CommandRecord, MutationRecord, VerificationRecord


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(slots=True)
class RunState:
    """Aggregate root retaining the original public field API for tool compatibility."""

    task: str
    workspace: Path
    session: SessionState = field(default_factory=SessionState)
    activity: WorkspaceActivity = field(default_factory=WorkspaceActivity)
    execution: ExecutionLog = field(default_factory=ExecutionLog)
    integrity: IntegrityTracker = field(default_factory=IntegrityTracker)

    @property
    def run_id(self) -> str:
        return self.session.run_id

    @property
    def turn(self) -> int:
        return self.session.turn

    @turn.setter
    def turn(self, value: int) -> None:
        self.session.turn = value

    @property
    def sequence(self) -> int:
        return self.session.sequence

    @sequence.setter
    def sequence(self, value: int) -> None:
        self.session.sequence = value

    @property
    def files_read(self) -> set[str]:
        return self.activity.files_read

    @property
    def files_modified(self) -> set[str]:
        return self.activity.files_modified

    @property
    def observations(self) -> list[dict[str, Any]]:
        return self.activity.observations

    @property
    def commands_run(self) -> list[CommandRecord]:
        return self.execution.commands

    @property
    def verification_runs(self) -> list[VerificationRecord]:
        return self.execution.verifications

    @property
    def approvals(self) -> list[dict[str, Any]]:
        return self.execution.approvals

    @property
    def policy_denials(self) -> int:
        return self.execution.policy_denials

    @policy_denials.setter
    def policy_denials(self, value: int) -> None:
        self.execution.policy_denials = value

    @property
    def tool_calls_requested(self) -> int:
        return self.execution.tool_calls_requested

    def record_tool_call_request(self) -> None:
        self.execution.tool_calls_requested += 1

    @property
    def mutations(self) -> list[MutationRecord]:
        return self.integrity.mutations

    @property
    def repeated_actions(self) -> dict[str, int]:
        return self.integrity.repeated_actions

    @property
    def protected_hashes(self) -> dict[str, str]:
        return self.integrity.protected_hashes

    @protected_hashes.setter
    def protected_hashes(self, value: dict[str, str]) -> None:
        self.integrity.protected_hashes = value

    @property
    def workspace_hashes(self) -> dict[str, str]:
        return self.integrity.workspace_hashes

    @workspace_hashes.setter
    def workspace_hashes(self, value: dict[str, str]) -> None:
        self.integrity.workspace_hashes = value

    @property
    def integrity_violations(self) -> list[dict[str, Any]]:
        return self.integrity.violations

    def next_sequence(self) -> int:
        return self.session.next_sequence()

    @property
    def latest_mutation(self) -> MutationRecord | None:
        return self.mutations[-1] if self.mutations else None

    @property
    def latest_verification(self) -> VerificationRecord | None:
        return self.verification_runs[-1] if self.verification_runs else None

    def record_file_read(self, path: Path) -> None:
        self.files_read.add(path.relative_to(self.workspace).as_posix())

    def record_mutation(self, path: Path, before_hash: str, after_hash: str) -> MutationRecord:
        record = MutationRecord(self.next_sequence(), path.relative_to(self.workspace).as_posix(), before_hash, after_hash, utc_now())
        self.mutations.append(record)
        self.files_modified.add(record.path)
        return record

    def record_command_mutations(self, changes: list[Any], command: str) -> None:
        if not changes:
            return
        paths: list[str] = []
        for change in changes:
            self.record_mutation(self.workspace / change.path, change.before_hash, change.after_hash)
            paths.append(change.path)
        self.record_integrity_violation("COMMAND_MUTATION", command, paths)

    def record_integrity_violation(self, code: str, source: str, paths: list[str]) -> None:
        self.integrity_violations.append({"code": code, "source": source, "paths": paths, "timestamp": utc_now()})

    def record_command(self, command: str, cwd: Path, exit_code: int | None, timed_out: bool, is_verification: bool) -> CommandRecord:
        record = CommandRecord(self.next_sequence(), command, cwd.relative_to(self.workspace).as_posix() or ".", exit_code, timed_out, is_verification, utc_now())
        self.commands_run.append(record)
        if is_verification:
            self.verification_runs.append(VerificationRecord(self.next_sequence(), record.sequence, not timed_out and exit_code == 0, exit_code, utc_now()))
        return record


__all__ = ["CommandRecord", "MutationRecord", "RunState", "VerificationRecord", "file_digest", "utc_now"]
