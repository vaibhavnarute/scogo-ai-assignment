"""Focused mutable components composed by :class:`RunState`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .state_records import CommandRecord, MutationRecord, VerificationRecord


@dataclass(slots=True)
class SessionState:
    run_id: str = field(default_factory=lambda: uuid4().hex)
    turn: int = 0
    sequence: int = 0

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence


@dataclass(slots=True)
class WorkspaceActivity:
    files_read: set[str] = field(default_factory=set)
    files_modified: set[str] = field(default_factory=set)
    observations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ExecutionLog:
    commands: list[CommandRecord] = field(default_factory=list)
    verifications: list[VerificationRecord] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    policy_denials: int = 0
    tool_calls_requested: int = 0


@dataclass(slots=True)
class IntegrityTracker:
    mutations: list[MutationRecord] = field(default_factory=list)
    repeated_actions: dict[str, int] = field(default_factory=dict)
    protected_hashes: dict[str, str] = field(default_factory=dict)
    workspace_hashes: dict[str, str] = field(default_factory=dict)
    violations: list[dict[str, Any]] = field(default_factory=list)
