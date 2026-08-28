"""Immutable records emitted by composed run-state components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MutationRecord:
    sequence: int
    path: str
    before_hash: str
    after_hash: str
    timestamp: str


@dataclass(frozen=True, slots=True)
class CommandRecord:
    sequence: int
    command: str
    cwd: str
    exit_code: int | None
    timed_out: bool
    is_verification: bool
    timestamp: str


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    sequence: int
    command_sequence: int
    passed: bool
    exit_code: int | None
    timestamp: str