from __future__ import annotations

import hashlib
from pathlib import Path

from harness.config import HarnessConfig
from harness.state import RunState
from harness.verify import Verifier, capture_integrity_baseline


def setup_state(workspace: Path) -> tuple[HarnessConfig, RunState]:
    config = HarnessConfig(workspace, verification_command="pytest -q")
    state = RunState("repair", workspace)
    capture_integrity_baseline(state, config)
    return config, state


def test_finish_without_verification_is_rejected(workspace: Path):
    config, state = setup_state(workspace)
    assert Verifier(config).evaluate_finish(state).error_code == "VERIFICATION_MISSING"


def test_finish_after_failed_verification_is_rejected(workspace: Path):
    config, state = setup_state(workspace)
    state.record_command("pytest -q", workspace, 1, False, True)
    assert Verifier(config).evaluate_finish(state).error_code == "VERIFICATION_FAILED"


def test_mutation_invalidates_older_verification(workspace: Path):
    config, state = setup_state(workspace)
    state.record_command("pytest -q", workspace, 0, False, True)
    path = workspace / "src" / "app.py"
    state.record_mutation(path, "before", "after")
    assert Verifier(config).evaluate_finish(state).error_code == "VERIFICATION_MISSING"


def test_valid_post_mutation_verification_is_accepted(workspace: Path):
    config, state = setup_state(workspace)
    path = workspace / "src" / "app.py"
    state.record_mutation(path, "before", "after")
    state.record_command("pytest -q", workspace, 0, False, True)
    assert Verifier(config).evaluate_finish(state).accepted


def test_protected_path_modification_is_rejected(workspace: Path):
    config, state = setup_state(workspace)
    protected = workspace / "tests" / "test_locked.py"
    protected.write_text("def test_locked(): assert False\n", encoding="utf-8")
    state.record_command("pytest -q", workspace, 0, False, True)
    assert Verifier(config).evaluate_finish(state).error_code == "INTEGRITY_VIOLATION"

