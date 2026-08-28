from __future__ import annotations

from pathlib import Path

import pytest

from harness.errors import HarnessError
from harness.state import RunState
from harness.state_components import ExecutionLog, IntegrityTracker, SessionState, WorkspaceActivity
from harness.tools.validation import ToolValidator


def test_run_state_composes_focused_independent_components(tmp_path: Path):
    first = RunState("first", tmp_path)
    second = RunState("second", tmp_path)
    assert isinstance(first.session, SessionState)
    assert isinstance(first.activity, WorkspaceActivity)
    assert isinstance(first.execution, ExecutionLog)
    assert isinstance(first.integrity, IntegrityTracker)
    first.files_read.add("one.py")
    first.policy_denials += 1
    assert second.files_read == set()
    assert second.policy_denials == 0
    assert first.run_id != second.run_id


def test_tool_validator_is_usable_without_registry_or_tracer():
    schema = {"properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}
    validator = ToolValidator()
    validator.validate(schema, {"path": "file.py"})
    with pytest.raises(HarnessError) as caught:
        validator.validate(schema, {"path": 4})
    assert caught.value.code == "INVALID_TOOL_ARGUMENTS"