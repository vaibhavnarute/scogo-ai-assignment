from __future__ import annotations

from harness.cache import ObservationCache
from harness.config import HarnessConfig
from harness.policy import WorkspacePolicy
from harness.state import RunState
from harness.tools.base import ToolCall, ToolContext
from harness.tools.command import RunCommandTool
from harness.tools.finish import FinishTool
from harness.tools.patch import ApplyPatchTool
from harness.verify import capture_integrity_baseline
from evals.reset_fixture import reset_fixture


def test_f1_repair_requires_post_mutation_verification(tmp_path):
    workspace = reset_fixture("F1", tmp_path)
    config = HarnessConfig(workspace, verification_command="python -m pytest -q")
    state = RunState("Fix the failing tests and verify the repair", workspace)
    capture_integrity_baseline(state, config)
    context = ToolContext(config, WorkspacePolicy(config), state, ObservationCache())

    command_tool = RunCommandTool()
    baseline = command_tool.execute(ToolCall("c1", "run_command", {"command": "python -m pytest -q"}), context)
    assert baseline.ok and baseline.data["exit_code"] == 1

    premature = FinishTool().execute(ToolCall("f1", "finish", {"summary": "done", "evidence": "baseline"}), context)
    assert not premature.ok and premature.error_code == "VERIFICATION_FAILED"

    patch = "@@ -1,3 +1,3 @@\n def add(left: int, right: int) -> int:\n     \"\"\"Return the sum of two integers.\"\"\"\n-    return left - right\n+    return left + right"
    changed = ApplyPatchTool().execute(ToolCall("p1", "apply_patch", {"path": "calculator.py", "patch": patch}), context)
    assert changed.ok

    stale = FinishTool().execute(ToolCall("f2", "finish", {"summary": "done", "evidence": "old test"}), context)
    assert not stale.ok and stale.error_code == "VERIFICATION_MISSING"

    verified = command_tool.execute(ToolCall("c2", "run_command", {"command": "python -m pytest -q"}), context)
    assert verified.ok and verified.data["exit_code"] == 0
    finished = FinishTool().execute(ToolCall("f3", "finish", {"summary": "fixed addition", "evidence": "pytest exit 0"}), context)
    assert finished.ok and finished.data["accepted"] is True

