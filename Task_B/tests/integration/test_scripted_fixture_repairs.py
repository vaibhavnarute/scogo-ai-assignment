from __future__ import annotations

from pathlib import Path

import pytest

from evals.reset_fixture import reset_fixture
from harness.agent import Agent, RunOutcome
from harness.config import HarnessConfig
from harness.providers.mock import MockProvider
from harness.providers.types import ModelResponse
from harness.tools.base import ToolCall

PATCHES = {
    "F2": ("pagination.py", "@@ -5,1 +5,1 @@\n-    return item_count // page_size + 1\n+    return (item_count + page_size - 1) // page_size"),
    "F3": ("accounts.py", "@@ -4,1 +4,1 @@\n-        raise TypeError(\"username is required\")\n+        raise ValueError(\"username is required\")"),
    "F4": ("shipping.py", "@@ -2,6 +2,6 @@\n     \"\"\"Return 0 for VIP or qualifying orders, otherwise the standard fee.\"\"\"\n     if order_total < 0:\n-        return 0\n-    if order_total > 50:\n+        raise ValueError(\"order total cannot be negative\")\n+    if vip or order_total >= 50:\n         return 0\n     return 8"),
    "F5": ("discounts.py", "@@ -3,1 +3,1 @@\n-    return price * (1 + percent / 100)\n+    return price * (1 - percent / 100)"),
}


def _provider(fixture: str) -> MockProvider:
    path, patch = PATCHES[fixture]
    return MockProvider([
        ModelResponse(tool_calls=[ToolCall(f"{fixture}-before", "run_command", {"command": "python -m pytest -q"})]),
        ModelResponse(tool_calls=[ToolCall(f"{fixture}-read", "read_file", {"path": path})]),
        ModelResponse(tool_calls=[ToolCall(f"{fixture}-patch", "apply_patch", {"path": path, "patch": patch})]),
        ModelResponse(tool_calls=[ToolCall(f"{fixture}-after", "run_command", {"command": "python -m pytest -q"})]),
        ModelResponse(tool_calls=[ToolCall(f"{fixture}-finish", "finish", {"summary": f"repaired {fixture}", "evidence": "pytest exit 0"})]),
    ])


@pytest.mark.parametrize("fixture", ["F2", "F3", "F4", "F5"])
def test_scripted_agent_repairs_each_remaining_fixture(fixture: str, tmp_path: Path):
    workspace = reset_fixture(fixture, tmp_path / "workspaces")
    config = HarnessConfig.from_fixture(workspace, approval_mode="yes", trace_dir=tmp_path / "traces")
    result = Agent(f"repair {fixture}", config, _provider(fixture)).run()
    assert result.outcome == RunOutcome.VERIFIED_SUCCESS
    assert result.exit_code == 0
    assert result.state.verification_runs[-1].passed
    assert result.state.files_modified == {PATCHES[fixture][0]}