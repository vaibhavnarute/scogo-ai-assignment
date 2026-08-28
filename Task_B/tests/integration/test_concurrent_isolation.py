from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evals.reset_fixture import reset_fixture
from harness.agent import Agent, RunOutcome
from harness.config import HarnessConfig
from harness.providers.mock import MockProvider
from harness.providers.types import ModelResponse
from harness.tools.base import ToolCall

PATCH = "@@ -1,3 +1,3 @@\n def add(left: int, right: int) -> int:\n     \"\"\"Return the sum of two integers.\"\"\"\n-    return left - right\n+    return left + right"


def _provider(label: str) -> MockProvider:
    return MockProvider([
        ModelResponse(tool_calls=[ToolCall(f"{label}-before", "run_command", {"command": "python -m pytest -q"})]),
        ModelResponse(tool_calls=[ToolCall(f"{label}-patch", "apply_patch", {"path": "calculator.py", "patch": PATCH})]),
        ModelResponse(tool_calls=[ToolCall(f"{label}-after", "run_command", {"command": "python -m pytest -q"})]),
        ModelResponse(tool_calls=[ToolCall(f"{label}-finish", "finish", {"summary": "fixed", "evidence": "pytest"})]),
    ])


def test_concurrent_runs_isolate_state_workspaces_and_traces(tmp_path: Path):
    trace_dir = tmp_path / "shared-traces"

    def run(label: str):
        workspace = reset_fixture("F1", tmp_path / label)
        config = HarnessConfig.from_fixture(workspace, approval_mode="yes", trace_dir=trace_dir)
        return Agent(f"repair-{label}", config, _provider(label)).run()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ("alpha", "beta")))

    assert all(result.outcome == RunOutcome.VERIFIED_SUCCESS for result in results)
    assert results[0].state is not results[1].state
    assert results[0].state.run_id != results[1].state.run_id
    assert results[0].trace_path != results[1].trace_path
    for result in results:
        assert result.state.files_modified == {"calculator.py"}
        assert "return left + right" in (result.state.workspace / "calculator.py").read_text(encoding="utf-8")
        records = [json.loads(line) for line in result.trace_path.read_text(encoding="utf-8").splitlines()]
        assert records and {record["run_id"] for record in records} == {result.state.run_id}