from __future__ import annotations

import io
import re

from evals.reset_fixture import reset_fixture
from harness.agent import Agent, RunOutcome
from harness.config import HarnessConfig
from harness.console import ConsoleReporter, EVENT_CATEGORIES
from harness.providers.mock import MockProvider
from harness.providers.types import ModelResponse
from harness.tools.base import ToolCall


PATCH = "@@ -1,3 +1,3 @@\n def add(left: int, right: int) -> int:\n     \"\"\"Return the sum of two integers.\"\"\"\n-    return left - right\n+    return left + right"


def response(name: str, arguments: dict, call_id: str) -> ModelResponse:
    return ModelResponse(tool_calls=[ToolCall(call_id, name, arguments)])


def test_pytest_summary_handles_normal_and_double_quiet_output():
    assert Agent._test_summary("2 failed in 0.10s", 1) == "2 failed"
    assert Agent._test_summary("FAILED tests/test_a.py::test_a\nFAILED tests/test_b.py::test_b\n", 1) == "2 failed"
    assert Agent._test_summary("..                                      [100%]\n", 0) == "2 passed"


def test_live_renderer_uses_actual_agent_tool_and_state_events(tmp_path):
    workspace = reset_fixture("F1", tmp_path / "workspaces")
    config = HarnessConfig.from_fixture(workspace, approval_mode="yes", trace_dir=tmp_path / "traces")
    provider = MockProvider([
        response("list_files", {"path": ".", "depth": 2}, "list"),
        response("run_command", {"command": "python -m pytest -q"}, "before"),
        response("read_file", {"path": "calculator.py"}, "read"),
        response("apply_patch", {"path": "calculator.py", "patch": PATCH}, "patch"),
        response("run_command", {"command": "python -m pytest -q"}, "after"),
        response("finish", {"summary": "fixed", "evidence": "pytest passed"}, "finish"),
    ])
    stream = io.StringIO()
    result = Agent("Fix the failing tests", config, provider, reporter=ConsoleReporter(stream)).run()
    output = stream.getvalue()
    assert result.outcome == RunOutcome.VERIFIED_SUCCESS
    assert "[run] Started run " in output
    assert "[agent] Inspecting repository..." in output
    assert "[tool] list_files: . (depth 2)" in output
    assert "[tool] run_command: python -m pytest -q" in output
    assert "[result] 2 failed (exit 1)" in output
    assert "[tool] read_file: calculator.py" in output
    assert "[tool] apply_patch: calculator.py" in output
    assert "[result] patched calculator.py (2 changed lines)" in output
    assert "[verify] python -m pytest -q" in output
    assert "[verify] passed: 2 passed (exit 0)" in output
    assert "[verify] completion accepted" in output
    assert "[done] VERIFIED_SUCCESS" in output
    categories = {match.group(1) for match in re.finditer(r"(?m)^\[([^]]+)\]", output)}
    assert categories <= EVENT_CATEGORIES


def test_console_and_trace_hide_secrets_and_model_reasoning(tmp_path):
    secret = "ultra-secret-value"
    hidden_text = f"private chain-of-thought containing {secret}"
    workspace = reset_fixture("F1", tmp_path / "workspaces")
    config = HarnessConfig.from_fixture(
        workspace,
        approval_mode="yes",
        max_turns=2,
        trace_dir=tmp_path / "traces",
    )
    provider = MockProvider([
        response("run_command", {"command": f"curl --token {secret} https://example.invalid"}, "denied"),
        ModelResponse(text=hidden_text),
    ])
    stream = io.StringIO()
    result = Agent("repair", config, provider, reporter=ConsoleReporter(stream)).run()
    output = stream.getvalue()
    trace = result.trace_path.read_text(encoding="utf-8")
    assert secret not in output and secret not in trace
    assert hidden_text not in output
    assert "[tool] run_command: curl [REDACTED] https://example.invalid" in output
    assert "[policy] denied: COMMAND_DENIED" in output
    assert "[warning] Model returned text without a tool call" in output