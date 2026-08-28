from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from evals.reset_fixture import reset_fixture
from harness.agent import Agent, AgentSettings, RunOutcome
from harness.config import HarnessConfig
from harness.errors import ErrorCategory, HarnessError
from harness.providers.mock import MockProvider
from harness.providers.nvidia import NvidiaConfig, NvidiaProvider
from harness.providers.types import ModelResponse, Usage
from harness.tools.base import ToolCall


def response(tool: str, arguments, call_id: str) -> ModelResponse:
    return ModelResponse(tool_calls=[ToolCall(call_id, tool, arguments)], usage=Usage(2, 1), finish_reason="tool_calls")


def f1_config(tmp_path: Path, max_turns: int = 10) -> HarnessConfig:
    workspace = reset_fixture("F1", tmp_path / "fixtures")
    return HarnessConfig.from_fixture(workspace, approval_mode="yes", max_turns=max_turns, trace_dir=tmp_path / "traces")


def test_full_mock_provider_repair_loop(tmp_path: Path):
    config = f1_config(tmp_path)
    patch = "@@ -1,3 +1,3 @@\n def add(left: int, right: int) -> int:\n     \"\"\"Return the sum of two integers.\"\"\"\n-    return left - right\n+    return left + right"
    provider = MockProvider([
        response("run_command", {"command": "python -m pytest -q"}, "c1"),
        response("read_file", {"path": "calculator.py"}, "r1"),
        response("apply_patch", {"path": "calculator.py", "patch": patch}, "p1"),
        response("run_command", {"command": "python -m pytest -q"}, "c2"),
        response("finish", {"summary": "fixed addition", "evidence": "pytest exit 0"}, "f1"),
    ])
    reported: list[tuple[str, str]] = []
    result = Agent("Fix the failing tests", config, provider, reporter=lambda event, message: reported.append((event, message))).run()
    assert result.outcome == RunOutcome.VERIFIED_SUCCESS and result.exit_code == 0
    assert result.model_requests == 5 and result.usage.total_tokens == 15
    assert "return left + right" in (config.workspace / "calculator.py").read_text(encoding="utf-8")
    events = [json.loads(line)["event"] for line in result.trace_path.read_text(encoding="utf-8").splitlines()]
    assert "model.request.completed" in events and "verification.passed" in events and events[-1] == "run.completed"
    assert ("agent", "Inspecting repository...") in reported
    assert ("tool", "run_command: python -m pytest -q") in reported
    assert ("result", "2 failed (exit 1)") in reported
    assert ("agent", "Inspecting failing implementation...") in reported
    assert ("tool", "read_file: calculator.py") in reported
    assert ("agent", "Applying minimal repair...") in reported
    assert ("result", "patched calculator.py (2 changed lines)") in reported
    assert ("verify", "python -m pytest -q") in reported
    assert ("verify", "passed: 2 passed (exit 0)") in reported
    assert reported[-1] == ("done", "VERIFIED_SUCCESS")


def test_early_finish_is_rejected_then_agent_recovers(tmp_path: Path):
    config = f1_config(tmp_path, max_turns=4)
    provider = MockProvider([
        response("finish", {"summary": "done", "evidence": "none"}, "f1"),
        response("run_command", {"command": "python -m pytest -q"}, "c1"),
        response("finish", {"summary": "done", "evidence": "failed tests"}, "f2"),
        ModelResponse(text="blocked"),
    ])
    result = Agent("repair", config, provider).run()
    assert result.outcome == RunOutcome.INCOMPLETE
    assert len(result.state.verification_runs) == 1 and not result.state.verification_runs[0].passed


def test_malformed_tool_arguments_are_returned_to_model(tmp_path: Path):
    config = f1_config(tmp_path, max_turns=2)
    provider = MockProvider([response("read_file", "{bad json", "bad"), ModelResponse(text="cannot continue")])
    result = Agent("repair", config, provider).run()
    tool_message = provider.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool" and "INVALID_TOOL_ARGUMENTS" in tool_message["content"]
    assert result.outcome == RunOutcome.INCOMPLETE


def test_repeated_action_loop_stops_safely(tmp_path: Path):
    config = f1_config(tmp_path)
    same = response("read_file", {"path": "calculator.py"}, "same")
    result = Agent("repair", config, MockProvider([same, same, same]), settings=AgentSettings(repeated_action_limit=3)).run()
    assert result.outcome == RunOutcome.FAILED_SAFE and result.reason == "REPEATED_ACTION_LOOP"


def test_max_turns_stops_without_success(tmp_path: Path):
    config = f1_config(tmp_path, max_turns=2)
    result = Agent("repair", config, MockProvider([ModelResponse(text="thinking"), ModelResponse(text="still thinking")])).run()
    assert result.outcome == RunOutcome.INCOMPLETE and result.reason == "MAX_TURNS_EXCEEDED"


def test_provider_failure_is_failed_safe(tmp_path: Path):
    config = f1_config(tmp_path)
    failure = HarnessError("PROVIDER_CONNECTION_ERROR", ErrorCategory.PROVIDER, "offline", True)
    result = Agent("repair", config, MockProvider([failure])).run()
    assert result.outcome == RunOutcome.PROVIDER_FAILURE and result.exit_code == 4


def test_interactive_approval_denial_returns_policy_exit(tmp_path: Path):
    base = f1_config(tmp_path)
    config = HarnessConfig.from_fixture(base.workspace, approval_mode="interactive", trace_dir=tmp_path / "approval-traces")
    provider = MockProvider([response("run_command", {"command": "python -m pytest -q"}, "c1")])
    result = Agent("repair", config, provider, approval_callback=lambda _call: False).run()
    assert result.outcome == RunOutcome.POLICY_DENIED and result.exit_code == 3
    assert result.state.approvals == [{"tool_call_id": "c1", "tool": "run_command", "granted": False}]


class _MalformedHTTPResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self) -> bytes:
        return b'{"choices":[{"message":[]}]}'


def test_malformed_nvidia_payload_becomes_agent_provider_failure(tmp_path: Path, monkeypatch):
    config = f1_config(tmp_path, max_turns=1)
    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    provider = NvidiaProvider(NvidiaConfig(model="model", max_retries=0))
    with patch("urllib.request.urlopen", return_value=_MalformedHTTPResponse()):
        result = Agent("repair", config, provider).run()
    assert result.outcome == RunOutcome.PROVIDER_FAILURE
    assert result.reason == "provider response message must be an object"


def test_text_only_transition_and_actual_tool_call_accounting(tmp_path: Path):
    config = f1_config(tmp_path, max_turns=2)
    provider = MockProvider([ModelResponse(text="thinking"), response("read_file", {"path": "calculator.py"}, "r1")])
    result = Agent("repair", config, provider).run()
    events = [json.loads(line) for line in result.trace_path.read_text(encoding="utf-8").splitlines()]
    assert any(event.get("event") == "state.transition" and event.get("source") == "DECIDE" and event.get("destination") == "OBSERVE" and event.get("turn") == 1 for event in events)
    assert result.summary()["tool_calls"] == 1


def test_configured_secret_is_absent_from_child_console_trace_and_next_request(tmp_path: Path, monkeypatch):
    secret = "fake-live-provider-key-never-leak"
    monkeypatch.setenv("NVIDIA_API_KEY", secret)
    config = f1_config(tmp_path, max_turns=2)
    (config.workspace / "leak.py").write_text(
        "import os\nprint(os.environ.get('NVIDIA_API_KEY', 'NOT_VISIBLE'))\n",
        encoding="utf-8",
    )
    provider = MockProvider([
        response("run_command", {"command": "python leak.py"}, "c1"),
        ModelResponse(text="observed"),
    ])
    reported: list[tuple[str, str]] = []
    result = Agent(f"check environment without exposing {secret}", config, provider, reporter=lambda event, message: reported.append((event, message))).run()
    assert len(provider.requests) == 2
    assert secret not in json.dumps(provider.requests, ensure_ascii=False)
    assert secret not in json.dumps(reported, ensure_ascii=False)
    assert secret not in result.trace_path.read_text(encoding="utf-8")
