from __future__ import annotations

import json
from pathlib import Path

from evals.run_baseline import run_baselines
from evals.run_eval import evaluate
from harness.errors import ErrorCategory, HarnessError
from harness.providers.mock import MockProvider
from harness.providers.types import ModelResponse
from harness.tools.base import ToolCall


PATCH = "@@ -1,3 +1,3 @@\n def add(left: int, right: int) -> int:\n     \"\"\"Return the sum of two integers.\"\"\"\n-    return left - right\n+    return left + right"


def scripted_agent_provider() -> MockProvider:
    return MockProvider(
        [
            ModelResponse(tool_calls=[ToolCall("c1", "run_command", {"command": "python -m pytest -q"})]),
            ModelResponse(tool_calls=[ToolCall("p1", "apply_patch", {"path": "calculator.py", "patch": PATCH})]),
            ModelResponse(tool_calls=[ToolCall("c2", "run_command", {"command": "python -m pytest -q"})]),
            ModelResponse(tool_calls=[ToolCall("f1", "finish", {"summary": "fixed", "evidence": "pytest"})]),
        ]
    )


def test_evaluation_runner_resets_and_records_every_run(tmp_path: Path):
    output = tmp_path / "results" / "runs.jsonl"
    records = evaluate(scripted_agent_provider, ["F1"], 2, output, workspace_root=tmp_path / "workspaces", trace_dir=tmp_path / "traces", implementation_revision="audit-hash")
    assert len(records) == 2 and all(record["outcome"] == "VERIFIED_SUCCESS" for record in records)
    assert [record["repetition"] for record in records] == [1, 2]
    assert all(record["failed_command_runs"] == 1 and record["recovery_rate"] == 1.0 for record in records)
    assert all(record["implementation_revision"] == "audit-hash" for record in records)
    assert all(record["raw_provider_tool_validity"] == 1.0 and record["normalized_executable_tool_validity"] == 1.0 for record in records)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["verified_success_rate"] == 1.0 and summary["cost"] is None


def test_one_shot_baseline_executes_only_one_model_patch(tmp_path: Path):
    output = tmp_path / "baseline.jsonl"
    records = run_baselines(lambda: MockProvider([ModelResponse(tool_calls=[ToolCall("p", "apply_patch", {"path": "calculator.py", "patch": PATCH})])]), ["F1"], output, repetitions=1, workspace_root=tmp_path / "baseline-workspaces", trace_dir=tmp_path / "traces")
    assert len(records) == 1 and records[0]["outcome"] == "VERIFIED_SUCCESS"
    assert records[0]["model_requests"] == 1 and records[0]["turns"] == 1
    assert output.with_suffix(".summary.json").is_file()

def test_one_shot_baseline_records_all_provider_failures(tmp_path: Path):
    output = tmp_path / "baseline-failures.jsonl"
    failure = HarnessError("AUTHENTICATION_FAILED", ErrorCategory.PROVIDER, "provider rejected authentication", False)
    records = run_baselines(
        lambda: MockProvider([failure]),
        ["F1"],
        output,
        repetitions=2,
        workspace_root=tmp_path / "baseline-workspaces",
        trace_dir=tmp_path / "traces",
    )
    assert len(records) == 2
    assert all(record["outcome"] == "PROVIDER_FAILURE" for record in records)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_one_shot_baseline_records_malformed_patch_without_aborting(tmp_path: Path):
    output = tmp_path / "baseline-malformed.jsonl"
    records = run_baselines(
        lambda: MockProvider([ModelResponse(tool_calls=[ToolCall("p", "apply_patch", {"path": "calculator.py", "patch": "not a patch"})])]),
        ["F1"],
        output,
        repetitions=1,
        workspace_root=tmp_path / "baseline-workspaces",
        trace_dir=tmp_path / "traces",
    )
    assert len(records) == 1
    assert records[0]["outcome"] == "INCOMPLETE"
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1
