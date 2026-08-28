from __future__ import annotations

import json
from pathlib import Path

from evals.common import summarize_records, trace_metrics


def test_recovery_metrics_require_later_success(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    events = [
        {"event": "tool.failed", "error": {"code": "INVALID_TOOL_ARGUMENTS", "recoverable": True}},
        {"event": "tool.completed", "tool": "read_file", "cached": False},
    ]
    trace.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    metrics = trace_metrics(trace)
    assert metrics["recoverable_tool_failures"] == 1
    assert metrics["recovered_tool_failures"] == 1
    assert metrics["recovery_rate"] == 1.0


def test_summary_aggregates_recovery_without_inventing_cost():
    records = [
        {
            "outcome": "INCOMPLETE",
            "turns": 2,
            "duration_ms": 10,
            "token_usage": {"input_tokens": 4, "output_tokens": 2},
            "policy_denials": 0,
            "recoverable_tool_failures": 2,
            "recovered_tool_failures": 1,
        }
    ]
    summary = summarize_records(records)
    assert summary["recovery_rate"] == 0.5
    assert summary["cost"] is None

def test_recovery_metrics_count_nonzero_command_then_passing_command(tmp_path: Path):
    trace = tmp_path / "command-trace.jsonl"
    events = [
        {"event": "tool.completed", "tool": "run_command", "result": {"exit_code": 1}, "cached": False},
        {"event": "tool.completed", "tool": "apply_patch", "result": {}, "cached": False},
        {"event": "tool.completed", "tool": "run_command", "result": {"exit_code": 0}, "cached": False},
    ]
    trace.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    metrics = trace_metrics(trace)
    assert metrics["failed_command_runs"] == 1
    assert metrics["recoverable_tool_failures"] == 1
    assert metrics["recovered_tool_failures"] == 1
    assert metrics["recovery_rate"] == 1.0


def test_normalized_provider_tool_name_remains_invalid_for_raw_validity_metric(tmp_path: Path):
    trace = tmp_path / "normalized-trace.jsonl"
    events = [
        {
            "event": "provider.tool_call.normalized",
            "tool_call_id": "c1",
            "raw_tool_name": "finish<|channel|>json",
            "normalized_tool_name": "finish",
        },
        {"event": "model.tool_call.requested", "tool_call_id": "c1", "tool": "finish"},
        {"event": "tool.completed", "tool_call_id": "c1", "tool": "finish", "cached": False},
    ]
    trace.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    metrics = trace_metrics(trace)
    assert metrics["invalid_tool_calls"] == 1
    assert metrics["tool_call_validity"] == 0.0