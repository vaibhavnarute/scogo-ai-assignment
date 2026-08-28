"""Shared provider construction and trace-derived evaluation metrics."""

from __future__ import annotations

import hashlib
import json
import statistics
import math
from pathlib import Path
from typing import Any

from harness.env import load_dotenv
from harness.providers.nvidia import DEFAULT_NVIDIA_MODEL, NvidiaConfig, NvidiaProvider


def create_provider(model: str | None = None) -> NvidiaProvider:
    load_dotenv()
    return NvidiaProvider(NvidiaConfig(model=model or DEFAULT_NVIDIA_MODEL))

def fixture_revision(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def trace_metrics(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    requested = [record for record in records if record["event"] == "model.tool_call.requested"]
    failures = [record for record in records if record["event"] == "tool.failed"]
    invalid_failure_ids = {
        record.get("tool_call_id")
        for record in failures
        if (record.get("error") or {}).get("code") in {"INVALID_TOOL_ARGUMENTS", "UNKNOWN_TOOL"}
    }
    normalized_ids = {
        record.get("tool_call_id")
        for record in records
        if record["event"] == "provider.tool_call.normalized"
    }
    invalid_call_ids = (invalid_failure_ids | normalized_ids) - {None}
    recoverable_failure_indices = [
        index
        for index, record in enumerate(records)
        if record["event"] == "tool.failed" and bool((record.get("error") or {}).get("recoverable"))
    ]
    failed_command_indices = [
        index
        for index, record in enumerate(records)
        if record["event"] == "tool.completed"
        and record.get("tool") == "run_command"
        and int((record.get("result") or {}).get("exit_code", 0)) != 0
    ]
    successful_tool_indices = [
        index
        for index, record in enumerate(records)
        if record["event"] == "tool.completed"
        and not (record.get("tool") == "run_command" and int((record.get("result") or {}).get("exit_code", 0)) != 0)
    ]
    successful_command_indices = [
        index
        for index, record in enumerate(records)
        if record["event"] == "tool.completed"
        and record.get("tool") == "run_command"
        and int((record.get("result") or {}).get("exit_code", 1)) == 0
    ]
    recovered = sum(any(success > failure for success in successful_tool_indices) for failure in recoverable_failure_indices)
    recovered += sum(any(success > failure for success in successful_command_indices) for failure in failed_command_indices)
    recoverable_count = len(recoverable_failure_indices) + len(failed_command_indices)
    provider_ms = sum(float(record.get("duration_ms", 0)) for record in records if record["event"] == "model.request.completed")
    command_ms = sum(
        float((record.get("result") or {}).get("duration_ms", 0))
        for record in records
        if record["event"] == "tool.completed" and record.get("tool") == "run_command"
    )
    provider_request_ids = [
        record["provider_request_id"]
        for record in records
        if record["event"] == "model.request.completed" and record.get("provider_request_id")
    ]
    integrity_events = [record for record in records if record["event"] == "integrity.violation"]
    outcome = next((record.get("outcome") for record in reversed(records) if record["event"] in {"run.completed", "run.failed"}), None)
    patch_events = [record for record in records if record["event"] == "model.tool_call.requested" and record.get("tool") == "apply_patch"]
    completed_patch_ids = {record.get("tool_call_id") for record in records if record["event"] == "tool.completed" and record.get("tool") == "apply_patch"}
    return {
        "tool_calls_requested": len(requested),
        "raw_provider_tool_validity": (len(requested) - len(invalid_call_ids)) / len(requested) if requested else None,
        "normalized_executable_tool_validity": (len(requested) - len(invalid_failure_ids)) / len(requested) if requested else None,
        "tool_call_validity": (len(requested) - len(invalid_call_ids)) / len(requested) if requested else None,
        "invalid_tool_calls": len(invalid_call_ids),
        "normalized_tool_calls": len(normalized_ids),
        "recoverable_tool_failures": recoverable_count,
        "recovered_tool_failures": recovered,
        "event_level_recovery_rate": recovered / recoverable_count if recoverable_count else None,
        "run_had_recoverable_failure": recoverable_count > 0,
        "run_recovered": recoverable_count > 0 and outcome == "VERIFIED_SUCCESS",
        "recovery_rate": recovered / recoverable_count if recoverable_count else None,
        "failed_command_runs": len(failed_command_indices),
        "provider_duration_ms": round(provider_ms, 3),
        "tool_command_duration_ms": round(command_ms, 3),
        "provider_request_ids": provider_request_ids,
        "patch_attempts": len(patch_events),
        "first_patch_succeeded": bool(patch_events and patch_events[0].get("tool_call_id") in completed_patch_ids),
        "integrity_violations": len(integrity_events),
        "integrity_violations_reaching_success": len(integrity_events) if outcome == "VERIFIED_SUCCESS" else 0,
        "cache_hits": sum(record.get("cached") is True for record in records if record["event"] == "tool.completed"),
        "trace_events": len(records),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [record for record in records if record["outcome"] == "VERIFIED_SUCCESS"]
    recoverable = sum(record.get("recoverable_tool_failures", 0) for record in records)
    recovered = sum(record.get("recovered_tool_failures", 0) for record in records)
    total_tool_calls = sum(record.get("tool_calls_requested", 0) for record in records)
    raw_valid_calls = sum(record.get("tool_calls_requested", 0) - record.get("invalid_tool_calls", 0) for record in records)
    normalized_valid_calls = sum(record.get("tool_calls_requested", 0) - max(0, record.get("invalid_tool_calls", 0) - record.get("normalized_tool_calls", 0)) for record in records)
    recovery_runs = [record for record in records if record.get("run_had_recoverable_failure")]
    durations = sorted(record["duration_ms"] for record in records)
    token_totals = [
        record["token_usage"].get(
            "total_tokens",
            record["token_usage"].get("input_tokens", 0)
            + record["token_usage"].get("output_tokens", 0)
            + record["token_usage"].get("cached_tokens", 0),
        )
        for record in records
    ]
    return {
        "runs": len(records),
        "verified_successes": len(successes),
        "verified_success_rate": len(successes) / len(records) if records else 0.0,
        "median_turns_to_success": statistics.median(record["turns"] for record in successes) if successes else None,
        "median_turns": statistics.median(record["turns"] for record in records) if records else None,
        "median_wall_clock_ms": statistics.median(record["duration_ms"] for record in records) if records else None,
        "p95_wall_clock_ms": durations[max(0, math.ceil(0.95 * len(durations)) - 1)] if durations else None,
        "provider_duration_ms": sum(record.get("provider_duration_ms", 0) for record in records),
        "tool_command_duration_ms": sum(record.get("tool_command_duration_ms", 0) for record in records),
        "input_tokens": sum(record["token_usage"]["input_tokens"] for record in records),
        "output_tokens": sum(record["token_usage"]["output_tokens"] for record in records),
        "cached_tokens": sum(record["token_usage"].get("cached_tokens", 0) for record in records),
        "total_tokens": sum(token_totals),
        "median_total_tokens": statistics.median(token_totals) if token_totals else None,
        "raw_provider_tool_validity": raw_valid_calls / total_tool_calls if total_tool_calls else None,
        "normalized_executable_tool_validity": normalized_valid_calls / total_tool_calls if total_tool_calls else None,
        "policy_denials": sum(record["policy_denials"] for record in records),
        "recoverable_tool_failures": recoverable,
        "recovered_tool_failures": recovered,
        "recovery_rate": recovered / recoverable if recoverable else None,
        "event_level_recovery_rate": recovered / recoverable if recoverable else None,
        "runs_with_recoverable_failures": len(recovery_runs),
        "runs_recovered": sum(bool(record.get("run_recovered")) for record in recovery_runs),
        "run_level_recovery_rate": sum(bool(record.get("run_recovered")) for record in recovery_runs) / len(recovery_runs) if recovery_runs else None,
        "integrity_violations": sum(record.get("integrity_violations", 0) for record in records),
        "integrity_violations_reaching_success": sum(record.get("integrity_violations_reaching_success", 0) for record in records),
        "cost": None,
        "cost_note": "Not computed without provider pricing and actual billing evidence.",
    }
