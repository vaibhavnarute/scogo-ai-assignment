"""Controlled one-shot patch baseline with no model execution feedback."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from harness.cache import ObservationCache
from harness.config import HarnessConfig
from harness.errors import HarnessError
from harness.policy import WorkspacePolicy
from harness.providers.base import ModelProvider
from harness.providers.types import Usage
from harness.state import RunState
from harness.tools.base import ToolCall, ToolContext
from harness.tools.command import RunCommandTool
from harness.tools.finish import FinishTool
from harness.tools.patch import ApplyPatchTool
from harness.tools.registry import ToolRegistry
from harness.trace import EventTracer
from harness.verify import capture_integrity_baseline

from .common import create_provider, fixture_revision, summarize_records, trace_metrics
from .reset_fixture import EVALS_ROOT, FIXTURE_NAMES, reset_fixture

BASELINE_FILES = {
    "F1": ("calculator.py", "tests/test_calculator.py"),
    "F2": ("pagination.py", "tests/test_pagination.py"),
    "F3": ("accounts.py", "tests/test_accounts.py"),
    "F4": ("shipping.py", "tests/test_shipping.py"),
    "F5": ("README.md", "discounts.py", "tests/test_discounts.py"),
}


def _snapshot_prompt(fixture: str, workspace: Path) -> str:
    sections = []
    for relative in BASELINE_FILES[fixture]:
        sections.append(f"--- {relative} ---\n{(workspace / relative).read_text(encoding='utf-8')}")
    return "Fix the failing tests with one patch. You receive no execution feedback.\n\n" + "\n".join(sections)


def run_baselines(
    provider_factory: Callable[[], ModelProvider],
    fixtures: Sequence[str],
    output_path: Path,
    *,
    repetitions: int = 3,
    workspace_root: Path | None = None,
    trace_dir: Path | None = None,
    implementation_revision: str | None = None,
) -> list[dict[str, Any]]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    unknown = [fixture for fixture in fixtures if fixture not in FIXTURE_NAMES]
    if unknown:
        raise ValueError(f"unknown fixture: {unknown[0]}")
    workspace_root = workspace_root or EVALS_ROOT / "baseline_workspaces"
    trace_dir = trace_dir or EVALS_ROOT.parent / ".harness_runs"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for fixture in fixtures:
            revision = fixture_revision(EVALS_ROOT / "fixtures" / fixture)
            for repetition in range(1, repetitions + 1):
                workspace = reset_fixture(fixture, workspace_root)
                config = HarnessConfig.from_fixture(workspace, approval_mode="yes", trace_dir=trace_dir)
                state = RunState("one-shot baseline", workspace)
                capture_integrity_baseline(state, config)
                provider = provider_factory()
                trace_path = config.trace_dir / f"baseline-{state.run_id}.jsonl"
                started = time.monotonic()
                usage = Usage()
                tool_calls = 0
                outcome = "INCOMPLETE"
                reason = "one-shot patch or verification failed"
                with EventTracer(trace_path, state.run_id) as tracer:
                    context = ToolContext(config, WorkspacePolicy(config), state, ObservationCache(), tracer)
                    registry = ToolRegistry([ApplyPatchTool(), RunCommandTool(), FinishTool()])
                    messages = [
                        {"role": "system", "content": "Return exactly one apply_patch tool call. You cannot inspect or execute anything."},
                        {"role": "user", "content": _snapshot_prompt(fixture, workspace)},
                    ]
                    tools = [{"name": ApplyPatchTool.name, "description": ApplyPatchTool.description, "parameters": ApplyPatchTool.schema}]
                    tracer.record("run.started", task="one-shot baseline", fixture=fixture, repetition=repetition, provider=provider.provider_name, model=provider.model)
                    tracer.record("model.request.started", turn=1, provider=provider.provider_name, model=provider.model, message_count=len(messages))
                    request_started = time.monotonic()
                    try:
                        response = provider.generate(messages, tools)
                    except HarnessError as exc:
                        tracer.record("model.request.failed", turn=1, duration_ms=round((time.monotonic() - request_started) * 1000, 3), error=exc.as_dict())
                        outcome = "PROVIDER_FAILURE"
                        reason = exc.message
                    else:
                        usage = response.usage or Usage()
                        tool_calls = len(response.tool_calls)
                        tracer.record("model.request.completed", turn=1, duration_ms=round((time.monotonic() - request_started) * 1000, 3), provider_request_id=response.provider_request_id, finish_reason=response.finish_reason, tool_calls=tool_calls, retry_count=response.retry_count, token_usage=response.usage.as_dict() if response.usage else None)
                        for repair in response.raw_metadata.get("tool_name_repairs", []):
                            tracer.record("provider.tool_call.normalized", turn=1, **repair)
                        patch_calls = [call for call in response.tool_calls if call.name == "apply_patch"]
                        for call in response.tool_calls:
                            tracer.record("model.tool_call.requested", turn=1, tool_call_id=call.id, tool=call.name, arguments=call.arguments)
                        patch_result = registry.dispatch(patch_calls[0], context) if len(patch_calls) == 1 else None
                        verification = registry.dispatch(ToolCall("baseline-verify", "run_command", {"command": config.verification_command}), context) if patch_result and patch_result.ok else None
                        finish = registry.dispatch(ToolCall("baseline-finish", "finish", {"summary": "one-shot patch", "evidence": "external verification"}), context) if verification else None
                        if finish and finish.ok:
                            outcome = "VERIFIED_SUCCESS"
                            reason = "verified"
                    tracer.record("run.completed" if outcome == "VERIFIED_SUCCESS" else "run.failed", outcome=outcome, reason=reason, turns=1, model_requests=1, token_usage=usage.as_dict())
                record = {
                    "fixture": fixture,
                    "fixture_revision": revision,
                    "repetition": repetition,
                    "implementation_revision": implementation_revision,
                    "evaluation_config": {"mode": "one-shot", "approval_mode": "yes", "verification_command": config.verification_command, "provider_requests": 1},
                    "provider": provider.provider_name,
                    "model": provider.model,
                    "run_id": state.run_id,
                    "outcome": outcome,
                    "reason": reason,
                    "turns": 1,
                    "model_requests": 1,
                    "tool_calls": tool_calls,
                    "files_read": list(BASELINE_FILES[fixture]),
                    "files_modified": sorted(state.files_modified),
                    "commands": len(state.commands_run),
                    "verification_attempts": len(state.verification_runs),
                    "policy_denials": state.policy_denials,
                    "duration_ms": round((time.monotonic() - started) * 1000, 3),
                    "token_usage": usage.as_dict(),
                    "trace_path": str(trace_path),
                    **trace_metrics(trace_path),
                }
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                output.flush()
    output_path.with_suffix(".summary.json").write_text(json.dumps(summarize_records(records), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--fixtures", default=",".join(FIXTURE_NAMES))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, default=EVALS_ROOT / "results" / "baseline.jsonl")
    parser.add_argument("--implementation-revision")
    args = parser.parse_args()
    fixtures = tuple(item.strip() for item in args.fixtures.split(",") if item.strip())
    factory = lambda: create_provider(args.model)
    records = run_baselines(factory, fixtures, args.output, repetitions=args.repetitions, implementation_revision=args.implementation_revision)
    print(json.dumps(summarize_records(records), indent=2, sort_keys=True))
    return 0 if all(record["outcome"] == "VERIFIED_SUCCESS" for record in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
