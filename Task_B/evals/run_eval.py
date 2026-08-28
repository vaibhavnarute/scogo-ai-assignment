"""Run every selected fixture from a clean reset and report every run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from harness.agent import Agent
from harness.config import HarnessConfig
from harness.providers.base import ModelProvider

from .common import create_provider, fixture_revision, summarize_records, trace_metrics
from .reset_fixture import EVALS_ROOT, FIXTURE_NAMES, reset_fixture


def evaluate(
    provider_factory: Callable[[], ModelProvider],
    fixtures: Sequence[str],
    repetitions: int,
    output_path: Path,
    *,
    workspace_root: Path | None = None,
    trace_dir: Path | None = None,
    max_turns: int = 20,
    implementation_revision: str | None = None,
) -> list[dict[str, Any]]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    workspace_root = workspace_root or EVALS_ROOT / "workspaces"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for fixture in fixtures:
            if fixture not in FIXTURE_NAMES:
                raise ValueError(f"unknown fixture: {fixture}")
            revision = fixture_revision(EVALS_ROOT / "fixtures" / fixture)
            for repetition in range(1, repetitions + 1):
                workspace = reset_fixture(fixture, workspace_root)
                config = HarnessConfig.from_fixture(workspace, approval_mode="yes", max_turns=max_turns, trace_dir=trace_dir)
                provider = provider_factory()
                result = Agent("Fix the failing tests in this repository and verify the repair.", config, provider).run()
                record = {"fixture": fixture, "fixture_revision": revision, "repetition": repetition, "implementation_revision": implementation_revision, "evaluation_config": {"max_turns": max_turns, "approval_mode": "yes", "verification_command": config.verification_command, "command_timeout_seconds": config.command_timeout_seconds}, "provider": provider.provider_name, "model": provider.model, **result.summary(), **trace_metrics(result.trace_path)}
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                output.flush()
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summarize_records(records), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--fixtures", default=",".join(FIXTURE_NAMES))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--output", type=Path, default=EVALS_ROOT / "results" / "runs.jsonl")
    parser.add_argument("--implementation-revision")
    args = parser.parse_args()
    fixtures = tuple(item.strip() for item in args.fixtures.split(",") if item.strip())
    factory = lambda: create_provider(args.model)
    records = evaluate(factory, fixtures, args.repetitions, args.output, max_turns=args.max_turns, implementation_revision=args.implementation_revision)
    print(json.dumps(summarize_records(records), indent=2, sort_keys=True))
    return 0 if all(record["outcome"] == "VERIFIED_SUCCESS" for record in records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
