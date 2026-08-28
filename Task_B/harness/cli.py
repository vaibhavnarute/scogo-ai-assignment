"""Terminal interface for the repository-repair harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .agent import Agent
from .config import HarnessConfig
from .console import ConsoleReporter, safe_console_text
from .env import load_dotenv
from .errors import HarnessError
from .providers.nvidia import DEFAULT_NVIDIA_MODEL, NvidiaConfig, NvidiaProvider
from .tools.base import ToolCall


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal policy-constrained repository-repair AI harness")
    parser.add_argument("task", nargs="?", help="Natural-language repair task; prompts interactively when omitted")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--provider", choices=["nvidia"], default="nvidia")
    parser.add_argument("--model")
    parser.add_argument("--env-file", type=Path, help="Credential file; defaults to Task_B/.env")
    parser.add_argument("--verification-command")
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--command-timeout", type=float, default=30.0)
    parser.add_argument("--provider-timeout", type=float, default=60.0)
    parser.add_argument("--provider-retries", type=int, default=2)
    parser.add_argument("--approval", choices=["safe", "interactive", "yes"], default="interactive")
    parser.add_argument("--yes", action="store_true", help="Approve non-denied mutations and commands non-interactively")
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--json-summary", action="store_true", help="Also print the machine-readable terminal summary")
    return parser


def _provider_from_args(args: argparse.Namespace) -> NvidiaProvider:
    return NvidiaProvider(
        NvidiaConfig(
            model=args.model or DEFAULT_NVIDIA_MODEL,
            timeout_seconds=args.provider_timeout,
            max_retries=args.provider_retries,
            temperature=args.temperature,
            max_tokens=args.max_output_tokens,
        )
    )

def _harness_config(args: argparse.Namespace) -> HarnessConfig:
    workspace = args.workspace.expanduser().resolve()
    approval_mode = "yes" if args.yes else args.approval
    common = {
        "approval_mode": approval_mode,
        "max_turns": args.max_turns,
        "command_timeout_seconds": args.command_timeout,
        "trace_dir": args.trace_dir,
    }
    return HarnessConfig(workspace=workspace, verification_command=args.verification_command or "python -m pytest -q", **common)


def _approval_prompt(call: ToolCall) -> bool:
    arguments = call.arguments if isinstance(call.arguments, dict) else {}
    detail = arguments.get("command", "") if call.name == "run_command" else arguments.get("path", "")
    prompt = safe_console_text(f"Allow {call.name} {detail!r}? [y/N]")
    answer = input(f"[policy] {prompt} ").strip().casefold()
    return answer in {"y", "yes"}


def _resolve_task(task: str | None) -> str:
    interactive = task is None
    if task is None:
        try:
            task = input("> ")
        except EOFError as exc:
            raise ValueError("task is required when stdin is not interactive") from exc
    task = task.strip()
    if interactive and task.startswith(">"):
        task = task[1:].lstrip()
    if not task:
        raise ValueError("task must not be empty")
    return task


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    reporter = ConsoleReporter()
    try:
        load_dotenv(args.env_file)
        config = _harness_config(args)
        provider = _provider_from_args(args)
        print("AI Harness")
        print(f"Workspace: {safe_console_text(config.workspace)}")
        print(f"Provider: {provider.provider_name.upper()}")
        print(f"Model: {safe_console_text(provider.model)}")
        interactive = args.task is None
        if interactive:
            print()
        task = _resolve_task(args.task)
        if interactive:
            print()
        else:
            print(f"\n> {safe_console_text(task)}\n")
        result = Agent(task, config, provider, approval_callback=_approval_prompt, reporter=reporter).run()
    except (HarnessError, ValueError) as exc:
        message = exc.message if isinstance(exc, HarnessError) else str(exc)
        reporter("warning", message)
        reporter("done", "FAILED_TO_START")
        return 4
    try:
        trace_display = result.trace_path.relative_to(Path.cwd())
    except ValueError:
        trace_display = result.trace_path
    modified = ", ".join(sorted(result.state.files_modified)) or "none"
    reporter(
        "done",
        f"Summary: outcome={result.outcome.value} exit={result.exit_code} turns={result.state.turn} "
        f"model_requests={result.model_requests} modified={modified} policy_denials={result.state.policy_denials}",
    )
    reporter(
        "done",
        f"Usage: input={result.usage.input_tokens} output={result.usage.output_tokens} "
        f"cached={result.usage.cached_tokens} total={result.usage.total_tokens} duration_ms={result.duration_ms}",
    )
    reporter("done", f"Trace: {trace_display}")
    if args.json_summary:
        reporter("done", "JSON " + json.dumps(result.summary(), ensure_ascii=False, sort_keys=True))
    return result.exit_code

if __name__ == "__main__":
    raise SystemExit(main())
