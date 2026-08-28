"""Run a deterministic end-to-end Task B repair demo and retain its JSONL trace."""

from __future__ import annotations

import argparse
from pathlib import Path

from evals.reset_fixture import EVALS_ROOT, reset_fixture
from harness.agent import Agent
from harness.config import HarnessConfig
from harness.console import ConsoleReporter
from harness.providers.mock import MockProvider
from harness.providers.types import ModelResponse
from harness.tools.base import ToolCall

F1_PATCH = """@@ -1,3 +1,3 @@
 def add(left: int, right: int) -> int:
     \"\"\"Return the sum of two integers.\"\"\"
-    return left - right
+    return left + right"""


def demo_provider() -> MockProvider:
    return MockProvider(
        [
            ModelResponse(tool_calls=[ToolCall("demo-list", "list_files", {"path": ".", "depth": 2})]),
            ModelResponse(tool_calls=[ToolCall("demo-test-before", "run_command", {"command": "python -m pytest -q"})]),
            ModelResponse(tool_calls=[ToolCall("demo-read", "read_file", {"path": "calculator.py"})]),
            ModelResponse(tool_calls=[ToolCall("demo-patch", "apply_patch", {"path": "calculator.py", "patch": F1_PATCH})]),
            ModelResponse(tool_calls=[ToolCall("demo-test-after", "run_command", {"command": "python -m pytest -q"})]),
            ModelResponse(tool_calls=[ToolCall("demo-finish", "finish", {"summary": "corrected addition", "evidence": "pytest passed"})]),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=EVALS_ROOT / "demo_workspaces")
    parser.add_argument("--trace-dir", type=Path, default=EVALS_ROOT.parent / ".harness_runs")
    args = parser.parse_args()
    workspace = reset_fixture("F1", args.workspace_root)
    config = HarnessConfig.from_fixture(workspace, approval_mode="yes", trace_dir=args.trace_dir)
    provider = demo_provider()
    task = "Fix the failing calculator tests and verify the repair."
    print("AI Harness")
    print(f"Workspace: {workspace}")
    print(f"Provider: {provider.provider_name.upper()}")
    print(f"Model: {provider.model}")
    print(f"\n> {task}\n")
    reporter = ConsoleReporter()
    result = Agent(task, config, provider, reporter=reporter).run()
    print(f"Turns: {result.state.turn}")
    print(f"Trace: {result.trace_path}")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())