from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from evals.reset_fixture import reset_fixture
from harness.cache import ObservationCache
from harness.command_line import split_command
from harness.config import HarnessConfig
from harness.policy import WorkspacePolicy
from harness.state import RunState
from harness.tools.base import ToolCall, ToolContext
from harness.tools.command import RunCommandTool
from harness.tools.files import ListFilesTool, ReadFileTool
from harness.tools.finish import FinishTool
from harness.tools.patch import ApplyPatchTool
from harness.tools.registry import ToolRegistry
from harness.trace import sanitize_for_log
from harness.verify import Verifier, capture_integrity_baseline


def make_context(workspace: Path, **config_values: object) -> ToolContext:
    config = HarnessConfig(workspace, approval_mode="yes", **config_values)
    state = RunState("repair", workspace)
    capture_integrity_baseline(state, config)
    return ToolContext(config, WorkspacePolicy(config), state, ObservationCache())


def test_command_side_mutation_blocks_finish(tmp_path: Path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
    (tmp_path / "source.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "mutate.py").write_text("from pathlib import Path\nPath('source.py').write_text('value = 2\\n')\n", encoding="utf-8")
    context = make_context(tmp_path)
    mutation = RunCommandTool().execute(ToolCall("c1", "run_command", {"command": "python mutate.py"}), context)
    assert mutation.ok and mutation.data["repository_changes"] == ["source.py"]
    verification = RunCommandTool().execute(ToolCall("c2", "run_command", {"command": "python -m pytest -q"}), context)
    assert verification.ok and verification.data["exit_code"] == 0
    finish = FinishTool().execute(ToolCall("f", "finish", {"summary": "done", "evidence": "pytest"}), context)
    assert not finish.ok and finish.error_code == "INTEGRITY_VIOLATION"


def test_unrecorded_change_after_verification_is_detected(workspace: Path):
    context = make_context(workspace, verification_command="pytest -q")
    context.state.record_command("pytest -q", workspace, 0, False, True)
    (workspace / "src" / "app.py").write_text("value = 99\n", encoding="utf-8")
    decision = Verifier(context.config).evaluate_finish(context.state)
    assert not decision.accepted and decision.error_code == "INTEGRITY_VIOLATION"


def test_fixture_contract_is_loaded_and_metadata_is_protected(tmp_path: Path):
    workspace = reset_fixture("F1", tmp_path)
    config = HarnessConfig.from_fixture(workspace)
    assert config.verification_command == "python -m pytest -q"
    assert "tests" in config.protected_paths and "fixture.json" in config.protected_paths
    context = ToolContext(config, WorkspacePolicy(config), RunState("repair", workspace), ObservationCache())
    result = ToolRegistry([ApplyPatchTool()]).dispatch(ToolCall("p", "apply_patch", {"path": "fixture.json", "patch": "@@ -1,1 +1,1 @@\n-old\n+new"}), context)
    assert result.error_code == "PROTECTED_PATH"


def test_windows_command_parser_preserves_backslashes_and_quotes():
    assert split_command(r'python "C:\temp\my script.py"', windows=True) == ["python", r"C:\temp\my script.py"]


def test_timeout_kills_descendant_processes(tmp_path: Path):
    (tmp_path / "child.py").write_text("import time\nfrom pathlib import Path\ntime.sleep(0.8)\nPath('survivor.txt').write_text('alive')\n", encoding="utf-8")
    (tmp_path / "parent.py").write_text("import subprocess, sys, time\nsubprocess.Popen([sys.executable, 'child.py'])\ntime.sleep(10)\n", encoding="utf-8")
    context = make_context(tmp_path, command_timeout_seconds=0.15)
    result = RunCommandTool().execute(ToolCall("c", "run_command", {"command": "python parent.py"}), context)
    assert result.error_code == "COMMAND_TIMEOUT"
    time.sleep(1.0)
    assert not (tmp_path / "survivor.txt").exists()


def test_token_usage_metrics_are_preserved_while_secrets_are_redacted():
    sanitized = sanitize_for_log({"token_usage": {"input_tokens": 10, "output_tokens": 5}, "access_token": "secret"})
    assert sanitized["token_usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert sanitized["access_token"] == "[REDACTED]"


def test_repository_subprocess_cannot_read_configured_provider_secret(tmp_path: Path, monkeypatch):
    secret = "fake-provider-key-credential-leak-test"
    monkeypatch.setenv("NVIDIA_API_KEY", secret)
    (tmp_path / "leak.py").write_text("import os\nprint(os.environ.get('NVIDIA_API_KEY', 'NOT_VISIBLE'))\n", encoding="utf-8")
    context = make_context(tmp_path)
    result = RunCommandTool().execute(ToolCall("c", "run_command", {"command": "python leak.py"}), context)
    assert result.ok and "NOT_VISIBLE" in result.data["stdout"]
    assert secret not in result.data["stdout"]
    assert secret not in str(sanitize_for_log({"observation": secret}))


def test_external_workspace_drift_is_rejected_before_patch_and_blocks_finish(tmp_path: Path):
    (tmp_path / "source.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "unrelated.py").write_text("stable = True\n", encoding="utf-8")
    context = make_context(tmp_path, verification_command="python -c \"raise SystemExit(0)\"")
    (tmp_path / "unrelated.py").write_text("stable = False\n", encoding="utf-8")
    with pytest.raises(Exception) as caught:
        ApplyPatchTool().execute(ToolCall("p", "apply_patch", {"path": "source.py", "patch": "@@ -1,1 +1,1 @@\n-value = 1\n+value = 2"}), context)
    assert getattr(caught.value, "code", None) == "INTEGRITY_VIOLATION"
    assert (tmp_path / "source.py").read_text(encoding="utf-8") == "value = 1\n"
    assert context.state.integrity_violations[0]["paths"] == ["unrelated.py"]
    context.state.record_command(context.config.verification_command, tmp_path, 0, False, True)
    finish = FinishTool().execute(ToolCall("f", "finish", {"summary": "done", "evidence": "exit 0"}), context)
    assert not finish.ok and finish.error_code == "INTEGRITY_VIOLATION"


def test_list_files_prunes_exclusions_and_does_not_follow_symlink(context: ToolContext, tmp_path_factory):
    excluded = context.config.workspace / ".git" / "deep"
    excluded.mkdir(parents=True)
    (excluded / "secret.txt").write_text("hidden", encoding="utf-8")
    outside = tmp_path_factory.mktemp("listing-outside")
    (outside / "outside.txt").write_text("outside", encoding="utf-8")
    link = context.config.workspace / "external-link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        link = None
    result = ListFilesTool().execute(ToolCall("l", "list_files", {"path": ".", "depth": 4}), context)
    paths = {entry["path"]: entry["type"] for entry in result.data["entries"]}
    assert not any(path.startswith(".git") for path in paths)
    assert "outside.txt" not in paths
    if link is not None:
        assert paths["external-link"] == "blocked_symlink"


def test_error_details_survive_registry_normalization(context: ToolContext):
    result = ToolRegistry([ReadFileTool()]).dispatch(ToolCall("r", "read_file", {"path": "missing.py"}), context)
    assert result.error_details == {"path": "missing.py"}
    assert result.as_dict()["error_details"] == {"path": "missing.py"}


def test_complete_git_diff_and_zero_count_insertion_are_supported(context: ToolContext):
    full_patch = "diff --git a/src/app.py b/src/app.py\nindex 123..456 100644\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1,1 +1,1 @@\n-value = 1\n+value = 2"
    assert ApplyPatchTool().execute(ToolCall("p1", "apply_patch", {"path": "src/app.py", "patch": full_patch}), context).ok
    insertion = "@@ -0,0 +1,1 @@\n+header = 0"
    assert ApplyPatchTool().execute(ToolCall("p2", "apply_patch", {"path": "src/app.py", "patch": insertion}), context).ok
    assert (context.config.workspace / "src" / "app.py").read_text(encoding="utf-8") == "header = 0\nvalue = 2\n"


def test_default_trace_directory_is_task_b_runtime_directory(workspace: Path):
    config = HarnessConfig(workspace)
    assert config.trace_dir.name == ".harness_runs"
    assert config.trace_dir.parent.name == "Task_B"
