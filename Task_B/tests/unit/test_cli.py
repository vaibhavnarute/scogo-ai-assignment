from pathlib import Path

import harness.cli as cli_module
from harness.cli import main
from harness.config import HarnessConfig
from harness.providers.mock import MockProvider
from harness.providers.types import ModelResponse


def _workspace(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text("def test_ok(): assert True\n", encoding="utf-8")


def test_cli_configuration_failure_returns_exit_four(tmp_path: Path, monkeypatch, capsys):
    _workspace(tmp_path)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    code = main([
        "repair",
        "--workspace", str(tmp_path),
        "--env-file", str(tmp_path / "missing.env"),
        "--trace-dir", str(tmp_path / "traces"),
        "--yes",
    ])
    captured = capsys.readouterr()
    assert code == 4
    assert "Provider: NVIDIA" in captured.out


def test_cli_prompts_for_task_when_omitted(tmp_path: Path, monkeypatch, capsys):
    _workspace(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "repair from interactive prompt")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    code = main([
        "--workspace", str(tmp_path),
        "--env-file", str(tmp_path / "missing.env"),
        "--trace-dir", str(tmp_path / "traces"),
        "--yes",
    ])
    assert code == 4
    assert "Provider: NVIDIA" in capsys.readouterr().out


def test_cli_rejects_empty_interactive_task(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _prompt: "   ")
    assert main([]) == 4
    output = capsys.readouterr().out
    assert "[warning] task must not be empty" in output
    assert "[done] FAILED_TO_START" in output

def _text_only_provider() -> MockProvider:
    return MockProvider([ModelResponse(text="No tool action")])


def _request_contains_task(provider: MockProvider, task: str) -> bool:
    return any(task in str(message.get("content", "")) for message in provider.requests[0]["messages"])


def test_interactive_input_becomes_real_agent_task(tmp_path: Path, monkeypatch, capsys):
    _workspace(tmp_path)
    provider = _text_only_provider()
    monkeypatch.setattr(cli_module, "_provider_from_args", lambda _args: provider)
    monkeypatch.setattr("builtins.input", lambda prompt: "> Repair the interactive bug" if prompt == "> " else "yes")
    code = main([
        "--workspace", str(tmp_path),
        "--max-turns", "1",
        "--trace-dir", str(tmp_path / "traces"),
        "--yes",
    ])
    output = capsys.readouterr().out
    assert code == 2
    assert _request_contains_task(provider, "Repair the interactive bug")
    assert not _request_contains_task(provider, "> Repair the interactive bug")
    assert "AI Harness" in output and "[run] Started run " in output
    assert "[done] Summary: outcome=INCOMPLETE" in output
    assert "[done] Trace:" in output


def test_direct_positional_task_becomes_real_agent_task(tmp_path: Path, monkeypatch, capsys):
    _workspace(tmp_path)
    provider = _text_only_provider()
    monkeypatch.setattr(cli_module, "_provider_from_args", lambda _args: provider)
    code = main([
        "Fix the failing tests",
        "--workspace", str(tmp_path),
        "--max-turns", "1",
        "--trace-dir", str(tmp_path / "traces"),
        "--yes",
    ])
    output = capsys.readouterr().out
    assert code == 2
    assert _request_contains_task(provider, "Fix the failing tests")
    assert "> Fix the failing tests" in output


def test_cli_reports_eof_without_starting_agent(monkeypatch, capsys):
    def raise_eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", raise_eof)
    assert main([]) == 4
    output = capsys.readouterr().out
    assert "[warning] task is required when stdin is not interactive" in output
    assert "[done] FAILED_TO_START" in output

def test_cli_accepts_explicit_nvidia_provider():
    args = cli_module.build_parser().parse_args(["--provider", "nvidia"])
    assert args.provider == "nvidia"


def test_production_cli_does_not_trust_workspace_fixture_verifier(tmp_path: Path):
    _workspace(tmp_path)
    (tmp_path / "fixture.json").write_text('{"verification_command":"python fake_verify.py"}', encoding="utf-8")
    args = cli_module.build_parser().parse_args(["repair", "--workspace", str(tmp_path), "--yes"])
    assert cli_module._harness_config(args).verification_command == "python -m pytest -q"
    assert HarnessConfig.from_fixture(tmp_path).verification_command == "python fake_verify.py"
