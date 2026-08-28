from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.config import HarnessConfig
from harness.errors import HarnessError
from harness.policy import PathIntent, WorkspacePolicy


def test_resolves_internal_path(workspace: Path):
    policy = WorkspacePolicy(HarnessConfig(workspace))
    assert policy.resolve_path("src/app.py", PathIntent.READ) == workspace / "src" / "app.py"


@pytest.mark.parametrize("requested", ["../outside.txt", "../../outside.txt"])
def test_blocks_traversal(workspace: Path, requested: str):
    with pytest.raises(HarnessError) as caught:
        WorkspacePolicy(HarnessConfig(workspace)).resolve_path(requested, PathIntent.READ)
    assert caught.value.code == "PATH_OUTSIDE_WORKSPACE"


def test_blocks_absolute_outside_path(workspace: Path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "file.py"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(HarnessError) as caught:
        WorkspacePolicy(HarnessConfig(workspace)).resolve_path(str(outside), PathIntent.READ)
    assert caught.value.code == "PATH_OUTSIDE_WORKSPACE"


@pytest.mark.parametrize("name", [".env", "private.pem", "service.key", "credentials.json", "secrets.txt"])
def test_blocks_sensitive_names(workspace: Path, name: str):
    path = workspace / name
    path.write_text("do-not-read", encoding="utf-8")
    with pytest.raises(HarnessError) as caught:
        WorkspacePolicy(HarnessConfig(workspace)).resolve_path(name, PathIntent.READ)
    assert caught.value.code == "SENSITIVE_PATH"


def test_blocks_protected_write(workspace: Path):
    with pytest.raises(HarnessError) as caught:
        WorkspacePolicy(HarnessConfig(workspace)).resolve_path("tests/test_locked.py", PathIntent.WRITE)
    assert caught.value.code == "PROTECTED_PATH"


def test_blocks_symlink_escape_when_supported(workspace: Path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("external")
    target = outside / "secret.txt"
    target.write_text("secret", encoding="utf-8")
    link = workspace / "link.txt"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable for this account")
    with pytest.raises(HarnessError) as caught:
        WorkspacePolicy(HarnessConfig(workspace)).resolve_path("link.txt", PathIntent.READ)
    assert caught.value.code == "PATH_OUTSIDE_WORKSPACE"


def test_denies_shell_and_network_commands(workspace: Path):
    policy = WorkspacePolicy(HarnessConfig(workspace, approval_mode="yes"))
    assert not policy.check_command("pytest -q && curl example.com", workspace).allowed
    assert not policy.check_command("curl example.com", workspace).allowed
    assert not policy.check_command("python -c 'print(1)'", workspace).allowed


def test_safe_mode_allows_only_exact_verification(workspace: Path):
    policy = WorkspacePolicy(HarnessConfig(workspace, verification_command="pytest -q"))
    assert policy.check_command("pytest -q", workspace).allowed
    assert not policy.check_command("python --version", workspace).allowed

