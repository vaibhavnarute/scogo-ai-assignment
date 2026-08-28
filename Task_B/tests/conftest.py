from __future__ import annotations

from pathlib import Path

import pytest

from harness.cache import ObservationCache
from harness.config import HarnessConfig
from harness.policy import WorkspacePolicy
from harness.state import RunState
from harness.tools.base import ToolContext
from harness.verify import capture_integrity_baseline


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_locked.py").write_text("def test_locked(): assert True\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("value = 1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def context(workspace: Path) -> ToolContext:
    config = HarnessConfig(workspace=workspace, approval_mode="yes")
    state = RunState("repair the fixture", workspace)
    capture_integrity_baseline(state, config)
    return ToolContext(config, WorkspacePolicy(config), state, ObservationCache())

