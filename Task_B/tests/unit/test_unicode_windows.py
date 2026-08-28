from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.cache import ObservationCache
from harness.config import HarnessConfig
from harness.policy import WorkspacePolicy
from harness.state import RunState
from harness.tools.base import ToolCall, ToolContext
from harness.tools.files import ListFilesTool, ReadFileTool
from harness.tools.patch import ApplyPatchTool
from harness.verify import capture_integrity_baseline


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific Unicode path regression")
def test_unicode_filename_can_be_listed_read_and_patched_on_windows(tmp_path: Path):
    path = tmp_path / "café_测试.py"
    path.write_text("message = 'before'\n", encoding="utf-8")
    config = HarnessConfig(tmp_path, approval_mode="yes")
    state = RunState("unicode repair", tmp_path)
    capture_integrity_baseline(state, config)
    context = ToolContext(config, WorkspacePolicy(config), state, ObservationCache())

    listing = ListFilesTool().execute(ToolCall("l", "list_files", {"path": ".", "depth": 1}), context)
    assert any(entry["path"] == path.name for entry in listing.data["entries"])
    read = ReadFileTool().execute(ToolCall("r", "read_file", {"path": path.name}), context)
    assert read.ok and "before" in read.data["content"]
    patch = "@@ -1,1 +1,1 @@\n-message = 'before'\n+message = 'after'"
    result = ApplyPatchTool().execute(ToolCall("p", "apply_patch", {"path": path.name, "patch": patch}), context)
    assert result.ok
    assert path.read_text(encoding="utf-8") == "message = 'after'\n"