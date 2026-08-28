from __future__ import annotations

from harness.tools.base import ToolCall, ToolContext
from harness.tools.files import ReadFileTool
from harness.tools.patch import ApplyPatchTool
from harness.tools.registry import ToolRegistry
from harness.integrity import snapshot_workspace


VALID_PATCH = "@@ -1,1 +1,1 @@\n-value = 1\n+value = 2"


def test_applies_valid_patch_and_updates_state(context: ToolContext):
    result = ApplyPatchTool().execute(ToolCall("p", "apply_patch", {"path": "src/app.py", "patch": VALID_PATCH}), context)
    assert result.ok
    assert (context.config.workspace / "src" / "app.py").read_text(encoding="utf-8") == "value = 2\n"
    assert context.state.files_modified == {"src/app.py"}
    assert result.data["changed_lines"] == 2


def test_patch_invalid(context: ToolContext):
    result = ToolRegistry([ApplyPatchTool()]).dispatch(ToolCall("p", "apply_patch", {"path": "src/app.py", "patch": "not a patch"}), context)
    assert result.error_code == "PATCH_INVALID"


def test_patch_conflict(context: ToolContext):
    patch = "@@ -1,1 +1,1 @@\n-wrong = 1\n+value = 2"
    result = ToolRegistry([ApplyPatchTool()]).dispatch(ToolCall("p", "apply_patch", {"path": "src/app.py", "patch": patch}), context)
    assert result.error_code == "PATCH_CONFLICT"


def test_patch_rejects_protected_path(context: ToolContext):
    patch = "@@ -1,1 +1,1 @@\n-def test_locked(): assert True\n+def test_locked(): assert False"
    result = ToolRegistry([ApplyPatchTool()]).dispatch(ToolCall("p", "apply_patch", {"path": "tests/test_locked.py", "patch": patch}), context)
    assert result.error_code == "PROTECTED_PATH"


def test_patch_invalidates_read_cache(context: ToolContext):
    reader = ReadFileTool()
    call = ToolCall("r", "read_file", {"path": "src/app.py"})
    reader.execute(call, context)
    assert len(context.cache) == 1
    result = ApplyPatchTool().execute(ToolCall("p", "apply_patch", {"path": "src/app.py", "patch": VALID_PATCH}), context)
    assert result.data["cache_entries_invalidated"] == 1
    assert not reader.execute(call, context).cached

def test_multi_hunk_conflict_does_not_apply_earlier_hunk(context: ToolContext):
    path = context.config.workspace / "src" / "multi.py"
    original = "first = 1\nsecond = 2\nthird = 3\nfourth = 4\n"
    path.write_text(original, encoding="utf-8")
    context.state.workspace_hashes = snapshot_workspace(context.config)
    patch = "@@ -1,1 +1,1 @@\n-first = 1\n+first = 10\n@@ -4,1 +4,1 @@\n-wrong = 4\n+fourth = 40"
    result = ToolRegistry([ApplyPatchTool()]).dispatch(ToolCall("multi", "apply_patch", {"path": "src/multi.py", "patch": patch}), context)
    assert not result.ok and result.error_code == "PATCH_CONFLICT"
    assert path.read_text(encoding="utf-8") == original
    assert "src/multi.py" not in context.state.files_modified


def test_applies_safe_model_patch_envelope(context: ToolContext):
    patch = """*** Begin Patch
*** Update File: src/app.py
@@
-value = 1
+value = 2
*** End Patch
*** End Patch"""
    result = ApplyPatchTool().execute(
        ToolCall("p", "apply_patch", {"path": "src/app.py", "patch": patch}), context
    )
    assert result.ok
    assert (context.config.workspace / "src" / "app.py").read_text(encoding="utf-8") == "value = 2\n"


def test_model_patch_envelope_path_must_match_tool_path(context: ToolContext):
    patch = """*** Begin Patch
*** Update File: tests/test_locked.py
@@
-value = 1
+value = 2
*** End Patch"""
    result = ToolRegistry([ApplyPatchTool()]).dispatch(
        ToolCall("p", "apply_patch", {"path": "src/app.py", "patch": patch}), context
    )
    assert result.error_code == "PATCH_PATH_MISMATCH"
    assert (context.config.workspace / "src" / "app.py").read_text(encoding="utf-8") == "value = 1\n"


def test_model_patch_envelope_requires_unique_context(context: ToolContext):
    path = context.config.workspace / "src" / "repeated.py"
    original = "value = 1\nvalue = 1\n"
    path.write_text(original, encoding="utf-8")
    context.state.workspace_hashes = snapshot_workspace(context.config)
    patch = """*** Begin Patch
*** Update File: src/repeated.py
@@
-value = 1
+value = 2
*** End Patch"""
    result = ToolRegistry([ApplyPatchTool()]).dispatch(
        ToolCall("p", "apply_patch", {"path": "src/repeated.py", "patch": patch}), context
    )
    assert result.error_code == "PATCH_CONFLICT"
    assert path.read_text(encoding="utf-8") == original


def test_model_patch_envelope_rejects_unsupported_operations(context: ToolContext):
    patch = """*** Begin Patch
*** Add File: src/new.py
+value = 1
*** End Patch"""
    result = ToolRegistry([ApplyPatchTool()]).dispatch(
        ToolCall("p", "apply_patch", {"path": "src/app.py", "patch": patch}), context
    )
    assert result.error_code == "PATCH_UNSUPPORTED_OPERATION"
    assert not (context.config.workspace / "src" / "new.py").exists()

def test_applies_pathless_update_envelope_using_validated_tool_target(context: ToolContext):
    patch = """*** Begin Patch
*** Update File
@@
-value = 1
+value = 2
*** End Patch"""
    result = ApplyPatchTool().execute(
        ToolCall("p", "apply_patch", {"path": "src/app.py", "patch": patch}), context
    )
    assert result.ok


def test_applies_unnumbered_git_diff_with_matching_metadata(context: ToolContext):
    patch = """--- a/src/app.py
+++ b/src/app.py
@@
-value = 1
+value = 2"""
    result = ApplyPatchTool().execute(
        ToolCall("p", "apply_patch", {"path": "src/app.py", "patch": patch}), context
    )
    assert result.ok


def test_unnumbered_git_diff_rejects_metadata_path_mismatch(context: ToolContext):
    patch = """--- a/tests/test_locked.py
+++ b/tests/test_locked.py
@@
-value = 1
+value = 2"""
    result = ToolRegistry([ApplyPatchTool()]).dispatch(
        ToolCall("p", "apply_patch", {"path": "src/app.py", "patch": patch}), context
    )
    assert result.error_code == "PATCH_PATH_MISMATCH"
    assert (context.config.workspace / "src" / "app.py").read_text(encoding="utf-8") == "value = 1\n"
