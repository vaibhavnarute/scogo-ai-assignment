from __future__ import annotations

from evals.reset_fixture import FIXTURE_NAMES, reset_fixture


def test_all_fixtures_reset_to_independent_workspaces(tmp_path):
    for name in FIXTURE_NAMES:
        destination = reset_fixture(name, tmp_path)
        assert (destination / "fixture.json").is_file()
        assert (destination / "tests").is_dir()
    changed = tmp_path / "F1" / "calculator.py"
    changed.write_text("broken reset", encoding="utf-8")
    reset_fixture("F1", tmp_path)
    assert "return left - right" in changed.read_text(encoding="utf-8")

