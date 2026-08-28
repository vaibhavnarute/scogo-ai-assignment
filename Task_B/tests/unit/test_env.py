from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.env import load_dotenv
from harness.errors import HarnessError


def test_load_dotenv_loads_nonempty_values_without_overriding(monkeypatch, tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# NVIDIA credential and generic parser coverage\n"
        "NVIDIA_API_KEY='file-nvidia-value'\n"
        'HARNESS_TEST_VALUE="nvidia=test=value"\n'
        "EMPTY_TEST_KEY=\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NVIDIA_API_KEY", "existing-nvidia-key")
    monkeypatch.delenv("HARNESS_TEST_VALUE", raising=False)
    monkeypatch.delenv("EMPTY_TEST_KEY", raising=False)
    loaded = load_dotenv(env_file)
    assert loaded == {"HARNESS_TEST_VALUE"}
    assert os.environ["NVIDIA_API_KEY"] == "existing-nvidia-key"
    assert os.environ["HARNESS_TEST_VALUE"] == "nvidia=test=value"
    assert "EMPTY_TEST_KEY" not in os.environ


def test_load_dotenv_override_is_explicit(monkeypatch, tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("NVIDIA_API_KEY=replacement\n", encoding="utf-8")
    monkeypatch.setenv("NVIDIA_API_KEY", "original")
    assert load_dotenv(env_file, override=True) == {"NVIDIA_API_KEY"}
    assert os.environ["NVIDIA_API_KEY"] == "replacement"


def test_load_dotenv_reports_line_without_exposing_value(tmp_path: Path):
    env_file = tmp_path / ".env"
    secret = "must-not-appear-in-error"
    env_file.write_text(f"INVALID-NAME={secret}\n", encoding="utf-8")
    with pytest.raises(HarnessError) as caught:
        load_dotenv(env_file)
    assert caught.value.code == "CONFIG_INVALID_ENV_FILE"
    assert caught.value.details["line"] == 1
    assert secret not in str(caught.value.as_dict())


def test_missing_dotenv_is_a_noop(tmp_path: Path):
    assert load_dotenv(tmp_path / "missing.env") == set()
