"""config._int_setting: config is imported by every script, so a bad value in
.env or the environment has to warn and fall back, never crash at import."""
import os
from pathlib import Path

import pytest

import config


def test_default_when_unset(monkeypatch):
    monkeypatch.delenv("PARALLEL_WORKERS", raising=False)
    assert config._int_setting("PARALLEL_WORKERS", 4) == 4


def test_reads_env_value(monkeypatch):
    monkeypatch.setenv("PARALLEL_WORKERS", "2")
    assert config._int_setting("PARALLEL_WORKERS", 4) == 2


def test_garbage_falls_back_with_warning(monkeypatch, capsys):
    monkeypatch.setenv("PARALLEL_WORKERS", "four")
    assert config._int_setting("PARALLEL_WORKERS", 4) == 4
    assert "WARNING" in capsys.readouterr().err


def test_zero_and_negative_clamp_to_minimum(monkeypatch):
    monkeypatch.setenv("PARALLEL_WORKERS", "0")
    assert config._int_setting("PARALLEL_WORKERS", 4) == 1
    monkeypatch.setenv("PARALLEL_WORKERS", "-3")
    assert config._int_setting("PARALLEL_WORKERS", 4) == 1


def test_custom_minimum(monkeypatch):
    monkeypatch.setenv("LLAMA_CTX", "512")
    assert config._int_setting("LLAMA_CTX", 8192, minimum=2048) == 2048


def test_app_support_dir_default_is_the_standard_macos_location():
    # Pinned so a refactor can't silently move models and data elsewhere.
    if os.environ.get("VMT_APP_SUPPORT_DIR"):
        pytest.skip("VMT_APP_SUPPORT_DIR overrides the default")
    assert config.APP_SUPPORT_DIR == (
        Path.home() / "Library" / "Application Support" / "VLM Media Tagger")
