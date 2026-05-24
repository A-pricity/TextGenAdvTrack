import os

from textgenadvtrack.config import default_model, default_provider


def test_default_provider_falls_back_to_mock(monkeypatch):
    monkeypatch.delenv("TEXTGENADVTRACK_DEFAULT_PROVIDER", raising=False)
    assert default_provider() == "mock"


def test_default_model_reads_environment(monkeypatch):
    monkeypatch.setenv("TEXTGENADVTRACK_DEFAULT_MODEL", "gpt-test-model")
    assert default_model() == "gpt-test-model"
