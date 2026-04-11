import os

import pytest


def test_config_loads_from_env_vars(monkeypatch):
    """Config should populate from environment variables."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")
    monkeypatch.setenv("NOTION_TOKEN", "ntn_test456")

    from pipeline import Config

    config = Config()
    assert config.anthropic_api_key == "sk-ant-test123"
    assert config.notion_token == "ntn_test456"


def test_config_validate_catches_missing_keys(monkeypatch):
    """Validate should return list of missing required keys."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("NOTION_TOKEN", raising=False)

    from pipeline import Config

    config = Config()
    missing = config.validate()
    assert "ANTHROPIC_API_KEY" in missing
    assert "NOTION_TOKEN" in missing


def test_config_validate_passes_when_keys_present(monkeypatch):
    """Validate should return empty list when all required keys exist."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")
    monkeypatch.setenv("NOTION_TOKEN", "ntn_test456")

    from pipeline import Config

    config = Config()
    assert config.validate() == []


def test_config_has_sensible_defaults():
    """Config should have defaults for non-required fields."""
    from pipeline import Config

    config = Config()
    assert config.default_location == "Denver, CO"
    assert config.min_fit_score == 7
    assert config.max_jobs == 10
    assert config.model == "claude-sonnet-4-20250514"
