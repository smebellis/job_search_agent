import json

import anthropic
import pytest


def test_claude_client_initializes_with_config():
    """ClaudeClient should accept a Config and store the model name."""
    from pipeline import ClaudeClient, Config

    config = Config()
    config.model = "claude-sonnet-4-20250514"
    client = ClaudeClient(config)

    assert client.model == "claude-sonnet-4-20250514"


def test_claude_client_ask_returns_string(monkeypatch):
    """ask() should return the text content from Claude's response."""
    from pipeline import ClaudeClient, Config

    # Mock the Anthropic API call
    class FakeContent:
        text = "Hello from Claude"

    class FakeResponse:
        content = [FakeContent()]

    class FakeMessages:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)

    config = Config()
    client = ClaudeClient(config)

    result = client.ask("You are helpful.", "Say hello")
    assert result == "Hello from Claude"
    assert isinstance(result, str)


def test_claude_client_ask_json_parses_response(monkeypatch):
    """ask_json() should parse JSON from Claude's response."""
    from pipeline import ClaudeClient, Config

    class FakeContent:
        text = '{"name": "Ryan", "skills": ["Python", "AWS"]}'

    class FakeResponse:
        content = [FakeContent()]

    class FakeMessages:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)

    config = Config()
    client = ClaudeClient(config)

    result = client.ask_json("Parse this.", "Give me JSON")
    assert result["name"] == "Ryan"
    assert "Python" in result["skills"]


def test_claude_client_ask_json_strips_markdown_fences(monkeypatch):
    """ask_json() should handle responses wrapped in ```json fences."""
    from pipeline import ClaudeClient, Config

    class FakeContent:
        text = '```json\n{"status": "ok"}\n```'

    class FakeResponse:
        content = [FakeContent()]

    class FakeMessages:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)

    config = Config()
    client = ClaudeClient(config)

    result = client.ask_json("Parse.", "JSON please")
    assert result["status"] == "ok"
