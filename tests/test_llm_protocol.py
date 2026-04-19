"""Tests that LLMClient Protocol exists and ClaudeClient satisfies it."""
import json
from typing import runtime_checkable

import anthropic
import pytest


def test_llm_client_protocol_exists():
    from pipeline import LLMClient
    import typing
    assert hasattr(LLMClient, "__protocol_attrs__") or hasattr(LLMClient, "_is_protocol")


def test_claude_client_satisfies_llm_protocol(monkeypatch):
    from pipeline import ClaudeClient, Config, LLMClient

    class FakeContent:
        text = '{"ok": true}'
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
    assert isinstance(client, LLMClient)


def test_resume_parser_accepts_any_llm_client(monkeypatch):
    """ResumeParser should work with any object that has ask/ask_json."""
    from pipeline import ResumeParser, ResumeProfile

    class StubLLM:
        def ask(self, system: str, user: str) -> str:
            return ""
        def ask_json(self, system: str, user: str) -> dict:
            return {
                "name": "Test", "title": "Dev", "location": "Denver",
                "skills": ["Python"], "experience_years": 5, "military_service": None,
            }

    parser = ResumeParser(StubLLM())
    profile = parser.parse("some resume text")
    assert isinstance(profile, ResumeProfile)
    assert profile.name == "Test"
