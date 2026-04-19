import json

import anthropic
import pytest


def test_resume_parser_returns_profile(monkeypatch):
    """ResumeParser should return a ResumeProfile from resume text."""
    from pipeline import ClaudeClient, Config, ResumeParser, ResumeProfile

    fake_response = {
        "name": "Ryan Ellis",
        "title": "Lead AI Engineer",
        "location": "Denver, CO",
        "skills": ["Python", "AWS", "LangGraph"],
        "experience_years": 20,
        "military_service": {"branch": "U.S. Navy", "years": 20},
    }

    class FakeContent:
        text = json.dumps(fake_response)

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
    claude = ClaudeClient(config)
    parser = ResumeParser(claude)

    profile = parser.parse("Ryan Ellis\nLead AI Engineer\nDenver, CO")

    assert isinstance(profile, ResumeProfile)
    assert profile.name == "Ryan Ellis"
    assert profile.location == "Denver, CO"
    assert "Python" in profile.skills
    assert profile.military_service["branch"] == "U.S. Navy"


def test_resume_parser_handles_missing_military(monkeypatch):
    """ResumeParser should handle resumes with no military service."""
    from pipeline import ClaudeClient, Config, ResumeParser

    fake_response = {
        "name": "Jane Doe",
        "title": "Software Engineer",
        "location": "Austin, TX",
        "skills": ["Java", "Spring Boot"],
        "experience_years": 5,
        "military_service": None,
    }

    class FakeContent:
        text = json.dumps(fake_response)

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
    claude = ClaudeClient(config)
    parser = ResumeParser(claude)

    profile = parser.parse("Jane Doe\nSoftware Engineer")

    assert profile.name == "Jane Doe"
    assert profile.military_service is None


def test_resume_parser_sends_resume_text_to_claude(monkeypatch):
    """ResumeParser should pass the resume text to Claude's user message."""
    from pipeline import ClaudeClient, Config, ResumeParser

    captured_kwargs = {}

    class FakeContent:
        text = '{"name": "", "title": "", "location": "", "skills": [], "experience_years": 0, "military_service": null}'

    class FakeResponse:
        content = [FakeContent()]

    class FakeMessages:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeResponse()

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)

    config = Config()
    claude = ClaudeClient(config)
    parser = ResumeParser(claude)

    parser.parse("THIS IS MY RESUME TEXT")

    user_message = captured_kwargs["messages"][0]["content"]
    assert "THIS IS MY RESUME TEXT" in user_message


def test_resume_parser_ignores_extra_keys_from_llm(monkeypatch):
    """parse() must not crash when LLM returns unexpected extra fields."""
    import json
    import anthropic
    from pipeline import Config, ClaudeClient, ResumeParser

    response_with_extras = {
        "name": "Ryan", "title": "Lead", "location": "Denver",
        "skills": ["Python"], "experience_years": 5, "military_service": None,
        "unexpected_field": "some value",  # extra key LLM hallucinated
    }

    class FakeContent:
        text = json.dumps(response_with_extras)
    class FakeResponse:
        content = [FakeContent()]
    class FakeMessages:
        def create(self, **kwargs): return FakeResponse()
    class FakeAnthropic:
        def __init__(self, **kwargs): self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    config = Config()
    parser = ResumeParser(ClaudeClient(config))
    profile = parser.parse("Ryan Ellis resume text")
    assert profile.name == "Ryan"
