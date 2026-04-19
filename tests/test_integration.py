"""
End-to-end smoke test: full pipeline wired with fakes, no real API calls.
Resume (txt) → parse → score → contacts → messages → Notion disabled.
"""
import json

import anthropic
import pytest


class FakeContent:
    def __init__(self, text):
        self.text = text


class FakeResponse:
    def __init__(self, text):
        self.content = [FakeContent(text)]


class FakeMessages:
    def __init__(self):
        self.responses = []
        self.call_count = 0

    def add_response(self, text):
        self.responses.append(text)

    def create(self, **kwargs):
        response = FakeResponse(self.responses[self.call_count])
        self.call_count += 1
        return response


class FakeAnthropic:
    def __init__(self, **kwargs):
        self.messages = FakeMessages()


def test_full_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    """
    Full pipeline smoke test: resume file → parse → contacts → messages.
    Notion is disabled (empty token). No real API calls made.
    """
    from pipeline import (
        ClaudeClient,
        Config,
        ContactDiscoverer,
        Job,
        JobScorer,
        MessageGenerator,
        NotionWriter,
        Pipeline,
        PipelineState,
        ResumeParser,
        load_resume,
    )

    # --- resume file ---
    resume_file = tmp_path / "resume.txt"
    resume_file.write_text(
        "Ryan Ellis\nLead AI Engineer\nDenver, CO\nSkills: Python, AWS, LangGraph"
    )

    # --- fake Claude responses (one per AI call in order) ---
    fake_anthropic = FakeAnthropic()

    # 1. ResumeParser.parse()
    fake_anthropic.messages.add_response(
        json.dumps({
            "name": "Ryan Ellis",
            "title": "Lead AI Engineer",
            "location": "Denver, CO",
            "skills": ["Python", "AWS", "LangGraph"],
            "experience_years": 20,
            "military_service": {"branch": "U.S. Navy", "years": 20},
        })
    )
    # 2. ContactDiscoverer.discover()
    fake_anthropic.messages.add_response(
        json.dumps([
            {"name": "Alice", "title": "TA Lead", "company": "Acme", "category": "Recruiter",
             "relevance_score": 8, "linkedin_url": "", "email": "", "branch": "", "notes": ""},
            {"name": "Bob", "title": "Director", "company": "Acme", "category": "Hiring Manager",
             "relevance_score": 9, "linkedin_url": "", "email": "", "branch": "", "notes": ""},
            {"name": "Charlie", "title": "Tech Lead", "company": "Acme", "category": "Veteran",
             "relevance_score": 8, "linkedin_url": "", "email": "", "branch": "U.S. Navy", "notes": ""},
        ])
    )
    # 3. MessageGenerator.generate()
    fake_anthropic.messages.add_response(
        json.dumps([
            {"index": 0, "message": "Hi Alice — great to connect."},
            {"index": 1, "message": "Hi Bob — saw your work at Acme."},
            {"index": 2, "message": "Hi Charlie — fellow Navy vet here."},
        ])
    )

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: fake_anthropic)

    # --- config: Notion disabled, no Apify ---
    config = Config()
    config.notion_token = ""
    config.min_relevance_score = 7
    config.min_fit_score = 7

    # --- wire pipeline manually (same as main()) ---
    resume_text = load_resume(resume_file)
    pipeline = Pipeline(config)

    parser = ResumeParser(pipeline.claude)
    pipeline._transition(PipelineState.PARSE_RESUME)
    profile = parser.parse(resume_text)

    pipeline.ctx.resume = profile
    pipeline.ctx.target_job = Job(
        title="AI Ops Lead", company="Acme", location="Denver, CO",
        url="https://linkedin.com/jobs/view/123", key_skills=["Python", "AWS"],
    )

    pipeline._run_contacts_pipeline()

    # --- assertions ---
    assert pipeline.ctx.state == PipelineState.COMPLETE
    assert profile.name == "Ryan Ellis"
    assert profile.military_service is not None

    contacts = pipeline.ctx.contacts
    assert len(contacts) == 3
    assert contacts[0].category == "Veteran"          # veteran boost active
    assert contacts[0].priority == 1
    assert all(c.connection_message for c in contacts)
    assert all(len(c.connection_message) <= 300 for c in contacts)
