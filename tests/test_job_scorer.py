import json

import anthropic
import pytest


def test_job_dataclass_has_required_fields():
    """Job should have all expected fields with defaults."""
    from pipeline import Job

    job = Job()
    assert job.title == ""
    assert job.company == ""
    assert job.location == ""
    assert job.url == ""
    assert job.source == ""
    assert job.fit_score == 0
    assert job.key_skills == []
    assert job.description == ""


def test_job_stores_values():
    """Job should accept and store values."""
    from pipeline import Job

    job = Job(
        title="AI Ops Lead Engineer",
        company="Infosys",
        location="Denver, CO",
        url="https://linkedin.com/jobs/view/123",
        source="LinkedIn",
        fit_score=9,
        key_skills=["Python", "AWS", "Bedrock"],
    )
    assert job.title == "AI Ops Lead Engineer"
    assert job.fit_score == 9
    assert "AWS" in job.key_skills


def test_job_scorer_filters_below_threshold(monkeypatch):
    """JobScorer should exclude jobs scoring below min_fit_score."""
    from pipeline import ClaudeClient, Config, JobScorer

    scores_response = [
        {"index": 0, "fit_score": 9, "key_skills_matched": ["Python", "AWS"]},
        {"index": 1, "fit_score": 5, "key_skills_matched": ["Java"]},
        {"index": 2, "fit_score": 8, "key_skills_matched": ["Python"]},
    ]

    class FakeContent:
        text = json.dumps(scores_response)

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
    config.min_fit_score = 7
    claude = ClaudeClient(config)
    scorer = JobScorer(claude, config)

    raw_jobs = [
        {
            "title": "AI Engineer",
            "company": "Acme",
            "location": "Denver",
            "source": "Google",
            "key_skills": [],
            "description_summary": "AI role",
        },
        {
            "title": "Java Dev",
            "company": "BigCorp",
            "location": "Denver",
            "source": "LinkedIn",
            "key_skills": [],
            "description_summary": "Java role",
        },
        {
            "title": "ML Engineer",
            "company": "StartupX",
            "location": "Denver",
            "source": "Google",
            "key_skills": [],
            "description_summary": "ML role",
        },
    ]

    from pipeline import ResumeProfile

    resume = ResumeProfile(name="Ryan", skills=["Python", "AWS"], experience_years=20)

    results = scorer.score(raw_jobs, resume)

    assert len(results) == 2
    assert all(j.fit_score >= 7 for j in results)


def test_job_scorer_sorts_by_score_descending(monkeypatch):
    """JobScorer should return results sorted highest score first."""
    from pipeline import ClaudeClient, Config, JobScorer, ResumeProfile

    scores_response = [
        {"index": 0, "fit_score": 7, "key_skills_matched": ["Python"]},
        {"index": 1, "fit_score": 9, "key_skills_matched": ["Python", "AWS"]},
        {"index": 2, "fit_score": 8, "key_skills_matched": ["AWS"]},
    ]

    class FakeContent:
        text = json.dumps(scores_response)

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
    config.min_fit_score = 7
    claude = ClaudeClient(config)
    scorer = JobScorer(claude, config)

    raw_jobs = [
        {
            "title": "Role A",
            "company": "Co A",
            "location": "Denver",
            "source": "Google",
            "key_skills": [],
            "description_summary": "",
        },
        {
            "title": "Role B",
            "company": "Co B",
            "location": "Denver",
            "source": "LinkedIn",
            "key_skills": [],
            "description_summary": "",
        },
        {
            "title": "Role C",
            "company": "Co C",
            "location": "Denver",
            "source": "Google",
            "key_skills": [],
            "description_summary": "",
        },
    ]

    resume = ResumeProfile(name="Ryan", skills=["Python"], experience_years=20)
    results = scorer.score(raw_jobs, resume)

    assert results[0].fit_score == 9
    assert results[1].fit_score == 8
    assert results[2].fit_score == 7


def test_job_scorer_respects_max_jobs(monkeypatch):
    """JobScorer should cap output at config.max_jobs."""
    from pipeline import ClaudeClient, Config, JobScorer, ResumeProfile

    scores_response = [
        {"index": i, "fit_score": 8, "key_skills_matched": ["Python"]}
        for i in range(10)
    ]

    class FakeContent:
        text = json.dumps(scores_response)

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
    config.min_fit_score = 7
    config.max_jobs = 3
    claude = ClaudeClient(config)
    scorer = JobScorer(claude, config)

    raw_jobs = [
        {
            "title": f"Role {i}",
            "company": f"Co {i}",
            "location": "Denver",
            "source": "Google",
            "key_skills": [],
            "description_summary": "",
        }
        for i in range(10)
    ]

    resume = ResumeProfile(name="Ryan", skills=["Python"], experience_years=20)
    results = scorer.score(raw_jobs, resume)

    assert len(results) == 3


def test_job_scorer_populates_key_skills_from_matched(monkeypatch):
    """Job.key_skills must be populated from key_skills_matched in the LLM response."""
    import json
    import anthropic
    from pipeline import Config, ClaudeClient, JobScorer, ResumeProfile

    scores_response = [{"index": 0, "fit_score": 9, "key_skills_matched": ["Python", "AWS"]}]

    class FakeContent:
        text = json.dumps(scores_response)
    class FakeResponse:
        content = [FakeContent()]
    class FakeMessages:
        def create(self, **kwargs): return FakeResponse()
    class FakeAnthropic:
        def __init__(self, **kwargs): self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    config = Config()
    config.min_fit_score = 7
    scorer = JobScorer(ClaudeClient(config), config)
    raw_jobs = [{"title": "AI Eng", "company": "Acme", "location": "Denver",
                 "source": "LinkedIn", "description_summary": ""}]
    results = scorer.score(raw_jobs, ResumeProfile(name="Ryan", skills=["Python", "AWS"]))
    assert results[0].key_skills == ["Python", "AWS"]
