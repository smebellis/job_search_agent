# Job Pipeline Orchestrator Agent — Session Transfer Document

> **Purpose:** This document transfers a TDD-guided build session to a new conversation. It contains everything needed to continue from exactly where we stopped: all code written, all tests passing, the student's struggle patterns, and the next module to build.

---

## 1. Who is the student

- **Name:** Ryan Ellis
- **Role:** Lead AI Engineer at Deloitte, Denver CO
- **Background:** 20+ years U.S. Navy (intelligence operator), Masters in AI (University of Michigan), BS in CS (CU Denver)
- **Core skills:** Python, FastAPI, LangGraph, LangChain, RAG, AWS, Azure, MLflow, Ray Serve, PyTorch, agentic AI, MLOps
- **Learning style:** Learns best through Socratic questioning and TDD. Wants to understand WHY, not just get working code. Prefers progressive hints (plain English → pseudocode → skeleton → answer).

---

## 2. What we are building

A Python CLI agent that automates job search outreach:
1. Parses resume → structured profile
2. Searches for matching jobs (Google + LinkedIn/Apify)
3. Scores jobs against resume (fit score 1-10)
4. Discovers networking contacts at target companies (4 categories: Recruiter, Hiring Manager, Veteran, Peer)
5. Generates personalized <300 char LinkedIn connection messages
6. Writes everything to Notion databases

**Execution:** Single command — `python pipeline.py "https://linkedin.com/jobs/view/..."`

**Persistence:** Two Notion databases (already created):
- Job Pipeline - Applications (DB ID: `5135bb9f-2ad7-4400-b42a-b07d35a37d43`)
- Job Pipeline - Contacts (DB ID: `7e736104-656e-470c-b301-7bcd3c8dee82`)

---

## 3. TDD methodology rules

Follow this exact sequence for each module:
1. Give failing tests first. Student runs pytest and sees red.
2. Ask guiding questions about what tests expect and why.
3. Student implements. Never give the answer immediately.
4. When stuck: progressive hints (plain English → pseudocode → skeleton code).
5. When wrong: explain WHY it's wrong and what concept is missing.
6. After green tests: highlight what was learned.
7. **After each module completes:** Generate a quiz (questions only, no answers) weighted toward struggle areas.
8. **After each module completes:** Update the struggle tracker.

---

## 4. Current code (pipeline.py) — all tests passing

```python
import json
from dataclasses import dataclass, field
from enum import Enum
from os import getenv
from typing import Any, Optional

import anthropic


@dataclass
class Config:
    anthropic_api_key: str = ""
    notion_token: str = ""
    default_location: str = "Denver, CO"
    min_fit_score: int = 7
    min_relevance_score: int = 7
    max_jobs: int = 10
    model: str = "claude-sonnet-4-20250514"

    def __post_init__(self) -> None:
        self.anthropic_api_key = getenv("ANTHROPIC_API_KEY", "")
        self.notion_token = getenv("NOTION_TOKEN", "")

    def validate(self) -> list[Any]:
        missing_keys = []
        required_fields = {
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
            "NOTION_TOKEN": self.notion_token,
        }
        for display_name, value in required_fields.items():
            if not value:
                missing_keys.append(display_name)
        return missing_keys


class PipelineState(Enum):
    IDLE = "idle"
    PARSE_RESUME = "parse_resume"
    SEARCH_JOBS = "search_jobs"
    SCORE_JOBS = "score_jobs"
    DISCOVER_CONTACTS = "discover_contacts"
    GENERATE_MESSAGES = "generate_messages"
    WRITE_NOTION = "write_notion"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class PipelineContext:
    state: PipelineState = PipelineState.IDLE
    jobs: list = field(default_factory=list)
    contacts: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    resume: None = None
    target_job: None = None
    job_url: None = None


@dataclass
class ResumeProfile:
    name: str = ""
    title: str = ""
    location: str = ""
    skills: list = field(default_factory=list)
    experience_years: int = 0
    military_service: Optional[dict] = None


@dataclass
class Job:
    title: str = ""
    company: str = ""
    location: str = ""
    url: str = ""
    source: str = ""
    posted_date: str = ""
    description: str = ""
    key_skills: list = field(default_factory=list)
    salary: str = ""
    fit_score: int = 0


@dataclass
class Contact:
    name: str = ""
    company: str = ""
    title: str = ""
    category: str = ""
    relevance_score: int = 0
    linkedin_url: str = ""
    email: str = ""
    branch: str = ""
    connection_message: str = ""
    priority: int = 0
    notes: str = ""


class ClaudeClient:
    def __init__(self, config: Config) -> None:
        self.model = config.model
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def ask(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            max_tokens=1024,
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def ask_json(self, system: str, user: str) -> dict:
        raw = self.ask(system, user)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        return json.loads(text)


class ResumeParser:
    def __init__(self, claude: ClaudeClient) -> None:
        self.claude = claude

    def parse(self, resume_text: str) -> ResumeProfile:
        system = "You are a resume parser. Extract structured data from the resume. Return ONLY valid JSON with these fields: name, title, location, skills (list), experience_years (number), military_service (dict with branch and years, or null)."
        result = self.claude.ask_json(system, resume_text)
        return ResumeProfile(**result)


class JobScorer:
    def __init__(self, claude: ClaudeClient, config: Config) -> None:
        self.claude = claude
        self.config = config

    def score(self, raw_jobs: list[dict[str, Any]], resume: ResumeProfile) -> list:
        system = 'You are a expert resume ranker. Given a list of job postings and a candidate resume. Score each roles out of 10 based on fit (role, responsibilities, skills, experience). The scoring criteria is role alignment, skills overlap, seniority match, experience requirements, location match. Return ONLY a valid JSON array like: [{"index": 0, "fit_score": 7, "key_skills_matched": ["Python"]}]'

        resume_info = {
            "name": resume.name,
            "skills": resume.skills,
            "experience_years": resume.experience_years,
            "location": resume.location,
        }

        user = f"Resume:\n{json.dumps(resume_info)}\n\nJobs:\n{json.dumps(raw_jobs)}"
        scores = self.claude.ask_json(system, user)

        passing_scores = [
            s for s in scores if s["fit_score"] >= self.config.min_fit_score
        ]

        sorted_scores = sorted(
            passing_scores,
            key=lambda s: s["fit_score"],
            reverse=True,
        )

        results = []
        for score in sorted_scores:
            raw = raw_jobs[score["index"]]
            results.append(
                Job(
                    title=raw.get("title", ""),
                    company=raw.get("company", ""),
                    location=raw.get("location", ""),
                    source=raw.get("source", ""),
                    fit_score=score["fit_score"],
                    key_skills=score.get("key_skills_matched", []),
                    description=raw.get("description_summary", ""),
                ),
            )

        return results[: self.config.max_jobs]


class ContactDiscoverer:
    def __init__(self, claude: ClaudeClient, config: Config) -> None:
        self.claude = claude
        self.config = config

    def discover(self, job: Job, resume: ResumeProfile) -> list:
        system = 'You are a expert networking strategist. You will receive a job posting and a candidate profile. Generate relevant contacts across four categories: Recruiter, Hiring Manager, Veteran, Peer. Return ONLY a valid JSON array like: [{"name": "", "company": "", "title": "", "category": "", "relevance_score": 0, "linkedin_url": "", "email": "", "branch": "", "notes": ""}]'

        job_info = {
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "key_skills": job.key_skills,
        }
        contact_info = {
            "name": resume.name,
            "skills": resume.skills,
            "experience_years": resume.experience_years,
            "location": resume.location,
            "branch": resume.military_service,
        }
        user = (
            f"Job:\n{json.dumps(job_info)}\n\nCandidate:\n{json.dumps(contact_info)} "
        )

        has_military = resume.military_service is not None

        if has_military:
            category_order = {
                "Veteran": 0,
                "Hiring Manager": 1,
                "Recruiter": 2,
                "Peer": 3,
            }
        else:
            category_order = {
                "Hiring Manager": 0,
                "Recruiter": 1,
                "Peer": 2,
                "Veteran": 3,
            }

        contacts = self.claude.ask_json(system, user)

        relevant_contacts = [
            c
            for c in contacts
            if c["relevance_score"] >= self.config.min_relevance_score
        ]
        sorted_contacts = sorted(
            relevant_contacts,
            key=lambda c: (category_order.get(c["category"], 4), -c["relevance_score"]),
        )

        results = []

        for i, contact in enumerate(sorted_contacts):
            results.append(
                Contact(
                    name=contact.get("name", ""),
                    company=contact.get("company", ""),
                    title=contact.get("title", ""),
                    category=contact.get("category", ""),
                    relevance_score=contact.get("relevance_score", 0),
                    linkedin_url=contact.get("linkedin_url", ""),
                    email=contact.get("email", ""),
                    branch=contact.get("branch", ""),
                    connection_message=contact.get("connection_message", ""),
                    priority=i + 1,
                    notes=contact.get("notes", ""),
                ),
            )

        return results
```

---

## 5. All passing tests

### test_config.py (4 tests — all green)
```python
import os
import pytest

def test_config_loads_from_env_vars(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")
    monkeypatch.setenv("NOTION_TOKEN", "ntn_test456")
    from pipeline import Config
    config = Config()
    assert config.anthropic_api_key == "sk-ant-test123"
    assert config.notion_token == "ntn_test456"

def test_config_validate_catches_missing_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("NOTION_TOKEN", raising=False)
    from pipeline import Config
    config = Config()
    missing = config.validate()
    assert "ANTHROPIC_API_KEY" in missing
    assert "NOTION_TOKEN" in missing

def test_config_validate_passes_when_keys_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test123")
    monkeypatch.setenv("NOTION_TOKEN", "ntn_test456")
    from pipeline import Config
    config = Config()
    assert config.validate() == []

def test_config_has_sensible_defaults():
    from pipeline import Config
    config = Config()
    assert config.default_location == "Denver, CO"
    assert config.min_fit_score == 7
    assert config.max_jobs == 10
    assert config.model == "claude-sonnet-4-20250514"
```

### test_pipeline_state.py (4 tests — all green)
```python
import pytest

def test_pipeline_state_has_expected_values():
    from pipeline import PipelineState
    assert PipelineState.IDLE.value == "idle"
    assert PipelineState.PARSE_RESUME.value == "parse_resume"
    assert PipelineState.SEARCH_JOBS.value == "search_jobs"
    assert PipelineState.SCORE_JOBS.value == "score_jobs"
    assert PipelineState.DISCOVER_CONTACTS.value == "discover_contacts"
    assert PipelineState.GENERATE_MESSAGES.value == "generate_messages"
    assert PipelineState.WRITE_NOTION.value == "write_notion"
    assert PipelineState.COMPLETE.value == "complete"
    assert PipelineState.ERROR.value == "error"

def test_pipeline_context_initializes_with_defaults():
    from pipeline import PipelineContext, PipelineState
    ctx = PipelineContext()
    assert ctx.state == PipelineState.IDLE
    assert ctx.jobs == []
    assert ctx.contacts == []
    assert ctx.errors == []
    assert ctx.resume is None
    assert ctx.target_job is None
    assert ctx.job_url is None

def test_pipeline_context_tracks_state_transitions():
    from pipeline import PipelineContext, PipelineState
    ctx = PipelineContext()
    assert ctx.state == PipelineState.IDLE
    ctx.state = PipelineState.PARSE_RESUME
    assert ctx.state == PipelineState.PARSE_RESUME
    ctx.state = PipelineState.SEARCH_JOBS
    assert ctx.state == PipelineState.SEARCH_JOBS

def test_pipeline_context_collections_are_independent():
    from pipeline import PipelineContext
    ctx1 = PipelineContext()
    ctx2 = PipelineContext()
    ctx1.errors.append("something broke")
    assert len(ctx1.errors) == 1
    assert len(ctx2.errors) == 0
```

### test_claude_client.py (4 tests — all green)
```python
import json
import anthropic
import pytest

def test_claude_client_initializes_with_config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    from pipeline import ClaudeClient, Config
    config = Config()
    client = ClaudeClient(config)
    assert client.model == "claude-sonnet-4-20250514"

def test_claude_client_ask_returns_string(monkeypatch):
    from pipeline import ClaudeClient, Config
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
```

### test_resume_profile.py (3 tests — all green)
```python
import pytest
from dataclasses import asdict

def test_resume_profile_has_required_fields():
    from pipeline import ResumeProfile
    profile = ResumeProfile()
    assert profile.name == ""
    assert profile.title == ""
    assert profile.location == ""
    assert profile.skills == []
    assert profile.experience_years == 0
    assert profile.military_service is None

def test_resume_profile_stores_values():
    from pipeline import ResumeProfile
    profile = ResumeProfile(
        name="Ryan Ellis",
        title="Lead AI Engineer",
        location="Denver, CO",
        skills=["Python", "AWS", "LangGraph"],
        experience_years=20,
        military_service={"branch": "U.S. Navy", "years": 20},
    )
    assert profile.name == "Ryan Ellis"
    assert "Python" in profile.skills
    assert profile.military_service["branch"] == "U.S. Navy"

def test_resume_profile_lists_are_independent():
    from pipeline import ResumeProfile
    p1 = ResumeProfile()
    p2 = ResumeProfile()
    p1.skills.append("Python")
    assert len(p1.skills) == 1
    assert len(p2.skills) == 0
```

### test_resume_parser.py (3 tests — all green)
```python
import json
import pytest
import anthropic

def test_resume_parser_returns_profile(monkeypatch):
    from pipeline import Config, ClaudeClient, ResumeParser, ResumeProfile
    fake_response = {"name": "Ryan Ellis", "title": "Lead AI Engineer", "location": "Denver, CO", "skills": ["Python", "AWS", "LangGraph"], "experience_years": 20, "military_service": {"branch": "U.S. Navy", "years": 20}}
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
    from pipeline import Config, ClaudeClient, ResumeParser
    fake_response = {"name": "Jane Doe", "title": "Software Engineer", "location": "Austin, TX", "skills": ["Java", "Spring Boot"], "experience_years": 5, "military_service": None}
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
    from pipeline import Config, ClaudeClient, ResumeParser
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
```

### test_job_scorer.py (5 tests — all green)
```python
import json
import pytest
import anthropic

def test_job_dataclass_has_required_fields():
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
    from pipeline import Job
    job = Job(title="AI Ops Lead Engineer", company="Infosys", location="Denver, CO", url="https://linkedin.com/jobs/view/123", source="LinkedIn", fit_score=9, key_skills=["Python", "AWS", "Bedrock"])
    assert job.title == "AI Ops Lead Engineer"
    assert job.fit_score == 9
    assert "AWS" in job.key_skills

def test_job_scorer_filters_below_threshold(monkeypatch):
    from pipeline import Config, ClaudeClient, JobScorer, ResumeProfile
    scores_response = [{"index": 0, "fit_score": 9, "key_skills_matched": ["Python", "AWS"]}, {"index": 1, "fit_score": 5, "key_skills_matched": ["Java"]}, {"index": 2, "fit_score": 8, "key_skills_matched": ["Python"]}]
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
    raw_jobs = [{"title": "AI Engineer", "company": "Acme", "location": "Denver", "source": "Google", "key_skills": [], "description_summary": "AI role"}, {"title": "Java Dev", "company": "BigCorp", "location": "Denver", "source": "LinkedIn", "key_skills": [], "description_summary": "Java role"}, {"title": "ML Engineer", "company": "StartupX", "location": "Denver", "source": "Google", "key_skills": [], "description_summary": "ML role"}]
    resume = ResumeProfile(name="Ryan", skills=["Python", "AWS"], experience_years=20)
    results = scorer.score(raw_jobs, resume)
    assert len(results) == 2
    assert all(j.fit_score >= 7 for j in results)

def test_job_scorer_sorts_by_score_descending(monkeypatch):
    from pipeline import Config, ClaudeClient, JobScorer, ResumeProfile
    scores_response = [{"index": 0, "fit_score": 7, "key_skills_matched": ["Python"]}, {"index": 1, "fit_score": 9, "key_skills_matched": ["Python", "AWS"]}, {"index": 2, "fit_score": 8, "key_skills_matched": ["AWS"]}]
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
    raw_jobs = [{"title": "Role A", "company": "Co A", "location": "Denver", "source": "Google", "key_skills": [], "description_summary": ""}, {"title": "Role B", "company": "Co B", "location": "Denver", "source": "LinkedIn", "key_skills": [], "description_summary": ""}, {"title": "Role C", "company": "Co C", "location": "Denver", "source": "Google", "key_skills": [], "description_summary": ""}]
    resume = ResumeProfile(name="Ryan", skills=["Python"], experience_years=20)
    results = scorer.score(raw_jobs, resume)
    assert results[0].fit_score == 9
    assert results[1].fit_score == 8
    assert results[2].fit_score == 7

def test_job_scorer_respects_max_jobs(monkeypatch):
    from pipeline import Config, ClaudeClient, JobScorer, ResumeProfile
    scores_response = [{"index": i, "fit_score": 8, "key_skills_matched": ["Python"]} for i in range(10)]
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
    raw_jobs = [{"title": f"Role {i}", "company": f"Co {i}", "location": "Denver", "source": "Google", "key_skills": [], "description_summary": ""} for i in range(10)]
    resume = ResumeProfile(name="Ryan", skills=["Python"], experience_years=20)
    results = scorer.score(raw_jobs, resume)
    assert len(results) == 3
```

### test_contact_discoverer.py (7 tests — all green)
```python
import json
import pytest
import anthropic

def test_contact_dataclass_has_required_fields():
    from pipeline import Contact
    contact = Contact()
    assert contact.name == ""
    assert contact.company == ""
    assert contact.title == ""
    assert contact.category == ""
    assert contact.relevance_score == 0
    assert contact.linkedin_url == ""
    assert contact.email == ""
    assert contact.branch == ""
    assert contact.connection_message == ""
    assert contact.priority == 0
    assert contact.notes == ""

def test_contact_stores_veteran_data():
    from pipeline import Contact
    contact = Contact(name="Gordon Feliciano", company="Infosys", title="Technology Lead", category="Veteran", relevance_score=9, branch="U.S. Navy", notes="Denver, CO; 8 yrs Navy; Gulf War vet")
    assert contact.category == "Veteran"
    assert contact.branch == "U.S. Navy"
    assert "Gulf War" in contact.notes

def test_contact_discoverer_returns_four_categories(monkeypatch):
    from pipeline import Config, ClaudeClient, ContactDiscoverer, Job, ResumeProfile
    fake_contacts = [
        {"name": "Alice", "title": "TA Manager", "category": "Recruiter", "relevance_score": 8, "linkedin_url": "", "email": "", "branch": "", "notes": "Internal recruiter", "profile_details": {}},
        {"name": "Bob", "title": "Engineering Director", "category": "Hiring Manager", "relevance_score": 9, "linkedin_url": "", "email": "", "branch": "", "notes": "Likely reports to", "profile_details": {}},
        {"name": "Charlie", "title": "Tech Lead", "category": "Veteran", "relevance_score": 8, "linkedin_url": "", "email": "", "branch": "U.S. Navy", "notes": "Navy vet", "profile_details": {}},
        {"name": "Diana", "title": "Cloud Architect", "category": "Peer", "relevance_score": 7, "linkedin_url": "", "email": "", "branch": "", "notes": "Same function", "profile_details": {}},
    ]
    class FakeContent:
        text = json.dumps(fake_contacts)
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
    config.min_relevance_score = 7
    claude = ClaudeClient(config)
    discoverer = ContactDiscoverer(claude, config)
    job = Job(title="AI Ops Lead", company="Infosys", location="Denver, CO", key_skills=["Python", "AWS"])
    resume = ResumeProfile(name="Ryan", skills=["Python"], military_service={"branch": "U.S. Navy", "years": 20})
    contacts = discoverer.discover(job, resume)
    categories = {c.category for c in contacts}
    assert "Recruiter" in categories
    assert "Hiring Manager" in categories
    assert "Veteran" in categories
    assert "Peer" in categories

def test_contact_discoverer_filters_below_threshold(monkeypatch):
    from pipeline import Config, ClaudeClient, ContactDiscoverer, Job, ResumeProfile
    fake_contacts = [
        {"name": "High", "title": "Director", "category": "Hiring Manager", "relevance_score": 9, "linkedin_url": "", "email": "", "branch": "", "notes": "", "profile_details": {}},
        {"name": "Low", "title": "Intern", "category": "Peer", "relevance_score": 4, "linkedin_url": "", "email": "", "branch": "", "notes": "", "profile_details": {}},
        {"name": "Medium", "title": "Recruiter", "category": "Recruiter", "relevance_score": 7, "linkedin_url": "", "email": "", "branch": "", "notes": "", "profile_details": {}},
    ]
    class FakeContent:
        text = json.dumps(fake_contacts)
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
    config.min_relevance_score = 7
    claude = ClaudeClient(config)
    discoverer = ContactDiscoverer(claude, config)
    job = Job(title="AI Ops Lead", company="Infosys", location="Denver, CO")
    resume = ResumeProfile(name="Ryan", skills=["Python"])
    contacts = discoverer.discover(job, resume)
    assert len(contacts) == 2
    assert all(c.relevance_score >= 7 for c in contacts)

def test_contact_discoverer_boosts_veterans(monkeypatch):
    from pipeline import Config, ClaudeClient, ContactDiscoverer, Job, ResumeProfile
    fake_contacts = [
        {"name": "Recruiter", "title": "TA Lead", "category": "Recruiter", "relevance_score": 9, "linkedin_url": "", "email": "", "branch": "", "notes": "", "profile_details": {}},
        {"name": "HM", "title": "Director", "category": "Hiring Manager", "relevance_score": 9, "linkedin_url": "", "email": "", "branch": "", "notes": "", "profile_details": {}},
        {"name": "Vet", "title": "Tech Lead", "category": "Veteran", "relevance_score": 8, "linkedin_url": "", "email": "", "branch": "U.S. Navy", "notes": "", "profile_details": {}},
        {"name": "Peer", "title": "Engineer", "category": "Peer", "relevance_score": 8, "linkedin_url": "", "email": "", "branch": "", "notes": "", "profile_details": {}},
    ]
    class FakeContent:
        text = json.dumps(fake_contacts)
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
    config.min_relevance_score = 7
    claude = ClaudeClient(config)
    discoverer = ContactDiscoverer(claude, config)
    job = Job(title="AI Ops Lead", company="Infosys", location="Denver, CO")
    resume = ResumeProfile(name="Ryan", skills=["Python"], military_service={"branch": "U.S. Navy", "years": 20})
    contacts = discoverer.discover(job, resume)
    assert contacts[0].category == "Veteran"
    assert contacts[0].priority == 1

def test_contact_discoverer_no_veteran_boost_without_military(monkeypatch):
    from pipeline import Config, ClaudeClient, ContactDiscoverer, Job, ResumeProfile
    fake_contacts = [
        {"name": "Recruiter", "title": "TA Lead", "category": "Recruiter", "relevance_score": 9, "linkedin_url": "", "email": "", "branch": "", "notes": "", "profile_details": {}},
        {"name": "HM", "title": "Director", "category": "Hiring Manager", "relevance_score": 9, "linkedin_url": "", "email": "", "branch": "", "notes": "", "profile_details": {}},
        {"name": "Vet", "title": "Tech Lead", "category": "Veteran", "relevance_score": 8, "linkedin_url": "", "email": "", "branch": "U.S. Army", "notes": "", "profile_details": {}},
    ]
    class FakeContent:
        text = json.dumps(fake_contacts)
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
    config.min_relevance_score = 7
    claude = ClaudeClient(config)
    discoverer = ContactDiscoverer(claude, config)
    job = Job(title="AI Ops Lead", company="Infosys", location="Denver, CO")
    resume = ResumeProfile(name="Jane", skills=["Python"], military_service=None)
    contacts = discoverer.discover(job, resume)
    assert contacts[0].category != "Veteran"

def test_contact_discoverer_assigns_sequential_priorities(monkeypatch):
    from pipeline import Config, ClaudeClient, ContactDiscoverer, Job, ResumeProfile
    fake_contacts = [
        {"name": "A", "title": "Recruiter", "category": "Recruiter", "relevance_score": 8, "linkedin_url": "", "email": "", "branch": "", "notes": "", "profile_details": {}},
        {"name": "B", "title": "Director", "category": "Hiring Manager", "relevance_score": 9, "linkedin_url": "", "email": "", "branch": "", "notes": "", "profile_details": {}},
        {"name": "C", "title": "Engineer", "category": "Peer", "relevance_score": 7, "linkedin_url": "", "email": "", "branch": "", "notes": "", "profile_details": {}},
    ]
    class FakeContent:
        text = json.dumps(fake_contacts)
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
    config.min_relevance_score = 7
    claude = ClaudeClient(config)
    discoverer = ContactDiscoverer(claude, config)
    job = Job(title="AI Ops Lead", company="Infosys", location="Denver, CO")
    resume = ResumeProfile(name="Ryan", skills=["Python"])
    contacts = discoverer.discover(job, resume)
    priorities = [c.priority for c in contacts]
    assert priorities == [1, 2, 3]
```

---

## 6. Struggle tracker (Modules 1-6)

### Concepts student is solid on:
- Dataclass structure, `field(default_factory=list)`, Enum usage
- Dependency injection pattern (ClaudeClient + Config parameters)
- Mock setup ordering (define fakes → patch → create client → test) — zero errors since Module 3
- Dict unpacking with `**result` into dataclass constructors
- Writing system prompts for structured LLM output (done 3 times: parser, scorer, discoverer)
- List comprehension filtering
- `json.loads()` and `json.dumps()` as bridge between Python and LLM prompts
- Config as the home for business rules/thresholds
- Composition over inheritance (questioned "couldn't you just inherit?" and understood the answer)
- `enumerate()` for positional index assignment
- Generation prompts vs parsing prompts — different LLM task types

### Concepts student struggled with:
- **Class-level vs instance-level access** — wrote `Config.model` instead of `config.model` (happened 3 times across early modules, improving — did not recur in Module 6)
- **os.getenv timing** — defaults evaluated at class definition, not instance creation. Required `__post_init__` pattern
- **validate() method design** — needed multiple rounds of hints to map display names to attribute lookups using a dict
- **split() vs strip()** — confused the two string methods
- **Import style for mockability** — `from X import Y` captures at import time, `import X` looks up at runtime
- **Abstraction layers** — wrote raw API message format (`{"role": "user", "content": }`) inside JobScorer instead of passing plain strings to `ask_json()`
- **Where data comes from** — pulled `fit_score` from raw job dict instead of from Claude's scoring response
- **sorted() key parameter** — passed a value (`self.config.min_fit_score`) instead of a callable (`lambda s: s["fit_score"]`)
- **Building LLM user prompts** — didn't realize user messages are just formatted strings you build with f-strings + json.dumps()
- **Variable shadowing** — named a variable `filter`, shadowing the Python built-in
- **Return type annotations** — put `-> list` on `__init__` and `-> None` on `discover()`
- **Using wrong config field** — initially filtered contacts by `min_fit_score` instead of `min_relevance_score`

### Patterns to watch for:
- When stuck, ask "which layer am I at?" and "which object owns this data?"
- Class-vs-instance issue has not recurred recently — pattern may be resolved
- Getting stronger at data source attribution but still needs attention
- Strongest when working with dataclass definitions and the DI pattern

### Growth notes:
- Mock ordering: zero errors in Modules 4, 5, and 6
- List comprehension filtering was correct on first attempt in Modules 5 and 6
- Dependency injection pattern now applied three times without hesitation
- Correctly identified missing `min_relevance_score` in Config before being told
- Conditional sorting with tuple keys was new but landed with guided help
- Understood generation vs parsing distinction after initial confusion
- Composition over inheritance: correctly questioned "couldn't you just inherit?" in Module 5

---

## 7. Where to continue: Module 7 — MessageGenerator

Module 6 (ContactDiscoverer) is COMPLETE — all 7 tests green. The student needs Module 7 next.

### Module 7: MessageGenerator
Generates personalized <300 char LinkedIn connection request messages. Different rules per contact category.

**Tests to provide (test_message_generator.py):**
1. `test_messages_under_300_chars` — Every generated message is under 300 characters
2. `test_veteran_messages_no_job_mention` — Veteran messages don't contain job title or "hiring" or "role"
3. `test_veteran_messages_no_reslink` — Veteran messages don't contain the Reslink URL
4. `test_hiring_manager_messages_include_reslink` — HM messages contain Reslink URL
5. `test_peer_messages_no_job_mention` — Peer messages don't mention the job or company hiring

**Message rules:**
- **All messages:** Under 300 chars, unique per person, references 1 specific profile detail
- **Veterans:** NO job mention, NO Reslink URL. Shared military bond only.
- **Hiring Managers/Recruiters:** Reference role exists (don't say "I applied"), 1-2 JD skills, include Reslink
- **Peers:** NO job mention, NO Reslink. Specific detail only (alma mater, tech stack, location)

**Key implementation details:**
- Takes `ClaudeClient` and `Config` (same DI pattern)
- `generate(contacts, job, resume)` returns the same contact list with `connection_message` populated
- System prompt must encode all per-category rules
- The Reslink URL comes from `self.config.reslink_url`

**Concepts to teach:**
- Prompt engineering with strict output constraints (char limits, per-category rules)
- Validation testing — testing properties of output (length, absence of keywords) rather than exact content
- Config field for the Reslink URL (need to add `reslink_url` to Config)

---

## 8. Remaining modules after Module 7

| Module | Class | Key concepts |
|--------|-------|-------------|
| 8 | NotionWriter | Notion API property schemas, graceful degradation when no token |
| 9 | Pipeline orchestrator | State machine transitions, composition over inheritance, CLI with sys.argv |

### Notion schemas for Module 8:
- Applications DB ID: `5135bb9f-2ad7-4400-b42a-b07d35a37d43`
- Contacts DB ID: `7e736104-656e-470c-b301-7bcd3c8dee82`
- Properties detailed in Section 2 context (already created in Notion workspace)

---

## 9. How to use this document

Paste this entire document into a new conversation, then say:

> "I'm continuing a TDD build session. The transfer document above has all context. Module 6 is complete. Help me implement Module 7 (MessageGenerator) using the same Socratic TDD approach described in the methodology section."

The model should:
1. Acknowledge the context and where you left off
2. Give failing tests for Module 7 (MessageGenerator)
3. Guide you through implementation with questions, not answers
4. Generate a quiz after all tests pass
5. Update the struggle tracker
