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


class MessageGenerator:
    def __init__(self, claude: ClaudeClient, config: Config) -> None:
        self.claude = claude
        self.config = config

    def generate(self, contacts: list, job: Job, resume: ResumeProfile):
        system = f'You are a networking message writer and you are given a list of contacts, create a connection message that is unique to a person, if nothing unique can be found then do not return a message. Return ONLY a valid JSON array like: [{{"index": 0, "message": ""}}]. For veterans: focus on shared military service. Do NOT mention the job title. Do Not include any URL. For hiring managers and recruiters: mention the role exists and highlight 1-2 skills, include this specific Reslink URL {self.config.reslink_url} add in the ResLink to the output. For Peers: focus on shared interests or background. Do NOT mention the job title or hiring. All messages must be under 300 characters.'

        contacts_data = [
            {
                "name": c.name,
                "company": c.company,
                "title": c.title,
                "category": c.category,
                "branch": c.branch,
                "notes": c.notes,
            }
            for i, c in enumerate(contacts)
        ]
        job_info = {
            "title": job.title,
            "company": job.company,
            "key_skills": job.key_skills,
        }
        resume_info = {
            "name": resume.name,
            "skills": resume.skills,
            "military_service": resume.military_service,
        }
        user = f"Contacts:\n{json.dumps(contacts_data)}\n\nJob:\n{json.dumps(job_info)}\n\nResume:\n{json.dumps(resume_info)}"

        results = self.claude.ask_json(system, user)

        for idx, contact in enumerate(results):
            contacts[idx].connection_message = contact["message"]

        return contacts


class Pipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.claude = ClaudeClient(config)
        self.resume_parser = ResumeParser(self.claude)
        self.job_scorer = JobScorer(self.claude, self.config)
        self.contact_discoverer = ContactDiscoverer(self.claude, self.config)
        self.message_generator = MessageGenerator(self.claude, self.config)
        self.notion_writer = NotionWriter(self.config)
        self.ctx = PipelineContext(state=PipelineState.IDLE)

    def _transition(self, new_state) -> None:
        self.ctx.state = new_state

    def _run_contacts_pipeline(self) -> None:
        try:
            self._transition(PipelineState.DISCOVER_CONTACTS)
            contacts = self.contact_discoverer.discover(
                self.ctx.target_job, self.ctx.resume
            )
            self.ctx.contacts.extend(contacts)

            self._transition(PipelineState.GENERATE_MESSAGES)
            self.ctx.contacts = self.message_generator.generate(
                self.ctx.contacts, self.ctx.target_job, self.ctx.resume
            )

            self._transition(PipelineState.WRITE_NOTION)
            self.notion_writer.write_job(self.ctx.target_job)
            for _, contact in enumerate(self.ctx.contacts):
                self.notion_writer.write_contact(contact, self.ctx.target_job.title)

            self._transition(PipelineState.COMPLETE)
        except Exception as e:
            self._transition(PipelineState.ERROR)
            self.ctx.errors.append(str(e))
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

### test_message_generator.py (5 tests — all green)
```python
import json
import pytest
import anthropic

def test_messages_under_300_chars(monkeypatch):
    from pipeline import Config, ClaudeClient, MessageGenerator, Contact, Job, ResumeProfile
    fake_messages = [{"index": 0, "message": "Hi Gordon — fellow Navy veteran here. 20 years USN out of Virginia Beach."}, {"index": 1, "message": "Hi Vaibhav — noticed your AWS work at Infosys CO. Quick intro: https://reslink.io/ryan"}, {"index": 2, "message": "Hi Sukhdeep — fellow Colorado tech person. Your AWS and microservices work mirrors my stack."}]
    class FakeContent:
        text = json.dumps(fake_messages)
    class FakeResponse:
        content = [FakeContent()]
    class FakeMessages:
        def create(self, **kwargs): return FakeResponse()
    class FakeAnthropic:
        def __init__(self, **kwargs): self.messages = FakeMessages()
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    config = Config()
    config.reslink_url = "https://reslink.io/ryan"
    claude = ClaudeClient(config)
    generator = MessageGenerator(claude, config)
    contacts = [Contact(name="Gordon", category="Veteran", title="Tech Lead", company="Infosys", branch="U.S. Navy"), Contact(name="Vaibhav", category="Hiring Manager", title="Cloud Architect", company="Infosys"), Contact(name="Sukhdeep", category="Peer", title="Dev Manager", company="Infosys")]
    job = Job(title="AI Ops Lead", company="Infosys", location="Denver, CO", key_skills=["Python", "AWS", "Bedrock"])
    resume = ResumeProfile(name="Ryan", skills=["Python"], military_service={"branch": "U.S. Navy", "years": 20})
    result = generator.generate(contacts, job, resume)
    for contact in result:
        assert len(contact.connection_message) <= 300

def test_veteran_messages_no_job_mention(monkeypatch):
    from pipeline import Config, ClaudeClient, MessageGenerator, Contact, Job, ResumeProfile
    fake_messages = [{"index": 0, "message": "Hi Gordon — fellow Navy veteran here. 20 years USN out of Virginia Beach. Great to see another vet at Infosys Denver."}]
    class FakeContent:
        text = json.dumps(fake_messages)
    class FakeResponse:
        content = [FakeContent()]
    class FakeMessages:
        def create(self, **kwargs): return FakeResponse()
    class FakeAnthropic:
        def __init__(self, **kwargs): self.messages = FakeMessages()
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    config = Config()
    config.reslink_url = "https://reslink.io/ryan"
    claude = ClaudeClient(config)
    generator = MessageGenerator(claude, config)
    contacts = [Contact(name="Gordon", category="Veteran", title="Tech Lead", company="Infosys", branch="U.S. Navy")]
    job = Job(title="AI Ops Lead", company="Infosys", key_skills=["Python"])
    resume = ResumeProfile(name="Ryan", military_service={"branch": "U.S. Navy", "years": 20})
    result = generator.generate(contacts, job, resume)
    msg = result[0].connection_message.lower()
    assert "ai ops lead" not in msg
    assert "hiring" not in msg
    assert "role" not in msg

def test_veteran_messages_no_reslink(monkeypatch):
    from pipeline import Config, ClaudeClient, MessageGenerator, Contact, Job, ResumeProfile
    fake_messages = [{"index": 0, "message": "Hi Gordon — fellow Navy veteran here. Served 20 years. Good to connect with another vet in Colorado tech."}]
    class FakeContent:
        text = json.dumps(fake_messages)
    class FakeResponse:
        content = [FakeContent()]
    class FakeMessages:
        def create(self, **kwargs): return FakeResponse()
    class FakeAnthropic:
        def __init__(self, **kwargs): self.messages = FakeMessages()
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    config = Config()
    config.reslink_url = "https://reslink.io/ryan"
    claude = ClaudeClient(config)
    generator = MessageGenerator(claude, config)
    contacts = [Contact(name="Gordon", category="Veteran", title="Tech Lead", company="Infosys", branch="U.S. Navy")]
    job = Job(title="AI Ops Lead", company="Infosys", key_skills=["Python"])
    resume = ResumeProfile(name="Ryan", military_service={"branch": "U.S. Navy", "years": 20})
    result = generator.generate(contacts, job, resume)
    assert config.reslink_url not in result[0].connection_message

def test_hiring_manager_messages_include_reslink(monkeypatch):
    from pipeline import Config, ClaudeClient, MessageGenerator, Contact, Job, ResumeProfile
    reslink = "https://reslink.io/ryan"
    fake_messages = [{"index": 0, "message": f"Hi Vaibhav — your AWS work at Infosys CO caught my eye. Quick intro: {reslink}"}]
    class FakeContent:
        text = json.dumps(fake_messages)
    class FakeResponse:
        content = [FakeContent()]
    class FakeMessages:
        def create(self, **kwargs): return FakeResponse()
    class FakeAnthropic:
        def __init__(self, **kwargs): self.messages = FakeMessages()
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    config = Config()
    config.reslink_url = reslink
    claude = ClaudeClient(config)
    generator = MessageGenerator(claude, config)
    contacts = [Contact(name="Vaibhav", category="Hiring Manager", title="Cloud Architect", company="Infosys")]
    job = Job(title="AI Ops Lead", company="Infosys", key_skills=["Python", "AWS"])
    resume = ResumeProfile(name="Ryan", skills=["Python"])
    result = generator.generate(contacts, job, resume)
    assert reslink in result[0].connection_message

def test_peer_messages_no_job_mention(monkeypatch):
    from pipeline import Config, ClaudeClient, MessageGenerator, Contact, Job, ResumeProfile
    fake_messages = [{"index": 0, "message": "Hi Sukhdeep — fellow Colorado tech person. Your AWS and microservices work mirrors my stack."}]
    class FakeContent:
        text = json.dumps(fake_messages)
    class FakeResponse:
        content = [FakeContent()]
    class FakeMessages:
        def create(self, **kwargs): return FakeResponse()
    class FakeAnthropic:
        def __init__(self, **kwargs): self.messages = FakeMessages()
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    config = Config()
    config.reslink_url = "https://reslink.io/ryan"
    claude = ClaudeClient(config)
    generator = MessageGenerator(claude, config)
    contacts = [Contact(name="Sukhdeep", category="Peer", title="Dev Manager", company="Infosys", notes="Boulder CO AWS")]
    job = Job(title="AI Ops Lead", company="Infosys", key_skills=["Python"])
    resume = ResumeProfile(name="Ryan", skills=["Python"])
    result = generator.generate(contacts, job, resume)
    msg = result[0].connection_message.lower()
    assert "ai ops lead" not in msg
    assert "hiring" not in msg
```

### test_notion_writer.py (4 tests — all green)
```python
import pytest

class FakePages:
    def __init__(self):
        self.created = []
        self.updated = []
    def create(self, **kwargs):
        self.created.append(kwargs)
        return {"id": "fake-page-id-123"}
    def update(self, **kwargs):
        self.updated.append(kwargs)
        return {"id": kwargs.get("page_id", "fake")}

class FakeNotionClient:
    def __init__(self, **kwargs):
        self.pages = FakePages()

def test_notion_writer_creates_job_page(monkeypatch):
    import notion_client
    from pipeline import Config, NotionWriter, Job
    monkeypatch.setattr(notion_client, "Client", FakeNotionClient)
    config = Config()
    config.notion_token = "ntn_fake"
    config.notion_jobs_db = "fake-jobs-db-id"
    writer = NotionWriter(config)
    job = Job(title="AI Ops Lead Engineer", company="Infosys", location="Denver, CO", url="https://linkedin.com/jobs/view/123", source="LinkedIn", fit_score=9, key_skills=["Python", "AWS"])
    page_id = writer.write_job(job)
    assert page_id == "fake-page-id-123"
    props = writer.client.pages.created[0]["properties"]
    assert props["Role"]["title"][0]["text"]["content"] == "AI Ops Lead Engineer"
    assert props["Company"]["rich_text"][0]["text"]["content"] == "Infosys"
    assert props["Fit Score"]["number"] == 9
    assert props["Job URL"]["url"] == "https://linkedin.com/jobs/view/123"
    assert props["Source"]["select"]["name"] == "LinkedIn"

def test_notion_writer_creates_contact_page(monkeypatch):
    import notion_client
    from pipeline import Config, NotionWriter, Contact
    monkeypatch.setattr(notion_client, "Client", FakeNotionClient)
    config = Config()
    config.notion_token = "ntn_fake"
    config.notion_contacts_db = "fake-contacts-db-id"
    writer = NotionWriter(config)
    contact = Contact(name="Gordon Feliciano", company="Infosys", title="Technology Lead", category="Veteran", relevance_score=9, linkedin_url="https://linkedin.com/in/gordonfeliciano", email="gordon.feliciano@infosys.com", branch="U.S. Navy", connection_message="Hi Gordon — fellow Navy veteran here.", priority=1, notes="Denver, CO; Gulf War vet")
    page_id = writer.write_contact(contact, "AI Ops Lead at Infosys")
    assert page_id == "fake-page-id-123"
    props = writer.client.pages.created[0]["properties"]
    assert props["Name"]["title"][0]["text"]["content"] == "Gordon Feliciano"
    assert props["Category"]["select"]["name"] == "Veteran"
    assert props["Relevance Score"]["number"] == 9
    assert props["Email"]["email"] == "gordon.feliciano@infosys.com"
    assert props["LinkedIn URL"]["url"] == "https://linkedin.com/in/gordonfeliciano"
    assert props["Priority"]["number"] == 1
    assert props["Message Sent"]["checkbox"] == False

def test_notion_writer_updates_status(monkeypatch):
    import notion_client
    from pipeline import Config, NotionWriter
    monkeypatch.setattr(notion_client, "Client", FakeNotionClient)
    config = Config()
    config.notion_token = "ntn_fake"
    writer = NotionWriter(config)
    writer.update_job_status("fake-page-id-123", "Messages Drafted")
    assert len(writer.client.pages.updated) == 1
    update_call = writer.client.pages.updated[0]
    assert update_call["page_id"] == "fake-page-id-123"
    assert update_call["properties"]["Status"]["select"]["name"] == "Messages Drafted"

def test_notion_writer_skips_when_no_token():
    from pipeline import Config, NotionWriter, Job, Contact
    config = Config()
    config.notion_token = ""
    writer = NotionWriter(config)
    job = Job(title="Test Role", company="Test Co")
    contact = Contact(name="Test Person", category="Peer")
    assert writer.write_job(job) is None
    assert writer.write_contact(contact, "Test Role at Test Co") is None
    assert writer.update_job_status("fake-id", "Queued") is None
```

### test_resume_loader.py (4 tests — all green)
```python
import pytest
from pathlib import Path

def test_load_resume_reads_txt_file(tmp_path):
    from pipeline import load_resume
    resume_file = tmp_path / "resume.txt"
    resume_file.write_text("Ryan Ellis\nLead AI Engineer\nDenver, CO")
    result = load_resume(resume_file)
    assert "Ryan Ellis" in result
    assert "Lead AI Engineer" in result

def test_load_resume_reads_pdf_file(tmp_path, monkeypatch):
    from pipeline import load_resume
    pdf_file = tmp_path / "resume.pdf"
    pdf_file.write_bytes(b"fake pdf content")
    class FakePage:
        def get_text(self): return "Ryan Ellis\nLead AI Engineer"
    class FakeDoc:
        def __init__(self, path): self.pages = [FakePage(), FakePage()]
        def __iter__(self): return iter(self.pages)
        def close(self): pass
    import pipeline
    monkeypatch.setattr(pipeline, "fitz", type("module", (), {"open": FakeDoc}))
    result = load_resume(pdf_file)
    assert "Ryan Ellis" in result
    assert "Lead AI Engineer" in result

def test_load_resume_raises_on_missing_file():
    from pipeline import load_resume
    with pytest.raises(FileNotFoundError):
        load_resume(Path("/nonexistent/resume.txt"))

def test_load_resume_raises_on_unsupported_format(tmp_path):
    from pipeline import load_resume
    docx_file = tmp_path / "resume.docx"
    docx_file.write_text("fake content")
    with pytest.raises(ValueError, match="Unsupported"):
        load_resume(docx_file)
```

### test_pipeline.py (4 tests — 2 passing, 2 failing / in progress)
```python
import json
import pytest
import anthropic

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

def test_pipeline_initializes_all_modules():
    from pipeline import Config, Pipeline
    config = Config()
    config.notion_token = ""
    pipeline = Pipeline(config)
    assert pipeline.config is config
    assert pipeline.claude is not None
    assert pipeline.resume_parser is not None
    assert pipeline.job_scorer is not None
    assert pipeline.contact_discoverer is not None
    assert pipeline.message_generator is not None
    assert pipeline.notion_writer is not None

def test_pipeline_starts_in_idle_state():
    from pipeline import Config, Pipeline, PipelineState
    config = Config()
    config.notion_token = ""
    pipeline = Pipeline(config)
    assert pipeline.ctx.state == PipelineState.IDLE

def test_pipeline_transitions_through_states(monkeypatch):
    from pipeline import Config, Pipeline, PipelineState, ResumeProfile, Job
    import notion_client
    transitions = []
    fake_anthropic = FakeAnthropic()
    fake_anthropic.messages.add_response(json.dumps([{"name": "Alice", "title": "Recruiter", "category": "Recruiter", "relevance_score": 8, "linkedin_url": "", "email": "", "branch": "", "notes": ""}]))
    fake_anthropic.messages.add_response(json.dumps([{"index": 0, "message": "Hi Alice — great to connect."}]))
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: fake_anthropic)
    class FakeNotionClient:
        def __init__(self, **kwargs):
            self.pages = type("P", (), {"create": lambda self, **kw: {"id": "fake-id"}, "update": lambda self, **kw: {"id": "fake-id"}})()
    monkeypatch.setattr(notion_client, "Client", FakeNotionClient)
    config = Config()
    config.notion_token = "ntn_fake"
    config.min_relevance_score = 7
    pipeline = Pipeline(config)
    original_transition = pipeline._transition
    def tracking_transition(new_state):
        transitions.append(new_state)
        original_transition(new_state)
    pipeline._transition = tracking_transition
    pipeline.ctx.resume = ResumeProfile(name="Ryan", skills=["Python"], military_service=None)
    pipeline.ctx.target_job = Job(title="AI Ops Lead", company="Infosys", location="Denver", key_skills=["Python"])
    pipeline._run_contacts_pipeline()
    assert PipelineState.DISCOVER_CONTACTS in transitions
    assert PipelineState.GENERATE_MESSAGES in transitions
    assert PipelineState.WRITE_NOTION in transitions
    assert PipelineState.COMPLETE in transitions

def test_pipeline_error_handling(monkeypatch):
    from pipeline import Config, Pipeline, PipelineState, ResumeProfile, Job
    fake_anthropic = FakeAnthropic()
    fake_anthropic.messages.add_response("NOT VALID JSON")
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: fake_anthropic)
    config = Config()
    config.notion_token = ""
    config.min_relevance_score = 7
    pipeline = Pipeline(config)
    pipeline.ctx.resume = ResumeProfile(name="Ryan", skills=["Python"])
    pipeline.ctx.target_job = Job(title="Test", company="TestCo", key_skills=["Python"])
    pipeline._run_contacts_pipeline()
    assert pipeline.ctx.state == PipelineState.ERROR
    assert len(pipeline.ctx.errors) > 0
```

---

## 6. Struggle tracker (FINAL — All 9 modules complete)

### Concepts student is solid on:
- Dataclass structure, `field(default_factory=list)`, Enum usage — automatic
- Dependency injection pattern — applied 5 times without hesitation (4 with Claude + 1 without)
- Mock setup ordering (define → patch → create → test) — zero errors since Module 3
- System prompt design for structured LLM output — done 4 times
- List comprehension filtering and dict building from objects
- `json.dumps()` as the bridge between Python objects and text
- Conditional sorting with tuple keys and category_order dicts
- Validation testing (properties, not exact content)
- Notion API property schemas
- Graceful degradation with early return pattern
- Mocking external SDKs (anthropic + notion_client) using import-module pattern
- f-string curly brace escaping

### Concepts student struggled with:
- **Class-level vs instance-level access** — 3 occurrences in early modules, fully resolved since Module 4
- **os.getenv timing** — `__post_init__` pattern
- **validate() method design** — mapping display names to attributes
- **split() vs strip()** — confused the two
- **Import style for mockability** — `from X import Y` vs `import X` (resolved by Module 8)
- **Abstraction layers** — wrote raw API message format in wrong layer (Module 5)
- **Pulling fields from wrong data source** — raw job vs score dict (Module 5), jobs DB vs contacts DB (Module 8)
- **sorted() key needs callable** — passed a value instead of lambda (Module 5)
- **Building LLM user prompts** — initially didn't realize they're just formatted strings (Module 5, resolved)
- **Variable shadowing** — named a variable `filter` (Module 5)
- **JSON serialization of dataclass objects** — needed to build dicts manually (Module 7, now resolved)
- **f-string escaping for JSON examples** — curly braces (Module 7, resolved quickly)
- **Return type annotations** — wrong types on `__init__` and `discover()` (Module 6)
- **State transitions vs creating new context objects** — passed `PipelineContext(state=...)` to `_transition()` instead of just `PipelineState.VALUE` (Module 9 in progress)

### Patterns to watch for:
- Tendency to create NEW objects when should be UPDATING existing ones (PipelineContext in Module 9, similar to connection_message pattern in Module 7)
- "Wrong data source" pattern has appeared in Modules 5 and 8 — ask "which object owns this field?"
- JSON serialization pattern fully locked in after Module 7
- Class-vs-instance issue fully resolved

### Growth notes:
- Modules 7 and 8 were the fastest — pattern recognition is accelerating
- DI pattern, mock ordering, and system prompt design are all automatic
- Student is correctly applying composition over inheritance without prompting
- Now comfortable questioning design decisions (asked about model-agnostic DI, inheritance vs composition)
- Proactively catches missing config fields before being told

---

## 7. Project status: POST-COMPLETION EXTENSIONS IN PROGRESS

Core 9-module build is complete. Extensions are being added for production readiness.

### Completed since original build (as of 2026-04-19):

| Item | Status | Notes |
|------|--------|-------|
| CLI entry point (`parse_arguments`, `main`) | ✅ Done | `sys.argv` parsing, runs full pipeline |
| `JobSearcher` with Apify LinkedIn scraper | ✅ Done | Requires `APIFY_API_TOKEN` env var |
| `load_resume` with pymupdf PDF parsing | ✅ Done | Handles `.txt` and `.pdf`; raises `FileNotFoundError` / `ValueError` for bad input |
| Fix `write_contact` to use `notion_contacts_db` | ✅ Done | Was using wrong DB |
| `json.dumps()` in `MessageGenerator` user message | ✅ Done | |

### Test counts (2026-04-19):
- **72 total tests**: all passing
- `test_resume_loader.py`: 4 tests
- `test_cli.py`: 9 tests (previously 2 failing — now fixed)
- `test_llm_protocol.py`: 3 tests (new)
- `test_integration.py`: 1 test (new)

### Still remaining:
- ~~Fix `test_cli.py`~~ ✅ Done — added `FakeAnthropic` / `FakeMessages` classes at module level
- ~~Add `LLMClient` Protocol~~ ✅ Done — `@runtime_checkable Protocol` in `pipeline.py`; `ClaudeClient` satisfies it; `test_llm_protocol.py` (3 tests)
- ~~End-to-end integration test~~ ✅ Done — `test_integration.py`: full pipeline with fakes (resume → parse → contacts → messages, Notion disabled); 1 test

**All 72 tests pass. No remaining backlog items.**

---

## 8. How to use this document

Core build is complete; production extensions are underway. To continue, paste this document into a new conversation and say:

> "I completed the TDD build described in this document. 66 of 68 tests are passing. The 2 failures are in test_cli.py — FakeAnthropic is not defined there. I want to [fix test_cli.py / add LLMClient Protocol / add integration tests / etc.]. Continue using the same Socratic TDD approach."
