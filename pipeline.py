import json
from dataclasses import dataclass, field
from enum import Enum
from os import getenv
from typing import Any

import anthropic
import notion_client


@dataclass
class Config:
    anthropic_api_key: str = ""
    notion_token: str = ""
    default_location: str = "Denver, CO"
    min_fit_score: int = 7
    min_relevance_score: int = 0
    max_jobs: int = 10
    model: str = "claude-sonnet-4-20250514"
    reslink_url: str = "[Reslink URL]"
    notion_jobs_db = ""
    notion_contacts_db = ""

    def __post_init__(self) -> None:
        self.anthropic_api_key: str = getenv("ANTHROPIC_API_KEY", "")
        self.notion_token: str = getenv("NOTION_TOKEN", "")

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


class ClaudeClient:
    def __init__(self, config: Config) -> None:
        # Store the model name
        # Create an anthropic.Anthropic instance using the api key
        self.model = config.model
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def ask(self, system: str, user: str) -> str:
        # Call self.client.messages.create(...)
        # Return the text from the first content block
        response = self.client.messages.create(
            max_tokens=1024,
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        return response.content[0].text

    def ask_json(self, system: str, user: str) -> dict:
        # Call self.ask() to get the raw text
        # Strip markdown fences if present
        # Parse with json.loads() and return
        raw = self.ask(system, user)

        text = raw.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        data = json.loads(text)
        return data


@dataclass
class ResumeProfile:
    name: str = ""
    title: str = ""
    location: str = ""
    skills: list = field(default_factory=list)
    experience_years: int = 0
    military_service: dict | None = None


class ResumeParser:
    def __init__(self, claude: ClaudeClient) -> None:
        self.claude = claude

    def parse(self, resume_text: str) -> ResumeProfile:
        # 1. Call self.claude.ask_json() with a system prompt and the resume text
        # 2. Map the returned dict onto a ResumeProfile dataclass
        # 3. Return the ResumeProfile
        system = "You are a resume parser. Extract structured data from the resume. Return ONLY valid JSON with these fields: name, title, location, skills (list), experience_years (number), military_service (dict with branch and years, or null)."

        result = self.claude.ask_json(system, resume_text)

        return ResumeProfile(**result)


@dataclass
class Job:
    title: str = ""
    company: str = ""
    location: str = ""
    url: str = ""
    source: str = ""
    fit_score: int = 0
    key_skills: list = field(default_factory=list)
    description: str = ""


class JobScorer:
    def __init__(self, claude: ClaudeClient, config: Config) -> None:
        self.claude = claude
        self.config = config

    def score(
        self,
        raw_jobs: list[dict[str, Any]],
        resume: ResumeProfile,
    ) -> list:
        # 1. Build system prompt (tell Claude to score and return JSON)
        # 2. Build user prompt (include resume + jobs)
        # 3. scores = self.claude.ask_json(system, user)
        # 4. Build Job objects for passing scores, filter + sort + cap
        system = 'You are a expert resume ranker.  Given a list of job postings and a candidate resume. Score each roles out of 10 based on fit (role, responsibilities, skills, experience). The scoring criteria is role alignment, skills overlap, seniority match, experience requirements, location match. Return ONLY a valid JSON array like: [{"index": 0, fit_score": 7, "key_skills_matched": ["Python"]}]'

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
                    key_skills=score.get("key_skills", []),
                    description=raw.get("description_summary", ""),
                ),
            )

        return results[: self.config.max_jobs]


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


class ContactDiscoverer:
    def __init__(self, claude: ClaudeClient, config: Config) -> None:
        self.claude = claude
        self.config = config

    def discover(self, job: Job, resume: ResumeProfile) -> list:
        system = 'You are a expert networking strategist. You will receive a job posting and a candidate profile.  Generate relevant contacts across four categories: Recuiter, Hiring Manager, Veteran, Peer. Return ONLY a valid JSON array like: [{"name": "", "company": "", "title": "", "category": "", "relevance_score": 0, "linkedin_url": "", "url": "", "email": "", "branch": "", "notes": ""}]'

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
                    relevance_score=contact.get("relevance_score", ""),
                    linkedin_url=contact.get("linked_url", ""),
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

    def generate(self, contacts: list[Contact], job: Job, resume: ResumeProfile):
        system = f'You are a networking message writer and you are given a list of contacts, create a connection message that is unique to a person, if nothing unique can be found then do not return a message.  Return ONLY a valid JSON array like: [{{"index": 0, "message": ""}}]. For veterans: focus on shared military service. Do NOT mention the job title. Do Not include any URL. For hiring managers and recuiters: mention the role exists and highlight 1-2 skills, include this specific Reslink URL {self.config.reslink_url} add in the ResLink to the output. For Peers: focus on shared interests or background. Do NOT mention the job title or hiring.  All messages must be under 300 characters.'

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


class NotionWriter:
    def __init__(self, config: Config):
        self.config = config
        if not config.notion_token:
            self.client = None
            self.enabled = False
        else:
            self.client = notion_client.Client(auth=config.notion_token)
            self.enabled = True

    def write_job(self, job: Job):
        if not self.enabled:
            return None
        response = self.client.pages.create(
            parent={"database_id": self.config.notion_jobs_db},
            properties={
                "Role": {"title": [{"text": {"content": job.title}}]},
                "Company": {"rich_text": [{"text": {"content": job.company}}]},
                "Fit Score": {"number": job.fit_score},
                "Job URL": {"url": job.url},
                "Source": {"select": {"name": job.source}},
            },
        )

        return response["id"]

    def write_contact(self, contact: Contact, title: str):
        if not self.enabled:
            return None
        response = self.client.pages.create(
            parent={"database_id": self.config.notion_contacts_db},
            properties={
                "Name": {"title": [{"text": {"content": contact.name}}]},
                "Category": {"select": {"name": contact.category}},
                "Relevance Score": {"number": contact.relevance_score},
                "Email": {"email": contact.email},
                "LinkedIn URL": {"url": contact.linkedin_url},
                "Priority": {"number": contact.priority},
                "Message Sent": {"checkbox": False},
            },
        )

        return response["id"]

    def update_job_status(self, id, status):
        if not self.enabled:
            return None
        response = self.client.pages.update(
            page_id=id,
            properties={"Status": {"select": {"name": status}}},
        )
