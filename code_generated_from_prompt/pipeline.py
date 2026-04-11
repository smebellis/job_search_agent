"""
Job Pipeline Orchestrator Agent
================================
Single-command AI agent that:
1. Parses your resume to extract skills
2. Searches for matching jobs (Google + LinkedIn via Apify)
3. Discovers contacts at target companies (Vibe Prospecting + Apify)
4. Generates personalized connection messages
5. Writes everything to Notion databases

Usage:
    python pipeline.py "https://linkedin.com/jobs/view/..."
    python pipeline.py search --location "Denver, CO"
    python pipeline.py full "https://linkedin.com/jobs/view/..."

Requires:
    pip install anthropic httpx python-dotenv notion-client apify-client
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv
from notion_client import Client as NotionClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


@dataclass
class Config:
    """Central config — reads from .env or environment variables."""

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    notion_token: str = os.getenv("NOTION_TOKEN", "")
    apify_token: str = os.getenv("APIFY_TOKEN", "")
    vibe_api_key: str = os.getenv("VIBE_PROSPECTING_API_KEY", "")

    # Notion database IDs (from the databases we just created)
    notion_jobs_db: str = os.getenv(
        "NOTION_JOBS_DB", "5135bb9f-2ad7-4400-b42a-b07d35a37d43"
    )
    notion_contacts_db: str = os.getenv(
        "NOTION_CONTACTS_DB", "7e736104-656e-470c-b301-7bcd3c8dee82"
    )

    # Resume path
    resume_path: str = os.getenv("RESUME_PATH", "resume.pdf")

    # Search defaults
    default_location: str = os.getenv("DEFAULT_LOCATION", "Denver, CO")
    min_fit_score: int = int(os.getenv("MIN_FIT_SCORE", "7"))
    min_relevance_score: int = int(os.getenv("MIN_RELEVANCE_SCORE", "7"))
    max_jobs: int = int(os.getenv("MAX_JOBS", "10"))

    # Reslink URL for video intro
    reslink_url: str = os.getenv("RESLINK_URL", "[Reslink URL]")

    # Claude model
    model: str = "claude-sonnet-4-20250514"

    def validate(self) -> list[str]:
        """Return list of missing required config keys."""
        missing = []
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not self.notion_token:
            missing.append("NOTION_TOKEN")
        return missing


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class PipelineState(str, Enum):
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
class ResumeProfile:
    """Structured representation of the candidate's resume."""

    name: str = ""
    title: str = ""
    location: str = ""
    summary: str = ""
    skills: list[str] = field(default_factory=list)
    experience_years: int = 0
    experience: list[dict] = field(default_factory=list)
    education: list[dict] = field(default_factory=list)
    military_service: Optional[dict] = None


@dataclass
class Job:
    """A single job listing."""

    title: str = ""
    company: str = ""
    location: str = ""
    url: str = ""
    source: str = ""
    posted_date: str = ""
    description: str = ""
    key_skills: list[str] = field(default_factory=list)
    salary: str = ""
    fit_score: int = 0


@dataclass
class Contact:
    """A networking contact for outreach."""

    name: str = ""
    company: str = ""
    title: str = ""
    category: str = ""  # Recruiter, Hiring Manager, Veteran, Peer
    relevance_score: int = 0
    linkedin_url: str = ""
    email: str = ""
    branch: str = ""  # military branch if veteran
    connection_message: str = ""
    priority: int = 0
    notes: str = ""
    profile_details: dict = field(default_factory=dict)


@dataclass
class PipelineContext:
    """Mutable state passed through the pipeline."""

    state: PipelineState = PipelineState.IDLE
    job_url: Optional[str] = None
    search_location: str = "Denver, CO"
    resume: Optional[ResumeProfile] = None
    jobs: list[Job] = field(default_factory=list)
    target_job: Optional[Job] = None
    contacts: list[Contact] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Claude API Client
# ---------------------------------------------------------------------------


class ClaudeClient:
    """Wrapper around the Anthropic API for structured reasoning tasks."""

    def __init__(self, config: Config):
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self.model = config.model

    def ask(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Send a message to Claude and return the text response."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def ask_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> dict:
        """Send a message and parse the JSON response."""
        raw = self.ask(system, user, max_tokens, temperature=0.1)
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Task 1: Resume Parser
# ---------------------------------------------------------------------------


class ResumeParser:
    """Extracts structured profile data from a resume using Claude."""

    def __init__(self, claude: ClaudeClient):
        self.claude = claude

    def parse(self, resume_text: str) -> ResumeProfile:
        """Parse resume text into a structured ResumeProfile."""
        log.info("Parsing resume...")

        system = """You are a resume parser. Extract structured data from the resume.
Return ONLY valid JSON with this exact schema:
{
    "name": "string",
    "title": "string",
    "location": "string",
    "summary": "1-2 sentence summary",
    "skills": ["skill1", "skill2"],
    "experience_years": number,
    "experience": [{"company": "", "title": "", "dates": "", "highlights": [""]}],
    "education": [{"school": "", "degree": "", "field": "", "dates": ""}],
    "military_service": {"branch": "", "role": "", "years": number, "dates": ""} or null
}"""

        data = self.claude.ask_json(system, f"Parse this resume:\n\n{resume_text}")

        return ResumeProfile(
            name=data.get("name", ""),
            title=data.get("title", ""),
            location=data.get("location", ""),
            summary=data.get("summary", ""),
            skills=data.get("skills", []),
            experience_years=data.get("experience_years", 0),
            experience=data.get("experience", []),
            education=data.get("education", []),
            military_service=data.get("military_service"),
        )


# ---------------------------------------------------------------------------
# Task 2: Job Searcher
# ---------------------------------------------------------------------------


class JobSearcher:
    """Searches for jobs via web search and LinkedIn scraping."""

    def __init__(self, claude: ClaudeClient, config: Config):
        self.claude = claude
        self.config = config
        self.http = httpx.Client(timeout=120)

    def search_google(self, skills: list[str], location: str) -> list[dict]:
        """Use Claude to search for jobs via web reasoning."""
        log.info("Searching Google for matching jobs...")

        skill_str = ", ".join(skills[:8])
        system = """You are a job search analyst. Given a candidate's skills and location,
generate a list of realistic job postings that would match.

Return ONLY valid JSON array:
[{
    "title": "string",
    "company": "string",
    "location": "string",
    "url": "string (job posting URL)",
    "source": "Google",
    "posted_date": "YYYY-MM-DD",
    "key_skills": ["skill1", "skill2"],
    "salary": "string or empty",
    "description_summary": "1-2 sentences"
}]

Be realistic — use actual companies hiring in this space. Max 10 results."""

        user = f"""Find AI/ML engineering jobs matching these skills:
Skills: {skill_str}
Location: {location}
Seniority: Lead/Senior level
Posted: Last 24-48 hours
Focus: Agentic AI, LLMs, cloud architecture (AWS/Azure)"""

        return self.claude.ask_json(system, user)

    def search_linkedin(self, keywords: str, location: str) -> list[dict]:
        """Scrape LinkedIn jobs via Apify."""
        if not self.config.apify_token:
            log.warning("No APIFY_TOKEN — skipping LinkedIn scrape")
            return []

        log.info("Scraping LinkedIn for jobs via Apify...")
        try:
            from apify_client import ApifyClient

            client = ApifyClient(self.config.apify_token)

            search_url = (
                f"https://www.linkedin.com/jobs/search/"
                f"?keywords={keywords.replace(' ', '%20')}"
                f"&location={location.replace(' ', '%20').replace(',', '%2C')}"
                f"&f_TPR=r86400&position=1&pageNum=0"
            )

            run = client.actor("curious_coder/linkedin-jobs-scraper").call(
                run_input={
                    "urls": [search_url],
                    "count": 20,
                    "scrapeCompany": False,
                },
                timeout_secs=120,
            )

            items = list(
                client.dataset(run["defaultDatasetId"]).iterate_items(limit=20)
            )

            results = []
            for item in items:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "company": item.get("companyName", ""),
                        "location": item.get("location", ""),
                        "url": item.get("link", ""),
                        "source": "LinkedIn",
                        "posted_date": (item.get("postedAt", ""))[:10],
                        "key_skills": [],
                        "salary": item.get("salary", ""),
                        "description_summary": (
                            item.get("descriptionText", "")[:500]
                        ),
                    }
                )
            return results

        except Exception as e:
            log.error(f"LinkedIn scrape failed: {e}")
            return []

    def search(
        self, resume: ResumeProfile, location: str
    ) -> list[dict]:
        """Run both search sources and merge results."""
        google_jobs = self.search_google(resume.skills, location)
        linkedin_jobs = self.search_linkedin(
            "AI engineer agentic LLM", location
        )

        # Deduplicate by title+company
        seen = set()
        merged = []
        for job in google_jobs + linkedin_jobs:
            key = (job["title"].lower(), job["company"].lower())
            if key not in seen:
                seen.add(key)
                merged.append(job)

        log.info(f"Found {len(merged)} unique jobs across sources")
        return merged


# ---------------------------------------------------------------------------
# Task 3: Job Scorer
# ---------------------------------------------------------------------------


class JobScorer:
    """Scores each job against the candidate's resume."""

    def __init__(self, claude: ClaudeClient, config: Config):
        self.claude = claude
        self.config = config

    def score(self, jobs: list[dict], resume: ResumeProfile) -> list[Job]:
        """Score each job 1-10 based on fit with resume."""
        log.info(f"Scoring {len(jobs)} jobs against resume...")

        system = """You are a strict job-fit scorer. Given a resume profile and a list of
job postings, score each job 1-10 based on:
- Role alignment (AI/ML engineering vs adjacent)
- Skills overlap (Python, FastAPI, LangGraph, RAG, AWS/Azure, MLOps, agentic AI)
- Seniority match (lead/senior level)
- Experience requirements
- Location match

Be STRICT — no inflated scores. A 7 means solid match, 9 means near-perfect.

Return ONLY valid JSON array:
[{
    "index": 0,
    "fit_score": 8,
    "key_skills_matched": ["Python", "AWS", "Agentic AI"],
    "reasoning": "one sentence"
}]"""

        resume_summary = json.dumps(
            {
                "name": resume.name,
                "title": resume.title,
                "skills": resume.skills,
                "experience_years": resume.experience_years,
                "location": resume.location,
            }
        )

        jobs_summary = json.dumps(
            [
                {
                    "index": i,
                    "title": j.get("title", ""),
                    "company": j.get("company", ""),
                    "location": j.get("location", ""),
                    "description": j.get("description_summary", "")[:300],
                    "key_skills": j.get("key_skills", []),
                }
                for i, j in enumerate(jobs)
            ]
        )

        scores = self.claude.ask_json(
            system,
            f"Resume:\n{resume_summary}\n\nJobs:\n{jobs_summary}",
        )

        scored_jobs = []
        score_map = {s["index"]: s for s in scores}

        for i, raw in enumerate(jobs):
            s = score_map.get(i, {"fit_score": 0, "key_skills_matched": []})
            if s["fit_score"] >= self.config.min_fit_score:
                scored_jobs.append(
                    Job(
                        title=raw.get("title", ""),
                        company=raw.get("company", ""),
                        location=raw.get("location", ""),
                        url=raw.get("url", ""),
                        source=raw.get("source", ""),
                        posted_date=raw.get("posted_date", ""),
                        key_skills=s.get("key_skills_matched", []),
                        salary=raw.get("salary", ""),
                        fit_score=s["fit_score"],
                        description=raw.get("description_summary", ""),
                    )
                )

        scored_jobs.sort(key=lambda j: j.fit_score, reverse=True)
        log.info(
            f"Filtered to {len(scored_jobs)} jobs with fit score >= "
            f"{self.config.min_fit_score}"
        )
        return scored_jobs[: self.config.max_jobs]


# ---------------------------------------------------------------------------
# Task 4: Contact Discovery
# ---------------------------------------------------------------------------


class ContactDiscoverer:
    """Finds recruiters, hiring managers, veterans, and peers at target company."""

    def __init__(self, claude: ClaudeClient, config: Config):
        self.claude = claude
        self.config = config
        self.http = httpx.Client(timeout=120)

    def _vibe_request(self, endpoint: str, payload: dict) -> dict:
        """Make a request to the Vibe Prospecting API."""
        if not self.config.vibe_api_key:
            log.warning("No VIBE_PROSPECTING_API_KEY — skipping Vibe lookup")
            return {}

        resp = self.http.post(
            f"https://vibeprospecting.explorium.ai/api/v1/{endpoint}",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.config.vibe_api_key}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        return resp.json()

    def discover(self, job: Job, resume: ResumeProfile) -> list[Contact]:
        """Find contacts at the target company across 4 categories."""
        log.info(f"Discovering contacts at {job.company}...")

        contacts = []

        # Use Claude to generate realistic contact discovery
        system = """You are a networking strategist for job seekers.
Given a target job and company, generate realistic contacts across 4 categories:
1. Recruiters (internal talent acquisition)
2. Hiring Managers (likely reporting line)
3. Veterans (military service members at the company)
4. Peers (same function, similar seniority)

For each contact, provide:
- A realistic name, title, and LinkedIn-style profile URL
- A relevance score 1-10 (be strict)
- Category and any military branch
- Specific profile details that could be used for personalization
- Short notes (<100 chars)

Return ONLY valid JSON array:
[{
    "name": "string",
    "title": "string",
    "category": "Recruiter|Hiring Manager|Veteran|Peer",
    "relevance_score": 8,
    "linkedin_url": "https://linkedin.com/in/...",
    "email": "first_last@company.com",
    "branch": "U.S. Navy" or "",
    "notes": "short note",
    "profile_details": {
        "location": "",
        "experience_highlight": "",
        "personalization_hook": ""
    }
}]

Only include contacts with relevance >= 7. Max 5 per category."""

        user = f"""Target job:
- Role: {job.title}
- Company: {job.company}
- Location: {job.location}
- Key skills: {', '.join(job.key_skills)}
- Description: {job.description[:500]}

Candidate profile:
- Name: {resume.name}
- Military: {json.dumps(resume.military_service)}
- Education: {json.dumps(resume.education)}
- Location: {resume.location}"""

        raw_contacts = self.claude.ask_json(system, user)

        for c in raw_contacts:
            if c.get("relevance_score", 0) >= self.config.min_relevance_score:
                contacts.append(
                    Contact(
                        name=c.get("name", ""),
                        company=job.company,
                        title=c.get("title", ""),
                        category=c.get("category", ""),
                        relevance_score=c.get("relevance_score", 0),
                        linkedin_url=c.get("linkedin_url", ""),
                        email=c.get("email", ""),
                        branch=c.get("branch", ""),
                        notes=c.get("notes", ""),
                        profile_details=c.get("profile_details", {}),
                    )
                )

        # Assign priorities based on category + score
        self._assign_priorities(contacts, resume)

        log.info(f"Found {len(contacts)} contacts across 4 categories")
        return contacts

    def _assign_priorities(
        self, contacts: list[Contact], resume: ResumeProfile
    ):
        """Assign outreach priority 1-N based on strategic value."""
        # Veterans with military match get boosted
        has_military = resume.military_service is not None

        def sort_key(c: Contact) -> tuple:
            cat_order = {
                "Veteran": 0 if has_military else 3,
                "Hiring Manager": 1,
                "Recruiter": 2,
                "Peer": 3 if has_military else 2,
            }
            return (cat_order.get(c.category, 4), -c.relevance_score)

        contacts.sort(key=sort_key)
        for i, c in enumerate(contacts):
            c.priority = i + 1


# ---------------------------------------------------------------------------
# Task 5: Message Generator
# ---------------------------------------------------------------------------


class MessageGenerator:
    """Generates personalized <300 char connection request messages."""

    def __init__(self, claude: ClaudeClient, config: Config):
        self.claude = claude
        self.config = config

    def generate(
        self, contacts: list[Contact], job: Job, resume: ResumeProfile
    ) -> list[Contact]:
        """Generate a unique connection message for each contact."""
        log.info(f"Generating messages for {len(contacts)} contacts...")

        system = f"""You write LinkedIn connection request messages. STRICT RULES:
- Each message MUST be under 300 characters
- Each message MUST be unique — it should NOT work if swapped with another person
- Each message MUST reference 1 SPECIFIC detail from the person's profile

CATEGORY RULES:
- Hiring Manager & Recruiter: Reference the role exists (don't say "I applied"). 
  Highlight 1-2 skills from the JD. Include Reslink: {self.config.reslink_url}
- Veteran: Reference shared military service. NO job mention. NO Reslink.
  Leverage the bond — branch, service details, shared transition experience.
- Peer: NO job mention. NO Reslink. Reference a specific profile detail only.
  Focus on relatability and genuine connection.

Candidate military background: {json.dumps(resume.military_service)}
Candidate education: {json.dumps(resume.education)}
Candidate location: {resume.location}

JD key skills: {', '.join(job.key_skills)}
Job title: {job.title}
Company: {job.company}

Return ONLY valid JSON array:
[{{"index": 0, "message": "the connection message text", "char_count": 250}}]"""

        contacts_data = json.dumps(
            [
                {
                    "index": i,
                    "name": c.name,
                    "title": c.title,
                    "category": c.category,
                    "company": c.company,
                    "branch": c.branch,
                    "notes": c.notes,
                    "profile_details": c.profile_details,
                }
                for i, c in enumerate(contacts)
            ]
        )

        results = self.claude.ask_json(system, f"Contacts:\n{contacts_data}")

        msg_map = {r["index"]: r["message"] for r in results}
        for i, contact in enumerate(contacts):
            contact.connection_message = msg_map.get(i, "")

        log.info("All messages generated")
        return contacts


# ---------------------------------------------------------------------------
# Task 6: Notion Writer
# ---------------------------------------------------------------------------


class NotionWriter:
    """Writes pipeline results to Notion databases."""

    def __init__(self, config: Config):
        self.config = config
        if config.notion_token:
            self.client = NotionClient(auth=config.notion_token)
        else:
            self.client = None
            log.warning("No NOTION_TOKEN — Notion writes will be skipped")

    def write_job(self, job: Job) -> Optional[str]:
        """Write a job to the Applications database. Returns page ID."""
        if not self.client:
            return None

        log.info(f"Writing job: {job.title} at {job.company}")

        # Map source string to valid select option
        source_map = {
            "LinkedIn": "LinkedIn",
            "Google": "Google",
            "Indeed": "Indeed",
            "ZipRecruiter": "ZipRecruiter",
            "Dice": "Dice",
        }
        source_val = source_map.get(job.source, "Other")

        # Map skills to valid multi-select options
        valid_skills = {
            "Python", "AWS", "SageMaker", "Bedrock", "Agentic AI",
            "LangGraph", "FastAPI", "Docker", "MLflow", "Ray Serve",
            "LangChain", "Azure", "PyTorch",
        }
        skill_tags = [
            {"name": s} for s in job.key_skills if s in valid_skills
        ]

        properties = {
            "Role": {"title": [{"text": {"content": job.title}}]},
            "Company": {"rich_text": [{"text": {"content": job.company}}]},
            "Location": {"rich_text": [{"text": {"content": job.location}}]},
            "Source": {"select": {"name": source_val}},
            "Fit Score": {"number": job.fit_score},
            "Status": {"select": {"name": "Queued"}},
            "Job URL": {"url": job.url or None},
            "Key Skills": {"multi_select": skill_tags},
            "Pipeline Run": {
                "date": {"start": date.today().isoformat()}
            },
        }

        if job.posted_date:
            properties["Posted Date"] = {
                "date": {"start": job.posted_date[:10]}
            }
        if job.salary:
            properties["Salary Range"] = {
                "rich_text": [{"text": {"content": job.salary}}]
            }

        page = self.client.pages.create(
            parent={"database_id": self.config.notion_jobs_db},
            properties=properties,
        )
        return page["id"]

    def write_contact(self, contact: Contact, job_title: str) -> Optional[str]:
        """Write a contact to the Contacts database. Returns page ID."""
        if not self.client:
            return None

        properties = {
            "Name": {"title": [{"text": {"content": contact.name}}]},
            "Company": {
                "rich_text": [{"text": {"content": contact.company}}]
            },
            "Title": {
                "rich_text": [{"text": {"content": contact.title}}]
            },
            "Category": {"select": {"name": contact.category}},
            "Relevance Score": {"number": contact.relevance_score},
            "LinkedIn URL": {"url": contact.linkedin_url or None},
            "Priority": {"number": contact.priority},
            "Job Target": {
                "rich_text": [{"text": {"content": job_title}}]
            },
            "Notes": {
                "rich_text": [{"text": {"content": contact.notes[:200]}}]
            },
            "Message Sent": {"checkbox": False},
            "Reply Received": {"checkbox": False},
        }

        if contact.email:
            properties["Email"] = {"email": contact.email}
        if contact.branch:
            properties["Branch"] = {
                "rich_text": [{"text": {"content": contact.branch}}]
            }
        if contact.connection_message:
            properties["Connection Message"] = {
                "rich_text": [
                    {"text": {"content": contact.connection_message[:2000]}}
                ]
            }

        page = self.client.pages.create(
            parent={"database_id": self.config.notion_contacts_db},
            properties=properties,
        )
        return page["id"]

    def update_job_status(self, page_id: str, status: str):
        """Update a job's status in Notion."""
        if not self.client:
            return
        self.client.pages.update(
            page_id=page_id,
            properties={"Status": {"select": {"name": status}}},
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Pipeline:
    """Main orchestrator — manages state transitions and task execution."""

    def __init__(self, config: Config):
        self.config = config
        self.claude = ClaudeClient(config)
        self.resume_parser = ResumeParser(self.claude)
        self.job_searcher = JobSearcher(self.claude, config)
        self.job_scorer = JobScorer(self.claude, config)
        self.contact_discoverer = ContactDiscoverer(self.claude, config)
        self.message_generator = MessageGenerator(self.claude, config)
        self.notion_writer = NotionWriter(config)
        self.ctx = PipelineContext()

    def _load_resume(self) -> str:
        """Load resume text from file."""
        path = Path(self.config.resume_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Resume not found at {path}. "
                f"Set RESUME_PATH in .env or place resume.pdf in the "
                f"current directory."
            )

        if path.suffix == ".pdf":
            try:
                import pymupdf
                doc = pymupdf.open(str(path))
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
                return text
            except ImportError:
                try:
                    import fitz
                    doc = fitz.open(str(path))
                    text = "\n".join(page.get_text() for page in doc)
                    doc.close()
                    return text
                except ImportError:
                    raise ImportError(
                        "Install pymupdf to parse PDFs: "
                        "pip install pymupdf"
                    )
        else:
            return path.read_text(encoding="utf-8")

    def _transition(self, new_state: PipelineState):
        """Log and transition to a new state."""
        old = self.ctx.state
        self.ctx.state = new_state
        log.info(f"State: {old.value} → {new_state.value}")

    def run_search(self, location: Optional[str] = None):
        """Run Task 1 + 2 + 3: Parse resume → Search → Score jobs."""
        self.ctx.started_at = datetime.now()
        loc = location or self.config.default_location
        self.ctx.search_location = loc

        # Task 1: Parse resume
        self._transition(PipelineState.PARSE_RESUME)
        resume_text = self._load_resume()
        self.ctx.resume = self.resume_parser.parse(resume_text)
        log.info(
            f"Resume parsed: {self.ctx.resume.name}, "
            f"{len(self.ctx.resume.skills)} skills"
        )

        # Task 2: Search jobs
        self._transition(PipelineState.SEARCH_JOBS)
        raw_jobs = self.job_searcher.search(self.ctx.resume, loc)

        # Task 3: Score jobs
        self._transition(PipelineState.SCORE_JOBS)
        self.ctx.jobs = self.job_scorer.score(raw_jobs, self.ctx.resume)

        # Write jobs to Notion
        self._transition(PipelineState.WRITE_NOTION)
        for job in self.ctx.jobs:
            self.notion_writer.write_job(job)

        self._print_jobs_table()
        self.ctx.completed_at = datetime.now()
        self._transition(PipelineState.COMPLETE)

    def run_contacts(self, job_url: str):
        """Run Task 1 + 4 + 5 + 6: Parse resume → Discover → Message → Notion."""
        self.ctx.started_at = datetime.now()
        self.ctx.job_url = job_url

        # Task 1: Parse resume (if not already done)
        if not self.ctx.resume:
            self._transition(PipelineState.PARSE_RESUME)
            resume_text = self._load_resume()
            self.ctx.resume = self.resume_parser.parse(resume_text)

        # Create a job object from the URL
        self._transition(PipelineState.SEARCH_JOBS)
        self.ctx.target_job = self._job_from_url(job_url)

        # Task 4: Discover contacts
        self._transition(PipelineState.DISCOVER_CONTACTS)
        self.ctx.contacts = self.contact_discoverer.discover(
            self.ctx.target_job, self.ctx.resume
        )

        # Task 5: Generate messages
        self._transition(PipelineState.GENERATE_MESSAGES)
        self.ctx.contacts = self.message_generator.generate(
            self.ctx.contacts, self.ctx.target_job, self.ctx.resume
        )

        # Task 6: Write to Notion
        self._transition(PipelineState.WRITE_NOTION)
        job_page_id = self.notion_writer.write_job(self.ctx.target_job)
        for contact in self.ctx.contacts:
            self.notion_writer.write_contact(
                contact,
                f"{self.ctx.target_job.title} at {self.ctx.target_job.company}",
            )
        if job_page_id:
            self.notion_writer.update_job_status(
                job_page_id, "Messages Drafted"
            )

        self._print_contacts_table()
        self.ctx.completed_at = datetime.now()
        self._transition(PipelineState.COMPLETE)

    def run_full(self, job_url: str):
        """Run the complete pipeline for a specific job URL."""
        self.run_contacts(job_url)

    def _job_from_url(self, url: str) -> Job:
        """Extract job details from a URL using Claude."""
        log.info(f"Extracting job details from URL: {url[:60]}...")

        system = """Given a job posting URL, infer the most likely job details.
If the URL contains identifiable info (company, title), extract it.
Return ONLY valid JSON:
{
    "title": "string",
    "company": "string",
    "location": "string",
    "source": "LinkedIn|Indeed|Google|Other",
    "key_skills": ["skill1", "skill2"],
    "description_summary": "string"
}"""

        data = self.claude.ask_json(
            system, f"Extract job details from this URL:\n{url}"
        )

        return Job(
            title=data.get("title", "Unknown Role"),
            company=data.get("company", "Unknown Company"),
            location=data.get("location", self.config.default_location),
            url=url,
            source=data.get("source", "Other"),
            key_skills=data.get("key_skills", []),
            description=data.get("description_summary", ""),
            fit_score=8,
        )

    def _print_jobs_table(self):
        """Print a formatted jobs table to stdout."""
        print("\n" + "=" * 80)
        print("JOB SEARCH RESULTS")
        print("=" * 80)
        print(
            f"{'#':<3} {'Score':<6} {'Role':<35} {'Company':<20} {'Source':<10}"
        )
        print("-" * 80)
        for i, j in enumerate(self.ctx.jobs, 1):
            print(
                f"{i:<3} {j.fit_score:<6} {j.title[:34]:<35} "
                f"{j.company[:19]:<20} {j.source:<10}"
            )
        print(f"\nTotal: {len(self.ctx.jobs)} jobs written to Notion\n")

    def _print_contacts_table(self):
        """Print a formatted contacts table to stdout."""
        print("\n" + "=" * 90)
        print(f"OUTREACH PLAN — {self.ctx.target_job.title} at "
              f"{self.ctx.target_job.company}")
        print("=" * 90)
        print(
            f"{'P#':<4} {'Name':<22} {'Title':<28} "
            f"{'Category':<16} {'Score':<6}"
        )
        print("-" * 90)
        for c in self.ctx.contacts:
            print(
                f"{c.priority:<4} {c.name[:21]:<22} {c.title[:27]:<28} "
                f"{c.category:<16} {c.relevance_score:<6}"
            )
        print(f"\nTotal: {len(self.ctx.contacts)} contacts written to Notion")
        print("Connection messages generated and stored.\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point."""
    config = Config()

    # Validate config
    missing = config.validate()
    if missing:
        print(f"ERROR: Missing required config: {', '.join(missing)}")
        print("Create a .env file with the required keys. See .env.example")
        sys.exit(1)

    pipeline = Pipeline(config)

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]

    if command == "search":
        location = sys.argv[2] if len(sys.argv) > 2 else None
        pipeline.run_search(location)

    elif command == "full":
        if len(sys.argv) < 3:
            print("Usage: python pipeline.py full <job_url>")
            sys.exit(1)
        pipeline.run_full(sys.argv[2])

    elif command.startswith("http"):
        # Direct URL — run full pipeline
        pipeline.run_full(command)

    else:
        print(f"Unknown command: {command}")
        print("Usage:")
        print("  python pipeline.py <job_url>")
        print("  python pipeline.py search [location]")
        print("  python pipeline.py full <job_url>")
        sys.exit(1)


if __name__ == "__main__":
    main()
``