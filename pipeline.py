import json
import logging
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from os import getenv
from pathlib import Path
from typing import Any

import anthropic
import apify_client
import notion_client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logger level
handler = logging.StreamHandler(sys.stdout)  # Print to stdout
formatter = logging.Formatter("%(message)s")  # Simple format
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass
class Config:
    anthropic_api_key: str = ""
    notion_token: str = ""
    apify_token: str = ""
    apify_linkedin_actor_id: str = "apify/linkedin-jobs-scraper"
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
        self.apify_token = getenv("APIFY_TOKEN", "")
        logger.debug(
            "Config loaded: anthropic_key_present=%s notion_token_present=%s model=%s",
            bool(self.anthropic_api_key),
            bool(self.notion_token),
            self.model,
        )

    def validate(self) -> list[Any]:
        missing_keys = []
        required_fields = {
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
            "NOTION_TOKEN": self.notion_token,
        }

        for display_name, value in required_fields.items():
            if not value:
                missing_keys.append(display_name)

        if missing_keys:
            logger.warning("Config missing required keys: %s", missing_keys)

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
        self.model = config.model
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        logger.info("ClaudeClient initialised with model=%s", self.model)

    def ask(self, system: str, user: str) -> str:
        logger.debug(
            "ClaudeClient.ask() model=%s system_chars=%d user_chars=%d",
            self.model,
            len(system),
            len(user),
        )
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
            logger.debug("ask_json() stripped markdown fence")
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("ask_json() JSON parse failed: %s", exc)
            raise

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
        logger.info("ResumeParser.parse() starting")
        system = "You are a resume parser. Extract structured data from the resume. Return ONLY valid JSON with these fields: name, title, location, skills (list), experience_years (number), military_service (dict with branch and years, or null)."

        result = self.claude.ask_json(system, resume_text)

        profile = ResumeProfile(**result)
        logger.info(
            "ResumeParser.parse() complete name=%r skills=%d experience_years=%d military=%s",
            profile.name,
            len(profile.skills),
            profile.experience_years,
            bool(profile.military_service),
        )
        return profile


@dataclass
class Job:
    title: str = ""
    company: str = ""
    location: str = ""
    url: str = ""
    source: str = ""
    salary: str = ""
    posted_date: str = ""
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
        logger.info(
            "JobScorer.score() input=%d min_fit_score=%d max_jobs=%d",
            len(raw_jobs),
            self.config.min_fit_score,
            self.config.max_jobs,
        )
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

        dropped = len(scores) - len(passing_scores)
        if dropped > 0:
            logger.warning(
                "JobScorer.score() dropped %d jobs below min_fit_score=%d",
                dropped,
                self.config.min_fit_score,
            )

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

        capped = results[: self.config.max_jobs]
        logger.info(
            "JobScorer.score() after_filter=%d after_cap=%d",
            len(results),
            len(capped),
        )
        return capped


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
        logger.info(
            "ContactDiscoverer.discover() job=%r company=%r", job.title, job.company
        )
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
        logger.info(
            "ContactDiscoverer.discover() raw=%d min_relevance=%d veteran_boost=%s",
            len(contacts),
            self.config.min_relevance_score,
            has_military,
        )

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

        if not results:
            logger.warning(
                "ContactDiscoverer.discover() all contacts filtered out for job=%r",
                job.title,
            )
        else:
            logger.info(
                "ContactDiscoverer.discover() returning %d contacts", len(results)
            )

        return results


class MessageGenerator:
    def __init__(self, claude: ClaudeClient, config: Config) -> None:
        self.claude = claude
        self.config = config

    def generate(self, contacts: list[Contact], job: Job, resume: ResumeProfile):
        logger.info(
            "MessageGenerator.generate() contacts=%d job=%r", len(contacts), job.title
        )
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

        assigned = sum(1 for c in contacts if c.connection_message)
        logger.info("MessageGenerator.generate() messages_assigned=%d", assigned)
        if assigned < len(contacts):
            logger.warning(
                "MessageGenerator.generate() %d contacts received no message",
                len(contacts) - assigned,
            )

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
        logger.info("NotionWriter enabled=%s", self.enabled)

    def write_job(self, job: Job):
        if not self.enabled:
            return None
        logger.info(
            "NotionWriter.write_job() title=%r company=%r fit_score=%d",
            job.title,
            job.company,
            job.fit_score,
        )
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
        logger.info("NotionWriter.write_job() created page_id=%s", response["id"])
        return response["id"]

    def write_contact(self, contact: Contact, title: str):
        if not self.enabled:
            return None
        logger.debug(
            "NotionWriter.write_contact() name=%r priority=%d",
            contact.name,
            contact.priority,
        )
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
        logger.debug("NotionWriter.write_contact() created page_id=%s", response["id"])
        return response["id"]

    def update_job_status(self, id, status):
        if not self.enabled:
            return None
        logger.debug(
            "NotionWriter.update_job_status() page_id=%s status=%r", id, status
        )
        response = self.client.pages.update(
            page_id=id,
            properties={"Status": {"select": {"name": status}}},
        )


class JobSearcher:
    def __init__(self, config: Config) -> None:
        self.config = config
        if not config.apify_token:
            self.client = None
            self.enabled = False
        else:
            self.client = apify_client.ApifyClient(config.apify_token)
            self.enabled = True
        logger.info("Apify enabled=%s", self.enabled)

    def search(self, linkedin_url: str) -> Job:
        """
        Fetch job details from LinkedIn URL using Apify scraper.

        Args:
            job_url: LinkedIn job posting URL

        Returns:
            Job object with scraped details, or None if token missing or no results

        """
        if not self.enabled:
            return None
        logger.info("JobSearcher.search() fetching %s", linkedin_url)
        actor = self.client.actor(self.config.apify_linkedin_actor_id)

        run = actor.call(
            run_input={
                "jobUrl": linkedin_url,
            }
        )
        dataset_id = run["defaultDatasetId"]
        dataset = self.client.dataset(dataset_id)
        items = list(dataset.iterate_items())
        if len(items) == 0:
            return None
        item = items[0]

        job = Job(url=linkedin_url, source="LinkedIn")
        job.title = item["title"]
        job.company = item["companyName"]
        job.location = item["location"]
        job.description = item["descriptionText"]
        job.salary = item["salary"]
        job.posted_date = item["postedAt"]
        logger.info("JobSearcher.search() found job: %s at %s", job.title, job.company)
        return job


class Pipeline:
    def __init__(self, config: Config) -> None:
        logger.info("Pipeline initialising")
        self.config = config
        self.claude = ClaudeClient(config)
        self.resume_parser = ResumeParser(self.claude)
        self.job_scorer = JobScorer(self.claude, self.config)
        self.contact_discoverer = ContactDiscoverer(self.claude, self.config)
        self.message_generator = MessageGenerator(self.claude, self.config)
        self.job_searcher = JobSearcher(config)
        self.notion_writer = NotionWriter(self.config)
        self.ctx = PipelineContext(state=PipelineState.IDLE)
        logger.info("Pipeline ready")

    def _transition(self, new_state) -> None:
        logger.info("Pipeline state %s → %s", self.ctx.state.name, new_state.name)
        self.ctx.state = new_state

    def _run_contacts_pipeline(self) -> None:
        logger.info("_run_contacts_pipeline() starting")
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
            logger.info("_run_contacts_pipeline() complete")
        except Exception as e:
            logger.exception("_run_contacts_pipeline() unhandled exception")
            self._transition(PipelineState.ERROR)
            self.ctx.errors.append(str(e))


def parse_arguments(argv):
    """
    Parse --job and --resume from command line.

    Args:
        argv: list like ['pipeline.py', '--job=URL', '--resume=FILE']

    Returns:
        dict with 'job' and 'resume' keys

    Raises:
        ValueError: if validation fails

    """
    args = {}
    pattern = r"https://.*linkedin\.com/jobs/view/\d+/?$"

    for item in argv[1:]:
        arg1, arg2 = item.split("=")
        arg1 = arg1.replace("--", "")
        if arg1 == "job" and not re.match(pattern, arg2):
            raise ValueError("--job must be a valid URL")
        if arg1 == "resume":
            p = Path(arg2)
            if not p.exists():
                raise ValueError("Resume file not found")
        args[arg1] = arg2

    if "job" not in args:
        raise ValueError("--job is required")
    if "resume" not in args:
        raise ValueError("--resume is required")

    return args


def load_resume(filepath: Path) -> str:
    """
    Load resume text from file.

    Args:
        filepath: path to resume file

    Returns:
        str: resume content

    Raises:
        FileNotFoundError: if file doesn't exist

    """
    try:
        with Path.open(filepath, "r") as f:
            return f.read()
    except FileNotFoundError as err:
        msg = f"Resume file not found: {filepath}"
        raise FileNotFoundError(msg) from err


def main() -> None:
    try:
        args = parse_arguments(sys.argv)
        resume_text = load_resume(args["resume"])

        config = Config()
        missing = config.validate()
        # if missing:
        #     raise ValueError(f"Missing environment Variable: {', '.join(missing)}")
        pipeline = Pipeline(config)
        parser = ResumeParser(pipeline.claude)
        pipeline._transition(PipelineState.PARSE_RESUME)
        print("PARSE_RESUME")
        profile = parser.parse(resume_text)
        pipeline.ctx.job_url = args["job"]
        pipeline.ctx.resume = profile
        pipeline.ctx.target_job = pipeline.job_searcher.search(args["job"])
        if pipeline.ctx.target_job is None:
            # Graceful fallback
            pipeline.ctx.target_job = Job(url=args["job"])

        pipeline._run_contacts_pipeline()

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
