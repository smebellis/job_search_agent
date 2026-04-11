from dataclasses import dataclass, field
from enum import Enum
from os import environ, getenv
from typing import Any


@dataclass
class Config:
    anthropic_api_key: str = ""
    notion_token: str = ""
    default_location: str = "Denver, CO"
    min_fit_score: int = 7
    max_jobs: int = 10
    model: str = "claude-sonnet-4-20250514"

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


class ClaudeClient(Config):
    def ask():
        pass

    def ask_json():
        pass
