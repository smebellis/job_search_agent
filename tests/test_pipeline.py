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


def test_pipeline_initializes_all_modules():
    """Pipeline should create instances of all module classes."""
    from pipeline import Config, Pipeline

    config = Config()
    config.notion_token = ""  # Disable Notion
    pipeline = Pipeline(config)

    assert pipeline.config is config
    assert pipeline.claude is not None
    assert pipeline.resume_parser is not None
    assert pipeline.job_scorer is not None
    assert pipeline.contact_discoverer is not None
    assert pipeline.message_generator is not None
    assert pipeline.notion_writer is not None


def test_pipeline_starts_in_idle_state():
    """Pipeline context should start in IDLE state."""
    from pipeline import Config, Pipeline, PipelineState

    config = Config()
    config.notion_token = ""
    pipeline = Pipeline(config)

    assert pipeline.ctx.state == PipelineState.IDLE


def test_pipeline_transitions_through_states(monkeypatch):
    """Pipeline should transition through states during run_full."""
    import notion_client

    from pipeline import Config, Pipeline, PipelineState

    # Track state transitions
    transitions = []

    fake_anthropic = FakeAnthropic()

    # Response 1: ContactDiscoverer - generate contacts
    fake_anthropic.messages.add_response(
        json.dumps(
            [
                {
                    "name": "Alice",
                    "title": "Recruiter",
                    "category": "Recruiter",
                    "relevance_score": 8,
                    "linkedin_url": "",
                    "email": "",
                    "branch": "",
                    "notes": "",
                },
            ]
        )
    )

    # Response 2: MessageGenerator - generate messages
    fake_anthropic.messages.add_response(
        json.dumps(
            [
                {"index": 0, "message": "Hi Alice — great to connect."},
            ]
        )
    )

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: fake_anthropic)

    class FakeNotionClient:
        def __init__(self, **kwargs):
            self.pages = type(
                "P",
                (),
                {
                    "create": lambda self, **kw: {"id": "fake-id"},
                    "update": lambda self, **kw: {"id": "fake-id"},
                },
            )()

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

    # Pre-set resume and job to skip parsing/searching
    from pipeline import Job, ResumeProfile

    pipeline.ctx.resume = ResumeProfile(
        name="Ryan", skills=["Python"], military_service=None
    )
    pipeline.ctx.target_job = Job(
        title="AI Ops Lead", company="Infosys", location="Denver", key_skills=["Python"]
    )

    pipeline._run_contacts_pipeline()

    assert PipelineState.DISCOVER_CONTACTS in transitions
    assert PipelineState.GENERATE_MESSAGES in transitions
    assert PipelineState.WRITE_NOTION in transitions
    assert PipelineState.COMPLETE in transitions


def test_pipeline_error_handling(monkeypatch):
    """Pipeline should catch errors and transition to ERROR state."""
    from pipeline import Config, Pipeline, PipelineState

    fake_anthropic = FakeAnthropic()
    fake_anthropic.messages.add_response("NOT VALID JSON")

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: fake_anthropic)

    config = Config()
    config.notion_token = ""
    config.min_relevance_score = 7
    pipeline = Pipeline(config)

    from pipeline import Job, ResumeProfile

    pipeline.ctx.resume = ResumeProfile(name="Ryan", skills=["Python"])
    pipeline.ctx.target_job = Job(title="Test", company="TestCo", key_skills=["Python"])

    pipeline._run_contacts_pipeline()

    assert pipeline.ctx.state == PipelineState.ERROR
    assert len(pipeline.ctx.errors) > 0
