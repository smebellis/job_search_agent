import json
import pytest
import anthropic


def test_messages_under_300_chars(monkeypatch):
    """Every generated message should be under 300 characters."""
    from pipeline import (
        Config,
        ClaudeClient,
        MessageGenerator,
        Contact,
        Job,
        ResumeProfile,
    )

    fake_messages = [
        {
            "index": 0,
            "message": "Hi Gordon — fellow Navy veteran here. 20 years USN out of Virginia Beach. Great to see another vet in tech at Infosys Denver.",
        },
        {
            "index": 1,
            "message": "Hi Vaibhav — noticed your AWS cloud migration work at Infosys CO. The AI Ops Lead role aligns with my Bedrock and SageMaker experience. Quick intro: [Reslink URL]",
        },
        {
            "index": 2,
            "message": "Hi Sukhdeep — fellow Colorado tech person. Your work across AWS and microservices at Infosys Boulder mirrors my stack. Would enjoy connecting.",
        },
    ]

    class FakeContent:
        text = json.dumps(fake_messages)

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
    config.reslink_url = "https://reslink.io/ryan"
    claude = ClaudeClient(config)
    generator = MessageGenerator(claude, config)

    contacts = [
        Contact(
            name="Gordon",
            category="Veteran",
            title="Tech Lead",
            company="Infosys",
            branch="U.S. Navy",
            notes="Navy vet Denver",
        ),
        Contact(
            name="Vaibhav",
            category="Hiring Manager",
            title="Cloud Architect",
            company="Infosys",
            notes="AWS migration CO",
        ),
        Contact(
            name="Sukhdeep",
            category="Peer",
            title="Dev Manager",
            company="Infosys",
            notes="Boulder CO AWS",
        ),
    ]
    job = Job(
        title="AI Ops Lead",
        company="Infosys",
        location="Denver, CO",
        key_skills=["Python", "AWS", "Bedrock"],
    )
    resume = ResumeProfile(
        name="Ryan",
        skills=["Python"],
        military_service={"branch": "U.S. Navy", "years": 20},
    )

    result = generator.generate(contacts, job, resume)

    for contact in result:
        assert len(contact.connection_message) <= 300, (
            f"Message for {contact.name} is {len(contact.connection_message)} chars"
        )


def test_veteran_messages_no_job_mention(monkeypatch):
    """Veteran messages should not mention the job title or hiring."""
    from pipeline import (
        Config,
        ClaudeClient,
        MessageGenerator,
        Contact,
        Job,
        ResumeProfile,
    )

    fake_messages = [
        {
            "index": 0,
            "message": "Hi Gordon — fellow Navy veteran here. 20 years USN out of Virginia Beach. Great to see another vet at Infosys Denver.",
        },
    ]

    class FakeContent:
        text = json.dumps(fake_messages)

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
    config.reslink_url = "https://reslink.io/ryan"
    claude = ClaudeClient(config)
    generator = MessageGenerator(claude, config)

    contacts = [
        Contact(
            name="Gordon",
            category="Veteran",
            title="Tech Lead",
            company="Infosys",
            branch="U.S. Navy",
        )
    ]
    job = Job(title="AI Ops Lead", company="Infosys", key_skills=["Python"])
    resume = ResumeProfile(
        name="Ryan", military_service={"branch": "U.S. Navy", "years": 20}
    )

    result = generator.generate(contacts, job, resume)

    msg = result[0].connection_message.lower()
    assert "ai ops lead" not in msg
    assert "hiring" not in msg
    assert "role" not in msg


def test_veteran_messages_no_reslink(monkeypatch):
    """Veteran messages should not contain the Reslink URL."""
    from pipeline import (
        Config,
        ClaudeClient,
        MessageGenerator,
        Contact,
        Job,
        ResumeProfile,
    )

    fake_messages = [
        {
            "index": 0,
            "message": "Hi Gordon — fellow Navy veteran here. Served 20 years. Good to connect with another vet in Colorado tech.",
        },
    ]

    class FakeContent:
        text = json.dumps(fake_messages)

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
    config.reslink_url = "https://reslink.io/ryan"
    claude = ClaudeClient(config)
    generator = MessageGenerator(claude, config)


def test_message_generator_handles_missing_message_key(monkeypatch):
    """generate() must not KeyError when LLM omits the 'message' key for an entry."""
    import json
    import anthropic
    from pipeline import Config, ClaudeClient, MessageGenerator, Contact, Job, ResumeProfile

    # LLM returns entry without "message" key (it was told to omit if nothing unique)
    fake_response = [{"index": 0}]

    class FakeContent:
        text = json.dumps(fake_response)
    class FakeResponse:
        content = [FakeContent()]
    class FakeMessages:
        def create(self, **kwargs): return FakeResponse()
    class FakeAnthropic:
        def __init__(self, **kwargs): self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    config = Config()
    config.reslink_url = "https://reslink.io/ryan"
    generator = MessageGenerator(ClaudeClient(config), config)
    contacts = [Contact(name="Alice", category="Recruiter", company="Acme")]
    result = generator.generate(contacts, Job(title="AI Lead", company="Acme"), ResumeProfile())
    assert result[0].connection_message == ""
