import json

import anthropic
import pytest


def test_contact_dataclass_has_required_fields() -> None:
    """Contact should have all expected fields with defaults."""
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


def test_contact_stores_veteran_data() -> None:
    """Contact should store military branch for veterans."""
    from pipeline import Contact

    contact = Contact(
        name="Gordon Feliciano",
        company="Infosys",
        title="Technology Lead",
        category="Veteran",
        relevance_score=9,
        branch="U.S. Navy",
        notes="Denver, CO; 8 yrs Navy; Gulf War vet",
    )
    assert contact.category == "Veteran"
    assert contact.branch == "U.S. Navy"
    assert "Gulf War" in contact.notes


def test_contact_discoverer_returns_four_categories(monkeypatch) -> None:
    """ContactDiscoverer should return contacts across all 4 categories."""
    from pipeline import ClaudeClient, Config, ContactDiscoverer, Job, ResumeProfile

    fake_contacts = [
        {
            "name": "Alice",
            "title": "TA Manager",
            "category": "Recruiter",
            "relevance_score": 8,
            "linkedin_url": "",
            "email": "",
            "branch": "",
            "notes": "Internal recruiter",
            "profile_details": {},
        },
        {
            "name": "Bob",
            "title": "Engineering Director",
            "category": "Hiring Manager",
            "relevance_score": 9,
            "linkedin_url": "",
            "email": "",
            "branch": "",
            "notes": "Likely reports to",
            "profile_details": {},
        },
        {
            "name": "Charlie",
            "title": "Tech Lead",
            "category": "Veteran",
            "relevance_score": 8,
            "linkedin_url": "",
            "email": "",
            "branch": "U.S. Navy",
            "notes": "Navy vet",
            "profile_details": {},
        },
        {
            "name": "Diana",
            "title": "Cloud Architect",
            "category": "Peer",
            "relevance_score": 7,
            "linkedin_url": "",
            "email": "",
            "branch": "",
            "notes": "Same function",
            "profile_details": {},
        },
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

    job = Job(
        title="AI Ops Lead",
        company="Infosys",
        location="Denver, CO",
        key_skills=["Python", "AWS"],
    )
    resume = ResumeProfile(
        name="Ryan",
        skills=["Python"],
        military_service={"branch": "U.S. Navy", "years": 20},
    )

    contacts = discoverer.discover(job, resume)

    categories = {c.category for c in contacts}
    assert "Recruiter" in categories
    assert "Hiring Manager" in categories
    assert "Veteran" in categories
    assert "Peer" in categories


def test_contact_discoverer_filters_below_threshold(monkeypatch):
    """ContactDiscoverer should exclude contacts below min_relevance_score."""
    from pipeline import ClaudeClient, Config, ContactDiscoverer, Job, ResumeProfile

    fake_contacts = [
        {
            "name": "High",
            "title": "Director",
            "category": "Hiring Manager",
            "relevance_score": 9,
            "linkedin_url": "",
            "email": "",
            "branch": "",
            "notes": "",
            "profile_details": {},
        },
        {
            "name": "Low",
            "title": "Intern",
            "category": "Peer",
            "relevance_score": 4,
            "linkedin_url": "",
            "email": "",
            "branch": "",
            "notes": "",
            "profile_details": {},
        },
        {
            "name": "Medium",
            "title": "Recruiter",
            "category": "Recruiter",
            "relevance_score": 7,
            "linkedin_url": "",
            "email": "",
            "branch": "",
            "notes": "",
            "profile_details": {},
        },
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
    """Veterans should be prioritized to top when candidate has military service."""
    from pipeline import ClaudeClient, Config, ContactDiscoverer, Job, ResumeProfile

    fake_contacts = [
        {
            "name": "Recruiter",
            "title": "TA Lead",
            "category": "Recruiter",
            "relevance_score": 9,
            "linkedin_url": "",
            "email": "",
            "branch": "",
            "notes": "",
            "profile_details": {},
        },
        {
            "name": "HM",
            "title": "Director",
            "category": "Hiring Manager",
            "relevance_score": 9,
            "linkedin_url": "",
            "email": "",
            "branch": "",
            "notes": "",
            "profile_details": {},
        },
        {
            "name": "Vet",
            "title": "Tech Lead",
            "category": "Veteran",
            "relevance_score": 8,
            "linkedin_url": "",
            "email": "",
            "branch": "U.S. Navy",
            "notes": "",
            "profile_details": {},
        },
        {
            "name": "Peer",
            "title": "Engineer",
            "category": "Peer",
            "relevance_score": 8,
            "linkedin_url": "",
            "email": "",
            "branch": "",
            "notes": "",
            "profile_details": {},
        },
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
    resume = ResumeProfile(
        name="Ryan",
        skills=["Python"],
        military_service={"branch": "U.S. Navy", "years": 20},
    )

    contacts = discoverer.discover(job, resume)

    assert contacts[0].category == "Veteran"
    assert contacts[0].priority == 1


def test_contact_discoverer_no_veteran_boost_without_military(monkeypatch):
    """Veterans should NOT be boosted when candidate has no military service."""
    from pipeline import ClaudeClient, Config, ContactDiscoverer, Job, ResumeProfile

    fake_contacts = [
        {
            "name": "Recruiter",
            "title": "TA Lead",
            "category": "Recruiter",
            "relevance_score": 9,
            "linkedin_url": "",
            "email": "",
            "branch": "",
            "notes": "",
            "profile_details": {},
        },
        {
            "name": "HM",
            "title": "Director",
            "category": "Hiring Manager",
            "relevance_score": 9,
            "linkedin_url": "",
            "email": "",
            "branch": "",
            "notes": "",
            "profile_details": {},
        },
        {
            "name": "Vet",
            "title": "Tech Lead",
            "category": "Veteran",
            "relevance_score": 8,
            "linkedin_url": "",
            "email": "",
            "branch": "U.S. Army",
            "notes": "",
            "profile_details": {},
        },
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


def test_contact_discoverer_assigns_sequential_priorities(monkeypatch) -> None:
    """Each contact should get a sequential priority number starting at 1."""
    from pipeline import ClaudeClient, Config, ContactDiscoverer, Job, ResumeProfile

    fake_contacts = [
        {
            "name": "A",
            "title": "Recruiter",
            "category": "Recruiter",
            "relevance_score": 8,
            "linkedin_url": "",
            "email": "",
            "branch": "",
            "notes": "",
            "profile_details": {},
        },
        {
            "name": "B",
            "title": "Director",
            "category": "Hiring Manager",
            "relevance_score": 9,
            "linkedin_url": "",
            "email": "",
            "branch": "",
            "notes": "",
            "profile_details": {},
        },
        {
            "name": "C",
            "title": "Engineer",
            "category": "Peer",
            "relevance_score": 7,
            "linkedin_url": "",
            "email": "",
            "branch": "",
            "notes": "",
            "profile_details": {},
        },
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
