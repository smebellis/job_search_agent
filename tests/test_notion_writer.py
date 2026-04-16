import pytest


class FakePages:
    """Mock for notion_client.Client().pages"""

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
    """Mock for notion_client.Client"""

    def __init__(self, **kwargs):
        self.pages = FakePages()


def test_notion_writer_creates_job_page(monkeypatch):
    """NotionWriter should create a job page with correct Notion property schema."""
    import notion_client

    from pipeline import Config, Job, NotionWriter

    monkeypatch.setattr(notion_client, "Client", FakeNotionClient)

    config = Config()
    config.notion_token = "ntn_fake"
    config.notion_jobs_db = "fake-jobs-db-id"
    writer = NotionWriter(config)

    job = Job(
        title="AI Ops Lead Engineer",
        company="Infosys",
        location="Denver, CO",
        url="https://linkedin.com/jobs/view/123",
        source="LinkedIn",
        fit_score=9,
        key_skills=["Python", "AWS"],
    )

    page_id = writer.write_job(job)

    assert page_id == "fake-page-id-123"
    assert len(writer.client.pages.created) == 1

    props = writer.client.pages.created[0]["properties"]

    # Title type
    assert props["Role"]["title"][0]["text"]["content"] == "AI Ops Lead Engineer"
    # Rich text type
    assert props["Company"]["rich_text"][0]["text"]["content"] == "Infosys"
    # Number type
    assert props["Fit Score"]["number"] == 9
    # URL type
    assert props["Job URL"]["url"] == "https://linkedin.com/jobs/view/123"
    # Select type
    assert props["Source"]["select"]["name"] == "LinkedIn"


def test_notion_writer_creates_contact_page(monkeypatch):
    """NotionWriter should create a contact page with correct property types."""
    import notion_client

    from pipeline import Config, Contact, NotionWriter

    monkeypatch.setattr(notion_client, "Client", FakeNotionClient)

    config = Config()
    config.notion_token = "ntn_fake"
    config.notion_contacts_db = "fake-contacts-db-id"
    writer = NotionWriter(config)

    contact = Contact(
        name="Gordon Feliciano",
        company="Infosys",
        title="Technology Lead",
        category="Veteran",
        relevance_score=9,
        linkedin_url="https://linkedin.com/in/gordonfeliciano",
        email="gordon.feliciano@infosys.com",
        branch="U.S. Navy",
        connection_message="Hi Gordon — fellow Navy veteran here.",
        priority=1,
        notes="Denver, CO; Gulf War vet",
    )

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
    """update_job_status should call pages.update with the correct status."""
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
    """All methods should return None gracefully when no Notion token is configured."""
    from pipeline import Config, Contact, Job, NotionWriter

    config = Config()
    config.notion_token = ""
    writer = NotionWriter(config)

    job = Job(title="Test Role", company="Test Co")
    contact = Contact(name="Test Person", category="Peer")

    assert writer.write_job(job) is None
    assert writer.write_contact(contact, "Test Role at Test Co") is None
    assert writer.update_job_status("fake-id", "Queued") is None
