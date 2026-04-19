import pytest


class FakeDataset:
    def __init__(self, items):
        self._items = items

    def iterate_items(self):
        return iter(self._items)


class FakeRun:
    def __init__(self, dataset_id="fake-dataset-123"):
        self.data = {"defaultDatasetId": dataset_id}

    def __getitem__(self, key):
        return self.data[key]


class FakeActor:
    def __init__(self, items):
        self._items = items
        self.last_input = None

    def call(self, run_input=None, **kwargs):
        self.last_input = run_input
        return FakeRun()


class FakeApifyClient:
    def __init__(self, token=None):
        self.token = token
        self._items = []
        self._actor = None

    def set_items(self, items):
        self._items = items
        self._actor = FakeActor(items)

    def actor(self, actor_id):
        return self._actor

    def dataset(self, dataset_id):
        return FakeDataset(self._items)


def test_job_searcher_requires_config():
    """JobSearcher should accept Config and store it."""
    from pipeline import Config, JobSearcher

    config = Config()
    config.apify_token = "fake-token"
    searcher = JobSearcher(config)

    assert searcher.config is config


def test_job_searcher_returns_job_from_linkedin(monkeypatch):
    """JobSearcher should return a Job object with fields from Apify response."""
    from pipeline import Config, JobSearcher, Job
    import apify_client

    fake_client = FakeApifyClient(token="fake")
    fake_client.set_items([
        {
            "title": "AI Ops Lead Engineer",
            "companyName": "Infosys",
            "location": "Denver, CO",
            "descriptionText": "Design and implement AIOps framework using AWS.",
            "salary": "$150,000 - $200,000",
            "link": "https://www.linkedin.com/jobs/view/4397667321",
            "postedAt": "2026-04-10T01:42:03.000Z",
        }
    ])

    monkeypatch.setattr(apify_client, "ApifyClient", lambda token: fake_client)

    config = Config()
    config.apify_token = "fake-token"
    searcher = JobSearcher(config)

    job = searcher.search("https://www.linkedin.com/jobs/view/4397667321")

    assert isinstance(job, Job)
    assert job.title == "AI Ops Lead Engineer"
    assert job.company == "Infosys"
    assert job.location == "Denver, CO"
    assert "AIOps" in job.description
    assert job.url == "https://www.linkedin.com/jobs/view/4397667321"
    assert job.source == "LinkedIn"


def test_job_searcher_passes_url_to_apify(monkeypatch):
    """JobSearcher should pass the job URL to the Apify actor input."""
    from pipeline import Config, JobSearcher
    import apify_client

    fake_client = FakeApifyClient(token="fake")
    fake_client.set_items([
        {
            "title": "Test Role",
            "companyName": "TestCo",
            "location": "Denver",
            "descriptionText": "A test job",
            "salary": "",
            "link": "https://www.linkedin.com/jobs/view/123",
            "postedAt": "2026-04-10",
        }
    ])

    monkeypatch.setattr(apify_client, "ApifyClient", lambda token: fake_client)

    config = Config()
    config.apify_token = "fake-token"
    searcher = JobSearcher(config)

    searcher.search("https://www.linkedin.com/jobs/view/123")

    actor_input = fake_client._actor.last_input
    assert "https://www.linkedin.com/jobs/view/123" in str(actor_input)


def test_job_searcher_graceful_when_no_token():
    """JobSearcher should return None when no Apify token is configured."""
    from pipeline import Config, JobSearcher

    config = Config()
    config.apify_token = ""
    searcher = JobSearcher(config)

    result = searcher.search("https://www.linkedin.com/jobs/view/123")
    assert result is None


def test_job_searcher_handles_empty_results(monkeypatch):
    """JobSearcher should return None when Apify returns no results."""
    from pipeline import Config, JobSearcher
    import apify_client

    fake_client = FakeApifyClient(token="fake")
    fake_client.set_items([])

    monkeypatch.setattr(apify_client, "ApifyClient", lambda token: fake_client)

    config = Config()
    config.apify_token = "fake-token"
    searcher = JobSearcher(config)

    result = searcher.search("https://www.linkedin.com/jobs/view/999")
    assert result is None