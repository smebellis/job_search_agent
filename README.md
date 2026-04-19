# Job Search Agent

A Python CLI agent that automates job search outreach. Given a LinkedIn job URL and a resume file, it parses your resume, fetches the job details, discovers networking contacts at the target company, generates personalized LinkedIn connection messages, and writes everything to Notion.

## What it does

1. Loads and parses your resume (`.txt` or `.pdf`)
2. Fetches job details from a LinkedIn URL via Apify scraper
3. Scores your resume fit against the role (1–10)
4. Discovers up to 4 contacts per job (Recruiter, Hiring Manager, Veteran, Peer)
5. Generates personalized <300 character LinkedIn connection messages per contact
6. Writes jobs and contacts to two Notion databases

## Quick start

```bash
# Install dependencies
uv sync

# Set required env vars
export ANTHROPIC_API_KEY=sk-ant-...
export NOTION_TOKEN=ntn_...
export APIFY_TOKEN=apify_...    # optional — job details fall back to URL-only if absent

# Run
python pipeline.py --job=https://linkedin.com/jobs/view/123456789 --resume=./resume.txt
```

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key for all AI calls |
| `NOTION_TOKEN` | Yes | Notion integration token |
| `APIFY_TOKEN` | No | Apify API token for LinkedIn scraping. If absent, job details fall back to a stub `Job` with the provided URL. |

## Installation

```bash
git clone <repo>
cd job_search_agent
uv sync
```

## Architecture

Everything lives in a single file: `pipeline.py`. All classes use dependency injection — dependencies are passed via `__init__`, never constructed internally.

### Data flow

```
resume file
    └─ load_resume()           # .txt read / .pdf via pymupdf
         └─ ResumeParser       # Claude → ResumeProfile
              └─ JobSearcher   # Apify LinkedIn scraper → Job
                   └─ ContactDiscoverer  # Claude → list[Contact]
                        └─ MessageGenerator  # Claude → Contact.connection_message
                             └─ NotionWriter  # Notion API → job page + contact pages
```

### Pipeline state machine

`Pipeline` drives a `PipelineContext` through `PipelineState` values:

```
IDLE → PARSE_RESUME → DISCOVER_CONTACTS → GENERATE_MESSAGES → WRITE_NOTION → COMPLETE
                                                                            ↘ ERROR
```

`_transition()` updates `ctx.state`. On any exception inside `_run_contacts_pipeline()`, the pipeline transitions to `ERROR` and records the message in `ctx.errors` — it does not crash.

### Components

| Class | Responsibility |
|-------|---------------|
| `Config` | Reads env vars at instantiation time (`__post_init__`). Provides defaults. |
| `LLMClient` | `typing.Protocol` — defines `ask()` / `ask_json()` interface for any LLM backend. |
| `ClaudeClient` | Concrete `LLMClient`. Wraps `anthropic.Anthropic`. `ask_json()` strips markdown fences before parsing. |
| `ResumeParser` | Sends resume text to Claude, returns `ResumeProfile` dataclass. |
| `JobSearcher` | Calls Apify `linkedin-jobs-scraper` actor. Gracefully disabled when `APIFY_TOKEN` is absent. |
| `JobScorer` | Scores a list of raw job dicts against a resume; filters below `min_fit_score`; sorts descending. |
| `ContactDiscoverer` | Discovers contacts across 4 categories. Applies **veteran priority boost**: when `resume.military_service` is set, `Veteran` contacts sort first; otherwise `Hiring Manager` sorts first. |
| `MessageGenerator` | Generates per-contact LinkedIn messages. Veterans: shared service, no job title, no URL. Hiring managers / recruiters: 1–2 skills + Reslink URL. Peers: shared interests, no job mention. |
| `NotionWriter` | Writes to two Notion databases. Gracefully disabled when `notion_token` is empty — all write methods return `None` silently. |

### Key design decisions

- **`Config.__post_init__`** reads env vars at instance creation, not class definition — required for `monkeypatch.setenv` to work in tests.
- **`ClaudeClient.ask_json()`** strips ` ```json ` fences before parsing — Claude sometimes wraps JSON responses in markdown code blocks.
- **`LLMClient` Protocol** is `@runtime_checkable` — any class with matching `ask` / `ask_json` methods satisfies it, enabling dependency injection of alternative LLM backends without subclassing.
- **`JobSearcher`** sets `self.enabled = False` when no Apify token is present, and `search()` returns `None`. `main()` falls back to a stub `Job(url=...)`.
- **`NotionWriter`** sets `self.enabled = False` when `notion_token` is empty — safe to instantiate in tests without mocking Notion.

## Notion databases

Two databases must be created in Notion before running:

| Database | ID |
|----------|----|
| Job Pipeline - Applications | `5135bb9f-2ad7-4400-b42a-b07d35a37d43` |
| Job Pipeline - Contacts | `7e736104-656e-470c-b301-7bcd3c8dee82` |

## Resume formats

`load_resume(filepath)` accepts:

- `.txt` — read as plain text
- `.pdf` — text extracted via [pymupdf](https://pymupdf.readthedocs.io/)

Raises `FileNotFoundError` for missing files, `ValueError` for any other extension.

## Configuration defaults

Defined in `Config`:

| Field | Default | Description |
|-------|---------|-------------|
| `model` | `claude-sonnet-4-20250514` | Claude model used for all AI calls |
| `min_fit_score` | `7` | Jobs scoring below this are dropped |
| `min_relevance_score` | `0` | Contacts scoring below this are dropped |
| `max_jobs` | `10` | Maximum jobs returned by `JobScorer` |
| `default_location` | `Denver, CO` | Fallback location for searches |
| `reslink_url` | `[Reslink URL]` | Included in hiring manager / recruiter messages |

## Testing

```bash
# Run all tests (72 total)
uv run pytest

# Single file
uv run pytest tests/test_pipeline.py

# Single test
uv run pytest tests/test_integration.py::test_full_pipeline_runs_end_to_end
```

### Test suite overview

| File | Tests | What it covers |
|------|-------|---------------|
| `test_config.py` | 4 | Env var loading, validation |
| `test_pipeline_state.py` | 4 | State enum, context dataclass |
| `test_claude_client.py` | 4 | `ask()`, `ask_json()`, markdown fence stripping |
| `test_resume_profile.py` | 3 | Dataclass fields and defaults |
| `test_resume_parser.py` | 3 | Claude call, JSON → `ResumeProfile` |
| `test_resume_loader.py` | 4 | txt/pdf loading, error cases |
| `test_job_scorer.py` | 5 | Filtering, sorting, `max_jobs` cap |
| `test_job_searcher.py` | 5 | Apify integration, graceful no-token |
| `test_contact_discovery.py` | 7 | Categories, filtering, veteran boost, priorities |
| `test_message_generator.py` | 5 | Per-category rules, length, Reslink |
| `test_notion_writer.py` | 4 | Page creation, status update, no-token skip |
| `test_pipeline.py` | 9 | State transitions, error handling, CLI parsing |
| `test_cli.py` | 9 | CLI argument parsing, validation, `main()` |
| `test_llm_protocol.py` | 3 | `LLMClient` Protocol, `ClaudeClient` satisfies it |
| `test_integration.py` | 1 | Full pipeline end-to-end with fakes |

### Test patterns

All tests mock `anthropic.Anthropic` via:
```python
monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
```

Use `import anthropic` (not `from anthropic import Anthropic`) so the patch is resolved at call time.

To disable Notion in tests, set `config.notion_token = ""` — `NotionWriter` detects this and skips all writes.

To mock PDF loading, monkeypatch `pipeline.fitz`:
```python
monkeypatch.setattr(pipeline, "fitz", type("module", (), {"open": FakeDoc}))
```

## Project structure

```
pipeline.py              # entire application — one file
tests/
  test_config.py
  test_pipeline_state.py
  test_claude_client.py
  test_resume_profile.py
  test_resume_parser.py
  test_resume_loader.py
  test_job_scorer.py
  test_job_searcher.py
  test_contact_discovery.py
  test_message_generator.py
  test_notion_writer.py
  test_pipeline.py
  test_cli.py
  test_llm_protocol.py
  test_integration.py
pyproject.toml
CLAUDE.md                # guidance for Claude Code
```
