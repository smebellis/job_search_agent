# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_pipeline.py

# Run a specific test
uv run pytest tests/test_pipeline.py::test_pipeline_starts_in_idle_state

# Install dependencies
uv sync
```

## Required Environment Variables

```
ANTHROPIC_API_KEY=...
NOTION_TOKEN=...
```

## Architecture

Everything lives in a single file: `pipeline.py`. The pattern is dependency injection throughout — each class receives its dependencies via `__init__` rather than constructing them internally.

### Data flow

```
resume file → load_resume → resume text
resume text → ResumeParser → ResumeProfile
raw job dicts → JobScorer → list[Job]
Job + ResumeProfile → ContactDiscoverer → list[Contact]
list[Contact] + Job + ResumeProfile → MessageGenerator → list[Contact] (with messages)
Job + list[Contact] → NotionWriter → Notion pages
```

All AI calls go through `ClaudeClient`, which wraps the Anthropic SDK. `ask()` returns raw text; `ask_json()` strips markdown fences and parses JSON.

### Pipeline state machine

`Pipeline` composes all components and drives a `PipelineContext` through `PipelineState` enum values: `IDLE → DISCOVER_CONTACTS → GENERATE_MESSAGES → WRITE_NOTION → COMPLETE` (or `ERROR` on exception). `_transition()` updates `ctx.state`.

### Key design decisions

- **`load_resume(filepath)`** accepts `.txt` (plain read) or `.pdf` (via `pymupdf`/`fitz`). Raises `FileNotFoundError` for missing files and `ValueError` for unsupported extensions. The module-level `fitz` import is used so tests can monkeypatch `pipeline.fitz`.
- **`JobSearcher`** scrapes LinkedIn via the Apify API. Requires `APIFY_API_TOKEN` env var.
- **`ClaudeClient.ask_json()`** strips ` ```json ` fences before parsing — Claude sometimes wraps JSON in markdown code blocks.
- **`ContactDiscoverer`** applies veteran priority boosting: when `resume.military_service is not None`, the sort order puts `"Veteran"` contacts first; otherwise hiring manager goes first.
- **`NotionWriter`** degrades gracefully — if `notion_token` is empty, `self.enabled = False` and all write methods return `None` silently. Tests rely on this to skip Notion calls.
- **`Config`** uses `__post_init__` to read env vars at instance creation time, not class definition time. This is required for `monkeypatch.setenv` to work in tests.

### Notion database IDs (already created)

- Job Pipeline - Applications: `5135bb9f-2ad7-4400-b42a-b07d35a37d43`
- Job Pipeline - Contacts: `7e736104-656e-470c-b301-7bcd3c8dee82`

## Test patterns

All tests mock `anthropic.Anthropic` via `monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)`. Use `import anthropic` (not `from anthropic import Anthropic`) so the patch is looked up at call time. The `FakeMessages` class in `test_pipeline.py` supports sequenced multi-call responses via `add_response()`.

When testing `NotionWriter`, set `config.notion_token = ""` to disable it or monkeypatch `notion_client.Client`.

When testing `load_resume` with a PDF, monkeypatch `pipeline.fitz` with a fake module that provides `open()`.

## Current status (2026-04-19)

**Implemented and tested:**
- `load_resume` — txt + PDF loading, error handling (66 passing, 2 failing in `test_cli.py`)
- `ResumeParser`, `JobScorer`, `ContactDiscoverer`, `MessageGenerator`, `NotionWriter`
- `Pipeline` state machine
- `JobSearcher` — Apify/LinkedIn scraping

**Known failures:**
- `tests/test_cli.py` — 2 tests fail with `NameError: name 'FakeAnthropic'`; `FakeAnthropic` is not imported/defined in that test module. Needs fix.

**What's missing:**
- End-to-end integration test
- CLI entrypoint wired to `load_resume` + full pipeline run
