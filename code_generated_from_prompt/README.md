# Job Pipeline Orchestrator Agent

Single-command AI agent that automates your entire job search outreach workflow.

## Architecture

```
                    ┌──────────────────────┐
                    │    CLI ENTRY POINT    │
                    │  python pipeline.py   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │    ORCHESTRATOR       │
                    │  (state machine)      │
                    │                       │
                    │  IDLE → PARSE →       │
                    │  SEARCH → SCORE →     │
                    │  DISCOVER → MESSAGE → │
                    │  NOTION → COMPLETE    │
                    └──────────┬───────────┘
                               │
         ┌─────────┬───────────┼───────────┬──────────┐
         ▼         ▼           ▼           ▼          ▼
    ┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
    │ Resume  │ │  Job   │ │Contact │ │Message │ │ Notion │
    │ Parser  │ │Searcher│ │Discover│ │  Gen   │ │ Writer │
    │         │ │        │ │        │ │        │ │        │
    │ Claude  │ │Claude +│ │Claude +│ │ Claude │ │ Notion │
    │  API    │ │Apify   │ │Vibe    │ │  API   │ │  API   │
    └─────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

## Setup

```bash
# 1. Clone/copy the project
cd job_pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Place your resume
cp ~/path/to/resume.pdf ./resume.pdf
```

## Usage

### Full pipeline for a specific job
```bash
python pipeline.py "https://linkedin.com/jobs/view/ai-ops-lead-engineer-4397667321"
```

### Search for matching jobs first
```bash
python pipeline.py search "Denver, CO"
```

### Explicit full pipeline
```bash
python pipeline.py full "https://linkedin.com/jobs/view/..."
```

## What it does

| Step | Task | Tools | Output |
|------|------|-------|--------|
| 1 | Parse resume | Claude API | Structured skills/experience profile |
| 2 | Search jobs | Claude + Apify LinkedIn scraper | Raw job listings |
| 3 | Score jobs | Claude API | Filtered & scored job table (fit ≥ 7) |
| 4 | Discover contacts | Claude + Vibe Prospecting | 4-category contact list with emails |
| 5 | Generate messages | Claude API | Personalized <300 char messages |
| 6 | Write to Notion | Notion API | Jobs DB + Contacts DB populated |

## Notion Databases

Two databases are auto-created:

### Job Pipeline - Applications
Tracks every job you're targeting with fit scores, status, and skills.

### Job Pipeline - Contacts  
Every networking contact with their category, personalized message, 
and outreach tracking (sent/replied checkboxes).

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `NOTION_TOKEN` | Yes | Notion integration token |
| `APIFY_TOKEN` | No | Enables LinkedIn job scraping |
| `VIBE_PROSPECTING_API_KEY` | No | Enables contact email enrichment |
| `RESUME_PATH` | No | Default: `resume.pdf` |
| `DEFAULT_LOCATION` | No | Default: `Denver, CO` |
| `MIN_FIT_SCORE` | No | Default: `7` |
| `RESLINK_URL` | No | Your video intro URL for messages |

## Contact Categories & Message Rules

| Category | Mentions job? | Includes Reslink? | Personalization focus |
|----------|--------------|-------------------|----------------------|
| Veteran | No | No | Shared military service bond |
| Hiring Manager | Yes (obliquely) | Yes | JD skills + their profile |
| Recruiter | Yes (obliquely) | Yes | Their recruiting approach |
| Peer | No | No | Shared detail (school, location, tech) |
```

## Extending

The pipeline is modular — each task is a standalone class:
- `ResumeParser` → swap in a different PDF parser
- `JobSearcher` → add Indeed, Glassdoor scrapers  
- `ContactDiscoverer` → add Apollo.io, Hunter.io
- `MessageGenerator` → fine-tune prompts per industry
- `NotionWriter` → swap for Airtable, Google Sheets
