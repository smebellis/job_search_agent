import pytest


def test_pipeline_state_has_expected_values():
    """PipelineState enum should define all pipeline stages."""
    from pipeline import PipelineState

    assert PipelineState.IDLE.value == "idle"
    assert PipelineState.PARSE_RESUME.value == "parse_resume"
    assert PipelineState.SEARCH_JOBS.value == "search_jobs"
    assert PipelineState.SCORE_JOBS.value == "score_jobs"
    assert PipelineState.DISCOVER_CONTACTS.value == "discover_contacts"
    assert PipelineState.GENERATE_MESSAGES.value == "generate_messages"
    assert PipelineState.WRITE_NOTION.value == "write_notion"
    assert PipelineState.COMPLETE.value == "complete"
    assert PipelineState.ERROR.value == "error"


def test_pipeline_context_initializes_with_defaults():
    """PipelineContext should start in IDLE state with empty collections."""
    from pipeline import PipelineContext, PipelineState

    ctx = PipelineContext()
    assert ctx.state == PipelineState.IDLE
    assert ctx.jobs == []
    assert ctx.contacts == []
    assert ctx.errors == []
    assert ctx.resume is None
    assert ctx.target_job is None
    assert ctx.job_url is None


def test_pipeline_context_tracks_state_transitions():
    """PipelineContext state should be mutable for transitions."""
    from pipeline import PipelineContext, PipelineState

    ctx = PipelineContext()
    assert ctx.state == PipelineState.IDLE

    ctx.state = PipelineState.PARSE_RESUME
    assert ctx.state == PipelineState.PARSE_RESUME

    ctx.state = PipelineState.SEARCH_JOBS
    assert ctx.state == PipelineState.SEARCH_JOBS


def test_pipeline_context_collections_are_independent():
    """Each PipelineContext instance should have its own lists."""
    from pipeline import PipelineContext

    ctx1 = PipelineContext()
    ctx2 = PipelineContext()

    ctx1.errors.append("something broke")

    assert len(ctx1.errors) == 1
    assert len(ctx2.errors) == 0
