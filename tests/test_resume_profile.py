from dataclasses import asdict

import pytest


def test_resume_profile_has_required_fields():
    """ResumeProfile should have all expected fields with defaults."""
    from pipeline import ResumeProfile

    profile = ResumeProfile()
    assert profile.name == ""
    assert profile.title == ""
    assert profile.location == ""
    assert profile.skills == []
    assert profile.experience_years == 0
    assert profile.military_service is None


def test_resume_profile_stores_values():
    """ResumeProfile should accept and store values."""
    from pipeline import ResumeProfile

    profile = ResumeProfile(
        name="Ryan Ellis",
        title="Lead AI Engineer",
        location="Denver, CO",
        skills=["Python", "AWS", "LangGraph"],
        experience_years=20,
        military_service={"branch": "U.S. Navy", "years": 20},
    )
    assert profile.name == "Ryan Ellis"
    assert "Python" in profile.skills
    assert profile.military_service["branch"] == "U.S. Navy"


def test_resume_profile_lists_are_independent():
    """Each ResumeProfile should have its own skill list."""
    from pipeline import ResumeProfile

    p1 = ResumeProfile()
    p2 = ResumeProfile()
    p1.skills.append("Python")

    assert len(p1.skills) == 1
    assert len(p2.skills) == 0
