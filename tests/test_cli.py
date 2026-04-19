import json

import anthropic
import pytest





def test_cli_parses_arguments():
    """CLI should extract --job and --resume from sys.argv"""
    from pipeline import parse_arguments

    args = parse_arguments(
        [
            "pipeline.py",
            "--job=https://linkedin.com/jobs/view/123",
            "--resume=./resume.txt",
        ]
    )
    assert args["job"] == "https://linkedin.com/jobs/view/123"
    assert args["resume"] == "./resume.txt"


def test_cli_raises_on_missing_job():
    """CLI should fail if --job is missing"""
    from pipeline import parse_arguments

    with pytest.raises(ValueError, match="--job is required"):
        parse_arguments(["pipeline.py", "--resume=./resume.txt"])


def test_cli_raises_on_missing_resume():
    """CLI should fail if --resume is missing"""
    from pipeline import parse_arguments

    with pytest.raises(ValueError, match="--resume is required"):
        parse_arguments(["pipeline.py", "--job=https://linkedin.com/jobs/view/123"])


def test_cli_validates_job_url_format():
    """CLI should reject invalid job URLs"""
    from pipeline import parse_arguments

    with pytest.raises(ValueError, match="--job must be a valid URL"):
        parse_arguments(["pipeline.py", "--job=not-a-url", "--resume=./resume.txt"])


def test_cli_validates_resume_file_exists(tmp_path):
    """CLI should reject non-existent resume files"""
    from pipeline import parse_arguments

    with pytest.raises(ValueError, match="Resume file not found"):
        parse_arguments(
            [
                "pipeline.py",
                "--job=https://linkedin.com/jobs/view/123",
                "--resume=/nonexistent/file.txt",
            ]
        )


def test_cli_validates_resume_file_exists_with_real_file(tmp_path):
    """CLI should accept existing resume files"""
    from pipeline import parse_arguments

    resume_file = tmp_path / "resume.txt"
    resume_file.write_text("Test resume content")
    args = parse_arguments(
        [
            "pipeline.py",
            f"--job=https://linkedin.com/jobs/view/123",
            f"--resume={str(resume_file)}",
        ]
    )
    assert args["resume"] == str(resume_file)


def test_cli_loads_resume_content():
    """CLI should read and return resume file content"""
    from pipeline import load_resume

    resume_content = load_resume("./test_resume.txt")
    assert isinstance(resume_content, str)
    assert len(resume_content) > 0


def test_cli_main_initializes_pipeline(monkeypatch, tmp_path):
    """main() should create Config, Pipeline, and run it"""
    import anthropic
    import notion_client

    # Setup fake resume file
    resume_file = tmp_path / "resume.txt"
    resume_file.write_text("Name: Alice\nSkills: Python, AWS")

    # Setup fakes
    fake_anthropic = FakeAnthropic()
    fake_anthropic.messages.add_response(
        json.dumps(
            {
                "name": "Alice",
                "title": "Engineer",
                "location": "Denver",
                "skills": ["Python"],
                "experience_years": 5,
                "military_service": None,
            }
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

    # Mock sys.argv and run main
    test_argv = [
        "pipeline.py",
        f"--job=https://linkedin.com/jobs/view/123",
        f"--resume={str(resume_file)}",
    ]
    monkeypatch.setattr("sys.argv", test_argv)

    from pipeline import main

    # Should not raise
    main()


def test_cli_prints_progress(monkeypatch, tmp_path, capsys):
    """main() should print stage transitions to stdout"""
    import anthropic
    import notion_client

    resume_file = tmp_path / "resume.txt"
    resume_file.write_text("Name: Alice\nSkills: Python")

    fake_anthropic = FakeAnthropic()
    fake_anthropic.messages.add_response(
        json.dumps(
            {
                "name": "Alice",
                "title": "Engineer",
                "location": "Denver",
                "skills": ["Python"],
                "experience_years": 5,
                "military_service": None,
            }
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

    test_argv = [
        "pipeline.py",
        f"--job=https://linkedin.com/jobs/view/123",
        f"--resume={str(resume_file)}",
    ]
    monkeypatch.setattr("sys.argv", test_argv)

    from pipeline import main

    main()

    captured = capsys.readouterr()
    assert "PARSE_RESUME" in captured.out or "parse" in captured.out.lower()
