import pytest
from pathlib import Path


def test_load_resume_reads_txt_file(tmp_path):
    """load_resume should read a plain text file and return its content."""
    from pipeline import load_resume

    resume_file = tmp_path / "resume.txt"
    resume_file.write_text("Ryan Ellis\nLead AI Engineer\nDenver, CO")

    result = load_resume(resume_file)

    assert "Ryan Ellis" in result
    assert "Lead AI Engineer" in result


def test_load_resume_reads_pdf_file(tmp_path, monkeypatch):
    """load_resume should extract text from a PDF file."""
    from pipeline import load_resume

    # Create a fake PDF file (just needs the .pdf extension for routing)
    pdf_file = tmp_path / "resume.pdf"
    pdf_file.write_bytes(b"fake pdf content")

    # Mock pymupdf since we don't need a real PDF for unit testing
    class FakePage:
        def get_text(self):
            return "Ryan Ellis\nLead AI Engineer"

    class FakeDoc:
        def __init__(self, path):
            self.pages = [FakePage(), FakePage()]

        def __iter__(self):
            return iter(self.pages)

        def close(self):
            pass

    import pipeline

    monkeypatch.setattr(pipeline, "fitz", type("module", (), {"open": FakeDoc}))

    result = load_resume(pdf_file)

    assert "Ryan Ellis" in result
    assert "Lead AI Engineer" in result


def test_load_resume_raises_on_missing_file():
    """load_resume should raise FileNotFoundError for missing files."""
    from pipeline import load_resume

    with pytest.raises(FileNotFoundError):
        load_resume(Path("/nonexistent/resume.txt"))


def test_load_resume_raises_on_unsupported_format(tmp_path):
    """load_resume should raise ValueError for unsupported file types."""
    from pipeline import load_resume

    docx_file = tmp_path / "resume.docx"
    docx_file.write_text("fake content")

    with pytest.raises(ValueError, match="Unsupported"):
        load_resume(docx_file)
