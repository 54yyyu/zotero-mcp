"""Regression test for #286 (Bug 2): force UTF-8 on the pdfminer subprocess.

On Windows, the console / subprocess stdout defaults to GBK or cp1252.
When pdfminer extracts a PDF that contains characters outside that
codec's range, ``sys.stdout.write`` raises ``UnicodeEncodeError``, the
child exits non-zero, and the parent marks the item ``has_fulltext='failed'``.
Subsequent incremental updates then skip the item until force-rebuild.

Fix: set ``PYTHONIOENCODING=utf-8`` + ``PYTHONUTF8=1`` in the child env
and read the child's stdout as UTF-8.
"""

from unittest.mock import MagicMock, patch

from zotero_mcp.local_db import LocalZoteroReader


class _BareReader(LocalZoteroReader):
    """LocalZoteroReader stub that skips DB init."""

    def __init__(self):
        self.db_path = "/dev/null"
        self._connection = None
        self.pdf_max_pages = 10
        self.pdf_timeout = 30


def test_pdf_extraction_subprocess_pins_utf8(tmp_path):
    """The pdfminer subprocess must receive a UTF-8 stdio env and the parent
    must read its stdout as UTF-8 — required for non-ASCII PDFs on Windows
    consoles (#286)."""
    reader = _BareReader()
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.touch()

    fake_result = MagicMock(returncode=0, stdout="hello", stderr="")
    with patch("subprocess.run", return_value=fake_result) as mock_run:
        text = reader._extract_text_from_pdf(fake_pdf)

    assert text == "hello"
    assert mock_run.call_count == 1
    _args, kwargs = mock_run.call_args

    # Parent must decode child stdout as UTF-8 with replacement (so undecodable
    # bytes don't kill the whole batch).
    assert kwargs.get("encoding") == "utf-8"
    assert kwargs.get("errors") == "replace"
    assert kwargs.get("text") is True

    # Child must use UTF-8 stdio.
    env = kwargs.get("env") or {}
    assert env.get("PYTHONIOENCODING") == "utf-8", (
        f"PYTHONIOENCODING missing/wrong in child env; got {env.get('PYTHONIOENCODING')!r}. "
        "Without this, pdfminer extracting non-ASCII text triggers "
        "UnicodeEncodeError on Windows consoles (issue #286)."
    )
    assert env.get("PYTHONUTF8") == "1"


def test_pdf_extraction_does_not_override_existing_pythonioencoding(tmp_path, monkeypatch):
    """``setdefault`` semantics: respect an existing PYTHONIOENCODING in the
    parent env rather than silently overwriting a deliberate user choice."""
    monkeypatch.setenv("PYTHONIOENCODING", "utf-16")
    reader = _BareReader()
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.touch()

    fake_result = MagicMock(returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=fake_result) as mock_run:
        reader._extract_text_from_pdf(fake_pdf)

    env = mock_run.call_args.kwargs.get("env") or {}
    assert env.get("PYTHONIOENCODING") == "utf-16"
