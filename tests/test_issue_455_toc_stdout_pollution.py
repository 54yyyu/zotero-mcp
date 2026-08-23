"""The PDF outline child's JSON channel survives a polluted stdout (#455).

`zotero_get_pdf_outline` failed on every PDF, whatever the file, with
"unreadable outline data: Expecting value: line 1 column 1 (char 0)". The
child script did `import fitz`; PyMuPDF >= 1.28 ends its legacy `fitz` shim
with a deprecation notice written to *stdout*, so the notice arrived ahead of
the JSON and `json.loads` choked on its first character.

The import is now `pymupdf`, which does not warn, and the payload is
sentinel-delimited so that a print from anywhere else in the child's
interpreter — a sitecustomize hook, a .pth file, a C-level MuPDF warning —
cannot break the channel either.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

from zotero_mcp.tools.write import (
    _TOC_CHILD_SCRIPT,
    _TOC_EXIT_NO_PYMUPDF,
    _TOC_SENTINEL,
    _extract_pdf_toc,
)

pymupdf = pytest.importorskip("pymupdf")

TOC = [[1, "Chapter One", 1], [2, "Section 1.1", 2], [1, "Chapter Two", 3]]


@pytest.fixture(scope="module")
def pdf_with_outline():
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page()
    doc.set_toc(TOC)
    path = os.path.join(tempfile.mkdtemp(), "outline.pdf")
    doc.save(path)
    doc.close()
    return path


def _run_child(pdf_path, env=None):
    return subprocess.run(
        [sys.executable, "-c", _TOC_CHILD_SCRIPT, str(pdf_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _noisy_env(message):
    """An environment whose interpreter prints *message* to stdout at startup.

    Stands in for PyMuPDF's own deprecation notice: both are plain prints that
    land on the child's stdout before it writes anything of its own, which is
    the whole of the #455 failure. Using usercustomize rather than monkey-
    patching PyMuPDF keeps the test meaningful on versions that don't warn.
    """
    site = tempfile.mkdtemp()
    with open(os.path.join(site, "usercustomize.py"), "w") as fh:
        fh.write(f"print({message!r})\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = site + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_child_imports_pymupdf_not_the_deprecated_fitz_shim():
    """The `fitz` name is what emits the notice; we must not ask for it first."""
    assert "import pymupdf as fitz" in _TOC_CHILD_SCRIPT
    # `fitz` survives only as the fallback for PyMuPDF < 1.24.3.
    assert _TOC_CHILD_SCRIPT.index("import pymupdf as fitz") < _TOC_CHILD_SCRIPT.index(
        "import fitz"
    )


def test_child_tags_its_payload(pdf_with_outline):
    proc = _run_child(pdf_with_outline)
    assert proc.returncode == 0
    assert _TOC_SENTINEL in proc.stdout
    assert json.loads(proc.stdout.rsplit(_TOC_SENTINEL, 1)[1]) == TOC


def test_outline_survives_a_deprecation_notice_on_stdout(pdf_with_outline):
    """The exact #455 shape: a notice printed to stdout ahead of the JSON."""
    notice = "warning: The `fitz` API is deprecated and will be removed in future."
    proc = _run_child(pdf_with_outline, env=_noisy_env(notice))

    assert proc.returncode == 0
    # The pollution really is present -- otherwise this test proves nothing.
    assert proc.stdout.startswith(notice)
    # ...and the payload is still recoverable.
    assert json.loads(proc.stdout.rsplit(_TOC_SENTINEL, 1)[1]) == TOC


def test_parent_discards_everything_before_the_sentinel(pdf_with_outline, monkeypatch):
    """End-to-end through _extract_pdf_toc, not just the child in isolation."""
    monkeypatch.setenv("PYTHONPATH", _noisy_env("noise on stdout")["PYTHONPATH"])
    outcome = _extract_pdf_toc(pdf_with_outline)
    assert outcome.status == "ok"
    assert outcome.toc == TOC


def test_clean_run_still_works(pdf_with_outline):
    outcome = _extract_pdf_toc(pdf_with_outline)
    assert outcome.status == "ok"
    assert outcome.toc == TOC


def test_pdf_without_an_outline_is_ok_not_an_error():
    """An empty TOC is a legitimate answer and must not read as a failure."""
    doc = pymupdf.open()
    doc.new_page()
    path = os.path.join(tempfile.mkdtemp(), "no_outline.pdf")
    doc.save(path)
    doc.close()

    outcome = _extract_pdf_toc(path)
    assert outcome.status == "ok"
    assert outcome.toc == []


def test_untagged_stdout_is_reported_as_missing_data_not_bad_json(monkeypatch):
    """A child that exits 0 without reaching its final write never produced an
    outline. Blaming the JSON there sent #455 reporters looking at the PDF."""
    class FakeProc:
        returncode = 0

        def communicate(self, timeout=None):
            return "some unrelated chatter\n", ""

    # _extract_pdf_toc imports subprocess inside the function, so patch the
    # module itself rather than an attribute on tools.write.
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProc())
    outcome = _extract_pdf_toc("ignored.pdf")
    assert outcome.status == "error"
    assert "not tagged" in outcome.detail
    assert "Expecting value" not in outcome.detail


def test_missing_pymupdf_still_reports_its_own_exit_code():
    """Both import names must fail before the child claims PyMuPDF is absent."""
    env = dict(os.environ)
    # Point the child at an empty directory as its only import root.
    empty = tempfile.mkdtemp()
    env["PYTHONPATH"] = empty
    env["PYTHONNOUSERSITE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-S", "-c", _TOC_CHILD_SCRIPT, "ignored.pdf"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    # Either PyMuPDF genuinely isn't importable under -S (then we get the
    # dedicated exit code) or it is (then the run fails on the missing file).
    assert proc.returncode != 0
    if proc.returncode == _TOC_EXIT_NO_PYMUPDF:
        assert _TOC_SENTINEL not in proc.stdout
