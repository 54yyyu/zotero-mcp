"""#621: clean_html decodes HTML entities. #642: nothing imports PyMuPDF as
``fitz`` (its deprecation notice goes to stdout and breaks ``--json``)."""

import ast
import os
import pathlib
import subprocess
import sys

import pytest

from zotero_mcp.utils import clean_html

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "zotero_mcp"


def test_clean_html_decodes_entities_after_stripping_tags():
    html = '<div class="csl-entry">Baur, D. G., &amp; Todorova, N. (2018). &#8211; &lt;b&gt;</div>'
    assert clean_html(html) == "Baur, D. G., & Todorova, N. (2018). – <b>"


def test_no_module_imports_fitz_by_its_deprecated_name():
    allowed = {"_pymupdf.py", "extract.py"}  # both only after `import pymupdf` failed
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.name in allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import) and any(a.name == "fitz" for a in node.names):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert offenders == []


def test_loading_pymupdf_prints_nothing_to_stdout():
    pytest.importorskip("pymupdf")
    env = {**os.environ, "PYTHONPATH": str(SRC.parent)}  # this checkout, not an installed copy
    out = subprocess.run([sys.executable, "-c", "from zotero_mcp._pymupdf import fitz; fitz.open"],
                         capture_output=True, text=True, timeout=60, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout == ""
