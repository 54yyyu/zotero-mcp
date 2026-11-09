"""semantic_search loads tiktoken's cl100k_base encoding on first use, not at import.

The module is imported on the main thread before the server answers its first
request on Windows (#485), and the encoding (about 100k BPE ranks, or a
download when it is not cached) is needed only by ``_truncate_to_tokens``.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("chromadb")
tiktoken = pytest.importorskip("tiktoken")

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_import_does_not_load_the_encoding():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {SRC!r})\n"
        "import tiktoken\n"
        "calls = []\n"
        "real = tiktoken.get_encoding\n"
        "tiktoken.get_encoding = lambda name: calls.append(name) or real(name)\n"
        "import zotero_mcp.semantic_search as s\n"
        "print(len(calls), s._tokenizer is None)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["0", "True"]


def test_first_use_loads_it_once(monkeypatch):
    from zotero_mcp import semantic_search as s

    calls = []
    real = tiktoken.get_encoding
    monkeypatch.setattr(s, "_tokenizer", None)
    monkeypatch.setattr(s.tiktoken, "get_encoding", lambda name: calls.append(name) or real(name))

    text = " ".join(["word"] * 50)
    first = s._truncate_to_tokens(text, max_tokens=5)
    second = s._truncate_to_tokens(text, max_tokens=5)

    assert first == second == real("cl100k_base").decode(real("cl100k_base").encode(text)[:5])
    assert calls == ["cl100k_base"]


def test_a_failed_load_falls_back_to_characters_and_is_not_retried(monkeypatch):
    from zotero_mcp import semantic_search as s

    calls = []

    def broken(name):
        calls.append(name)
        raise OSError("no network to fetch the BPE file")

    monkeypatch.setattr(s, "_tokenizer", None)
    monkeypatch.setattr(s.tiktoken, "get_encoding", broken)

    text = "x" * 100
    assert s._truncate_to_tokens(text, max_tokens=10) == "x" * 20
    assert s._truncate_to_tokens(text, max_tokens=10) == "x" * 20
    assert calls == ["cl100k_base"]
