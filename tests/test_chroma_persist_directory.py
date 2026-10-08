"""``semantic_search.persist_directory`` moves the ChromaDB index (#617).

ChromaDB fails to write its HNSW files under a non-ASCII path on Windows, and
the index used to be pinned to ``~/.config/zotero-mcp/chroma_db``. The writer
and status round trip is in ``test_chroma_client_real.py``.
"""

import builtins
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from zotero_mcp import setup_helper
from zotero_mcp.config import resolve_chroma_dir


def test_resolve_chroma_dir_defaults_under_home(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert resolve_chroma_dir(None) == tmp_path / ".config" / "zotero-mcp" / "chroma_db"
    assert resolve_chroma_dir("~/idx") == Path("~/idx").expanduser()


def test_setup_keeps_persist_directory(monkeypatch):
    """Setup rebuilds ``semantic_search`` from its prompts, and none of them
    asks for the index path."""
    answers = iter([
        "n",  # don't keep the existing configuration
        "1",  # default embedding model
        "1",  # manual updates
        "",  # default PDF max pages
        "",  # default extraction workers
        "",  # auto-detect DB path
    ])
    monkeypatch.setattr(builtins, "input", lambda *args: next(answers))

    config, _db_path = setup_helper.setup_semantic_search(
        {"embedding_model": "default", "persist_directory": "D:/zotero-mcp/chroma_db"}
    )

    assert config["persist_directory"] == "D:/zotero-mcp/chroma_db"


@pytest.mark.parametrize(
    "cwd, path, warns",
    [
        ("work", "chroma_db", False),
        ("work", "王林澜", True),
        # A relative path counts from the working directory.
        ("王林澜", "chroma_db", True),
    ],
)
def test_windows_warns_about_a_non_ascii_index_path(monkeypatch, caplog, tmp_path, cwd, path, warns):
    chroma_client = pytest.importorskip("zotero_mcp.chroma_client")
    (tmp_path / cwd).mkdir()
    monkeypatch.chdir(tmp_path / cwd)

    class Opened(Exception):
        pass

    def stop_before_opening(**kwargs):
        raise Opened

    monkeypatch.setattr(chroma_client, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(chroma_client.chromadb, "PersistentClient", stop_before_opening)

    with caplog.at_level(logging.WARNING, logger=chroma_client.__name__), pytest.raises(Opened):
        chroma_client.ChromaClient(persist_directory=path)

    assert ("persist_directory" in caplog.text) is warns
