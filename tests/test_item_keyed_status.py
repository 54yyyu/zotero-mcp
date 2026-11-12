"""Status counts an item-keyed index (#826) by record kind and says what legacy records remain.

Real persisted indexes in temp directories, with given vectors, so nothing embeds.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 14):
    pytest.skip("chromadb is incompatible with Python 3.14+", allow_module_level=True)
if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

import numpy as np

from zotero_mcp.chroma_client import ChromaClient, level_status_rows, read_collection_status

NAME = "zotero_library"
BIB = {"date_modified": "d"}
ATT = {"date_modified": "d", "file_stamp": "1:1", "file_hash": "h"}
LEGACY = {"title": "Old", "has_fulltext": True}


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("ZOTERO_EMBEDDING_MODEL", raising=False)
    return home


@pytest.fixture()
def index(home):
    directory = home / "chroma"
    config = home / ".config" / "zotero-mcp" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps({"semantic_search": {"persist_directory": str(directory), "embedding_model": "default"}})
    )
    return str(directory), str(config)


def _write(directory, records):
    client = ChromaClient(collection_name=NAME, persist_directory=directory)
    rng = np.random.default_rng(1)
    ids = [i for i, _ in records]
    client.upsert_embeddings(
        documents=[f"text of {i}" for i in ids],
        metadatas=[{"group_id": 0, "item_key": i.split("#")[0], **m} for i, m in records],
        ids=ids,
        embeddings=[rng.normal(size=4).astype(float).tolist() for _ in ids],
    )


def _passages(key, n=2, **extra):
    return [(f"{key}#{i}", {"chunk_index": i, "n_chunks": n, **extra}) for i in range(n)]


def test_a_new_shape_index_is_counted_per_record_kind(index):
    directory, config = index
    _write(directory, [("BIB1", BIB), ("BIB2", BIB), ("ATT1", ATT), ("ATT2", ATT)] + _passages("ATT1"))

    info = read_collection_status(config)

    assert (info["article_records"], info["passage_records"]) == (4, 2)
    assert info["bibliographic_records"] == 2
    assert info["attachment_records"] == 2
    assert info["article_items"] == 2  # attachments with an article record
    assert info["passage_items"] == 1  # attachments with passages
    assert info["legacy_records"] == 0
    assert info["legacy_passages"] == 0
    assert legacy_note(info) == []


def legacy_note(info):
    from zotero_mcp.chroma_client import legacy_status_note

    return legacy_status_note(info)


def test_a_mixed_legacy_and_new_index(index):
    directory, config = index
    _write(
        directory,
        [("BIB1", BIB), ("ATT1", ATT), ("OLD1", LEGACY), ("OLD2", {"title": "No text", "has_fulltext": False})]
        + _passages("ATT1")
        + _passages("OLD1", title="Old"),
    )

    info = read_collection_status(config)

    assert info["bibliographic_records"] == 1
    assert info["attachment_records"] == 1
    assert info["legacy_records"] == 2
    assert info["article_items"] == 2  # ATT1, and OLD1 (built with full text)
    assert info["passage_items"] == 2
    assert info["legacy_passages"] == 2
    assert info["legacy_passage_items"] == 1
    note = " ".join(legacy_note(info))
    assert "2 passages of 1 items" in note
    assert "migrate-index" in note


def test_the_rows_say_items_only_while_legacy_records_exist(index):
    directory, config = index
    _write(directory, [("ATT1", ATT)] + _passages("ATT1"))
    labels = [label for label, _ in level_status_rows(read_collection_status(config))]
    assert "Attachments with passages" in labels
    assert not any(label.startswith("Legacy") for label in labels)

    _write(directory, [("OLD1", LEGACY)] + _passages("OLD1", title="Old"))
    labels = [label for label, _ in level_status_rows(read_collection_status(config))]
    assert "Items with passages" in labels
    assert "Legacy records" in labels
    assert "Legacy passages" in labels


def _status_tool():
    from zotero_mcp.tools import search as search_tools

    tool = search_tools.get_search_database_status
    fn = tool.fn if hasattr(tool, "fn") else tool

    class Ctx:
        def info(self, *_):
            pass

        def error(self, *_):
            pass

    return fn(ctx=Ctx())


def test_the_status_tool_lists_the_kinds_and_the_legacy_note_only_when_legacy_passages_exist(index):
    directory, _config = index
    _write(directory, [("BIB1", BIB), ("ATT1", ATT)] + _passages("ATT1"))

    out = _status_tool()

    assert "**Bibliographic Records:** 1" in out
    assert "**Attachment Records:** 1" in out
    assert "**Attachments with Passages:** 1" in out
    assert "migrate-index" not in out

    _write(directory, [("OLD1", LEGACY)] + _passages("OLD1", title="Old"))

    out = _status_tool()

    assert "## Legacy Records" in out
    assert "zotero-mcp migrate-index" in out
    assert "**Legacy Passages:** 2" in out


@pytest.mark.parametrize("which", ["db-status", "zotero-cli"])
def test_the_clis_show_the_same(index, monkeypatch, capsys, which):
    import zotero_mcp.cli as cli
    import zotero_mcp.cli_standalone as standalone
    import zotero_mcp.semantic_search as semantic_search

    directory, config = index
    _write(directory, [("BIB1", BIB), ("OLD1", LEGACY)] + _passages("OLD1", title="Old"))
    info = ChromaClient(collection_name=NAME, persist_directory=directory).get_collection_info()
    search = type(
        "S",
        (),
        {
            "get_database_status": lambda self: {
                "collection_info": info,
                "update_config": {},
                "openai_batch": {},
                "chunking": {},
            }
        },
    )()
    monkeypatch.setattr(cli, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(standalone, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda *a, **k: search)
    if which == "db-status":
        monkeypatch.setattr(sys, "argv", ["zotero-mcp", "db-status", "--config-path", config])
        cli.main()
    else:
        monkeypatch.setattr(sys, "argv", ["zotero-cli", "db", "status", "--config-path", config])
        standalone.main()
    out = capsys.readouterr().out

    assert "Bibliographic records: 1" in out
    assert "Legacy records: 1" in out
    assert "Items with passages: 1" in out
    assert "zotero-mcp migrate-index" in out
