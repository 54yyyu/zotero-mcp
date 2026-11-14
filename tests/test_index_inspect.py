"""db-inspect on an item-keyed index (#826): record kinds from stored fields,
item types and --filter text from a reachable local Zotero database."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if sys.version_info >= (3, 14):
    pytest.skip("chromadb is incompatible with Python 3.14+", allow_module_level=True)
if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

import numpy as np

from zotero_mcp.chroma_client import ChromaClient
from zotero_mcp.index_inspect import inspect_index

NAME = "zotero_library"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def _passage(n):
    return {"chunk_index": n, "n_chunks": 2}


@pytest.fixture()
def client(home):
    client = ChromaClient(collection_name=NAME, persist_directory=str(home / "chroma"))
    records = [
        ("BIB1", {"date_modified": "d"}),
        ("ATT1", {"date_modified": "d", "file_stamp": "1:1"}),
        ("ATT1#0", _passage(0)),
        ("ATT1#1", _passage(1)),
        ("OLD1", {"title": "Legacy paper", "creators": "Doe, J.", "item_type": "book", "has_fulltext": True}),
    ]
    rng = np.random.default_rng(1)
    ids = [i for i, _ in records]
    client.upsert_embeddings(
        documents=[f"text of {i}" for i in ids],
        metadatas=[{"group_id": 0, "item_key": i.split("#")[0], **m} for i, m in records],
        ids=ids,
        embeddings=[rng.normal(size=4).astype(float).tolist() for _ in ids],
    )
    return client


class Reader:
    """Stands in for LocalZoteroReader.get_full_items."""

    def __init__(self):
        self.asked = []

    def get_full_items(self, keys, *, group_id=0, **_):
        self.asked.append(sorted(keys))
        data = {
            "BIB1": {
                "itemType": "journalArticle",
                "title": "Live title",
                "creators": [{"lastName": "Smith", "firstName": "A"}],
            },
            "ATT1": {"itemType": "attachment", "title": "Smith 2020.pdf", "creators": []},
        }
        return {k: {"key": k, "data": data[k]} for k in keys if k in data}


def _search(client, reader):
    return SimpleNamespace(chroma_client=client, _open_local_reader=lambda: reader)


def _run(client, reader, **kwargs):
    lines = []
    args = {"stats": False, "limit": 50, "filter_text": None, "show_documents": False, **kwargs}
    inspect_index(_search(client, reader), out=lines.append, **args)
    return "\n".join(lines)


def test_stats_count_record_kinds_from_stored_fields_and_item_types_from_zotero(client):
    reader = Reader()

    out = _run(client, reader, stats=True)

    assert "bibliographic: 1" in out
    assert "attachment: 1" in out
    assert "passage: 2" in out
    assert "legacy: 1" in out
    assert "journalArticle: 1" in out
    assert "attachment: 1" in out
    # Item types come from the live items; a passage resolves through its attachment's key.
    assert ["ATT1", "BIB1"] in reader.asked
    # Legacy records still report what they stored.
    assert "book: 1" in out


def test_stats_without_a_reachable_zotero_say_so_and_keep_the_kinds(client):
    out = _run(client, None, stats=True)

    assert "bibliographic: 1" in out
    assert "passage: 2" in out
    assert "local Zotero database is not reachable" in out
    assert "journalArticle" not in out
    assert "book: 1" in out


def test_a_reader_that_fails_counts_as_unreachable(client):
    class Broken(Reader):
        def get_full_items(self, keys, **_):
            raise RuntimeError("database is locked")

    out = _run(client, Broken(), stats=True)

    assert "local Zotero database is not reachable" in out
    assert "bibliographic: 1" in out


def test_filter_matches_live_title_and_creators(client):
    reader = Reader()

    by_title = _run(client, reader, filter_text="live title")
    by_creator = _run(client, reader, filter_text="smith")

    assert "Live title" in by_title and "Smith 2020.pdf" not in by_title
    assert "Live title" in by_creator and "Smith 2020.pdf" in by_creator
    assert "Legacy paper" not in by_creator


def test_filter_still_matches_a_legacy_record_on_its_stored_fields(client):
    out = _run(client, None, filter_text="legacy")

    assert "Legacy paper" in out
    assert "Doe, J." in out
    assert "cannot be matched without the local Zotero database" in out


def test_without_a_filter_every_record_is_listed_with_its_kind(client):
    out = _run(client, Reader())

    for expected in ("BIB1", "ATT1", "ATT1#0", "OLD1", "bibliographic", "attachment", "passage", "legacy"):
        assert expected in out
