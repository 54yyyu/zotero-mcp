"""An attachment with only some of its passages has a passage gap (#826, fix round 2b).

A failed row or request can leave ``PDFAAAAA#3`` missing while ``#0..#2`` are
there; the next explicit update writes that attachment's passages and nothing
else. Real Chroma in a temp dir, a recording embedder, a fake reader; HOME,
Path.home and the config path point into tmp_path.
"""

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 14):
    pytest.skip("chromadb needs pydantic v1 paths on 3.14+", allow_module_level=True)

from _scan_fakes import target  # noqa: E402
from test_item_keyed_local_scan import Item  # noqa: E402
from test_rekeyed_index_updates import ATT, ITEM, NAME, PASSAGES, Reader, World  # noqa: E402

from zotero_mcp import semantic_search  # noqa: E402
from zotero_mcp.chroma_client import ChromaClient  # noqa: E402


@pytest.fixture()
def world(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    return World(monkeypatch, tmp_path)


def _reader():
    return Reader([Item(ITEM, 1)], targets={1: [target(ATT)]})


def test_a_missing_passage_is_refilled_by_the_next_explicit_update(world):
    world.update(_reader())
    passages = sorted(world.stored(PASSAGES))
    assert len(passages) == 5
    ChromaClient(collection_name=NAME, persist_directory=world.directory).client.get_collection(PASSAGES).delete(
        ids=[f"{ATT}#3"]
    )
    world.embedded.clear()

    world.update(_reader())

    assert sorted(world.stored(PASSAGES)) == passages
    assert world.embedded, "the attachment's passages were written again"
    assert all("Body sentence" in text for text in world.embedded), "and nothing else: no record of the item"
    assert ITEM in world.stored(NAME)
