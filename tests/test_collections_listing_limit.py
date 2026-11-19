"""zotero_get_collections builds its tree from every collection and says when
the listing is cut. It used to cut the name-sorted list to `limit` first, so a
subcollection whose parent fell past the cut vanished without a word."""

from unittest.mock import Mock

from zotero_mcp import library
from zotero_mcp.tools import retrieval


class _Backend:
    def __init__(self, collections):
        self._collections = collections

    def list_collections(self, *, include_trashed=False):
        return list(self._collections)


def _coll(key, name, parent=False):
    return {"key": key, "data": {"key": key, "name": name, "parentCollection": parent}}


def _listing(monkeypatch, collections, **kwargs):
    monkeypatch.setattr(library, "get_library_backend", lambda *a, **k: _Backend(collections))
    return retrieval.get_collections(ctx=Mock(), **kwargs)


def test_child_is_listed_when_its_parent_sorts_after_the_cut(monkeypatch):
    # Name-sorted, as the SQLite backend returns them: the parent is last.
    collections = [_coll("CHILD001", "Alpha child", parent="AAPARENT")]
    collections += [_coll(f"FILL{n:04d}", f"Filler {n:03d}") for n in range(4)]
    collections += [_coll("AAPARENT", "Zeta parent")]

    out = _listing(monkeypatch, collections, limit=5)

    assert "  - **Alpha child** (Key: CHILD001)" in out
    assert out.count("(Key: ") == 5
    assert "Showing 5 of 6 collections" in out


def test_no_notice_when_everything_fits(monkeypatch):
    collections = [_coll("PARENT01", "Parent"), _coll("CHILD001", "Child", parent="PARENT01")]
    out = _listing(monkeypatch, collections)
    assert out.count("(Key: ") == 2
    assert "Showing" not in out
