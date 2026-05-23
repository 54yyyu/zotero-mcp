"""Regression tests for #233: trashed-collection visibility & validation.

Three behaviors covered:

- ``zotero_get_collections(include_trashed=True)`` shows trashed
  collections (annotated as such), while the default ``False`` preserves
  the existing tree.
- ``zotero_search_collections(include_trashed=True)`` matches trashed
  collections too.
- ``zotero_manage_collections`` pre-validates each ``add_to`` /
  ``remove_from`` key and refuses if the key is in the Trash or missing,
  instead of silently filing items into an invisible bucket.
"""

from unittest.mock import MagicMock

from conftest import DummyContext, FakeZotero, _FakeResponse

from zotero_mcp import server
from zotero_mcp.tools import _helpers

# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


def _coll(key: str, name: str, deleted: bool = False, parent: str | None = None) -> dict:
    data = {"name": name, "key": key, "parentCollection": parent or False}
    if deleted:
        data["deleted"] = True
    return {"key": key, "version": 1, "data": data}


class _CollectionFakeZotero(FakeZotero):
    """FakeZotero with collection() lookups, trash fetch, and addto support."""

    def __init__(self, live: list[dict], trashed: list[dict]):
        super().__init__()
        self._collections = live
        self._trashed = trashed
        self._all_by_key = {c["key"]: c for c in (live + trashed)}
        self.addto_calls: list[tuple[str, str]] = []

    def collection(self, key, **_kw):
        if key in self._all_by_key:
            return self._all_by_key[key]
        raise KeyError(key)

    # pyzotero's _retrieve_data returns an httpx.Response-like object.
    # The helper only calls .json() on it.
    def _retrieve_data(self, request: str, params=None):
        if request.endswith("/collections/trash"):
            resp = MagicMock()
            resp.json.return_value = self._trashed
            return resp
        raise NotImplementedError(request)

    def addto_collection(self, collection_key, item, **_kw):
        key = item["key"] if isinstance(item, dict) else item
        self.addto_calls.append((collection_key, key))
        return _FakeResponse(204)

    def deletefrom_collection(self, collection_key, item, **_kw):
        return _FakeResponse(204)


def _patch_clients(monkeypatch, zot):
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: zot)
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (zot, zot)
    )


# ---------------------------------------------------------------------------
# fetch_trashed_collections / is_collection_trashed helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_fetch_trashed_returns_list(self):
        z = _CollectionFakeZotero(
            live=[_coll("LIVE0000", "Live")],
            trashed=[_coll("DEAD0000", "Dead", deleted=True)],
        )
        out = _helpers.fetch_trashed_collections(z)
        assert [c["key"] for c in out] == ["DEAD0000"]

    def test_fetch_trashed_swallows_errors(self):
        class Boom:
            library_id = "1"
            library_type = "users"

            def _retrieve_data(self, *_a, **_kw):
                raise RuntimeError("offline")

        assert _helpers.fetch_trashed_collections(Boom()) == []

    def test_is_collection_trashed_returns_true(self):
        z = _CollectionFakeZotero(
            live=[],
            trashed=[_coll("DEAD0000", "Dead", deleted=True)],
        )
        assert _helpers.is_collection_trashed(z, "DEAD0000") is True

    def test_is_collection_trashed_returns_false_for_live(self):
        z = _CollectionFakeZotero(
            live=[_coll("LIVE0000", "Live")], trashed=[]
        )
        assert _helpers.is_collection_trashed(z, "LIVE0000") is False

    def test_is_collection_trashed_returns_none_for_missing(self):
        z = _CollectionFakeZotero(live=[], trashed=[])
        assert _helpers.is_collection_trashed(z, "GONE0000") is None


# ---------------------------------------------------------------------------
# zotero_manage_collections validation (the main user-visible bug)
# ---------------------------------------------------------------------------


class TestManageCollectionsValidation:
    def test_rejects_trashed_collection_in_add_to(self, monkeypatch):
        z = _CollectionFakeZotero(
            live=[_coll("LIVE0000", "Live")],
            trashed=[_coll("DEAD0000", "Trashed", deleted=True)],
        )
        _patch_clients(monkeypatch, z)

        result = server.manage_collections(
            item_keys=["ITEM0001"],
            add_to=["DEAD0000"],
            ctx=DummyContext(),
        )
        assert "Trash" in result
        assert "DEAD0000" in result
        # And nothing was actually filed.
        assert z.addto_calls == []

    def test_rejects_missing_collection_in_add_to(self, monkeypatch):
        z = _CollectionFakeZotero(live=[], trashed=[])
        _patch_clients(monkeypatch, z)

        result = server.manage_collections(
            item_keys=["ITEM0001"],
            add_to=["NONESUCH"],
            ctx=DummyContext(),
        )
        assert "not found" in result
        assert "NONESUCH" in result
        assert z.addto_calls == []

    def test_live_collection_proceeds_normally(self, monkeypatch):
        z = _CollectionFakeZotero(
            live=[_coll("LIVE0000", "Live")], trashed=[]
        )
        _patch_clients(monkeypatch, z)

        result = server.manage_collections(
            item_keys=["ITEM0001"],
            add_to=["LIVE0000"],
            ctx=DummyContext(),
        )
        assert "Added ITEM0001 to LIVE0000" in result
        assert z.addto_calls == [("LIVE0000", "ITEM0001")]

    def test_validation_includes_both_add_and_remove_keys(self, monkeypatch):
        z = _CollectionFakeZotero(
            live=[_coll("LIVE0000", "Live")],
            trashed=[_coll("DEAD0000", "Dead", deleted=True)],
        )
        _patch_clients(monkeypatch, z)

        result = server.manage_collections(
            item_keys=["ITEM0001"],
            add_to=["LIVE0000"],
            remove_from=["DEAD0000"],
            ctx=DummyContext(),
        )
        # DEAD0000 is in remove_from → still rejected.
        assert "Trash" in result
        assert "DEAD0000" in result
        # And nothing was filed because validation gates ALL collection keys.
        assert z.addto_calls == []


# ---------------------------------------------------------------------------
# zotero_get_collections / zotero_search_collections include_trashed
# ---------------------------------------------------------------------------


class TestGetCollections:
    def test_default_excludes_trashed(self, monkeypatch):
        z = _CollectionFakeZotero(
            live=[_coll("LIVE0000", "Live")],
            trashed=[_coll("DEAD0000", "Dead", deleted=True)],
        )
        _patch_clients(monkeypatch, z)

        result = server.get_collections(ctx=DummyContext())
        assert "Live" in result
        assert "Dead" not in result

    def test_include_trashed_shows_them(self, monkeypatch):
        z = _CollectionFakeZotero(
            live=[_coll("LIVE0000", "Live")],
            trashed=[_coll("DEAD0000", "Dead", deleted=True)],
        )
        _patch_clients(monkeypatch, z)

        result = server.get_collections(include_trashed=True, ctx=DummyContext())
        assert "Live" in result
        assert "Dead" in result
        assert "*[trashed]*" in result


class TestSearchCollections:
    def test_search_excludes_trashed_by_default(self, monkeypatch):
        z = _CollectionFakeZotero(
            live=[_coll("LIVE0000", "Unrelated Topic")],
            trashed=[_coll("DEAD0000", "Important Stuff", deleted=True)],
        )
        _patch_clients(monkeypatch, z)

        result = server.search_collections(query="important", ctx=DummyContext())
        assert "No collections found matching" in result
        assert "DEAD0000" not in result

    def test_search_with_include_trashed_returns_it(self, monkeypatch):
        z = _CollectionFakeZotero(
            live=[],
            trashed=[_coll("DEAD0000", "Important Stuff", deleted=True)],
        )
        _patch_clients(monkeypatch, z)

        result = server.search_collections(
            query="important", include_trashed=True, ctx=DummyContext()
        )
        assert "Important Stuff" in result
        assert "DEAD0000" in result
        assert "*[trashed]*" in result
