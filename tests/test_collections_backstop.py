"""Regression tests for #235: collections parameter routing on zotero_add_by_doi.

The reported symptom (silent failure when ``collections="KEY"``) is not a
normalization bug — ``_normalize_str_list_input`` already handles the string
form — but a routing bug downstream: pyzotero's atomic ``item["collections"]``
filing on ``create_items`` intermittently no-ops, leaving the new item in
My Library instead of the requested collection.

The fix is a deterministic backstop: after create, read the item back, diff
the actual collection membership against the requested set, and explicitly
``addto_collection`` for any that didn't take.
"""

from unittest.mock import MagicMock

from conftest import DummyContext, FakeZotero, _FakeResponse

from zotero_mcp import server
from zotero_mcp.tools import _helpers


def _make_crossref_response():
    msg = {
        "type": "journal-article",
        "title": ["Some Paper"],
        "DOI": "10.1234/test",
        "author": [{"given": "A", "family": "Author"}],
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "ok", "message": msg}
    resp.raise_for_status = MagicMock()
    return resp


class _RecordingZotero(FakeZotero):
    """FakeZotero that records addto_collection calls and lets the test
    decide whether pyzotero's atomic filing 'worked' on create_items.
    """

    def __init__(self, atomic_filing_works: bool = True):
        super().__init__()
        self._atomic = atomic_filing_works
        self.addto_calls: list[tuple[str, str]] = []
        self._created_items: dict[str, dict] = {}

    def create_items(self, items, **kwargs):
        self.created.extend(items)
        result = {}
        for i, item in enumerate(items):
            key = f"KEY{i:04d}"
            result[str(i)] = key
            # Stash the created item so item(key) can read it back.
            stored_collections = (
                list(item.get("collections") or []) if self._atomic else []
            )
            self._created_items[key] = {
                "key": key,
                "version": 1,
                "data": {**item, "key": key, "collections": stored_collections},
            }
        return {"success": result, "successful": {}, "failed": {}}

    def item(self, item_key):
        if item_key in self._created_items:
            return self._created_items[item_key]
        return super().item(item_key)

    def addto_collection(self, collection_key, item, **kwargs):
        key = item["key"] if isinstance(item, dict) else item
        self.addto_calls.append((collection_key, key))
        # Update the stored item so subsequent reads see the new membership.
        stored = self._created_items.get(key)
        if stored:
            cols = stored["data"].setdefault("collections", [])
            if collection_key not in cols:
                cols.append(collection_key)
        return _FakeResponse(204)


# ---------------------------------------------------------------------------
# ensure_collection_membership helper
# ---------------------------------------------------------------------------


class TestEnsureCollectionMembership:
    def test_empty_keys_is_noop(self):
        z = _RecordingZotero()
        assert _helpers.ensure_collection_membership(z, "X", []) == []
        assert z.addto_calls == []

    def test_when_atomic_filing_worked_no_addto_called(self, monkeypatch):
        z = _RecordingZotero(atomic_filing_works=True)
        z.create_items([{"itemType": "journalArticle", "collections": ["A75DWWBH"]}])
        failed = _helpers.ensure_collection_membership(z, "KEY0000", ["A75DWWBH"])
        assert failed == []
        assert z.addto_calls == []  # nothing to do

    def test_when_atomic_filing_failed_addto_is_called(self):
        z = _RecordingZotero(atomic_filing_works=False)
        z.create_items([{"itemType": "journalArticle", "collections": ["A75DWWBH"]}])
        failed = _helpers.ensure_collection_membership(z, "KEY0000", ["A75DWWBH"])
        assert failed == []
        assert z.addto_calls == [("A75DWWBH", "KEY0000")]

    def test_partial_membership_only_files_missing(self):
        z = _RecordingZotero(atomic_filing_works=False)
        z.create_items([{"itemType": "journalArticle", "collections": ["A75DWWBH"]}])
        # Pre-seed one membership so only the second key needs addto.
        z._created_items["KEY0000"]["data"]["collections"] = ["A75DWWBH"]
        failed = _helpers.ensure_collection_membership(
            z, "KEY0000", ["A75DWWBH", "BBBBBBBB"]
        )
        assert failed == []
        assert z.addto_calls == [("BBBBBBBB", "KEY0000")]


# ---------------------------------------------------------------------------
# add_by_doi end-to-end
# ---------------------------------------------------------------------------


def _patch_write_client(monkeypatch, zot):
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (zot, zot)
    )
    monkeypatch.setattr("requests.get", lambda *a, **kw: _make_crossref_response())
    # Bypass the OA-PDF lookup (it would hit Unpaywall / Semantic Scholar over the network).
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._try_attach_oa_pdf",
        lambda *a, **kw: "skipped (test)",
    )


def test_add_by_doi_string_collection_routes_correctly(monkeypatch):
    """The bare-string ``collections="KEY"`` form must produce the same routing
    as the array form (#235)."""
    z = _RecordingZotero(atomic_filing_works=True)
    z._collections = [
        {"key": "A75DWWBH", "data": {"name": "Target", "parentCollection": False}},
    ]
    _patch_write_client(monkeypatch, z)

    result = server.add_by_doi(
        doi="10.1234/test", collections="A75DWWBH", ctx=DummyContext()
    )

    item = z.created[0]
    assert item["collections"] == ["A75DWWBH"]
    assert "Filed in ['A75DWWBH']" in result


def test_add_by_doi_string_collection_with_atomic_failure_backstops(monkeypatch):
    """If pyzotero's atomic filing on ``create_items`` no-ops, the backstop
    must explicitly ``addto_collection`` so the item still lands correctly."""
    z = _RecordingZotero(atomic_filing_works=False)
    z._collections = [
        {"key": "A75DWWBH", "data": {"name": "Target", "parentCollection": False}},
    ]
    _patch_write_client(monkeypatch, z)

    result = server.add_by_doi(
        doi="10.1234/test", collections="A75DWWBH", ctx=DummyContext()
    )

    assert z.addto_calls == [("A75DWWBH", "KEY0000")]
    assert "Filed in ['A75DWWBH']" in result


def test_add_by_doi_no_collections_reports_my_library(monkeypatch):
    z = _RecordingZotero()
    _patch_write_client(monkeypatch, z)

    result = server.add_by_doi(doi="10.1234/test", ctx=DummyContext())

    assert z.addto_calls == []
    assert "My Library (no collection)" in result
