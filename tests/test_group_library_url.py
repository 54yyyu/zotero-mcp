"""Regression tests for #272: group-library URL normalization.

Verifies that write-path code does not leak the singular ``group`` form of
``library_type`` into pyzotero clients, which builds ``/group/{id}/...`` URLs
that 404 against the Zotero Web API (which expects ``/groups/{id}/...``).
"""

import pytest

from zotero_mcp import server
from zotero_mcp.tools import _helpers

# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, library_id="0", library_type="users"):
        self.library_id = library_id
        self.library_type = library_type


class TestApplyLibraryOverride:
    def test_no_override_is_noop(self):
        zot = _FakeClient(library_id="1", library_type="users")
        _helpers.apply_library_override(zot, None)
        assert zot.library_id == "1"
        assert zot.library_type == "users"

    def test_empty_override_is_noop(self):
        zot = _FakeClient(library_id="1", library_type="users")
        _helpers.apply_library_override(zot, {})
        assert zot.library_id == "1"
        assert zot.library_type == "users"

    def test_group_singular_is_normalized_to_plural(self):
        zot = _FakeClient()
        _helpers.apply_library_override(zot, {"library_id": "5910265", "library_type": "group"})
        assert zot.library_id == "5910265"
        assert zot.library_type == "groups"

    def test_user_singular_is_normalized_to_plural(self):
        zot = _FakeClient()
        _helpers.apply_library_override(zot, {"library_id": "99", "library_type": "user"})
        assert zot.library_type == "users"

    def test_already_plural_is_unchanged(self):
        zot = _FakeClient()
        _helpers.apply_library_override(zot, {"library_id": "99", "library_type": "groups"})
        assert zot.library_type == "groups"

    def test_library_id_only_keeps_existing_library_type(self):
        zot = _FakeClient(library_type="users")
        _helpers.apply_library_override(zot, {"library_id": "42"})
        assert zot.library_id == "42"
        assert zot.library_type == "users"


# ---------------------------------------------------------------------------
# End-to-end: confirm switch_library + create_note routes to /groups/...
# ---------------------------------------------------------------------------


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


class FakeGroupZotero:
    """Captures library_type when create_items is invoked."""

    def __init__(self):
        self.library_id = "0"
        self.library_type = "users"
        self.create_calls = []

    def item(self, _item_key):
        return {"data": {"title": "Parent Item"}}

    def create_items(self, items):
        self.create_calls.append({"library_id": self.library_id, "library_type": self.library_type})
        return {"success": {"0": "NOTEKEY01"}}


@pytest.fixture
def force_local_mode(monkeypatch):
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: True)
    monkeypatch.setattr("zotero_mcp.tools.annotations._utils.is_local_mode", lambda: True)


def test_create_note_against_group_uses_plural_url(monkeypatch, force_local_mode):
    """Switching to a group library must route create_note via /groups/{id}/items."""
    from zotero_mcp import client as zclient

    fake = FakeGroupZotero()
    monkeypatch.setattr(zclient, "get_zotero_client", lambda: fake)
    monkeypatch.setattr(zclient, "get_web_zotero_client", lambda: fake)

    # Simulate the runtime override left by zotero_switch_library(..., "group").
    zclient.set_active_library(library_id="5910265", library_type="group")
    try:
        result = server.create_note(
            item_key="ITEM0001",
            note_title="t",
            note_text="body",
            ctx=DummyContext(),
        )
    finally:
        zclient.clear_active_library()

    assert "Successfully created note" in result, result
    assert fake.create_calls, "create_items was not called"
    call = fake.create_calls[0]
    assert call["library_id"] == "5910265"
    assert call["library_type"] == "groups", (
        f"library_type must be normalized to plural before the POST (got {call['library_type']!r}); see issue #272"
    )
