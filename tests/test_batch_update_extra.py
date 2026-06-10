"""Tests for zotero_batch_update_extra (issue #232).

Batch upsert / removal of `Key: value` lines in the Extra field across
multiple items, parallel to zotero_batch_update_tags.
"""

from zotero_mcp import server
from zotero_mcp.tools.write import _apply_extra_edits


class DummyContext:
    def info(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


class FakeZoteroForExtra:
    """Fake client serving items by key and recording updates."""

    def __init__(self, items):
        self._items = {it["key"]: it for it in items}
        self.updated = []

    def item(self, item_key):
        it = self._items.get(item_key)
        if it is None:
            raise KeyError(item_key)
        return it

    def update_item(self, item):
        self.updated.append(item)
        return True


# ---------------------------------------------------------------------------
# Pure line-editing helper
# ---------------------------------------------------------------------------


def test_apply_extra_edits_upserts_into_empty_extra():
    new_extra, changed = _apply_extra_edits(
        "", set_keys={"tex.otscore": "2"}, remove_keys=[], replace=False
    )
    assert new_extra == "tex.otscore: 2"
    assert changed is True


def test_apply_extra_edits_replaces_existing_key_case_insensitive():
    extra = "Citation Key: smith2020\ntex.otscore: 1\nfree-form note line"
    new_extra, changed = _apply_extra_edits(
        extra, set_keys={"TEX.OTSCORE": "2"}, remove_keys=[], replace=False
    )
    assert new_extra == "Citation Key: smith2020\nTEX.OTSCORE: 2\nfree-form note line"
    assert changed is True


def test_apply_extra_edits_appends_new_key_at_end():
    extra = "Citation Key: smith2020"
    new_extra, changed = _apply_extra_edits(
        extra,
        set_keys={"tex.provenance": "coursework:waldstreicher-80010"},
        remove_keys=[],
        replace=False,
    )
    assert new_extra == (
        "Citation Key: smith2020\n"
        "tex.provenance: coursework:waldstreicher-80010"
    )
    assert changed is True


def test_apply_extra_edits_removes_matching_lines():
    extra = "tex.otscore: 2\nCitation Key: smith2020\nTex.OtScore: 3"
    new_extra, changed = _apply_extra_edits(
        extra, set_keys={}, remove_keys=["tex.otscore"], replace=False
    )
    assert new_extra == "Citation Key: smith2020"
    assert changed is True


def test_apply_extra_edits_no_change_when_nothing_matches():
    extra = "Citation Key: smith2020"
    new_extra, changed = _apply_extra_edits(
        extra, set_keys={}, remove_keys=["tex.otscore"], replace=False
    )
    assert new_extra == extra
    assert changed is False


def test_apply_extra_edits_no_change_when_value_already_set():
    extra = "tex.otscore: 2"
    new_extra, changed = _apply_extra_edits(
        extra, set_keys={"tex.otscore": "2"}, remove_keys=[], replace=False
    )
    assert new_extra == extra
    assert changed is False


def test_apply_extra_edits_replace_rebuilds_from_set_keys():
    extra = "Citation Key: smith2020\nfree-form note line"
    new_extra, changed = _apply_extra_edits(
        extra, set_keys={"tex.otscore": "2"}, remove_keys=[], replace=True
    )
    assert new_extra == "tex.otscore: 2"
    assert changed is True


def test_apply_extra_edits_preserves_freeform_lines():
    extra = "this is not a key-value line\ntex.otscore: 1"
    new_extra, changed = _apply_extra_edits(
        extra, set_keys={"tex.otscore": "2"}, remove_keys=[], replace=False
    )
    assert new_extra == "this is not a key-value line\ntex.otscore: 2"
    assert changed is True


# ---------------------------------------------------------------------------
# Tool-level behavior
# ---------------------------------------------------------------------------


def _make_items():
    return [
        {
            "key": "ITEM0001",
            "data": {
                "itemType": "journalArticle",
                "extra": "Citation Key: smith2020",
            },
        },
        {
            "key": "ITEM0002",
            "data": {"itemType": "preprint", "extra": ""},
        },
        {
            "key": "ATTACH01",
            "data": {"itemType": "attachment", "extra": ""},
        },
    ]


def _setup(monkeypatch, items):
    fake = FakeZoteroForExtra(items)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake)
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)
    return fake


def test_batch_update_extra_updates_multiple_items(monkeypatch):
    fake = _setup(monkeypatch, _make_items())

    result = server.batch_update_extra(
        item_keys=["ITEM0001", "ITEM0002"],
        set_keys={"tex.otscore": "2"},
        ctx=DummyContext(),
    )

    assert len(fake.updated) == 2
    for it in fake.updated:
        assert "tex.otscore: 2" in it["data"]["extra"]
    # ITEM0001 keeps its existing line
    assert "Citation Key: smith2020" in fake.updated[0]["data"]["extra"]
    assert "Items updated: 2" in result


def test_batch_update_extra_removes_keys(monkeypatch):
    items = _make_items()
    items[0]["data"]["extra"] = "Citation Key: smith2020\ntex.otscore: 2"
    fake = _setup(monkeypatch, items)

    result = server.batch_update_extra(
        item_keys=["ITEM0001"],
        remove_keys=["tex.otscore"],
        ctx=DummyContext(),
    )

    assert len(fake.updated) == 1
    assert fake.updated[0]["data"]["extra"] == "Citation Key: smith2020"
    assert "Items updated: 1" in result


def test_batch_update_extra_skips_attachments(monkeypatch):
    fake = _setup(monkeypatch, _make_items())

    result = server.batch_update_extra(
        item_keys=["ITEM0001", "ATTACH01"],
        set_keys={"tex.otscore": "2"},
        ctx=DummyContext(),
    )

    assert len(fake.updated) == 1
    assert "Items skipped: 1" in result


def test_batch_update_extra_accepts_json_string_set_keys(monkeypatch):
    fake = _setup(monkeypatch, _make_items())

    server.batch_update_extra(
        item_keys='["ITEM0002"]',
        set_keys='{"tex.otscore": "2"}',
        ctx=DummyContext(),
    )

    assert len(fake.updated) == 1
    assert fake.updated[0]["data"]["extra"] == "tex.otscore: 2"


def test_batch_update_extra_requires_item_keys(monkeypatch):
    _setup(monkeypatch, _make_items())

    result = server.batch_update_extra(
        item_keys=[], set_keys={"tex.otscore": "2"}, ctx=DummyContext()
    )

    assert result.startswith("Error")


def test_batch_update_extra_requires_an_action(monkeypatch):
    _setup(monkeypatch, _make_items())

    result = server.batch_update_extra(
        item_keys=["ITEM0001"], ctx=DummyContext()
    )

    assert result.startswith("Error")


def test_batch_update_extra_replace_incompatible_with_remove_keys(monkeypatch):
    _setup(monkeypatch, _make_items())

    result = server.batch_update_extra(
        item_keys=["ITEM0001"],
        set_keys={"tex.otscore": "2"},
        remove_keys=["tex.provenance"],
        replace=True,
        ctx=DummyContext(),
    )

    assert result.startswith("Error")


def test_batch_update_extra_continues_after_missing_item(monkeypatch):
    fake = _setup(monkeypatch, _make_items())

    result = server.batch_update_extra(
        item_keys=["NOSUCHKEY", "ITEM0002"],
        set_keys={"tex.otscore": "2"},
        ctx=DummyContext(),
    )

    assert len(fake.updated) == 1
    assert "Items updated: 1" in result
    assert "Items skipped: 1" in result
