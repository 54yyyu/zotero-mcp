"""Tag writes must keep existing tag dicts (incl. automatic type 1) verbatim."""

from zotero_mcp.tools import _helpers


def test_add_keeps_automatic_tags():
    existing = [{"tag": "auto", "type": 1}, {"tag": "manual"}]
    out = _helpers._apply_tag_changes(existing, [{"tag": "x"}])
    assert out == [{"tag": "auto", "type": 1}, {"tag": "manual"}, {"tag": "x"}]


def test_remove_keeps_other_types():
    existing = [{"tag": "auto", "type": 1}, {"tag": "gone", "type": 1}]
    out = _helpers._apply_tag_changes(existing, [], {"gone"})
    assert out == [{"tag": "auto", "type": 1}]


def test_add_existing_name_does_not_duplicate_or_retype():
    existing = [{"tag": "auto", "type": 1}]
    out = _helpers._apply_tag_changes(existing, [{"tag": "auto"}])
    assert out == [{"tag": "auto", "type": 1}]
