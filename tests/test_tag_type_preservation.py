"""Tag writes must keep existing tag dicts (incl. automatic type 1) verbatim."""

import pytest

from zotero_mcp.tools import _helpers


def test_add_keeps_automatic_tags():
    existing = [{"tag": "auto", "type": 1}, {"tag": "manual"}]
    out = _helpers._apply_tag_changes(existing, _helpers._normalize_tag_write_input(["x"], "add_tags"))
    assert out == [{"tag": "auto", "type": 1}, {"tag": "manual"}, {"tag": "x"}]


def test_remove_keeps_other_types():
    existing = [{"tag": "auto", "type": 1}, {"tag": "gone", "type": 1}]
    out = _helpers._apply_tag_changes(existing, [], {"gone"})
    assert out == [{"tag": "auto", "type": 1}]


def test_add_existing_name_does_not_duplicate_or_retype():
    existing = [{"tag": "auto", "type": 1}]
    out = _helpers._apply_tag_changes(existing, [{"tag": "auto"}])
    assert out == [{"tag": "auto", "type": 1}]


def test_typed_tag_input_shapes():
    assert _helpers._normalize_tag_write_input([{"tag": "a", "type": 1}, "b"]) == [
        {"tag": "a", "type": 1},
        {"tag": "b"},
    ]
    assert _helpers._normalize_tag_write_input('[{"tag": "a", "type": 1}]') == [{"tag": "a", "type": 1}]
    assert _helpers._normalize_tag_write_input("a, b") == [{"tag": "a"}, {"tag": "b"}]
    assert _helpers._normalize_tag_write_input(None) == []


@pytest.mark.parametrize("bad", [2, -1, 1.5, "1", True])
def test_tag_type_must_be_0_or_1(bad):
    with pytest.raises(ValueError, match="tag type must be 0 or 1"):
        _helpers._normalize_tag_write_input([{"tag": "a", "type": bad}])
