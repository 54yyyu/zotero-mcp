import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from zotero_mcp.tools import annotations


class _MetadataClient:
    def __init__(self, item):
        self._item = item
        self.item_calls = []

    def item(self, key):
        self.item_calls.append(key)
        return self._item


def test_dry_run_annotation_metadata_uses_selected_local_group(monkeypatch):
    group = _MetadataClient({
        "key": "GROUPPDF",
        "data": {
            "itemType": "attachment",
            "contentType": "application/pdf",
            "filename": "synthetic.pdf",
        },
    })
    personal = _MetadataClient({"key": "GROUPPDF", "data": {"itemType": "journalArticle"}})
    fake_pdf = SimpleNamespace(is_pdf=True, close=lambda: None)
    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace(open=lambda _path: fake_pdf))
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "123456")
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "group")
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
    monkeypatch.setattr(annotations._client, "get_active_library", lambda: {})
    monkeypatch.setattr(annotations._client, "get_local_zotero_client", lambda: personal)
    monkeypatch.setattr(annotations._client, "get_web_zotero_client", lambda: None)
    monkeypatch.setattr(annotations._client, "get_zotero_client", lambda: group)
    monkeypatch.setattr(
        annotations,
        "_download_attachment_for_processing",
        lambda *_args, **_kwargs: ("synthetic.pdf", None),
    )
    monkeypatch.setattr(
        annotations,
        "_annotation_payload",
        lambda *_args, **_kwargs: ({"itemType": "annotation"}, {}),
    )
    monkeypatch.setattr(
        annotations,
        "_get_note_write_client",
        lambda *_args, **_kwargs: pytest.fail("dry run must not request write access"),
    )

    (result,) = annotations.create_annotations(
        "GROUPPDF",
        [{"page": 1, "rect": [0.1, 0.1, 0.2, 0.2]}],
        ctx=MagicMock(),
        dry_run=True,
    )

    assert result["ok"]
    assert group.item_calls == ["GROUPPDF"]
    assert personal.item_calls == []
