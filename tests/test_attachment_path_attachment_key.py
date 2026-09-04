"""``zotero_get_attachment_path`` given an ATTACHMENT key said "No attachments found".

Same root cause as #372: the tool only walked ``get_attachment_paths(parent_key)``,
which goes through ``get_item_by_key`` — a query that excludes the 'attachment'
item type — so the attachment's own key resolved to nothing. The reader already
had ``get_attachment_by_key`` (added for #372); the path tool never used it.
Keys handed out by ``zotero_get_item_children`` are attachment keys, so this was
the natural next call to make, and it failed every time.
"""

import types

from conftest import DummyContext
from test_issue_372_pdf_tools import ATTACHMENT_KEY, PARENT_KEY, make_library

from zotero_mcp.tools import retrieval as retrieval_tools


def use_local_library(monkeypatch, db_path):
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: True)
    monkeypatch.setattr(
        retrieval_tools,
        "load_config",
        lambda: types.SimpleNamespace(resolve_zotero_db_path=lambda: str(db_path)),
    )


def test_attachment_key_resolves_to_its_own_path(monkeypatch, tmp_path):
    db_path, pdf_path = make_library(tmp_path)
    use_local_library(monkeypatch, db_path)

    result = retrieval_tools.get_attachment_path(item_key=ATTACHMENT_KEY, ctx=DummyContext())

    assert "No attachments found" not in result
    assert f"## `{ATTACHMENT_KEY}` (application/pdf)" in result
    assert f"- Local path: `{pdf_path}`" in result
    assert "missing on disk" not in result


def test_parent_key_still_lists_children(monkeypatch, tmp_path):
    db_path, pdf_path = make_library(tmp_path)
    use_local_library(monkeypatch, db_path)

    result = retrieval_tools.get_attachment_path(item_key=PARENT_KEY, ctx=DummyContext())

    assert f"## `{ATTACHMENT_KEY}` (application/pdf)" in result
    assert f"- Local path: `{pdf_path}`" in result


def test_unknown_key_still_reports_nothing(monkeypatch, tmp_path):
    db_path, _ = make_library(tmp_path)
    use_local_library(monkeypatch, db_path)

    result = retrieval_tools.get_attachment_path(item_key="NOSUCHKEY", ctx=DummyContext())

    assert result == "No attachments found for item `NOSUCHKEY`."
