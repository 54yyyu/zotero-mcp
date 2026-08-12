"""Regression tests for A5b: the Zotero API lock must cover only the actual
Zotero API calls in add_by_doi / add_by_url / _add_by_arxiv, not the
third-party CrossRef/arXiv/OA-PDF network work sandwiched between them.

Before this fix, each of these three functions was wrapped whole in
@with_zotero_api_lock, so the process-wide lock stayed held across CrossRef
metadata fetches and OA-PDF download+upload — the exact shape that let a
244-DOI import hold the lock continuously and starve every other MCP tool
call for the whole batch.

_lock_is_free() below MUST check from a separate thread: the lock is a
threading.RLock, so a same-thread acquire() always succeeds once it is
already held reentrantly by that thread, which would make these checks
pass whether or not the surrounding code still holds the lock.
"""

import threading
from unittest.mock import MagicMock

import pytest
from conftest import DummyContext, FakeZotero

from zotero_mcp import client as _client
from zotero_mcp.tools import write


def _lock_is_free() -> bool:
    """True iff no thread currently holds the shared Zotero API RLock."""
    result = {}

    def probe():
        acquired = _client._zotero_api_lock.acquire(blocking=False)
        result["acquired"] = acquired
        if acquired:
            _client._zotero_api_lock.release()

    t = threading.Thread(target=probe)
    t.start()
    t.join(timeout=5)
    return result.get("acquired", False)


def _make_crossref_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "status": "ok",
        "message": {
            "type": "journal-article",
            "title": ["A Paper"],
            "DOI": "10.1234/test",
            "author": [{"given": "Jane", "family": "Smith"}],
        },
    }
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def dummy_ctx():
    return DummyContext()


@pytest.fixture
def fake_zot():
    return FakeZotero()


class TestAddByDoiLockScope:
    def test_crossref_fetch_does_not_hold_lock(self, monkeypatch, fake_zot, dummy_ctx):
        """The lock must be free while the CrossRef metadata request is in flight."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        observed = {"lock_free_during_crossref": None}

        def fake_get(*args, **kwargs):
            observed["lock_free_during_crossref"] = _lock_is_free()
            return _make_crossref_response()

        monkeypatch.setattr("requests.get", fake_get)

        write.add_item(source="10.1234/test", source_type="doi", ctx=dummy_ctx)

        assert observed["lock_free_during_crossref"] is True

    def test_pdf_attach_does_not_hold_lock(self, monkeypatch, fake_zot, dummy_ctx):
        """The lock must be free while _try_attach_oa_pdf (OA lookup + PDF
        download/upload) runs — it happens strictly after the item is
        already created."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: _make_crossref_response())

        observed = {"lock_free_during_pdf_attach": None}

        def fake_try_attach(write_zot, item_key, doi, ctx, **kwargs):
            observed["lock_free_during_pdf_attach"] = _lock_is_free()
            return "stubbed"

        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._try_attach_oa_pdf", fake_try_attach
        )

        write.add_item(source="10.1234/test", source_type="doi", ctx=dummy_ctx)

        assert observed["lock_free_during_pdf_attach"] is True

    def test_create_items_runs_under_the_lock(self, monkeypatch, fake_zot, dummy_ctx):
        """Sanity check the other half: the actual Zotero write call must
        still be serialized — this isn't a no-op removal of the lock."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: _make_crossref_response())

        observed = {"lock_held_during_create": None}
        real_create_items = fake_zot.create_items

        def spying_create_items(items, **kwargs):
            observed["lock_held_during_create"] = not _lock_is_free()
            return real_create_items(items, **kwargs)

        monkeypatch.setattr(fake_zot, "create_items", spying_create_items)

        write.add_item(source="10.1234/test", source_type="doi", ctx=dummy_ctx)

        assert observed["lock_held_during_create"] is True


class TestAddByUrlLockScope:
    def test_generic_webpage_creates_item_under_lock(self, monkeypatch, fake_zot, dummy_ctx):
        """The generic-webpage branch has no third-party fetch; confirm it
        still serializes its Zotero calls (no regression from removing the
        outer decorator)."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )

        observed = {"lock_held_during_create": None}
        real_create_items = fake_zot.create_items

        def spying_create_items(items, **kwargs):
            observed["lock_held_during_create"] = not _lock_is_free()
            return real_create_items(items, **kwargs)

        monkeypatch.setattr(fake_zot, "create_items", spying_create_items)

        result = write.add_item(
            source="https://example.com/some-article", source_type="url", ctx=dummy_ctx
        )

        assert len(fake_zot.created) == 1
        assert "Created webpage item" in result
        assert observed["lock_held_during_create"] is True


ARXIV_ATOM_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <updated>2024-01-01T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title>A Paper</title>
    <summary>An abstract.</summary>
    <author><name>Alice Smith</name></author>
  </entry>
</feed>
"""


def _make_arxiv_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.text = ARXIV_ATOM_XML
    resp.content = ARXIV_ATOM_XML.encode("utf-8")
    resp.raise_for_status = MagicMock()
    return resp


class TestAddByArxivLockScope:
    def test_arxiv_metadata_fetch_does_not_hold_lock(self, monkeypatch, fake_zot, dummy_ctx):
        """The lock must be free while the arXiv API request is in flight."""
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        observed = {"lock_free_during_arxiv_fetch": None}

        def fake_get(url, *args, **kwargs):
            if "export.arxiv.org" in url:
                observed["lock_free_during_arxiv_fetch"] = _lock_is_free()
            return _make_arxiv_response()

        monkeypatch.setattr("requests.get", fake_get)

        write.add_item(
            source="https://arxiv.org/abs/2401.00001", source_type="url", ctx=dummy_ctx
        )

        assert observed["lock_free_during_arxiv_fetch"] is True

    def test_create_items_runs_under_the_lock(self, monkeypatch, fake_zot, dummy_ctx):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (fake_zot, fake_zot)
        )
        monkeypatch.setattr("requests.get", lambda *a, **kw: _make_arxiv_response())

        observed = {"lock_held_during_create": None}
        real_create_items = fake_zot.create_items

        def spying_create_items(items, **kwargs):
            observed["lock_held_during_create"] = not _lock_is_free()
            return real_create_items(items, **kwargs)

        monkeypatch.setattr(fake_zot, "create_items", spying_create_items)

        write.add_item(
            source="https://arxiv.org/abs/2401.00001", source_type="url", ctx=dummy_ctx
        )

        assert len(fake_zot.created) == 1
        assert observed["lock_held_during_create"] is True
