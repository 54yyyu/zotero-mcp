"""Tests for idempotent adds (#4): find_existing_items + if_exists semantics.

if_exists contract on the add_by_* family:

- 'duplicate' (default): today's behavior — always create, even when an
  identical identifier exists.
- 'file': converge. Reuse the existing item, add it to any requested
  collections it isn't in, add any missing tags. Nothing is ever removed.
  Re-running the same command is a no-op.
- 'skip': report the existing item, change nothing.
"""

from unittest.mock import MagicMock

import pytest

from conftest import DummyContext, FakeZotero, _FakeResponse
from zotero_mcp import server
from zotero_mcp.tools import _helpers


DOI = "10.1234/test.2024.001"


def _make_crossref_response():
    msg = {
        "type": "journal-article",
        "title": ["Fresh Paper"],
        "DOI": DOI,
        "author": [{"given": "A", "family": "Author"}],
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "ok", "message": msg}
    resp.raise_for_status = MagicMock()
    return resp


class FakeZoteroIdem(FakeZotero):
    """FakeZotero with addto/update tracking against stored items."""

    def __init__(self):
        super().__init__()
        self.addto_calls = []

    def addto_collection(self, collection_key, item, **kwargs):
        key = item["key"] if isinstance(item, dict) else item
        self.addto_calls.append((collection_key, key))
        for it in self._items:
            if it.get("key") == key:
                cols = it["data"].setdefault("collections", [])
                if collection_key not in cols:
                    cols.append(collection_key)
        return _FakeResponse(204)


@pytest.fixture
def fake_zot():
    z = FakeZoteroIdem()
    z._collections = [
        {"key": "COLA0001", "data": {"name": "Old Coll", "parentCollection": False}},
        {"key": "COLB0001", "data": {"name": "Target", "parentCollection": False}},
    ]
    z._items = [
        {
            "key": "EXIST001",
            "version": 5,
            "data": {
                "itemType": "journalArticle",
                "title": "Existing Paper",
                "DOI": DOI,
                "collections": ["COLA0001"],
                "tags": [{"tag": "old"}],
            },
        },
    ]
    return z


@pytest.fixture
def dummy_ctx():
    return DummyContext()


def _patch_clients(monkeypatch, zot):
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (zot, zot)
    )
    monkeypatch.setattr(
        "requests.get", lambda *a, **kw: _make_crossref_response()
    )
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._try_attach_oa_pdf",
        lambda *a, **kw: "skipped (test)",
    )


# ---------------------------------------------------------------------------
# find_existing_items
# ---------------------------------------------------------------------------

class TestFindExistingItems:
    def test_doi_match(self, fake_zot):
        out = _helpers.find_existing_items(fake_zot, doi=DOI)
        assert [i["key"] for i in out] == ["EXIST001"]

    def test_doi_match_with_prefixed_stored_value(self, fake_zot):
        fake_zot._items[0]["data"]["DOI"] = f"https://doi.org/{DOI}"
        out = _helpers.find_existing_items(fake_zot, doi=DOI)
        assert [i["key"] for i in out] == ["EXIST001"]

    def test_doi_no_match(self, fake_zot):
        assert _helpers.find_existing_items(fake_zot, doi="10.9999/other") == []

    def test_attachments_excluded(self, fake_zot):
        fake_zot._items.append({
            "key": "ATTACH01",
            "version": 1,
            "data": {"itemType": "attachment", "DOI": DOI},
        })
        out = _helpers.find_existing_items(fake_zot, doi=DOI)
        assert [i["key"] for i in out] == ["EXIST001"]

    def test_arxiv_match_via_url(self, fake_zot):
        fake_zot._items.append({
            "key": "ARXIV001",
            "version": 1,
            "data": {
                "itemType": "preprint",
                "url": "https://arxiv.org/abs/2401.00001",
                "extra": "",
            },
        })
        out = _helpers.find_existing_items(fake_zot, arxiv_id="2401.00001")
        assert [i["key"] for i in out] == ["ARXIV001"]

    def test_arxiv_match_via_extra(self, fake_zot):
        fake_zot._items.append({
            "key": "ARXIV002",
            "version": 1,
            "data": {"itemType": "preprint", "url": "", "extra": "arXiv:2401.00002"},
        })
        out = _helpers.find_existing_items(fake_zot, arxiv_id="2401.00002")
        assert [i["key"] for i in out] == ["ARXIV002"]

    def test_isbn_match_across_10_13_forms(self, fake_zot):
        # ISBN-10 0306406152 == ISBN-13 9780306406157
        fake_zot._items.append({
            "key": "BOOK0001",
            "version": 1,
            "data": {"itemType": "book", "ISBN": "0-306-40615-2 9999999999"},
        })
        out = _helpers.find_existing_items(fake_zot, isbn="9780306406157")
        assert [i["key"] for i in out] == ["BOOK0001"]

    def test_url_match_modulo_trailing_slash(self, fake_zot):
        fake_zot._items.append({
            "key": "PAGE0001",
            "version": 1,
            "data": {"itemType": "webpage", "url": "https://example.com/post/"},
        })
        out = _helpers.find_existing_items(fake_zot, url="https://example.com/post")
        assert [i["key"] for i in out] == ["PAGE0001"]

    def test_search_failure_returns_empty(self, dummy_ctx):
        class Boom(FakeZotero):
            def items(self, **kw):
                raise RuntimeError("api down")

        assert _helpers.find_existing_items(Boom(), doi=DOI, ctx=dummy_ctx) == []

    def test_no_identifier_returns_empty(self, fake_zot):
        assert _helpers.find_existing_items(fake_zot) == []


# ---------------------------------------------------------------------------
# add_by_doi × if_exists
# ---------------------------------------------------------------------------

class TestAddByDoiIfExists:
    def test_file_mode_reuses_and_converges(self, monkeypatch, fake_zot, dummy_ctx):
        _patch_clients(monkeypatch, fake_zot)

        result = server.add_by_doi(
            doi=DOI, collections=["COLB0001"], tags=["new-tag"],
            if_exists="file", ctx=dummy_ctx,
        )

        assert fake_zot.created == []                      # no duplicate item
        assert ("COLB0001", "EXIST001") in fake_zot.addto_calls
        assert len(fake_zot.updated) == 1                  # tags update
        new_tags = {t["tag"] for t in fake_zot.updated[0]["data"]["tags"]}
        assert new_tags == {"old", "new-tag"}
        assert "Already in library" in result
        assert "EXIST001" in result
        assert "added to ['COLB0001']" in result

    def test_file_mode_second_run_is_noop(self, monkeypatch, fake_zot, dummy_ctx):
        _patch_clients(monkeypatch, fake_zot)

        server.add_by_doi(doi=DOI, collections=["COLB0001"], tags=["new-tag"],
                          if_exists="file", ctx=dummy_ctx)
        addto_after_first = list(fake_zot.addto_calls)
        updates_after_first = len(fake_zot.updated)

        result = server.add_by_doi(doi=DOI, collections=["COLB0001"],
                                   tags=["new-tag"], if_exists="file",
                                   ctx=dummy_ctx)

        assert fake_zot.created == []
        assert fake_zot.addto_calls == addto_after_first   # nothing re-filed
        assert len(fake_zot.updated) == updates_after_first  # no tag rewrite
        assert "already in ['COLB0001']" in result

    def test_skip_mode_touches_nothing(self, monkeypatch, fake_zot, dummy_ctx):
        _patch_clients(monkeypatch, fake_zot)

        result = server.add_by_doi(
            doi=DOI, collections=["COLB0001"], tags=["new-tag"],
            if_exists="skip", ctx=dummy_ctx,
        )

        assert fake_zot.created == []
        assert fake_zot.addto_calls == []
        assert fake_zot.updated == []
        assert "No changes made" in result

    def test_duplicate_default_still_creates(self, monkeypatch, fake_zot, dummy_ctx):
        _patch_clients(monkeypatch, fake_zot)

        result = server.add_by_doi(doi=DOI, ctx=dummy_ctx)

        assert len(fake_zot.created) == 1
        assert "Successfully added" in result

    def test_file_mode_creates_when_no_match(self, monkeypatch, fake_zot, dummy_ctx):
        fake_zot._items = []          # nothing in the library
        _patch_clients(monkeypatch, fake_zot)

        result = server.add_by_doi(
            doi=DOI, collections=["COLB0001"], if_exists="file", ctx=dummy_ctx,
        )

        assert len(fake_zot.created) == 1
        assert "Successfully added" in result

    def test_invalid_if_exists_rejected(self, monkeypatch, fake_zot, dummy_ctx):
        _patch_clients(monkeypatch, fake_zot)
        result = server.add_by_doi(doi=DOI, if_exists="bogus", ctx=dummy_ctx)
        assert "if_exists" in result
        assert fake_zot.created == []


# ---------------------------------------------------------------------------
# add_by_url × if_exists (arXiv + webpage routing)
# ---------------------------------------------------------------------------

class TestAddByUrlIfExists:
    def test_arxiv_reused_without_network(self, monkeypatch, fake_zot, dummy_ctx):
        fake_zot._items.append({
            "key": "ARXIV001",
            "version": 2,
            "data": {
                "itemType": "preprint",
                "title": "An arXiv Paper",
                "url": "https://arxiv.org/abs/2401.00001",
                "extra": "arXiv:2401.00001",
                "collections": [],
                "tags": [],
            },
        })
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client",
            lambda ctx: (fake_zot, fake_zot),
        )

        def _no_network(*a, **kw):
            raise AssertionError("network must not be hit when reusing")

        monkeypatch.setattr("zotero_mcp.tools.write.requests.get", _no_network)

        result = server.add_by_url(
            url="https://arxiv.org/abs/2401.00001",
            collections=["COLB0001"], if_exists="file", ctx=dummy_ctx,
        )

        assert fake_zot.created == []
        assert ("COLB0001", "ARXIV001") in fake_zot.addto_calls
        assert "Already in library" in result

    def test_webpage_reused_by_url(self, monkeypatch, fake_zot, dummy_ctx):
        fake_zot._items.append({
            "key": "PAGE0001",
            "version": 3,
            "data": {
                "itemType": "webpage",
                "title": "A Post",
                "url": "https://example.com/post/",
                "collections": [],
                "tags": [],
            },
        })
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client",
            lambda ctx: (fake_zot, fake_zot),
        )

        result = server.add_by_url(
            url="https://example.com/post", collections=["COLB0001"],
            if_exists="file", ctx=dummy_ctx,
        )

        assert fake_zot.created == []
        assert ("COLB0001", "PAGE0001") in fake_zot.addto_calls
        assert "Already in library" in result


# ---------------------------------------------------------------------------
# add_by_isbn × if_exists
# ---------------------------------------------------------------------------

class TestAddByIsbnIfExists:
    def test_existing_isbn_reused_across_forms(self, monkeypatch, fake_zot, dummy_ctx):
        fake_zot._items.append({
            "key": "BOOK0001",
            "version": 4,
            "data": {
                "itemType": "book",
                "title": "A Book",
                "ISBN": "0-306-40615-2",   # ISBN-10 form of 9780306406157
                "collections": [],
                "tags": [],
            },
        })
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client",
            lambda ctx: (fake_zot, fake_zot),
        )

        result = server.add_by_isbn(
            isbn="9780306406157", collections=["COLB0001"],
            if_exists="file", ctx=dummy_ctx,
        )

        assert fake_zot.created == []
        assert ("COLB0001", "BOOK0001") in fake_zot.addto_calls
        assert "Already in library" in result


# ---------------------------------------------------------------------------
# add_by_bibtex × if_exists (batch: mixed existing/new)
# ---------------------------------------------------------------------------

class TestAddByBibtexIfExists:
    def test_mixed_batch_reuses_and_creates(self, monkeypatch, fake_zot, dummy_ctx):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client",
            lambda ctx: (fake_zot, fake_zot),
        )
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._try_attach_oa_pdf",
            lambda *a, **kw: "skipped (test)",
        )

        bib = (
            "@article{exists, title={Existing Paper}, author={A, B}, "
            "year={2024}, doi={" + DOI + "}}\n"
            "@article{fresh, title={Fresh Paper}, author={C, D}, year={2024}}"
        )
        result = server.add_by_bibtex(
            bibtex=bib, collections=["COLB0001"], if_exists="file",
            ctx=dummy_ctx,
        )

        # Only the DOI-less entry creates a new item.
        assert len(fake_zot.created) == 1
        assert ("COLB0001", "EXIST001") in fake_zot.addto_calls
        assert "1 already existed" in result
        assert "reused existing" in result

    def test_skip_mode_reports_without_changes(self, monkeypatch, fake_zot, dummy_ctx):
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client",
            lambda ctx: (fake_zot, fake_zot),
        )

        bib = ("@article{exists, title={Existing Paper}, author={A, B}, "
               "year={2024}, doi={" + DOI + "}}")
        result = server.add_by_bibtex(
            bibtex=bib, collections=["COLB0001"], if_exists="skip",
            ctx=dummy_ctx,
        )

        assert fake_zot.created == []
        assert fake_zot.addto_calls == []
        assert "skipped — already in library" in result
