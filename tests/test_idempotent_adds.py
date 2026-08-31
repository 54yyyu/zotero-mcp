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

    def test_malformed_entries_are_skipped(self, fake_zot, monkeypatch):
        """A live batch import got an int back inside the items() list. The
        try/except upstream only wraps the call, not this iteration, so the
        AttributeError escaped and aborted the whole import — a junk entry
        must cost at most its own match."""
        real_items = fake_zot.items

        def _items_with_junk(**kwargs):
            return [0, None, "junk", {"no_data_key": True},
                    {"data": 7}, *real_items(**kwargs)]

        monkeypatch.setattr(fake_zot, "items", _items_with_junk)

        out = _helpers.find_existing_items(fake_zot, doi=DOI, ctx=DummyContext())
        assert [i["key"] for i in out] == ["EXIST001"]

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

    # -- arXiv identity across storage sites and versions ------------------
    #
    # Zotero records an arXiv identity in a different field depending on how
    # the item arrived. Matching only url+extra misses the rest, so a re-add
    # of a paper already in the library silently creates a duplicate.

    def test_arxiv_match_via_archive_id(self, fake_zot):
        """Browser-connector imports put the ID in archiveID, not url/extra."""
        fake_zot._items.append({
            "key": "ARXIV003",
            "version": 1,
            "data": {
                "itemType": "preprint",
                "archiveID": "arXiv:2401.00003",
                "url": "",
                "extra": "",
            },
        })
        out = _helpers.find_existing_items(fake_zot, arxiv_id="2401.00003")
        assert [i["key"] for i in out] == ["ARXIV003"]

    def test_arxiv_match_via_datacite_doi(self, fake_zot):
        """An item added by DOI carries only arXiv's 10.48550 DataCite DOI."""
        fake_zot._items.append({
            "key": "ARXIV004",
            "version": 1,
            "data": {
                "itemType": "preprint",
                "DOI": "10.48550/arXiv.2401.00004",
                "url": "",
                "extra": "",
            },
        })
        out = _helpers.find_existing_items(fake_zot, arxiv_id="2401.00004")
        assert [i["key"] for i in out] == ["ARXIV004"]

    def test_arxiv_versioned_add_matches_bare_stored_id(self, fake_zot):
        """Adding .../abs/2401.00005v2 must reuse the stored unversioned item."""
        fake_zot._items.append({
            "key": "ARXIV005",
            "version": 1,
            "data": {
                "itemType": "preprint",
                "url": "https://arxiv.org/abs/2401.00005",
                "extra": "",
            },
        })
        out = _helpers.find_existing_items(fake_zot, arxiv_id="2401.00005v2")
        assert [i["key"] for i in out] == ["ARXIV005"]

    def test_arxiv_bare_add_matches_versioned_stored_id(self, fake_zot):
        """And the reverse: a stored v1 is the same paper as a bare re-add."""
        fake_zot._items.append({
            "key": "ARXIV006",
            "version": 1,
            "data": {
                "itemType": "preprint",
                "url": "https://arxiv.org/abs/2401.00006v1",
                "extra": "",
            },
        })
        out = _helpers.find_existing_items(fake_zot, arxiv_id="2401.00006")
        assert [i["key"] for i in out] == ["ARXIV006"]

    def test_arxiv_does_not_match_a_different_paper(self, fake_zot):
        """The version-insensitive compare must not collapse distinct IDs."""
        fake_zot._items.append({
            "key": "ARXIV007",
            "version": 1,
            "data": {
                "itemType": "preprint",
                "url": "https://arxiv.org/abs/2401.00007",
                "DOI": "10.48550/arXiv.2401.00007",
                "archiveID": "arXiv:2401.00007",
                "extra": "arXiv:2401.00007 [cs]",
            },
        })
        assert _helpers.find_existing_items(fake_zot, arxiv_id="2401.00008") == []

    def test_arxiv_ignores_a_non_arxiv_doi(self, fake_zot):
        """A publisher DOI must not be read as an arXiv identity."""
        fake_zot._items.append({
            "key": "JOURNAL01",
            "version": 1,
            "data": {
                "itemType": "journalArticle",
                "DOI": "10.1038/nature12373",
                "url": "",
                "extra": "",
            },
        })
        assert _helpers.find_existing_items(fake_zot, arxiv_id="2401.00009") == []

    # -- title fallback ----------------------------------------------------
    #
    # Zotero's `q` "searches titles and individual creator fields"; the API
    # docs add that "searching of other fields will be possible in the
    # future". So DOI/url/archiveID/extra are NOT searchable server side and
    # an identifier query returns nothing for an item that IS present. The
    # title is the one field the API does index, so it supplies the
    # candidates; the identifier still decides.

    def test_title_fallback_finds_item_identifier_search_cannot(self):
        """The real-world failure: identifier query returns nothing."""
        item = {
            "key": "TITLE001",
            "version": 1,
            "data": {
                "itemType": "preprint",
                "title": "RL's Razor",
                "DOI": "10.48550/arXiv.2509.04259",
                "url": "",
                "extra": "",
            },
        }

        class IdentifierBlindZotero(FakeZoteroIdem):
            """Models the Web API: only a title/creator query matches."""

            def items(self, **kwargs):
                q = (kwargs.get("q") or "").lower()
                if kwargs.get("qmode") == "titleCreatorYear" and "razor" in q:
                    return [item]
                return []

        z = IdentifierBlindZotero()
        # Without the title the identifier query finds nothing at all.
        assert _helpers.find_existing_items(z, arxiv_id="2509.04259") == []
        # With it, the item is found and confirmed by its DOI.
        out = _helpers.find_existing_items(
            z, arxiv_id="2509.04259", title="RL's Razor"
        )
        assert [i["key"] for i in out] == ["TITLE001"]

    def test_title_fallback_still_requires_the_identifier_to_match(self, fake_zot):
        """A same-title different-paper must NOT be treated as the same item.

        The title only supplies candidates; the identifier decides. Without
        this the fallback would merge unrelated papers that share a title.
        """
        fake_zot._items.append({
            "key": "OTHER001",
            "version": 1,
            "data": {
                "itemType": "preprint",
                "title": "Generalized Linear Models",
                "DOI": "10.48550/arXiv.1111.11111",
                "url": "",
                "extra": "",
            },
        })
        out = _helpers.find_existing_items(
            fake_zot, arxiv_id="2222.22222", title="Generalized Linear Models"
        )
        assert out == []

    def test_title_fallback_is_skipped_when_identifier_already_matched(self):
        """No second query when the identifier query already confirmed."""
        item = {
            "key": "IDENT001",
            "version": 1,
            "data": {
                "itemType": "preprint",
                "url": "https://arxiv.org/abs/2401.00010",
                "extra": "",
            },
        }

        class CountingZotero(FakeZoteroIdem):
            def __init__(self):
                super().__init__()
                self.queries = []

            def items(self, **kwargs):
                self.queries.append(kwargs.get("qmode"))
                return [item]

        z = CountingZotero()
        out = _helpers.find_existing_items(
            z, arxiv_id="2401.00010", title="Whatever"
        )
        assert [i["key"] for i in out] == ["IDENT001"]
        assert z.queries == ["everything"]

    def test_title_fallback_survives_crossref_jats_markup(self):
        """A marked-up CrossRef title must not take the lookup to zero.

        Zotero's quick search ANDs whitespace-separated tokens, so a literal
        '<i>' or '&amp;' carried over from CrossRef's title[0] is a token
        that matches nothing and the whole fallback returns empty against an
        item whose stored title is clean. Measured against the live API:
        the exact title hits, the same title with one <i> pair scores zero.
        """
        item = {
            "key": "JATS001",
            "version": 1,
            "data": {
                "itemType": "journalArticle",
                "title": "Horizontal transfer in Escherichia coli & kin",
                "DOI": "10.1234/jats.2024.001",
                "url": "",
                "extra": "",
            },
        }

        class TokenAndingZotero(FakeZoteroIdem):
            """Models quick search: every token must appear in the title."""

            def items(self, **kwargs):
                if kwargs.get("qmode") != "titleCreatorYear":
                    return []
                stored = item["data"]["title"].lower()
                tokens = (kwargs.get("q") or "").lower().split()
                return [item] if tokens and all(t in stored for t in tokens) else []

        z = TokenAndingZotero()
        raw = "Horizontal transfer in <i>Escherichia coli</i> &amp; kin"
        # The raw CrossRef spelling matches nothing...
        assert _helpers.find_existing_items(
            z, doi="10.1234/jats.2024.001", title=None
        ) == []
        # ...but the normalized query finds it, and the DOI confirms it.
        out = _helpers.find_existing_items(
            z, doi="10.1234/jats.2024.001", title=raw
        )
        assert [i["key"] for i in out] == ["JATS001"]

    def test_title_fallback_skipped_when_title_is_only_markup(self):
        """Nothing usable left after stripping means no second query."""

        class CountingZotero(FakeZoteroIdem):
            def __init__(self):
                super().__init__()
                self.queries = []

            def items(self, **kwargs):
                self.queries.append(kwargs.get("qmode"))
                return []

        z = CountingZotero()
        assert _helpers.find_existing_items(
            z, doi="10.1234/nope", title="<i></i>   "
        ) == []
        assert z.queries == ["everything"]

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
