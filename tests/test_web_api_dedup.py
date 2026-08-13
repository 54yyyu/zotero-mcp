"""Web-API dedup for the add_by_* family (#441).

The Zotero *web* API's quick search indexes only title, creator and year
(plus attachment full text under ``qmode=everything``) — it never reaches
metadata fields like DOI, ISBN, url or extra. The *local* API's
``qmode=everything`` does reach metadata. ``find_existing_items`` therefore
missed every identifier match over the web transport, and ``if_exists``
silently created duplicates.

The fix: once the metadata lookup (CrossRef / arXiv / Open Library) has
produced a title, a second quick search by title (``qmode=titleCreatorYear``,
indexed on both transports) runs, still confirmed by the normalized
identifier comparison client-side.

These tests use a fake that reproduces the web API's search semantics, so a
regression to identifier-only quick search fails loudly.
"""

from unittest.mock import MagicMock

import pytest

from conftest import DummyContext, FakeZotero
from zotero_mcp import server  # noqa: F401 — registers tools/monkeypatch targets
from zotero_mcp.tools import _helpers, write

DOI = "10.1002/smj.3597"
TITLE = "Strategic Renewal in Practice"


class WebApiSearchFake(FakeZotero):
    """FakeZotero with the web API's quick-search semantics.

    ``items(q=..., qmode=...)`` matches only against title, creators and
    date — regardless of qmode — mirroring api.zotero.org, where
    ``everything`` adds full-text/notes but still not metadata fields.
    """

    def __init__(self):
        super().__init__()
        self.search_calls: list[dict] = []

    def items(self, **kwargs):
        q = kwargs.get("q")
        if q is None:
            return self._items
        self.search_calls.append(dict(kwargs))
        ql = str(q).lower()
        hits = []
        for it in self._items:
            data = it.get("data", {})
            creators = " ".join(
                f"{c.get('firstName', '')} {c.get('lastName', '')} {c.get('name', '')}"
                for c in data.get("creators", [])
            )
            haystack = " ".join(
                [data.get("title", ""), creators, data.get("date", "")]
            ).lower()
            if ql in haystack:
                hits.append(it)
        return hits


@pytest.fixture
def web_zot():
    z = WebApiSearchFake()
    z._items = [
        {
            "key": "EXIST001",
            "version": 5,
            "data": {
                "itemType": "journalArticle",
                "title": TITLE,
                "DOI": DOI,
                "ISBN": "978-0-306-40615-7",
                "url": "https://arxiv.org/abs/2401.12345",
                "extra": "arXiv:2401.12345",
                "date": "2024",
                "collections": [],
                "tags": [],
            },
        },
    ]
    return z


@pytest.fixture
def dummy_ctx():
    return DummyContext()


# ---------------------------------------------------------------------------
# find_existing_items: title-hint fallback
# ---------------------------------------------------------------------------

class TestTitleHintFallback:
    def test_identifier_search_alone_misses_on_web_api(self, web_zot):
        """Documents the #441 failure mode this module guards against."""
        assert _helpers.find_existing_items(web_zot, doi=DOI) == []

    def test_title_hint_recovers_doi_match(self, web_zot):
        out = _helpers.find_existing_items(web_zot, doi=DOI, title_hint=TITLE)
        assert [i["key"] for i in out] == ["EXIST001"]

    def test_title_hint_match_is_confirmed_by_identifier(self, web_zot):
        """A title hit whose DOI differs must not count as existing."""
        out = _helpers.find_existing_items(
            web_zot, doi="10.9999/other", title_hint=TITLE
        )
        assert out == []

    def test_title_hint_recovers_isbn_match(self, web_zot):
        out = _helpers.find_existing_items(
            web_zot, isbn="9780306406157", title_hint=TITLE
        )
        assert [i["key"] for i in out] == ["EXIST001"]

    def test_title_hint_recovers_arxiv_match(self, web_zot):
        out = _helpers.find_existing_items(
            web_zot, arxiv_id="2401.12345", title_hint=TITLE
        )
        assert [i["key"] for i in out] == ["EXIST001"]

    def test_identifier_search_false_skips_identifier_query(self, web_zot):
        _helpers.find_existing_items(
            web_zot, doi=DOI, title_hint=TITLE, identifier_search=False
        )
        assert [c["q"] for c in web_zot.search_calls] == [TITLE]
        assert web_zot.search_calls[0]["qmode"] == "titleCreatorYear"

    def test_title_search_not_run_when_identifier_search_hits(self):
        """On the local API the identifier pass matches; no second query."""

        class LocalApiSearchFake(WebApiSearchFake):
            def items(self, **kwargs):
                q = kwargs.get("q")
                if q is None:
                    return self._items
                self.search_calls.append(dict(kwargs))
                # Local API: qmode=everything reaches metadata fields.
                ql = str(q).lower()
                return [
                    it for it in self._items
                    if ql in str(it.get("data", {})).lower()
                ]

        z = LocalApiSearchFake()
        z._items = [
            {
                "key": "EXIST001",
                "version": 5,
                "data": {"itemType": "journalArticle", "title": TITLE, "DOI": DOI},
            }
        ]
        out = _helpers.find_existing_items(z, doi=DOI, title_hint=TITLE)
        assert [i["key"] for i in out] == ["EXIST001"]
        assert len(z.search_calls) == 1


# ---------------------------------------------------------------------------
# add_by_doi end-to-end over the web-API fake
# ---------------------------------------------------------------------------

def _crossref_response(title=TITLE, doi=DOI):
    msg = {
        "type": "journal-article",
        "title": [title],
        "DOI": doi,
        "author": [{"given": "A", "family": "Author"}],
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "ok", "message": msg}
    resp.raise_for_status = MagicMock()
    return resp


def _patch_clients(monkeypatch, zot):
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._get_write_client", lambda ctx: (zot, zot)
    )
    monkeypatch.setattr("requests.get", lambda *a, **kw: _crossref_response())
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._try_attach_oa_pdf",
        lambda *a, **kw: "skipped (test)",
    )


class TestAddByDoiWebApi:
    def test_skip_dedupes_after_metadata_fetch(
        self, monkeypatch, web_zot, dummy_ctx
    ):
        _patch_clients(monkeypatch, web_zot)
        result = write.add_by_doi(doi=DOI, if_exists="skip", ctx=dummy_ctx)
        assert "EXIST001" in result
        assert web_zot.created == []

    def test_file_converges_instead_of_creating(
        self, monkeypatch, web_zot, dummy_ctx
    ):
        _patch_clients(monkeypatch, web_zot)
        result = write.add_by_doi(doi=DOI, if_exists="file", ctx=dummy_ctx)
        assert "EXIST001" in result
        assert web_zot.created == []

    def test_duplicate_mode_still_creates(self, monkeypatch, web_zot, dummy_ctx):
        _patch_clients(monkeypatch, web_zot)
        result = write.add_by_doi(doi=DOI, if_exists="duplicate", ctx=dummy_ctx)
        assert "Successfully added" in result
        assert len(web_zot.created) == 1

    def test_new_doi_still_creates_under_skip(
        self, monkeypatch, web_zot, dummy_ctx
    ):
        _patch_clients(monkeypatch, web_zot)
        monkeypatch.setattr(
            "requests.get",
            lambda *a, **kw: _crossref_response(
                title="A Genuinely New Paper", doi="10.5555/brand.new"
            ),
        )
        result = write.add_by_doi(
            doi="10.5555/brand.new", if_exists="skip", ctx=dummy_ctx
        )
        assert "Successfully added" in result
        assert len(web_zot.created) == 1


# ---------------------------------------------------------------------------
# batch import path (_maybe_reuse_existing)
# ---------------------------------------------------------------------------

class TestBatchReuseWebApi:
    def test_batch_entry_with_title_dedupes(self, web_zot, dummy_ctx):
        item_data = {"itemType": "journalArticle", "title": TITLE, "DOI": DOI}
        out = write._maybe_reuse_existing(
            web_zot, web_zot, item_data, [], None, "skip", dummy_ctx
        )
        assert out is not None
        assert out["key"] == "EXIST001"
        assert out["existed"].startswith("skipped")

    def test_batch_entry_without_title_still_creates(self, web_zot, dummy_ctx):
        item_data = {"itemType": "journalArticle", "title": "", "DOI": DOI}
        out = write._maybe_reuse_existing(
            web_zot, web_zot, item_data, [], None, "skip", dummy_ctx
        )
        assert out is None
