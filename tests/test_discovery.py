"""Tests for the discovery tools (zotero_find_related_papers, zotero_library_coverage).

Network calls to OpenAlex are stubbed by monkeypatching
``zotero_mcp.tools.discovery.requests.get``. The Zotero client is stubbed by
monkeypatching ``zotero_mcp.client.get_zotero_client``.
"""

from conftest import DummyContext, FakeZotero

from zotero_mcp.tools import discovery

# --- Fake HTTP plumbing ----------------------------------------------------


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def _work(work_id, title, year, doi, cited_by, authors=None, **extra):
    w = {
        "id": work_id,
        "title": title,
        "publication_year": year,
        "doi": f"https://doi.org/{doi}" if doi else None,
        "cited_by_count": cited_by,
        "authorships": [{"author": {"display_name": a}} for a in (authors or [])],
    }
    w.update(extra)
    return w


# --- Fake Zotero -----------------------------------------------------------


class FakeZoteroDiscovery(FakeZotero):
    """FakeZotero that supports add_parameters/items for membership checks and
    children() for coverage."""

    def __init__(self):
        super().__init__()
        self._params = {}
        self._by_doi = {}  # normalized DOI -> list of items returned by items()

    def add_parameters(self, **kwargs):
        self._params = kwargs

    def items(self, **kwargs):
        q = (kwargs.get("q") or self._params.get("q") or "").strip().lower()
        if q and q in self._by_doi:
            return self._by_doi[q]
        if q:
            return []
        return self._items


def _make_zot(monkeypatch):
    zot = FakeZoteroDiscovery()
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: zot)
    return zot


def _patch_requests(monkeypatch, handler):
    def fake_get(url, params=None, timeout=None, **kwargs):
        return handler(url, params or {})

    monkeypatch.setattr(discovery.requests, "get", fake_get)


# NOTE: DOIs below use the real `10.NNNN/...` shape because the tool normalizes
# DOIs via _helpers._normalize_doi, which (correctly) rejects malformed ones.


# --- find_related_papers: references --------------------------------------


def test_references_direction(monkeypatch):
    _make_zot(monkeypatch)

    source = {
        "id": "https://openalex.org/W1",
        "title": "Source Paper",
        "display_name": "Source Paper",
        "referenced_works": [
            "https://openalex.org/W10",
            "https://openalex.org/W11",
        ],
        "cited_by_api_url": "https://api.openalex.org/works?filter=cites:W1",
    }
    ref_results = {
        "results": [
            _work("https://openalex.org/W10", "Ref A", 2010, "10.1234/refa", 5, ["Alice A"]),
            _work("https://openalex.org/W11", "Ref B", 2012, "10.1234/refb", 9, ["Bob B"]),
        ]
    }

    def handler(url, params):
        if url.endswith("/works/https://doi.org/10.1234/x"):
            return FakeResponse(200, source)
        if url.endswith("/works") and "openalex_id" in params.get("filter", ""):
            return FakeResponse(200, ref_results)
        return FakeResponse(404, {})

    _patch_requests(monkeypatch, handler)

    out = discovery.find_related_papers("10.1234/x", direction="references", ctx=DummyContext())
    assert "References" in out
    assert "Ref A" in out
    assert "Ref B" in out
    # references keep referenced_works order: Ref A before Ref B
    assert out.index("Ref A") < out.index("Ref B")
    assert "Citations" not in out


# --- find_related_papers: citations + sorting -----------------------------


def test_citations_direction_sorted_by_citation_count(monkeypatch):
    _make_zot(monkeypatch)

    cited_url = "https://api.openalex.org/works?filter=cites:W1"
    source = {
        "id": "https://openalex.org/W1",
        "title": "Source Paper",
        "referenced_works": [],
        "cited_by_api_url": cited_url,
    }
    citing_results = {
        "results": [
            _work("https://openalex.org/W20", "Citer Low", 2020, "10.1234/low", 3, ["X"]),
            _work("https://openalex.org/W21", "Citer High", 2021, "10.1234/high", 99, ["Y"]),
        ]
    }

    def handler(url, params):
        if url.endswith("/works/https://doi.org/10.1234/x"):
            return FakeResponse(200, source)
        if url == cited_url:
            return FakeResponse(200, citing_results)
        return FakeResponse(404, {})

    _patch_requests(monkeypatch, handler)

    out = discovery.find_related_papers("10.1234/x", direction="citations", ctx=DummyContext())
    assert "Citations" in out
    assert "References" not in out
    # Higher citation count first
    assert out.index("Citer High") < out.index("Citer Low")


# --- find_related_papers: library membership flagging ---------------------


def test_library_membership_flagging(monkeypatch):
    zot = _make_zot(monkeypatch)
    # Mark 10.1234/refa as already in library.
    zot._by_doi["10.1234/refa"] = [{"key": "INLIB001", "data": {"itemType": "journalArticle", "DOI": "10.1234/refa"}}]

    source = {
        "id": "https://openalex.org/W1",
        "title": "Source Paper",
        "referenced_works": ["https://openalex.org/W10", "https://openalex.org/W11"],
        "cited_by_api_url": "https://api.openalex.org/works?filter=cites:W1",
    }
    ref_results = {
        "results": [
            _work("https://openalex.org/W10", "Ref A", 2010, "10.1234/refa", 5, ["Alice"]),
            _work("https://openalex.org/W11", "Ref B", 2012, "10.1234/refb", 9, ["Bob"]),
        ]
    }

    def handler(url, params):
        if url.endswith("/works/https://doi.org/10.1234/x"):
            return FakeResponse(200, source)
        if url.endswith("/works") and "openalex_id" in params.get("filter", ""):
            return FakeResponse(200, ref_results)
        return FakeResponse(404, {})

    _patch_requests(monkeypatch, handler)

    out = discovery.find_related_papers("10.1234/x", direction="references", ctx=DummyContext())
    assert "in library ✓" in out
    assert "not in library" in out
    assert "1 already in library" in out


# --- find_related_papers: no DOI error path -------------------------------


def test_no_doi_error_path(monkeypatch):
    _make_zot(monkeypatch)
    # requests.get should never be called, but stub it to a clear failure.
    _patch_requests(monkeypatch, lambda url, params: FakeResponse(404, {}))

    out = discovery.find_related_papers("not a doi at all", ctx=DummyContext())
    assert "Could not resolve a DOI" in out


def test_item_key_without_doi(monkeypatch):
    zot = _make_zot(monkeypatch)
    zot._items = [{"key": "ABCD1234", "data": {"itemType": "journalArticle"}}]
    _patch_requests(monkeypatch, lambda url, params: FakeResponse(404, {}))

    out = discovery.find_related_papers("ABCD1234", ctx=DummyContext())
    assert "Could not resolve a DOI" in out


def test_item_key_with_doi_resolves(monkeypatch):
    zot = _make_zot(monkeypatch)
    zot._items = [{"key": "ABCD1234", "data": {"itemType": "journalArticle", "DOI": "10.1234/x"}}]
    source = {
        "id": "https://openalex.org/W1",
        "title": "Source Paper",
        "referenced_works": [],
        "cited_by_api_url": "https://api.openalex.org/works?filter=cites:W1",
    }

    def handler(url, params):
        if url.endswith("/works/https://doi.org/10.1234/x"):
            return FakeResponse(200, source)
        return FakeResponse(200, {"results": []})

    _patch_requests(monkeypatch, handler)

    out = discovery.find_related_papers("ABCD1234", direction="both", ctx=DummyContext())
    assert "Related Papers for: Source Paper" in out


# --- library_coverage -----------------------------------------------------


class FakeZoteroCoverage(FakeZoteroDiscovery):
    def collection_items(self, key, start=0, limit=100, **kwargs):
        return self._items[start : start + limit]

    def items(self, start=None, limit=None, **kwargs):
        if start is not None:
            return self._items[start : start + (limit or 100)]
        return super().items(**kwargs)


def _make_coverage_zot(monkeypatch):
    zot = FakeZoteroCoverage()
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: zot)
    return zot


def test_library_coverage_missing_and_present(monkeypatch):
    zot = _make_coverage_zot(monkeypatch)
    zot._items = [
        {
            "key": "PAP1",
            "data": {"itemType": "journalArticle", "title": "Has PDF", "date": "2021", "DOI": "10.1234/haspdf"},
        },
        {
            "key": "PAP2",
            "data": {"itemType": "journalArticle", "title": "No PDF", "date": "2022", "DOI": "10.1234/nopdf"},
        },
    ]
    zot._children = {
        "PAP1": [{"key": "ATT1", "data": {"itemType": "attachment", "contentType": "application/pdf"}}],
        "PAP2": [{"key": "NOTE1", "data": {"itemType": "note"}}],
    }

    out = discovery.library_coverage(ctx=DummyContext())
    assert "Items scanned: 2" in out
    assert "With PDF: 1" in out
    assert "Missing PDF: 1" in out
    assert "Coverage: 50.0%" in out
    assert "No PDF" in out
    assert "10.1234/nopdf" in out
    # Item that has a PDF should not appear in the missing list.
    assert "Has PDF" not in out.split("Items Missing")[1] if "Items Missing" in out else True


def test_library_coverage_standalone_pdf_counts(monkeypatch):
    zot = _make_coverage_zot(monkeypatch)
    zot._items = [
        {
            "key": "STAND1",
            "data": {"itemType": "attachment", "contentType": "application/pdf", "filename": "loose.pdf"},
        },
    ]
    out = discovery.library_coverage(ctx=DummyContext())
    assert "Items scanned: 1" in out
    assert "With PDF: 1" in out
    assert "Coverage: 100.0%" in out


def test_library_coverage_scoped_collection(monkeypatch):
    zot = _make_coverage_zot(monkeypatch)
    zot._items = [
        {
            "key": "PAP2",
            "data": {"itemType": "journalArticle", "title": "No PDF", "date": "2022", "DOI": "10.1234/nopdf"},
        },
    ]
    zot._children = {"PAP2": []}
    out = discovery.library_coverage(collection_key="ABCD1234", ctx=DummyContext())
    assert "collection ABCD1234" in out
    assert "Missing PDF: 1" in out


def test_library_coverage_children_error_tolerated(monkeypatch):
    zot = _make_coverage_zot(monkeypatch)
    zot._items = [
        {"key": "PAP3", "data": {"itemType": "journalArticle", "title": "Boom", "date": "2020"}},
    ]

    def boom(key, **kwargs):
        raise RuntimeError("network down")

    zot.children = boom
    out = discovery.library_coverage(ctx=DummyContext())
    # Treated as missing, not an error.
    assert "Missing PDF: 1" in out
    assert "Boom" in out
