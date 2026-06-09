"""Tests for the Scite integration (scite_client + tools/scite).

Covers two API-contract bugs:

1. ``get_papers_batch`` must POST a bare JSON array to ``/papers``. The
   object form ``{"dois": [...]}`` returns HTTP 400 ("Input should be a
   valid list"), which silently breaks paper-metadata enrichment and
   ``scite_check_retractions`` entirely.

2. Scite lowercases DOI keys in its responses (both ``/papers`` and
   ``/tallies``), while the tools look results up by the original-case DOI
   from ``_normalize_doi`` (which preserves case). A retraction on an
   uppercase DOI (e.g. Wakefield 1998, ``10.1016/S0140-6736(97)11096-0``)
   was therefore missed, producing a false "all clear".
"""

from zotero_mcp import scite_client
from zotero_mcp.tools import scite as scite_tools

# A real DOI with uppercase characters whose Scite record carries editorial
# notices (the retracted Wakefield 1998 MMR paper).
UPPER_DOI = "10.1016/S0140-6736(97)11096-0"
LOWER_DOI = UPPER_DOI.lower()


class _FakeResp:
    """Minimal ``requests.Response`` stub."""

    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# Bug 1 — POST /papers requires a bare JSON array body
# ---------------------------------------------------------------------------


def test_get_papers_batch_posts_bare_doi_array(monkeypatch):
    """The request body must be a bare list, not ``{"dois": [...]}``."""
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["json"] = json
        return _FakeResp(200, {"papers": {"10.1162/tacl_a_00638": {"title": "X"}}})

    monkeypatch.setattr(scite_client.requests, "post", fake_post)

    result = scite_client.get_papers_batch(["10.1162/tacl_a_00638"])

    assert captured["json"] == ["10.1162/tacl_a_00638"]
    assert result == {"10.1162/tacl_a_00638": {"title": "X"}}


# ---------------------------------------------------------------------------
# Bug 2 — case-insensitive DOI lookup against Scite's lowercased keys
# ---------------------------------------------------------------------------


def test_enrich_items_matches_lowercased_scite_keys(monkeypatch):
    """enrich_items must match an uppercase-DOI item to Scite's lowercase keys."""
    monkeypatch.setattr(
        scite_tools._scite,
        "get_tallies_batch",
        lambda dois: {
            LOWER_DOI: {
                "supporting": 1,
                "contradicting": 2,
                "mentioning": 3,
                "citingPublications": 6,
            }
        },
    )
    monkeypatch.setattr(
        scite_tools._scite,
        "get_papers_batch",
        lambda dois: {
            LOWER_DOI: {
                "editorialNotices": [
                    {"type": "retraction", "sourceDoi": "10.1016/x"}
                ]
            }
        },
    )

    items = [{"data": {"DOI": UPPER_DOI, "title": "Wakefield 1998"}}]
    result = scite_tools.enrich_items(items)

    # Result is keyed by the original-case DOI (what enrich_search looks up).
    assert UPPER_DOI in result
    assert "Scite" in result[UPPER_DOI]
    assert "Editorial Notices" in result[UPPER_DOI]


def test_check_retractions_flags_uppercase_doi(monkeypatch, dummy_ctx, fake_zot):
    """check_retractions must flag a retracted paper whose DOI is uppercase."""
    fake_zot._items = [
        {"key": "ITEM0001", "data": {"DOI": UPPER_DOI, "title": "Wakefield 1998"}}
    ]
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)
    monkeypatch.setattr(
        scite_tools._scite,
        "get_papers_batch",
        lambda dois: {
            LOWER_DOI: {
                "editorialNotices": [
                    {"type": "retraction", "sourceDoi": "10.1016/x"}
                ]
            }
        },
    )

    result = scite_tools.check_retractions(ctx=dummy_ctx)

    assert "Editorial Notice Alerts" in result
    assert "Retraction" in result
    assert "All clear" not in result
