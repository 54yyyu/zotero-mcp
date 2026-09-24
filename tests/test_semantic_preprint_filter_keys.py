"""T1.7: the local-scan preprint filter uses the shared match keys (#496).

``_get_items_from_local_db`` drops a preprint when a journalArticle of the
same work is also present in the scan, so the semantic index keeps one copy.
Before this change, "same work" was decided by a local ``norm()`` that glued
together all whitespace in the DOI/title; after, it goes through the shared
``metadata_match_keys()``, filtered to the ``doi``/``title`` kinds — the same
primitive the duplicate detector uses. That is a deliberate narrowing in one
respect (punctuation is folded to a *separator*, not deleted), which the
third test below pins.

Note on the DOI fixture: the plan's own example DOI (``10.1/x``) has a
one-digit registrant code and is rejected by this project's pre-existing
``DOI_RE`` (``^10\\.\\d{4,9}/\\S+$``), so ``doi_match_key("10.1/x")`` is
``None`` for both forms and would not exercise the DOI branch at all. Using
a valid DOI (``10.1000/x``) is required to actually test doi_match_key's
case-fold + URL-strip behavior.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search
from zotero_mcp.local_db import ZoteroItem


class _FakeChromaClient:
    """No existing documents — every candidate is treated as new."""

    def get_document_metadata(self, key):
        return None


class _FakeReader:
    """LocalZoteroReader double serving a fixed item list to the local
    scan's phase-1 (metadata-only) path, which is where the preprint filter
    runs."""

    attachment_priority = ("pdf", "html")

    def __init__(self, items):
        self._items = items

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_all_item_keys(self):
        return {it.key for it in self._items}

    def get_key_group_map(self):
        return ({it.key: 0 for it in self._items}, set())

    def get_items_with_text(self, limit=None, include_fulltext=False, collection_keys=None):
        return list(self._items)


def _search(monkeypatch, items):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    monkeypatch.setattr(semantic_search, "LocalZoteroReader", lambda **kw: _FakeReader(items))
    return semantic_search.ZoteroSemanticSearch(chroma_client=_FakeChromaClient())


def _journal(key, **kw):
    return ZoteroItem(item_id=1, key=key, item_type_id=1, item_type="journalArticle", **kw)


def _preprint(key, **kw):
    return ZoteroItem(item_id=2, key=key, item_type_id=2, item_type="preprint", **kw)


def test_doi_case_and_url_form_are_the_same_work(monkeypatch):
    """journalArticle DOI given as a doi.org URL vs preprint DOI given bare
    with different case: doi_match_key folds both to the same key, so the
    preprint is dropped."""
    journal = _journal("J1", doi="https://doi.org/10.1000/x", title="Some Title")
    preprint = _preprint("P1", doi="10.1000/X", title="Some Other Title")
    search = _search(monkeypatch, [journal, preprint])

    items = search._get_items_from_local_db(extract_fulltext=False)

    assert {it["key"] for it in items} == {"J1"}


def test_en_dash_vs_hyphen_title_pair_is_the_same_work(monkeypatch):
    """Titles differing only by en-dash vs hyphen: normalize_title_for_matching
    folds both dash characters to a space, so the two titles normalize to the
    same key and the preprint is dropped."""
    journal = _journal("J2", title="Pre–training Language Models")
    preprint = _preprint("P2", title="Pre-training Language Models")
    search = _search(monkeypatch, [journal, preprint])

    items = search._get_items_from_local_db(extract_fulltext=False)

    assert {it["key"] for it in items} == {"J2"}


def test_self_attention_vs_selfattention_is_kept(monkeypatch):
    """Pins the narrowing. The old glue-all-whitespace norm() reduced both
    "self attention" and "selfattention" to "selfattention" and would have
    dropped the preprint. normalize_title_for_matching maps a space to a
    separator (a no-op on a space) rather than deleting it, so the two
    titles stay distinct and the preprint is kept."""
    journal = _journal("J3", title="self attention")
    preprint = _preprint("P3", title="selfattention")
    search = _search(monkeypatch, [journal, preprint])

    items = search._get_items_from_local_db(extract_fulltext=False)

    assert {it["key"] for it in items} == {"J3", "P3"}
