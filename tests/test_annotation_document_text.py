"""Regression test for #287: annotation document text in semantic index.

The previous ``_create_document_text`` template was
``title + creators + abstract + tags + note``. For annotations:

- ``title`` is empty (annotations don't have titles)
- ``abstract`` is empty
- ``creators`` is ``[]`` → ``format_creators([])`` returns ``"No authors listed"``

so every annotation in the library was embedded with the literal string
``"No authors listed"``, producing identical embedding vectors. With
thousands of annotations all collapsed to one vector, semantic search
returned those entries for every query — blocking real items from
surfacing.

Fix: route annotations through a dedicated builder that uses
``annotationText`` + ``annotationComment`` + tags.
"""

from unittest.mock import MagicMock

import pytest

# semantic_search depends on chromadb; skip the module on environments
# where chroma isn't installed (it's an optional extra: ``[semantic]``).
chromadb = pytest.importorskip("chromadb")  # noqa: F841


from zotero_mcp.semantic_search import ZoteroSemanticSearch  # noqa: E402


@pytest.fixture
def search(monkeypatch):
    # Avoid network / env requirements: stub both clients.
    monkeypatch.setattr(
        "zotero_mcp.semantic_search.get_zotero_client", lambda: MagicMock()
    )
    return ZoteroSemanticSearch(chroma_client=MagicMock())


# ---------------------------------------------------------------------------
# Annotation document text
# ---------------------------------------------------------------------------


def _annotation_item(text: str = "", comment: str = "", tags: list[str] | None = None) -> dict:
    return {
        "key": "ANNO0001",
        "version": 1,
        "data": {
            "key": "ANNO0001",
            "itemType": "annotation",
            "annotationText": text,
            "annotationComment": comment,
            "annotationType": "highlight",
            "tags": [{"tag": t} for t in (tags or [])],
            "title": "",
            "creators": [],
            "abstractNote": "",
        },
    }


class TestAnnotationDocumentText:
    def test_uses_annotation_text_not_no_authors_listed(self, search):
        """The headline regression for #287."""
        ann = _annotation_item(
            text="The pre-Columbian Pueblo economy depended on …"
        )
        out = search._create_document_text(ann)
        assert "No authors listed" not in out
        assert "Pueblo economy" in out

    def test_includes_comment_when_present(self, search):
        ann = _annotation_item(
            text="The headline finding.",
            comment="My critique: small sample size.",
        )
        out = search._create_document_text(ann)
        assert "headline finding" in out
        assert "critique" in out

    def test_includes_tags(self, search):
        ann = _annotation_item(text="text", tags=["important", "review"])
        out = search._create_document_text(ann)
        assert "important" in out
        assert "review" in out

    def test_empty_annotation_returns_empty_string(self, search):
        """An annotation with no text or comment must yield empty document
        text so the upsert loop skips it instead of indexing noise."""
        ann = _annotation_item(text="", comment="")
        assert search._create_document_text(ann) == ""

    def test_two_annotations_produce_distinct_text(self, search):
        """The whole bug: identical text → identical embeddings → cluster
        of duplicates at the top of every search."""
        a = _annotation_item(text="A unique highlighted passage about X.")
        b = _annotation_item(text="A completely different passage about Y.")
        assert search._create_document_text(a) != search._create_document_text(b)


# ---------------------------------------------------------------------------
# Non-annotation items still use the standard template
# ---------------------------------------------------------------------------


def test_journal_article_unchanged(search):
    """Other item types must keep using title + creators + abstract."""
    item = {
        "key": "JART0001",
        "data": {
            "itemType": "journalArticle",
            "title": "Some Paper",
            "abstractNote": "We show that …",
            "creators": [{"firstName": "A", "lastName": "Author", "creatorType": "author"}],
            "tags": [],
        },
    }
    out = search._create_document_text(item)
    assert "Some Paper" in out
    assert "Author" in out
    assert "We show that" in out
