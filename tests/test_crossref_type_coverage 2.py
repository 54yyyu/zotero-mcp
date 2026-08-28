"""Every CrossRef type must map to a Zotero type, or say that it did not.

An unmapped CrossRef type is not a labelling problem. ``add_by_doi`` writes
its fields into an item template, and the ``document`` template has no
``publicationTitle``, ``volume``, ``issue`` or ``pages`` — so an article that
fell through arrived with all four missing and nothing reported.

Seventeen of CrossRef's thirty types were missing from the table. The one
that surfaced it: DOI 10.13165/vpa-20-19-2-04, an ordinary journal article
its publisher registered as ``journal-issue``, sitting in a real library as a
``document`` with no journal, volume, issue or pages.
"""

import pytest

from zotero_mcp.tools._helpers import CROSSREF_TYPE_MAP, crossref_type_note

# https://api.crossref.org/types
CROSSREF_VOCABULARY = [
    "book-section", "monograph", "report-component", "report", "peer-review",
    "book-track", "journal-article", "book-part", "other", "book",
    "journal-volume", "book-set", "reference-entry", "proceedings-article",
    "journal", "component", "book-chapter", "proceedings-series",
    "report-series", "proceedings", "database", "standard", "reference-book",
    "posted-content", "journal-issue", "dissertation", "grant", "dataset",
    "book-series", "edited-book",
]

# Zotero's item types, as of schema version 41.
ZOTERO_ITEM_TYPES = {
    "artwork", "audioRecording", "bill", "blogPost", "book", "bookSection",
    "case", "computerProgram", "conferencePaper", "dataset", "dictionaryEntry",
    "document", "email", "encyclopediaArticle", "film", "forumPost", "hearing",
    "instantMessage", "interview", "journalArticle", "letter",
    "magazineArticle", "manuscript", "map", "newspaperArticle", "patent",
    "podcast", "preprint", "presentation", "radioBroadcast", "report",
    "standard", "statute", "thesis", "tvBroadcast", "videoRecording",
    "webpage",
}


class TestVocabularyCoverage:
    @pytest.mark.parametrize("cr_type", CROSSREF_VOCABULARY)
    def test_every_crossref_type_is_mapped(self, cr_type):
        assert cr_type in CROSSREF_TYPE_MAP, (
            f"CrossRef type {cr_type!r} falls through to 'document', which "
            "has no publicationTitle/volume/issue/pages — those values are "
            "dropped silently"
        )

    @pytest.mark.parametrize("zot_type", sorted(set(CROSSREF_TYPE_MAP.values())))
    def test_every_target_is_a_real_zotero_type(self, zot_type):
        assert zot_type in ZOTERO_ITEM_TYPES


class TestSpecificMappings:
    def test_journal_issue_keeps_article_fields(self):
        """The reported case. Publishers do register articles as
        'journal-issue'; journalArticle has somewhere to put the journal,
        volume, issue and pages that document does not."""
        assert CROSSREF_TYPE_MAP["journal-issue"] == "journalArticle"

    def test_book_chapter_is_a_book_section(self):
        assert CROSSREF_TYPE_MAP["book-chapter"] == "bookSection"

    def test_every_book_part_spelling_agrees(self):
        targets = {
            CROSSREF_TYPE_MAP[t]
            for t in ("book-chapter", "book-part", "book-section", "book-track")
        }
        assert targets == {"bookSection"}

    def test_proceedings_article_is_not_the_proceedings_volume(self):
        assert CROSSREF_TYPE_MAP["proceedings-article"] == "conferencePaper"
        assert CROSSREF_TYPE_MAP["proceedings"] == "book"

    def test_native_zotero_types_are_used(self):
        """dataset and standard used to land on 'document' even though Zotero
        has both types."""
        assert CROSSREF_TYPE_MAP["dataset"] == "dataset"
        assert CROSSREF_TYPE_MAP["standard"] == "standard"

    def test_types_with_no_equivalent_stay_document(self):
        """document is honest for these — a component is a figure, a grant is
        funding. The point is that they are decided, not defaulted."""
        for cr_type in ("component", "grant", "peer-review", "other"):
            assert CROSSREF_TYPE_MAP[cr_type] == "document"


class TestUnmappedTypeIsReported:
    def test_mapped_type_produces_no_note(self):
        assert crossref_type_note("journal-article") == ""

    def test_empty_type_produces_no_note(self):
        assert crossref_type_note("") == ""

    def test_unknown_type_is_named_with_the_remedy(self):
        """CrossRef adds types over time. A future one must not repeat the
        silent-loss failure — the caller is told what happened and how to fix
        it."""
        note = crossref_type_note("holographic-monograph")
        assert "holographic-monograph" in note
        assert "document" in note
        assert "item_type" in note
