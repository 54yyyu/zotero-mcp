"""The landing page can outrank CrossRef on authorship, and only on that.

CrossRef is the registry of record for where a paper was published. It is
not a reliable record of who wrote it: it serves what the publisher
deposited, and older deposits are routinely truncated.

10.1006/bulm.1999.0141 (Bull. Math. Biol., 1999) is the worked case.
CrossRef deposits ONE author for a paper with four, and no abstract.
SpringerLink's page for the same DOI carries all four and the abstract.
The fixture below is that page's real ``<head>``, retrieved 2026-09-03 and
trimmed to the tags this reader looks at -- including the duplicated
``dc.creator``/``citation_author`` block, which is what a real publisher
page looks like and what a hand-built fixture would have hidden.
"""

from unittest.mock import MagicMock, patch

import pytest
from conftest import DummyContext, FakeZotero

from zotero_mcp.html_metadata import extract_embedded_metadata
from zotero_mcp.tools import write

SPRINGER_HEAD = """<html><head>
<meta name="dc.description" content="The possibility of chaos control in
 biological systems has been stimulated by recent advances."/>
<meta name="citation_pdf_url" content="https://link.springer.com/content/pdf/10.1006/bulm.1999.0141.pdf"/>
<meta name="citation_journal_title" content="Bulletin of Mathematical Biology"/>
<meta name="citation_publisher" content="Springer-Verlag"/>
<meta name="citation_issn" content="1522-9602"/>
<meta name="citation_title" content="Controlling chaos in ecology: From deterministic to individual-based models"/>
<meta name="citation_volume" content="61"/>
<meta name="citation_issue" content="6"/>
<meta name="citation_publication_date" content="1999/11"/>
<meta name="citation_firstpage" content="1187"/>
<meta name="citation_lastpage" content="1207"/>
<meta name="dc.creator" content="Sol&#233;, Ricard V."/>
<meta name="dc.creator" content="Gamarra, Javier G. P."/>
<meta name="dc.creator" content="Ginovart, Marta"/>
<meta name="dc.creator" content="L&#243;pez, Daniel"/>
<meta name="citation_author" content="Sol&#233;, Ricard V."/>
<meta name="citation_author" content="Gamarra, Javier G. P."/>
<meta name="citation_author" content="Ginovart, Marta"/>
<meta name="citation_author" content="L&#243;pez, Daniel"/>
</head><body></body></html>
"""

# CrossRef's actual record for the same DOI: one author, no abstract.
STALE_CROSSREF = {
    "type": "journal-article",
    "title": ["Controlling Chaos in Ecology: From Deterministic to Individual-based Models"],
    "container-title": ["Bulletin of Mathematical Biology"],
    "DOI": "10.1006/bulm.1999.0141",
    "volume": "61", "page": "1187-1207",
    "author": [{"given": "R", "family": "SOLE"}],
    "published": {"date-parts": [[1999]]},
}


def _creators(cr_authors):
    return [{"creatorType": "author", "firstName": a.get("given", ""),
             "lastName": a["family"]} for a in cr_authors]


def _page(*names):
    metas = "".join(f'<meta name="citation_author" content="{n}"/>' for n in names)
    return extract_embedded_metadata(f"<html><head>{metas}</head></html>").authors


class TestTheRealCase:
    """End to end, through add_by_doi, on the real page."""

    @pytest.fixture
    def fake_zot(self):
        return FakeZotero()

    def _run(self, fake_zot, crossref_msg, page_html=SPRINGER_HEAD):
        def _get(url, **kwargs):
            if "api.crossref.org" in url:
                r = MagicMock()
                r.status_code = 200
                r.raise_for_status = MagicMock()
                r.json.return_value = {"status": "ok", "message": crossref_msg}
                return r
            raise AssertionError(f"unexpected fetch: {url}")

        with patch("zotero_mcp.tools._helpers._get_write_client",
                   return_value=(fake_zot, fake_zot)), \
             patch("zotero_mcp.tools.write.requests.get", side_effect=_get), \
             patch("zotero_mcp.tools._helpers._try_attach_oa_pdf",
                   return_value="skipped"):
            write.add_by_doi(
                doi="10.1006/bulm.1999.0141",
                supplemental=extract_embedded_metadata(page_html),
                ctx=DummyContext(),
            )
        return fake_zot.created[0]

    def test_the_missing_three_authors_are_recovered(self, fake_zot):
        item = self._run(fake_zot, STALE_CROSSREF)
        # "SOLE" is CrossRef's own spelling, kept; the other three are the
        # ones it never deposited.
        assert [c["lastName"] for c in item["creators"]] == [
            "SOLE", "Gamarra", "Ginovart", "López"
        ]

    def test_crossref_entry_is_preserved_verbatim(self, fake_zot):
        """CrossRef deposited "SOLE" -- shouted, and with the accent
        dropped -- and that is what is kept. This change is about people
        the registry is MISSING, not about how it spells the ones it has.

        The cost is real and accepted: a diacritic CrossRef lost stays
        lost, because preferring the page's surname while keeping
        CrossRef's forename would be splicing one person out of two
        records on a guess. Repairing shouted names is a separate concern
        with its own commit."""
        item = self._run(fake_zot, STALE_CROSSREF)
        assert item["creators"][0]["lastName"] == "SOLE"

    def test_the_duplicated_dc_creator_block_does_not_double_the_list(
        self, fake_zot
    ):
        """The page emits every author twice, under two tag families."""
        item = self._run(fake_zot, STALE_CROSSREF)
        assert len(item["creators"]) == 4

    def test_crossref_keeps_every_field_it_populated(self, fake_zot):
        item = self._run(fake_zot, STALE_CROSSREF)
        assert item["title"].startswith("Controlling Chaos in Ecology")
        assert item["publicationTitle"] == "Bulletin of Mathematical Biology"
        assert item["volume"] == "61"

    def test_the_abstract_crossref_lacks_comes_from_the_page(self, fake_zot):
        item = self._run(fake_zot, STALE_CROSSREF)
        assert item["abstractNote"].startswith("The possibility of chaos control")

    def test_an_abstract_crossref_supplied_is_not_replaced(self, fake_zot):
        item = self._run(fake_zot,
                         dict(STALE_CROSSREF, abstract="<p>CrossRef's own.</p>"))
        assert item["abstractNote"] == "CrossRef's own."


class TestTheMergeRule:
    """Unit tests for _merge_page_authors. ``None`` means "leave CrossRef alone"."""

    def test_page_naming_more_people_supersedes(self):
        merged = write._merge_page_authors(
            _creators([{"given": "R", "family": "Sole"}]),
            _page("Solé, Ricard V.", "Gamarra, Javier G. P."),
        )
        assert [c["lastName"] for c in merged] == ["Sole", "Gamarra"]

    def test_accents_do_not_make_a_person_look_missing(self):
        """CrossRef says SOLE, the page says Solé. Same person."""
        merged = write._merge_page_authors(
            _creators([{"given": "R", "family": "Sole"}]),
            _page("Solé, Ricard V."),
        )
        assert merged is None

    def test_crossref_keeps_its_forenames_for_people_both_sources_know(self):
        """Recovering a missing author must not cost the others their
        forenames -- initials-only citation_author is house style at several
        large publishers."""
        merged = write._merge_page_authors(
            _creators([{"given": "Ricard V.", "family": "Solé"},
                       {"given": "Javier G. P.", "family": "Gamarra"}]),
            _page("Solé, R. V.", "Gamarra, J. G. P.", "Ginovart, M."),
        )
        assert [c["firstName"] for c in merged] == [
            "Ricard V.", "Javier G. P.", "M."
        ]

    def test_a_shorter_page_list_is_ignored(self):
        merged = write._merge_page_authors(
            _creators([{"given": "A", "family": "One"},
                       {"given": "B", "family": "Two"}]),
            _page("One, A."),
        )
        assert merged is None

    def test_a_differently_spelled_list_of_equal_length_is_ignored(self):
        """Not a missing person -- a spelling disagreement. CrossRef wins."""
        merged = write._merge_page_authors(
            _creators([{"given": "A", "family": "Smith"}]),
            _page("Smyth, A."),
        )
        assert merged is None

    def test_a_collaboration_is_never_expanded_into_its_members(self):
        """CrossRef deposits large collaborations as one single-field
        creator. A page listing 300 individuals is making a different claim
        about authorship, not a fuller one."""
        merged = write._merge_page_authors(
            [{"creatorType": "author", "name": "The ATLAS Collaboration"}],
            _page(*[f"Author, Number {i}" for i in range(300)]),
        )
        assert merged is None

    def test_editors_are_left_alone(self):
        """EmbeddedMetadata.authors folds citation_editor in with the
        authors, so on a book chapter the page's flat list cannot be
        reconciled with CrossRef's typed one without promoting editors."""
        cr = _creators([{"given": "A", "family": "One"}])
        cr.append({"creatorType": "editor", "firstName": "E", "lastName": "Ditor"})
        merged = write._merge_page_authors(
            cr, _page("One, A.", "Ditor, E.", "Two, B."),
        )
        assert merged is None

    def test_two_authors_sharing_a_surname_are_matched_in_order(self):
        merged = write._merge_page_authors(
            _creators([{"given": "Ann", "family": "Smith"},
                       {"given": "Bob", "family": "Smith"}]),
            _page("Smith, A.", "Smith, B.", "Jones, C."),
        )
        assert [c["firstName"] for c in merged] == ["Ann", "Bob", "C."]

    def test_an_empty_page_list_changes_nothing(self):
        assert write._merge_page_authors(
            _creators([{"given": "A", "family": "One"}]), []
        ) is None

    def test_a_page_list_for_a_record_with_no_creators_is_taken(self):
        merged = write._merge_page_authors([], _page("One, A."))
        assert [c["lastName"] for c in merged] == ["One"]
