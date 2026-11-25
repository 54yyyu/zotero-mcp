"""CrossRef ``container-title`` lands in each item type's own container field.

bookSection and conferencePaper templates have no ``publicationTitle``; their
container is ``bookTitle`` / ``proceedingsTitle``. Writing only to
``publicationTitle`` dropped the book or proceedings name for every chapter
and conference paper added by DOI.
"""

import pytest

from zotero_mcp.tools.write import _crossref_to_item_data

_TEMPLATES = {
    "journalArticle": {"itemType": "journalArticle", "title": "", "creators": [],
                       "publicationTitle": "", "volume": "", "pages": "",
                       "date": "", "DOI": "", "url": ""},
    "bookSection": {"itemType": "bookSection", "title": "", "creators": [],
                    "bookTitle": "", "publisher": "", "pages": "", "ISBN": "",
                    "date": "", "DOI": "", "url": ""},
    "conferencePaper": {"itemType": "conferencePaper", "title": "", "creators": [],
                        "proceedingsTitle": "", "publisher": "", "pages": "",
                        "date": "", "DOI": "", "url": ""},
}


def _tmpl(zot_type):
    return dict(_TEMPLATES[zot_type])


@pytest.mark.parametrize("cr_type,field", [
    ("journal-article", "publicationTitle"),
    ("book-chapter", "bookTitle"),
    ("proceedings-article", "proceedingsTitle"),
])
def test_container_title_goes_to_type_specific_field(cr_type, field):
    cr = {"type": cr_type, "title": ["A Paper"],
          "container-title": ["The Container"], "page": "1-10"}
    item, _, _ = _crossref_to_item_data(cr, "10.1000/x", _tmpl)
    assert item[field] == "The Container"


# A chapter in a book series: CrossRef lists the series first and the book
# last, and the chapter's ISSN is the series'. Taking the first title filed
# "Lecture Notes in Computer Science" as U-Net's book (CrossRef record of
# 10.1007/978-3-319-24574-4_28, trimmed).
_SERIES_CHAPTER = {
    "type": "book-chapter",
    "title": ["U-Net: Convolutional Networks for Biomedical Image Segmentation"],
    "container-title": ["Lecture Notes in Computer Science",
                        "Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015"],
    "ISSN": ["0302-9743", "1611-3349"], "ISBN": ["9783319245737", "9783319245744"],
    "publisher-location": "Cham", "page": "234-241",
}


def _tmpl_full(zot_type):
    t = dict(_TEMPLATES[zot_type])
    t.update({"series": "", "place": "", "ISBN": "", "ISSN": ""})
    return t


def test_series_chapter_book_title_is_the_book_not_the_series():
    item, _, _ = _crossref_to_item_data(_SERIES_CHAPTER, "10.1007/978-3-319-24574-4_28", _tmpl_full)
    assert item["bookTitle"] == "Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015"
    assert item["series"] == "Lecture Notes in Computer Science"
    assert item["ISBN"] == "9783319245737 9783319245744"
    assert item["place"] == "Cham"


def test_two_titles_without_issn_keep_the_first():
    cr = {"type": "proceedings-article", "title": ["A Paper"],
          "container-title": ["Proc. of the X Conference", "X Conference Proceedings"]}
    item, _, _ = _crossref_to_item_data(cr, "10.1000/x", _tmpl_full)
    assert item["proceedingsTitle"] == "Proc. of the X Conference"
    assert item["series"] == ""


def test_parent_editors_filled_from_the_book(monkeypatch):
    from zotero_mcp.tools import write
    asked = []

    def fake_fetch(doi, ctx):
        asked.append(doi)
        return ("fetched", {"editor": [{"given": "Nassir", "family": "Navab"}]})

    monkeypatch.setattr(write, "_fetch_one_doi_metadata", fake_fetch)
    cr = {**_SERIES_CHAPTER, "relation": {"is-part-of": [
        {"id-type": "doi", "id": "https://doi.org/10.1007/978-3-319-24574-4"}]}}
    out = write._with_parent_editors(cr, ctx=None)
    assert asked == ["10.1007/978-3-319-24574-4"]
    assert out["editor"] == [{"given": "Nassir", "family": "Navab"}]
    # a record that already has editors, or no parent, is left alone
    assert write._with_parent_editors({**cr, "editor": [{"family": "X"}]}, None)["editor"] == [{"family": "X"}]
    assert write._with_parent_editors(_SERIES_CHAPTER, None) is _SERIES_CHAPTER
