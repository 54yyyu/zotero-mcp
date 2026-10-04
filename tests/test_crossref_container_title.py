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
