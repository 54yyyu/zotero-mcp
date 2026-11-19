"""Every date-typed field leaves the SQLite reader the way the API returns it."""

from zotero_mcp.local_db import _row_to_full_api_item

_ROW = {"key": "CASE0001", "version": 3, "dateAdded": "2024-01-02 03:04:05",
        "dateModified": "2024-01-02 03:04:05", "parentKey": None, "deleted": 0}


def _item(item_type, fields):
    return _row_to_full_api_item({**_ROW, "itemType": item_type}, fields, [], [], [], {}, None)["data"]


def test_type_specific_date_fields_lose_the_sort_prefix():
    assert _item("case", {"dateDecided": "2011-03-00 March 2011"})["dateDecided"] == "March 2011"
    assert _item("patent", {"issueDate": "2019-07-04 July 4, 2019"})["issueDate"] == "July 4, 2019"
    assert _item("journalArticle", {"date": "2020-05-01 2020-05-01"})["date"] == "2020-05-01"


def test_access_date_is_an_iso_timestamp():
    assert _item("webpage", {"accessDate": "2024-01-02 03:04:05"})["accessDate"] == "2024-01-02T03:04:05Z"


def test_other_fields_with_a_space_are_untouched():
    assert _item("journalArticle", {"title": "2020-05-01 A Day"})["title"] == "2020-05-01 A Day"
