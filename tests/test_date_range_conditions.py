"""Date range conditions compare real dates, never display text (#551).

On the API path `date isAfter "2024"` compared Zotero's display string
lexically, so "Nov/Dec 1990" (n sorts after 2) matched, and an item with no
date matched `isBefore "1900"`. Both paths now compare the 00-padded ISO date.
"""

import pytest

from zotero_mcp import search_semantics as sem


@pytest.mark.parametrize("parsed, display, expected", [
    ("2009-06", "June 2009", "2009-06-00"),
    ("1983-07", "Jul., 1983", "1983-07-00"),
    ("2022-09", "9/2022", "2022-09-00"),
    ("2020-03-01", "2020/03/01", "2020-03-01"),
    ("2017-09-15", "September 15-17, 2017", "2017-09-15"),
    ("2024", "2024", "2024-00-00"),
    ("0000", "in press", "0000-00-00"),
    (None, "Nov/Dec 1990", "1990-00-00"),
    (None, "in press", "0000-00-00"),
    (None, "", None),
    ("", None, None),
])
def test_date_range_key(parsed, display, expected):
    assert sem.date_range_key(parsed, display) == expected


# Display forms from the report, with the parsedDate Zotero returns for them.
_REPORT = [
    ("October 1, 2016", "2016-10-01"),
    ("Nov/Dec 1990", "1990-12"),
    ("9/1993", "1993-09"),
    ("Sep 2022", "2022-09"),
    ("Jun 8-Jun 11, 2016", "2016-06-08"),
    ("2018-05-01", "2018-05-01"),
    ("2019", "2019"),
    ("03/2021", "2021-03"),
    ("2022/11/28", "2022-11-28"),
]


def _match(parsed, display, value, op):
    key = sem.date_range_key(parsed, display)
    return sem.matches([key] if key else [], value, op)


@pytest.mark.parametrize("display, parsed", _REPORT)
def test_nothing_before_2024_matches_after_2024(display, parsed):
    assert _match(parsed, display, "2024", "isAfter") is False


@pytest.mark.parametrize("display, parsed", _REPORT)
def test_nothing_after_1900_matches_before_1900(display, parsed):
    assert _match(parsed, display, "1900", "isBefore") is False


def test_an_item_with_no_date_matches_no_range():
    for op, value in [("isBefore", "1900"), ("isAfter", "1900"), ("isLessThan", "2030")]:
        assert _match(None, "", value, op) is False


def test_year_only_date_orders_like_zoteros_sql():
    """SQL compares "2024-00-00" > "2024", so a year-only 2024 item is after "2024"."""
    assert _match("2024", "2024", "2024", "isAfter") is True
    assert _match("2024", "2024", "2024-06-01", "isAfter") is False
