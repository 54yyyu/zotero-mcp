"""find_by_citation_key on the SQLite backend, which looks the key up from the
value side instead of walking every item's fields."""

import pytest

from _full_db import FullDb
from zotero_mcp.local_db import LocalZoteroReader


@pytest.fixture
def reader(tmp_path):
    db = FullDb(tmp_path / "zotero.sqlite")
    db.item("FIELD001", fields={"title": "Field key", "citationKey": "smith2020"})
    db.item("TRASHED1", fields={"citationKey": "gone2019"}, deleted=True)
    db.item("LEGACY01", fields={"extra": "PMID: 1\nCitation Key: doe2018\nfoo: bar"})
    db.item("TITLE001", fields={"title": "brewer2011"})
    # Thirty near misses, then the real one: the old query took 25 LIKE
    # candidates in scan order and gave up.
    for n in range(30):
        db.item(f"NEAR{n:04d}", fields={"extra": f"Citation Key: lee2015x{n:02d}"})
    db.item("ZZLEGACY", fields={"extra": "Citation Key: lee2015"})
    with LocalZoteroReader(db_path=db.done()) as r:
        yield r


def _key(item):
    return item and item["key"]


def test_finds_the_citation_key_field(reader):
    item = reader.find_by_citation_key("smith2020")
    assert _key(item) == "FIELD001"
    assert item["data"]["title"] == "Field key"


def test_finds_a_legacy_extra_line(reader):
    assert _key(reader.find_by_citation_key("doe2018")) == "LEGACY01"


def test_legacy_key_is_found_behind_more_than_25_near_misses(reader):
    assert _key(reader.find_by_citation_key("lee2015")) == "ZZLEGACY"


def test_match_is_exact_and_on_the_right_field(reader):
    assert reader.find_by_citation_key("smith20") is None
    assert reader.find_by_citation_key("doe201") is None
    assert reader.find_by_citation_key("brewer2011") is None  # a title, not a key
    assert reader.find_by_citation_key("lee2015x") is None


def test_trashed_items_and_other_libraries_are_not_returned(reader):
    assert reader.find_by_citation_key("gone2019") is None
    assert reader.find_by_citation_key("smith2020", group_id=4242) is None
