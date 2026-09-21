"""Live: the SQLite backend must resolve base-field-mapped titles (#570).

Zotero maps the base field "title" onto a type-specific column for a few
item types — a statute's title lives in ``nameOfAct``, a case's in
``caseName``, an email's in ``subject`` (the exact set is discovered here via
``baseFieldMappingsCombined`` rather than hardcoded, since a Zotero release
can add to it). ``_ITEM_HYDRATION_SELECT_TEMPLATE`` in local_db.py — the
query behind ``search_items_sql``, ``advanced_search_sql`` and
``get_items_by_keys`` — only ever joins the literal ``title`` field
(fieldID 1), unlike the ``date``/``DOI``/``publicationTitle`` joins three
lines below it, which correctly resolve their fieldID by name. For any item
of one of these types that means:

* a title-substring search never matches it, even though the title text is
  really there (#570's first symptom); and
* the hydrated row's title is empty, and none of the alternate field is
  present either, so ``item_display_title`` has nothing left to resolve and
  falls back to "Untitled" (#570's second symptom) — even though a
  single-item read (``get_full_items``, which fetches every field generically
  by name) renders the same item correctly.

Ground truth comes straight from SQL against zotero.sqlite, never through the
reader under test — see tests/live/test_sqlite_only_reads.py for why that
matters. Gated by ZOTERO_MCP_LIVE_TESTS=1 (see conftest.py); skips cleanly,
rather than failing, in any library that happens to have none of these item
types.
"""

import re
import sqlite3

import pytest

_WORD_RE = re.compile(r"[A-Za-z]{5,}")


def _distinctive_word(title: str) -> str:
    match = _WORD_RE.search(title)
    return match.group(0) if match else title.split()[0]


@pytest.fixture(scope="session")
def base_field_title_candidates(sql_reader) -> list[dict]:
    """One real (key, itemType, actual_field, title) row per base-field
    -mapped item type this library's personal library actually has data for.
    """
    if sql_reader is None:
        pytest.skip("no readable zotero.sqlite on this machine")
    conn = sqlite3.connect(f"file:{sql_reader.db_path}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        library_row = conn.execute(
            "SELECT libraryID FROM libraries WHERE type='user'"
        ).fetchone()
        if library_row is None:
            pytest.skip("database has no personal library")
        library_id = library_row["libraryID"]

        mapped_fields = conn.execute(
            """
            SELECT DISTINCT it.typeName AS item_type, f.fieldName AS actual_field
            FROM baseFieldMappingsCombined bfm
            JOIN itemTypes it ON it.itemTypeID = bfm.itemTypeID
            JOIN fields f ON f.fieldID = bfm.fieldID
            JOIN fields base_f ON base_f.fieldID = bfm.baseFieldID
            WHERE base_f.fieldName = 'title' AND f.fieldName != 'title'
            """
        ).fetchall()

        candidates = []
        for mapping in mapped_fields:
            row = conn.execute(
                """
                SELECT i.key AS key, idv.value AS title
                FROM items i
                JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
                JOIN itemData id ON id.itemID = i.itemID
                JOIN itemDataValues idv ON idv.valueID = id.valueID
                JOIN fields f ON f.fieldID = id.fieldID AND f.fieldName = ?
                WHERE it.typeName = ? AND i.libraryID = ?
                  AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
                  AND LENGTH(idv.value) >= 5
                ORDER BY i.itemID LIMIT 1
                """,
                (mapping["actual_field"], mapping["item_type"], library_id),
            ).fetchone()
            if row is not None:
                candidates.append({
                    "key": row["key"],
                    "item_type": mapping["item_type"],
                    "actual_field": mapping["actual_field"],
                    "title": row["title"],
                })
        return candidates
    finally:
        conn.close()


@pytest.mark.timeout(60)
def test_search_items_sql_finds_base_field_mapped_titles(
    sql_reader, base_field_title_candidates
):
    """A title-substring search must find items whose title lives under a
    type-specific field, not just items using the literal 'title' column."""
    if not base_field_title_candidates:
        pytest.skip(
            "personal library has no items of a base-field-mapped type "
            "(case/email/statute/...) to test with"
        )
    from zotero_mcp.client import get_active_group_id

    failures = []
    for candidate in base_field_title_candidates:
        word = _distinctive_word(candidate["title"])
        result = sql_reader.search_items_sql(
            word, qmode="titleCreatorYear", item_type=candidate["item_type"],
            limit=50, group_id=get_active_group_id(),
        )
        found_keys = {item["key"] for item in (result or [])}
        if candidate["key"] not in found_keys:
            failures.append(
                f"{candidate['item_type']} {candidate['key']}: searching "
                f"{word!r} (from {candidate['actual_field']}={candidate['title']!r}) "
                f"did not find it"
            )
    assert not failures, "title search missed base-field-mapped items:\n" + "\n".join(
        failures
    )


@pytest.mark.timeout(60)
def test_hydrated_row_resolves_base_field_mapped_title(
    sql_reader, base_field_title_candidates
):
    """A hydrated search/list row must render the item's real title, not
    'Untitled', for a base-field-mapped item type."""
    if not base_field_title_candidates:
        pytest.skip(
            "personal library has no items of a base-field-mapped type "
            "(case/email/statute/...) to test with"
        )
    from zotero_mcp.utils import item_display_title

    keys = [candidate["key"] for candidate in base_field_title_candidates]
    hydrated = sql_reader.get_items_by_keys(keys)

    failures = []
    for candidate in base_field_title_candidates:
        item = hydrated.get(candidate["key"])
        if item is None:
            failures.append(
                f"{candidate['item_type']} {candidate['key']}: not returned by "
                f"get_items_by_keys"
            )
            continue
        displayed = item_display_title(item["data"])
        if displayed != candidate["title"]:
            failures.append(
                f"{candidate['item_type']} {candidate['key']}: displayed "
                f"{displayed!r}, expected {candidate['title']!r}"
            )
    assert not failures, "hydrated rows lost base-field-mapped titles:\n" + "\n".join(
        failures
    )


@pytest.mark.timeout(60)
def test_advanced_search_condition_matches_base_field_mapped_titles(
    sql_reader, base_field_title_candidates
):
    """A `title` *condition* must match these items, not just a free-text search.

    This is the half of #570 that outlived the first fix. The hydration
    projection resolved the base field, but ``_SIMPLE_FIELD_SQL["title"]``
    still compared a hardcoded fieldID 1 — so one statement contradicted
    itself: ``advanced_search`` could return a row whose displayed title was
    exactly the string its own WHERE clause had just failed to match on.
    """
    if not base_field_title_candidates:
        pytest.skip(
            "personal library has no items of a base-field-mapped type "
            "(case/email/statute/...) to test with"
        )
    from zotero_mcp.client import get_active_group_id

    failures = []
    for candidate in base_field_title_candidates:
        word = _distinctive_word(candidate["title"])
        result = sql_reader.advanced_search_sql(
            [{"field": "title", "operation": "contains", "value": word}],
            group_id=get_active_group_id(),
        )
        if result is None:
            # None means the backend declined and the caller would fall back
            # to pyzotero. That is not a pass — it means this assertion never
            # examined the SQL path at all.
            failures.append(
                f"{candidate['item_type']} {candidate['key']}: SQL backend "
                f"declined the query, so nothing was verified"
            )
            continue
        if candidate["key"] not in {item["key"] for item in result}:
            failures.append(
                f"{candidate['item_type']} {candidate['key']}: condition "
                f"title contains {word!r} (from {candidate['actual_field']}="
                f"{candidate['title']!r}) did not match it"
            )
    assert not failures, (
        "title conditions missed base-field-mapped items:\n" + "\n".join(failures)
    )
