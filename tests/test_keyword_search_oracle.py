"""`search_items_sql` against an independent statement of what it should match.

The SQL finds each term's items from the value side (title rows, distinct
creators, tags, notes) and intersects them, which is fast and easy to get
subtly wrong. The oracle below restates the rule in plain Python over the same
declared items: every word of a variant must match the title, a creator, the
year, or in 'everything' mode the abstract, a tag or a note; variants are
alternatives. A seeded random corpus and query set keeps the comparison broad
without being flaky.
"""

from __future__ import annotations

import random
import re
import sqlite3

import pytest

from _search_corpus import (
    CORPUS, DELETED_KEYS, GROUP_ID, GROUP_LIBRARY_ID, USER_LIBRARY_ID, Item,
    _FIELD_IDS, _ITEM_TYPE_IDS, build_sqlite,
)
from zotero_mcp import search_semantics as semantics
from zotero_mcp.local_db import LocalZoteroReader
from zotero_mcp.utils import _generate_search_variants

WORDS = [
    "quantum", "network", "Müller", "muller", "méthode", "learning", "deep",
    "Cladder-Micus", "50%", "under_score", "attention", "is", "all", "you",
    "need", "review", "王", "Ångström", "brain", "2011", "2019", "may",
]
FIRST = ["Hans", "Åsa", "Wei", "Mira", "Jud", None]
LAST = ["Müller", "Muller", "Brewer", "Cladder-Micus", "Van Der Berg", "王", "Big Org"]
NOTE_TEXT = "Some note text mentioning mindfulness practices"  # build_sqlite's
CHILD_NOTE_MODIFIED = "2023-01-01 00:00:00"


def _random_corpus(rng: random.Random) -> tuple[list[Item], dict[str, list[tuple[str, str]]]]:
    """Items with unique dateModified, plus child notes as {parent: [(key, text)]}."""
    items: list[Item] = []
    child_notes: dict[str, list[tuple[str, str]]] = {}
    for n in range(160):
        kind = rng.choice(["journalArticle"] * 6 + ["book", "case", "attachment", "note"])
        year = rng.choice([2011, 2012, 2019, 2020])
        month = rng.randint(1, 12)
        item = Item(
            key=f"RND{n:05d}",
            item_type=kind,
            title=None if kind == "note" else " ".join(rng.sample(WORDS, rng.randint(1, 5))),
            date=rng.choice([None, f"{year}-{month:02d}-00 {month}/{year}", f"{year}"]),
            abstract=rng.choice([None, " ".join(rng.sample(WORDS, 6))]),
            creators=[] if kind in ("note", "attachment") else [
                (rng.choice(FIRST), rng.choice(LAST)) for _ in range(rng.randint(0, 3))
            ],
            tags=rng.sample(WORDS, rng.randint(0, 2)),
            library_id=GROUP_LIBRARY_ID if n % 11 == 0 else USER_LIBRARY_ID,
            date_modified=f"2024-01-01 00:{n // 60:02d}:{n % 60:02d}",
        )
        # A creator pair repeated on one item would break the fixture's
        # (itemID, orderIndex) bookkeeping for nothing.
        item.creators = list(dict.fromkeys(item.creators))
        items.append(item)
        if kind == "journalArticle" and rng.random() < 0.3:
            child_notes[item.key] = [
                (f"CHN{n:05d}", "<p>" + " ".join(rng.sample(WORDS, 4)) + "</p>")
            ]
    return items, child_notes


def _build(tmp_path, items, child_notes, deleted=()):
    db = tmp_path / "zotero.sqlite"
    build_sqlite(db, items)
    conn = sqlite3.connect(db)
    ids = {key: item_id for item_id, key in conn.execute("SELECT itemID, key FROM items")}
    next_id = max(ids.values()) + 1
    for parent, notes in child_notes.items():
        for key, text in notes:
            conn.execute(
                "INSERT INTO items (itemID, key, itemTypeID, libraryID, dateAdded, dateModified) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (next_id, key, _ITEM_TYPE_IDS["note"], USER_LIBRARY_ID,
                 CHILD_NOTE_MODIFIED, CHILD_NOTE_MODIFIED),
            )
            conn.execute("INSERT INTO itemNotes VALUES (?, ?, ?)", (next_id, ids[parent], text))
            next_id += 1
    for key in deleted:
        conn.execute("INSERT OR IGNORE INTO deletedItems VALUES (?)", (ids[key],))
    conn.commit()
    conn.close()
    return db


def _terms(variant: str) -> list[str]:
    out = []
    for phrase, word in re.findall(r'"([^"]+)"|(\S+)', variant):
        term = phrase.strip() or word.strip(",;:.()[]\"'")
        if term:
            out.append(term)
    return out


def _oracle(items, child_notes, deleted, query, qmode, library_id=USER_LIBRARY_ID):
    """Keys `search_items_sql(query, qmode)` must return, newest-modified first."""
    variants = [semantics.normalize(v) for v in _generate_search_variants(query)]
    variants = [_terms(v) for v in dict.fromkeys(variants) if v.strip()]
    variants = [terms for terms in variants if terms]

    def haystacks(item: Item) -> tuple[list[str], str]:
        texts = [item.title or ""]
        texts += [f"{first or ''} {last or ''}".strip() for first, last in item.creators]
        if qmode == "everything":
            texts.append(item.abstract or "")
            texts += item.tags
            if item.item_type == "note":
                texts.append(NOTE_TEXT)
            texts += [text for _key, text in child_notes.get(item.key, [])]
        return [semantics.normalize(t) for t in texts], (item.date or "")[:4]

    def matches(texts: list[str], year: str) -> bool:
        return any(
            all(
                any(term in text for text in texts)
                or (re.fullmatch(r"\d{4}", term) is not None and term == year)
                for term in terms
            )
            for terms in variants
        )

    # (dateModified, insertion order) is the result order, newest first.
    hits = [
        (item.date_modified, order, item.key)
        for order, item in enumerate(items)
        if item.library_id == library_id
        and item.key not in deleted
        and item.item_type != "attachment"
        and matches(*haystacks(item))
    ]
    # A child note is an item too, and in 'everything' mode matches its own text.
    if qmode == "everything" and library_id == USER_LIBRARY_ID:
        order = len(items)
        for notes in child_notes.values():
            for key, text in notes:
                if matches([semantics.normalize(text)], ""):
                    hits.append((CHILD_NOTE_MODIFIED, order, key))
                order += 1
    return [key for _modified, _order, key in sorted(hits, reverse=True)]


def _random_query(rng: random.Random) -> str:
    shape = rng.random()
    if shape < 0.15:
        return f'"{rng.choice(WORDS)} {rng.choice(WORDS)}"'
    if shape < 0.3:
        last = rng.choice(LAST)
        return f"{last}, {rng.choice([f for f in FIRST if f])}"
    if shape < 0.4:
        return rng.choice(LAST)[: rng.randint(2, 5)] + " " + rng.choice(["2011", "2019", "may"])
    words = rng.sample(WORDS + LAST, rng.randint(1, 3))
    return " ".join(rng.choice([w, w.upper(), w[:4]]) for w in words)


@pytest.mark.parametrize("qmode", ["titleCreatorYear", "everything"])
def test_random_queries_match_the_oracle(tmp_path, qmode):
    rng = random.Random(20261118)
    items, child_notes = _random_corpus(rng)
    deleted = {item.key for item in items[::17]}
    db = _build(tmp_path, items, child_notes, deleted)

    hit_queries = 0
    with LocalZoteroReader(db_path=str(db)) as reader:
        for _ in range(250):
            query = _random_query(rng)
            expected = _oracle(items, child_notes, deleted, query, qmode)
            got = [i["key"] for i in reader.search_items_sql(query, qmode=qmode, limit=1000)]
            assert got == expected, f"{qmode} {query!r}"
            page = [i["key"] for i in reader.search_items_sql(query, qmode=qmode, limit=3)]
            assert page == expected[:3], f"{qmode} {query!r} (limit)"
            hit_queries += bool(expected)
    # The comparison is only worth something if queries actually hit.
    assert hit_queries > 100


@pytest.mark.parametrize("qmode", ["titleCreatorYear", "everything"])
def test_group_library_scope_matches_the_oracle(tmp_path, qmode):
    rng = random.Random(7)
    items, child_notes = _random_corpus(rng)
    db = _build(tmp_path, items, child_notes)
    with LocalZoteroReader(db_path=str(db)) as reader:
        for _ in range(60):
            query = _random_query(rng)
            expected = _oracle(
                items, child_notes, set(), query, qmode, library_id=GROUP_LIBRARY_ID
            )
            got = [
                i["key"] for i in reader.search_items_sql(
                    query, qmode=qmode, limit=1000, group_id=GROUP_ID
                )
            ]
            assert got == expected, f"{qmode} {query!r}"


@pytest.mark.parametrize("qmode", ["titleCreatorYear", "everything"])
def test_shared_corpus_matches_the_oracle(tmp_path, qmode):
    """The hand-written corpus every other search test uses. Its items share
    one dateModified, so sets are compared, not order."""
    db = tmp_path / "zotero.sqlite"
    build_sqlite(db)
    queries = [
        "muller", "Müller", "MÜLLER hans", "Mueller", "50%", "under_score", "Underscore_",
        "cladder micus", "Cladder-Micus", "van der berg", "quantum", "quantum 2024",
        "Doe quantum", "Doe, Jane", '"Quantum Networks"', '"Networks Quantum"',
        "marbury", "Marshall Marbury", "2018", "smith 2018", "may", "physics",
        "mindfulness", "wang", "王", "Big Organization", "monograph nordberg 2012",
    ]
    with LocalZoteroReader(db_path=str(db)) as reader:
        for query in queries:
            expected = _oracle(CORPUS, {}, DELETED_KEYS, query, qmode)
            got = {i["key"] for i in reader.search_items_sql(query, qmode=qmode, limit=1000)}
            assert got == set(expected), f"{qmode} {query!r}"


def test_title_is_read_from_the_field_the_item_type_uses(tmp_path):
    """A value sitting in another type's title field is not this item's title:
    the join form resolved the field per item type, and so must the value-side
    form (#570)."""
    db = tmp_path / "zotero.sqlite"
    build_sqlite(db, [
        Item("CASEOK01", item_type="case", title="Zebra v. Board"),
        Item("ARTICLE1", title="Plain article"),
        Item("CASEBAD1", item_type="case", title="Other case"),
    ])
    conn = sqlite3.connect(db)
    ids = {key: item_id for item_id, key in conn.execute("SELECT itemID, key FROM items")}
    conn.execute("INSERT INTO itemDataValues (valueID, value) VALUES (9001, 'Zebra stray value')")
    # An article carrying a caseName row, and a case carrying a plain title row.
    conn.execute("INSERT INTO itemData VALUES (?, ?, 9001)", (ids["ARTICLE1"], _FIELD_IDS["caseName"]))
    conn.execute("INSERT INTO itemData VALUES (?, ?, 9001)", (ids["CASEBAD1"], _FIELD_IDS["title"]))
    conn.commit()
    conn.close()

    with LocalZoteroReader(db_path=str(db)) as reader:
        assert [i["key"] for i in reader.search_items_sql("zebra")] == ["CASEOK01"]


def test_results_are_newest_modified_first_with_a_stable_tiebreak(tmp_path):
    db = tmp_path / "zotero.sqlite"
    build_sqlite(db, [Item(f"SAME{n:04d}", title="Tiebreak paper") for n in range(6)])
    with LocalZoteroReader(db_path=str(db)) as reader:
        first = [i["key"] for i in reader.search_items_sql("tiebreak", limit=4)]
        assert first == ["SAME0005", "SAME0004", "SAME0003", "SAME0002"]
        assert first == [i["key"] for i in reader.search_items_sql("tiebreak", limit=4)]
