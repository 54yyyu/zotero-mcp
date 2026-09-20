#!/usr/bin/env python3
"""Measure what the shared match keys change about duplicate detection.

Before #496 every detection site normalised identifiers its own way. The
duplicate grouper in `tools/write.py` keyed DOI groups on the stored field
lowercased and title groups on a rule that *deleted* punctuation;
`tools/_helpers.find_existing_items` compared DOIs case-sensitively; the
preprint filter in `semantic_search.py` glued all whitespace out of the
title. All of them now derive their keys from `zotero_mcp.identifiers`.

This counts what that does to a real library instead of arguing it from
invented examples. Both rules are run over the same items and the groups
they produce are diffed, so every number the PR quotes can be reproduced by
re-running this script. The old title rule is inlined below as
`_old_normalize_title` because the rewrite deleted it; it is kept here for
comparison only and nothing in the package calls it any more.

What this deliberately does NOT measure is whether a group is *right*. Two
items whose DOIs are equal after canonicalisation are the same work by
definition, so the DOI columns can be read as correctness. Two items whose
titles fold together may be a series, a reprint, a translation or two
unrelated papers, so the title columns measure movement only -- which is
also why they are reported in both directions.

    python scripts/measure_match_keys.py            # table
    python scripts/measure_match_keys.py --json     # machine-readable
    python scripts/measure_match_keys.py --top 10   # more examples
    python scripts/measure_match_keys.py --limit 500  # quick smoke run

Reads `zotero.sqlite` through `LocalZoteroReader`, which snapshots the WAL
when Zotero holds the database open, so it works with Zotero running. Reads
only; it never writes to the database or to Zotero. Needs a readable
zotero.sqlite (set ZOTERO_DB_PATH if the data directory is in a custom
location).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

#: How many example values to print per finding, unless --top says otherwise.
DEFAULT_TOP = 5

#: Titles are shown to make a count credible, not to be read in full.
_TITLE_WIDTH = 62

_OLD_ARTICLES = ("a ", "an ", "the ")


def _old_normalize_title(title):
    """The pre-#496 title rule, kept only for this comparison.

    Verbatim `_normalize_dup_title` from `tools/write.py` at 8778e62, the
    commit before the rewrite: lowercase, delete every non-word non-space
    character, collapse runs of whitespace, drop a leading article. Deleting
    punctuation rather than replacing it with a space is the behaviour the
    diff below is measuring.
    """
    t = (title or "").lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    for article in _OLD_ARTICLES:
        if t.startswith(article):
            t = t[len(article) :]
    return t


def _old_doi_key(raw):
    """The pre-#496 DOI rule: the stored field, stripped and lowercased."""
    return (raw or "").strip().lower()


def _multi_item_groups(pairs):
    """{key: [item_id, ...]} keeping only keys carried by two or more items.

    That is the same ">= 2 members" cut `_collect_duplicate_groups` makes,
    so a group counted here is a group the tool would offer for review.
    """
    groups = defaultdict(list)
    for key, item_id in pairs:
        groups[key].append(item_id)
    return {key: ids for key, ids in groups.items() if len(ids) >= 2}


def _identities(groups):
    """The membership of each group, as a set of frozensets of item ids.

    Comparing memberships rather than keys is what makes the diff below
    meaningful: the two rules produce different key strings for the same
    items, so keys cannot be compared, but who ends up together can.
    """
    return {frozenset(ids) for ids in groups.values()}


def _pairs(groups):
    """Every unordered pair of items that share a group.

    The group-level diff cannot tell a group that *grew* from one that
    disappeared: a three-item group that gains a fourth member counts as one
    group under each rule, so it shows up on both sides of the diff. Pairs
    do not have that problem. A pair present only under the new rule is a
    match gained; a pair present only under the old rule is a match lost,
    and that is the honest size of the narrowing.
    """
    out = set()
    for ids in groups.values():
        members = sorted(ids)
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                out.add((first, second))
    return out


def _shorten(text):
    text = " ".join((text or "").split())
    return text if len(text) <= _TITLE_WIDTH else text[: _TITLE_WIDTH - 1] + "…"


def _group_examples(identities, titles, top):
    """The largest groups from a membership set, rendered as their titles."""
    rows = []
    for members in sorted(identities, key=lambda m: (-len(m), sorted(m)))[:top]:
        rows.append(
            {
                "items": len(members),
                "titles": [_shorten(titles.get(i, "")) for i in sorted(members)][:3],
            }
        )
    return rows


def _pair_examples(pairs, titles, top):
    """Pairs rendered as the two titles involved, for reading by eye."""
    return [{"titles": [_shorten(titles.get(a, "")), _shorten(titles.get(b, ""))]} for a, b in sorted(pairs)[:top]]


def _diff(old_groups, new_groups, titles, top):
    """Both directions of a grouping change, at group and at pair level."""
    old_ids, new_ids = _identities(old_groups), _identities(new_groups)
    old_pairs, new_pairs = _pairs(old_groups), _pairs(new_groups)
    gained, lost = new_pairs - old_pairs, old_pairs - new_pairs
    return {
        "old": len(old_groups),
        "new": len(new_groups),
        "old_items": sum(len(v) for v in old_groups.values()),
        "new_items": sum(len(v) for v in new_groups.values()),
        "groups_only_new": len(new_ids - old_ids),
        "groups_only_old": len(old_ids - new_ids),
        "groups_unchanged": len(old_ids & new_ids),
        "pairs_gained": len(gained),
        "pairs_lost": len(lost),
        "groups_only_new_examples": _group_examples(new_ids - old_ids, titles, top),
        "pairs_lost_examples": _pair_examples(lost, titles, top),
    }


def _scope(reader, items, library):
    """Restrict the scan to one library, the way the tools scan one at a time.

    `zotero_find_duplicates` groups within the *active* library; a database
    holding group libraries as well would otherwise report groups that no
    single call can return. `get_key_group_map` gives each key its
    library's group id (0 is personal), and drops feeds and "My
    Publications", which have no group id and are not part of any library a
    caller can switch to.
    """
    key_groups, excluded = reader.get_key_group_map()
    if library == "all":
        return [it for it in items if it.key not in excluded], "all libraries"
    try:
        group_id = 0 if library == "personal" else int(library)
    except ValueError:
        raise SystemExit(f"--library takes 'personal', 'all' or a group id, not {library!r}") from None
    names = {gid: name for gid, name in reader.get_library_labels().values()}
    if group_id not in names:
        raise SystemExit(f"No library with group id {group_id} in this database.")
    return [it for it in items if key_groups.get(it.key) == group_id], names[group_id]


def collect(top, library="personal", limit=None):
    """Run both rules over the library and return every number the PR quotes."""
    os.environ.setdefault("ZOTERO_LOCAL", "true")

    from zotero_mcp.identifiers import (
        doi_match_key,
        metadata_match_keys,
        normalize_doi,
    )
    from zotero_mcp.local_db import get_local_zotero_reader

    reader = get_local_zotero_reader()
    if reader is None:
        raise SystemExit(
            "No readable zotero.sqlite. Set ZOTERO_LOCAL=true, and ZOTERO_DB_PATH "
            "if your Zotero data directory is in a custom location."
        )

    db_path = reader.db_path
    items, scope = _scope(reader, reader.get_items_with_text(limit=limit, include_fulltext=False), library)
    reader.close()
    if not items:
        raise SystemExit(f"No items to measure against in {scope}.")

    titles = {it.item_id: it.title or "" for it in items}

    old_doi_pairs = []
    old_title_pairs = []
    new_doi_pairs = []
    new_title_pairs = []
    for it in items:
        if key := _old_doi_key(it.doi):
            old_doi_pairs.append((key, it.item_id))
        if key := _old_normalize_title(it.title):
            old_title_pairs.append((key, it.item_id))
        # The new keys come from the function the detectors call, not from a
        # re-implementation of it, so this measures the shipped behaviour.
        for kind, value in metadata_match_keys(it):
            if kind == "doi":
                new_doi_pairs.append((value, it.item_id))
            elif kind == "title":
                new_title_pairs.append((value, it.item_id))

    old_doi = _multi_item_groups(old_doi_pairs)
    new_doi = _multi_item_groups(new_doi_pairs)
    old_title = _multi_item_groups(old_title_pairs)
    new_title = _multi_item_groups(new_title_pairs)

    # Groups that existed only because malformed DOI strings compared equal.
    garbage = {k: v for k, v in old_doi.items() if doi_match_key(k) is None}

    # Every item whose DOI field holds something, and yields no key: the
    # narrowing. The garbage groups above are this same set, counted by
    # distinct value instead of by item.
    narrowed = [it for it in items if (it.doi or "").strip() and doi_match_key(it.doi) is None]

    # A stored DOI that canonicalises to a different case is what made
    # find_existing_items miss an item it already had, and create a duplicate.
    cased = [it for it in items if doi_match_key(it.doi) and normalize_doi(it.doi) != doi_match_key(it.doi)]

    return {
        "db_path": db_path,
        "scope": scope,
        "library_items": len(items),
        "doi_groups": _diff(old_doi, new_doi, titles, top),
        "title_groups": _diff(old_title, new_title, titles, top),
        "garbage_doi": {
            "groups": len(garbage),
            "largest": max((len(v) for v in garbage.values()), default=0),
            "items": sum(len(v) for v in garbage.values()),
            # Compared against the DOI pairs lost overall: if the two agree,
            # every DOI match the new rule gives up came from a string that
            # was never a DOI, and no pair of real, equal DOIs was lost.
            "pairs": sum(len(v) * (len(v) - 1) // 2 for v in garbage.values()),
            "examples": [
                {"value": k, "items": len(v)}
                for k, v in sorted(garbage.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:top]
            ],
        },
        "doi_narrowing": {
            "items": len(narrowed),
            "shared_values": len(garbage),
        },
        "doi_case": {
            "items": len(cased),
            "examples": [{"stored": normalize_doi(it.doi), "canonical": doi_match_key(it.doi)} for it in cased[:top]],
        },
    }


def _delta(old, new):
    return f"{new - old:+,}"


def _examples(lines, rows, indent="        "):
    for row in rows:
        for offset, title in enumerate(row["titles"]):
            lines.append(f"{indent}{'·' if offset else '-'} {title}")


def render(result):
    lines = []
    add = lines.append

    add(f"Database: {result['db_path']}")
    add(f"Scanned: {result['library_items']:,} items in {result['scope']} (read-only)")
    add("")
    add("Everything in scope is grouped at once. zotero_find_duplicates refuses a")
    add("library over 5,000 items and asks for a collection_key, so this is what a")
    add("scan would find with that cap lifted, not what one call returns.")
    add("")

    header = f"{'':<30}{'old':>9}{'new':>9}{'change':>9}"
    add(header)
    add("-" * len(header))
    for label, block in (("DOI", "doi_groups"), ("title", "title_groups")):
        row = result[block]
        add(f"{label + ' groups':<30}{row['old']:>9,}{row['new']:>9,}{_delta(row['old'], row['new']):>9}")
        add(
            f"{'  items in them':<30}{row['old_items']:>9,}{row['new_items']:>9,}"
            f"{_delta(row['old_items'], row['new_items']):>9}"
        )
        add(
            f"{'  pairs gained / lost':<30}{'':>9}{'':>9}"
            f"{'+' + format(row['pairs_gained'], ',') + ' / -' + format(row['pairs_lost'], ','):>9}"
        )
    add("")

    garbage = result["garbage_doi"]
    add(
        f"Garbage-DOI groups removed: {garbage['groups']:,} "
        f"({garbage['items']:,} items, largest group {garbage['largest']:,})"
    )
    for row in garbage["examples"]:
        add(f"    {row['items']:>5,} × {row['value']!r}")
    lost = result["doi_groups"]["pairs_lost"]
    add(
        f"    They account for {garbage['pairs']:,} of the {lost:,} DOI pairs lost"
        + (" — every one, so no pair of real, equal DOIs stopped matching." if garbage["pairs"] == lost else ".")
    )
    add("")

    narrowing = result["doi_narrowing"]
    add(f"DOI narrowing: {narrowing['items']:,} items carry a non-empty DOI field that yields no key;")
    add(f"    {narrowing['shared_values']:,} distinct lowercased raw values among them are shared by two or")
    add("    more items — those are exactly the groups removed above.")
    add("")

    cased = result["doi_case"]
    add(f"Stored DOIs that differ in case from their canonical form: {cased['items']:,}")
    add("    Re-adding any of these in lowercase used to create a second copy.")
    for row in cased["examples"]:
        add(f"      {row['stored']}  ->  {row['canonical']}")
    add("")

    diff = result["title_groups"]
    add("Title grouping moves in both directions:")
    add(f"    groups only under the new rule:  {diff['groups_only_new']:,}")
    add(f"    groups only under the old rule:  {diff['groups_only_old']:,}")
    add(f"    groups identical under both:     {diff['groups_unchanged']:,}")
    add("    A group that only grew is counted on both sides, so read the pair counts for")
    add(
        f"    the real movement: {diff['pairs_gained']:,} item pairs now match that "
        f"did not, {diff['pairs_lost']:,} no longer match."
    )
    add("")
    add("    Groups that exist only under the new rule:")
    _examples(lines, diff["groups_only_new_examples"])
    add("")
    add("    Pairs that matched before and no longer do:")
    _examples(lines, diff["pairs_lost_examples"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of a table")
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help=f"how many examples to show per finding (default {DEFAULT_TOP})",
    )
    parser.add_argument(
        "--library",
        default="personal",
        help="which library to scan: 'personal' (default — the active library a "
        "duplicate scan sees), 'all', or a Zotero group id",
    )
    parser.add_argument("--limit", type=int, default=None, help="scan only the N most recently modified items")
    args = parser.parse_args()

    result = collect(max(0, args.top), args.library, args.limit)
    print(json.dumps(result, indent=2) if args.json else render(result))


if __name__ == "__main__":
    main()
