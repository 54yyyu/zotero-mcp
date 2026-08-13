"""Runtime discovery of real, non-hardcoded query values.

Live parity tests must run against ANY tester's Zotero library, so no
collection name, title, author, or other library-specific string may be
hardcoded. Instead, pull a small real sample from the connected library and
derive one usable value per condition field from whatever the sample
actually contains — a field with no usable data in the sample maps to None,
and callers must skip that specific sub-test rather than fail.
"""

import re

# A whole word, not an arbitrary substring — mid-word slices (e.g. "rspe"
# from "perspective") can match differently across backends depending on
# how each one tokenizes/searches, which produces divergences that reflect
# the query being nonsensical rather than a real backend disagreement.
_WORD_RE = re.compile(r"[A-Za-z]{5,}")

# Matches a plausible 4-digit publication year (1000-2099) anywhere in a
# display-format date string, e.g. "October 1, 2016" -> "2016". Mirrors
# tools/search.py's own year-extraction regex on branches that have one;
# inlined here since this branch predates that helper.
_YEAR_RE = re.compile(r"(?<!\d)(1\d{3}|20\d{2})(?!\d)")

CONDITION_FIELDS = [
    "title", "creator", "year", "date", "dateAdded", "dateModified",
    "itemType", "tag", "publicationTitle", "abstractNote", "DOI", "collection",
]


def discover(zot) -> dict[str, str | None]:
    """Sample the connected library and return {field: value | None} for
    every field in CONDITION_FIELDS, plus a free-text 'query' entry."""
    values: dict[str, str | None] = {field: None for field in CONDITION_FIELDS}
    values["query"] = None

    try:
        items = zot.items(limit=25, itemType="-attachment")
    except Exception:
        items = []

    for item in items:
        data = item.get("data", {}) or {}

        if values["title"] is None:
            title = (data.get("title") or "").strip()
            match = _WORD_RE.search(title)
            if match:
                values["title"] = match.group(0)
                values["query"] = values["title"]

        if values["creator"] is None:
            for creator in data.get("creators") or []:
                last = creator.get("lastName")
                if last:
                    values["creator"] = last
                    break

        date = (data.get("date") or "").strip()
        if date:
            if values["date"] is None:
                values["date"] = date
            if values["year"] is None:
                match = _YEAR_RE.search(date)
                if match:
                    values["year"] = match.group(0)

        if values["dateAdded"] is None:
            date_added = data.get("dateAdded") or ""
            if len(date_added) >= 10:
                values["dateAdded"] = date_added[:10]  # YYYY-MM-DD prefix

        if values["dateModified"] is None:
            date_modified = data.get("dateModified") or ""
            if len(date_modified) >= 10:
                values["dateModified"] = date_modified[:10]

        if values["itemType"] is None:
            item_type = data.get("itemType")
            if item_type:
                values["itemType"] = item_type

        if values["tag"] is None:
            for tag_entry in data.get("tags") or []:
                tag = tag_entry.get("tag")
                if tag:
                    values["tag"] = tag
                    break

        if values["publicationTitle"] is None:
            pub_title = data.get("publicationTitle")
            if pub_title:
                values["publicationTitle"] = pub_title

        if values["abstractNote"] is None:
            abstract = (data.get("abstractNote") or "").strip()
            if len(abstract) >= 10:
                values["abstractNote"] = abstract[:10]

        if values["DOI"] is None:
            doi = data.get("DOI")
            if doi:
                values["DOI"] = doi

    try:
        collections = zot.collections(limit=5)
    except Exception:
        collections = []
    for collection in collections:
        key = (collection.get("data") or {}).get("key") or collection.get("key")
        if key:
            values["collection"] = key
            break

    return values
