import os
import re
import sys
from contextlib import contextmanager

from unidecode import unidecode

html_re = re.compile(r"<.*?>")


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout temporarily."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def format_creators(creators: list[dict[str, str] | str]) -> str:
    """
    Format creator names into a string.

    Args:
        creators: List of creator objects from Zotero.  Each element is
            typically a dict with firstName/lastName or name keys, but may
            also be a plain string (e.g. from BetterBibTeX results).

    Returns:
        Formatted string with creator names.
    """
    names = []
    for creator in creators:
        if isinstance(creator, str):
            names.append(creator)
        elif "firstName" in creator and "lastName" in creator:
            names.append(f"{creator['lastName']}, {creator['firstName']}")
        elif "name" in creator:
            names.append(creator["name"])
    return "; ".join(names) if names else "No authors listed"


def is_local_mode() -> bool:
    """Return True if running in local mode.

    Local mode is enabled when environment variable `ZOTERO_LOCAL` is set to a
    truthy value ("true", "yes", or "1", case-insensitive).
    """
    value = os.getenv("ZOTERO_LOCAL", "")
    return value.lower() in {"true", "yes", "1"}

def format_item_result(
    item: dict,
    index: int | None = None,
    abstract_len: int | None = 200,
    include_tags: bool = True,
    extra_fields: dict[str, str] | None = None,
) -> list[str]:
    """Format a single Zotero item as markdown lines.

    Args:
        item: Zotero item dict (with ``data`` and ``key`` keys).
        index: 1-based position for numbered headings; omit for unnumbered.
        abstract_len: Max characters for abstract (``None`` = full text,
            ``0`` = omit entirely).
        include_tags: Whether to append tags.
        extra_fields: Additional ``**Label:** value`` pairs inserted after
            authors (e.g. ``{"Similarity Score": "0.912"}``).

    Returns:
        List of markdown lines (caller joins with ``"\\n"``).
    """
    data = item.get("data", {})
    title = data.get("title", "Untitled")
    heading = f"## {index}. {title}" if index is not None else f"## {title}"
    lines: list[str] = [
        heading,
        f"**Type:** {data.get('itemType', 'unknown')}",
        f"**Item Key:** {item.get('key', '')}",
        f"**Date:** {data.get('date', 'No date')}",
        f"**Authors:** {format_creators(data.get('creators', []))}",
    ]

    # Trash status. pyzotero's default list endpoints filter trashed items
    # out, but not every call site does (e.g. includeTrashed=1, direct
    # item() lookups routed through this formatter). Defense in depth —
    # surface the flag whenever data.deleted is set so agents never silently
    # reason about a trashed paper as if it were live.
    if data.get("deleted"):
        lines.append("**Status:** 🗑️ In Trash")

    if extra_fields:
        for label, value in extra_fields.items():
            lines.append(f"**{label}:** {value}")

    if abstract_len != 0:
        abstract = data.get("abstractNote", "")
        if abstract:
            if abstract_len and len(abstract) > abstract_len:
                abstract = abstract[:abstract_len] + "..."
            lines.append(f"**Abstract:** {abstract}")

    if include_tags:
        if tags := data.get("tags"):
            tag_list = [f"`{t['tag']}`" for t in tags]
            if tag_list:
                lines.append(f"**Tags:** {' '.join(tag_list)}")

    lines.append("")  # blank separator
    return lines


def clean_html(raw_html: str, collapse_whitespace: bool = False) -> str:
    """Remove HTML/XML tags from a string.

    Args:
        raw_html: String containing HTML content.
        collapse_whitespace: If True, collapse runs of whitespace into a
            single space and strip leading/trailing whitespace. Useful for
            cleaning JATS XML from CrossRef abstracts.
    Returns:
        Cleaned string without HTML tags.
    """
    if not raw_html:
        return ""
    clean_text = re.sub(html_re, "", raw_html)
    if collapse_whitespace:
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text


# ---------------------------------------------------------------------------
# Search normalization utilities
# ---------------------------------------------------------------------------

# German umlaut expansions (common in academic literature)
_UMLAUT_MAP = {
    'ü': 'ue', 'ö': 'oe', 'ä': 'ae', 'ß': 'ss',
    'Ü': 'Ue', 'Ö': 'Oe', 'Ä': 'Ae',
}

# Dash-like Unicode characters to normalize to ASCII hyphen-minus
_DASH_PATTERN = re.compile(r'[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]')

MAX_SEARCH_VARIANTS = 15


def _normalize_for_search(text: str) -> str:
    """Normalize text for fuzzy matching: transliterate to ASCII, normalize dashes.

    Uses ``unidecode`` for broad Unicode transliteration (handles CJK, Greek,
    Cyrillic, diacritics, etc.) and a regex for dash-like characters.
    """
    if not text:
        return text
    result = unidecode(text)
    result = _DASH_PATTERN.sub('-', result)
    return result


def _generate_search_variants(query: str) -> list[str]:
    """Generate variant forms of a search query for fuzzy matching.

    Returns a deduplicated list of query variants, capped at
    ``MAX_SEARCH_VARIANTS``.  Typically produces 2-5 variants for real
    author names.
    """
    if not query or not query.strip():
        return [query] if query else []

    variants: set[str] = {query}

    # ASCII transliteration (Müller → Muller, 王 → Wang)
    ascii_form = _normalize_for_search(query)
    if ascii_form != query:
        variants.add(ascii_form)

    # Dashes to spaces (Cladder-Micus → Cladder Micus)
    dash_to_space = query.replace('-', ' ')
    if dash_to_space != query:
        variants.add(dash_to_space)
    dash_to_space_norm = ascii_form.replace('-', ' ')
    if dash_to_space_norm not in variants:
        variants.add(dash_to_space_norm)

    # German umlaut expansions (Müller → Mueller)
    umlaut_expanded = query
    for char, expansion in _UMLAUT_MAP.items():
        umlaut_expanded = umlaut_expanded.replace(char, expansion)
    if umlaut_expanded != query:
        variants.add(umlaut_expanded)

    # Spaces to dashes (Cladder Micus → Cladder-Micus)
    if ' ' in query and '-' not in query:
        space_to_dash = query.replace(' ', '-')
        variants.add(space_to_dash)

    # Cap variants
    result = list(variants)
    if len(result) > MAX_SEARCH_VARIANTS:
        result = result[:MAX_SEARCH_VARIANTS]

    return result