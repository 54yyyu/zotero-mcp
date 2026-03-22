import os
import re
import sys
from contextlib import contextmanager

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