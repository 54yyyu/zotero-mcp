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

def format_creators(creators: list[dict[str, str]]) -> str:
    """
    Format creator names into a string.

    Args:
        creators: List of creator objects from Zotero.

    Returns:
        Formatted string with creator names.
    """
    names = []
    for creator in creators:
        if "firstName" in creator and "lastName" in creator:
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