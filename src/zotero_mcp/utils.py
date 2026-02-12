from typing import List, Dict
import os
import re

html_re = re.compile(r"<.*?>")

def format_creators(creators: list[dict[str, str]], concise: bool = False) -> str:
    """
    Format creator names into a string.

    Args:
        creators: List of creator objects from Zotero.
        concise: If True, uses abbreviated format (e.g., "Smith et al." for multiple authors).

    Returns:
        Formatted string with creator names.
    """
    if not creators:
        return "No authors listed"

    if concise:
        # In concise mode, show first author and "et al." if multiple
        if len(creators) == 1:
            creator = creators[0]
            if "lastName" in creator:
                return creator["lastName"]
            elif "name" in creator:
                return creator["name"].split(",")[0].split(" ")[0]
        else:
            creator = creators[0]
            if "lastName" in creator:
                return f"{creator['lastName']} et al."
            elif "name" in creator:
                return f"{creator['name'].split(',')[0]} et al."

    # Full mode (existing behavior)
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

def clean_html(raw_html: str) -> str:
    """
    Remove HTML tags from a string.

    Args:
        raw_html: String containing HTML content.
    Returns:
        Cleaned string without HTML tags.
    """
    clean_text = re.sub(html_re, "", raw_html)
    return clean_text
