import os
import re
import sys
import threading
from contextlib import contextmanager

from unidecode import unidecode

html_re = re.compile(r"<.*?>")

# Distribution name on PyPI, used to build install/upgrade hints.
PACKAGE_NAME = "zotero-mcp-server"


def detect_install_flavor() -> str | None:
    """Best-effort detection of how this package was installed.

    ``uv tool install`` places the package under ``.../uv/tools/<name>/lib/...``
    and pipx under ``.../pipx/venvs/<name>/lib/...``. Anything else (venv,
    conda, system site-packages) is most likely pip-managed, but we cannot
    prove it, so it is reported as unknown (``None``).

    Returns:
        ``"uv"``, ``"pipx"``, or ``None`` when the flavor is undetermined.
    """
    path = os.path.abspath(__file__).replace("\\", "/")
    if "/uv/tools/" in path:
        return "uv"
    if "/pipx/venvs/" in path:
        return "pipx"
    return None


def install_command(extra: str | None = None, flavor: str | None = None) -> str:
    """Return the command that installs/upgrades the package with *extra*.

    Args:
        extra: Optional extras name (e.g. ``"semantic"``, ``"pdf"``).
        flavor: Override for the detected installer ("uv", "pipx", "pip").
    """
    target = f"{PACKAGE_NAME}[{extra}]" if extra else PACKAGE_NAME
    flavor = flavor or detect_install_flavor()
    if flavor == "uv":
        return f"uv tool install --upgrade '{target}'"
    if flavor == "pipx":
        return f"pipx install --force '{target}'"
    return f"pip install '{target}'"


def install_hint(extra: str | None = None) -> str:
    """Install instruction matching how zotero-mcp was actually installed.

    A hardcoded ``pip install`` line is wrong — and silently does nothing
    useful — for ``uv tool``/pipx installs (issue #388). When the flavor is
    unambiguous we print only the command that works there; otherwise we print
    the pip command together with the uv and pipx equivalents so no user is
    left with a command that cannot work for them.
    """
    flavor = detect_install_flavor()
    if flavor:
        return f"Install it with: {install_command(extra, flavor)}"
    return (
        f"Install it with: {install_command(extra, 'pip')} "
        f"(uv: {install_command(extra, 'uv')}; "
        f"pipx: {install_command(extra, 'pipx')})"
    )


# State for suppress_stdout. It swaps the process-global sys.stdout, so
# concurrent users have to be counted rather than each saving and restoring
# their own idea of "the real stdout" (#431).
_stdout_lock = threading.Lock()
_stdout_depth = 0
_stdout_devnull = None
_stdout_original = None


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout temporarily.

    Reference-counted under a lock. Two MCP tool threads running a semantic
    search at the same time used to interleave their save/restore of the
    global ``sys.stdout``: the one that exited last restored a value it had
    captured while stdout was already redirected, leaving the global pointing
    at a closed devnull. Every later write to stdout then failed, which on the
    stdio transport reads as the server dropping the connection (#431). Only
    the first entrant redirects and only the last one restores; the lock is
    held for the bookkeeping alone, never for the body.
    """
    global _stdout_depth, _stdout_devnull, _stdout_original

    with _stdout_lock:
        if _stdout_depth == 0:
            _stdout_original = sys.stdout
            _stdout_devnull = open(os.devnull, "w")
            sys.stdout = _stdout_devnull
        _stdout_depth += 1
    try:
        yield
    finally:
        with _stdout_lock:
            _stdout_depth -= 1
            if _stdout_depth == 0:
                sys.stdout = _stdout_original
                devnull, _stdout_devnull = _stdout_devnull, None
                _stdout_original = None
                if devnull is not None:
                    try:
                        devnull.close()
                    except Exception:
                        pass

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


# ---------------------------------------------------------------------------
# Pagination helper
# ---------------------------------------------------------------------------

def _paginate(zot_method, *args, max_items=None, **kwargs):
    """Fetch all results from a pyzotero method using manual pagination.

    Avoids zot.everything() which can cause RLock pickling in MCP contexts.
    Accepts the same positional and keyword arguments as the wrapped method,
    plus an optional max_items to cap the total results.
    """
    items = []
    start = 0
    page_size = 100
    while True:
        batch = zot_method(*args, start=start, limit=page_size, **kwargs)
        if not batch:
            break
        items.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
        if max_items and len(items) >= max_items:
            break
    # Trimmed on the way out rather than only on the early-exit path. The cap
    # used to be applied inside the loop, which the last page skips: a run
    # that ended on a short batch returned everything it had fetched, however
    # small max_items was. Callers that pass it to bound a *response* — not
    # just the fetching — then over-reported (#453).
    if max_items:
        return items[:max_items]
    return items


def get_search_backend() -> str:
    """Return the configured metadata search backend: ``"sqlite"`` or ``"api"``.

    Controlled by ``ZOTERO_SEARCH_BACKEND`` (#167); any value other than
    ``"sqlite"`` — including unset — falls back to ``"api"``, the pyzotero-based
    path every deployment already uses. The ``sqlite`` backend additionally
    requires local mode and a readable ``zotero.sqlite``; callers fall back to
    ``"api"`` at the query site when that's not the case.
    """
    value = os.getenv("ZOTERO_SEARCH_BACKEND", "").strip().lower()
    return "sqlite" if value == "sqlite" else "api"


def item_display_title(data: dict) -> str:
    """The title to show for an item, whatever field it actually lives in.

    Three item shapes do not keep their title under ``title`` and each used to
    render as "Untitled" here:

    * Type-specific base fields — a statute's title is ``nameOfAct``, a case's
      is ``caseName``, an email's is ``subject``. ``schema.resolve_field``
      maps ``title`` onto whichever key the type uses (#452).
    * Standalone attachments, which have a ``filename`` and no title.
    * Notes, whose title is the first line of their body (#447).

    Shared with :func:`zotero_mcp.client.format_item_metadata` so a search
    result and an item lookup never disagree about what a paper is called.
    """
    item_type = data.get("itemType", "")

    if item_type == "note":
        return note_title(data.get("note", ""))

    if item_type:
        try:
            from zotero_mcp import schema as _schema

            resolved = _schema.resolve_field(item_type, "title")
        except Exception:  # schema unavailable — fall back to the plain field
            resolved = "title"
        if title := data.get(resolved):
            return title

    return data.get("title") or data.get("filename") or "Untitled"


def item_display_date(data: dict) -> str:
    """The date to show for an item, whatever field it actually lives in.

    The mirror of :func:`item_display_title` for the ``date`` base field, and
    the other half of #452. A case's date is ``dateDecided``, a statute's is
    ``dateEnacted``, a patent's is ``issueDate``; reading ``data["date"]``
    directly found nothing and rendered "No date" over a date Zotero holds
    perfectly well.

    Returns "" when there is genuinely no date, so callers choose their own
    placeholder.
    """
    item_type = data.get("itemType", "")
    if item_type:
        try:
            from zotero_mcp import schema as _schema

            resolved = _schema.resolve_field(item_type, "date")
        except Exception:  # schema unavailable — fall back to the plain field
            resolved = "date"
        if date := data.get(resolved):
            return str(date)

    return str(data.get("date") or "")


def library_label(item: dict) -> str | None:
    """Human-readable source library for one item, or None if unattributed.

    Reads the top-level ``library`` key that pyzotero items carry and the
    SQLite backend stamps via ``row_to_api_item``. Renders the personal
    library as "My Library (personal)" and a group as
    "<name> (groupID=<id>)", so a global search's results say where each hit
    lives (#163).
    """
    library = item.get("library")
    if not isinstance(library, dict):
        return None
    name = str(library.get("name") or "").strip()
    library_id = library.get("id")
    if library.get("type") in ("user", "users"):
        return f"{name or 'My Library'} (personal)"
    if library_id is None:
        return name or None
    return f"{name or 'Group'} (groupID={library_id})"


def format_item_result(
    item: dict,
    index: int | None = None,
    abstract_len: int | None = 200,
    include_tags: bool = True,
    extra_fields: dict[str, str] | None = None,
    show_library: bool = False,
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
        show_library: Add a ``**Library:**`` line naming the item's source
            library. Off by default so single-library output is unchanged;
            global searches turn it on, where the label is the whole point.

    Returns:
        List of markdown lines (caller joins with ``"\\n"``).
    """
    data = item.get("data", {})
    title = item_display_title(data)
    heading = f"## {index}. {title}" if index is not None else f"## {title}"
    lines: list[str] = [
        heading,
        f"**Type:** {data.get('itemType', 'unknown')}",
        f"**Item Key:** {item.get('key', '')}",
        f"**Date:** {item_display_date(data) or 'No date'}",
        f"**Authors:** {format_creators(data.get('creators', []))}",
    ]

    if show_library and (label := library_label(item)):
        lines.insert(3, f"**Library:** {label}")

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


#: Closing tags of line-level blocks. Dropped rather than turned into a
#: newline: the *opening* tag of the next item already supplies one, and
#: emitting both would put a blank line between every pair of list items.
_LINE_BREAK_CLOSE_RE = re.compile(r"</(?:li|tr|dt|dd)\s*>", re.IGNORECASE)

#: Tags that end a line but not a paragraph — a list item, a table row, an
#: explicit break. These get a single newline.
_LINE_BREAK_RE = re.compile(r"<(?:br|li|tr|dt|dd)\b[^>]*>", re.IGNORECASE)

#: Tags that end a paragraph-level block. These get a blank line, so prose
#: stays readable as markdown rather than becoming one wall of text.
_PARA_BREAK_RE = re.compile(
    r"</?(?:p|div|ul|ol|table|h[1-6]|blockquote|pre|section|article)\b[^>]*>",
    re.IGNORECASE,
)


def html_to_text(raw_html: str) -> str:
    """Strip HTML to plain text, preserving block structure as line breaks.

    :func:`clean_html` removes tags without putting anything in their place,
    which is right for an inline fragment and wrong for a document. Zotero
    stores notes as HTML with no newlines between blocks, so
    ``<p>Title</p><p>Body</p>`` came back as ``TitleBody`` — which, among
    other things, made the note's "first line" the whole note (#447).
    """
    if not raw_html:
        return ""
    text = _PARA_BREAK_RE.sub("\n\n", raw_html)
    text = _LINE_BREAK_CLOSE_RE.sub("", text)
    text = _LINE_BREAK_RE.sub("\n", text)
    text = clean_html(text)
    # Substitution leaves runs of blank lines behind (a `</p><p>` pair emits
    # four newlines). Normalise to at most one blank line, and drop the
    # trailing spaces each line may have picked up.
    lines = [line.strip() for line in text.splitlines()]
    out: list[str] = []
    for line in lines:
        if not line and (not out or not out[-1]):
            continue  # leading, or a second consecutive blank
        out.append(line)
    return "\n".join(out).strip()


def note_title(note_html: str, max_chars: int = 80) -> str:
    """Derive a display title for a note from its content.

    Notes are the one item type with no `title` field: Zotero's own client
    shows the note's first line instead, and the API has nothing to offer in
    its place. Formatting a note through the generic item path therefore
    rendered every one of them as "Untitled" (#447).
    """
    text = html_to_text(note_html or "")
    if not text:
        return "Untitled Note"
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if not first_line:
        return "Untitled Note"
    if len(first_line) > max_chars:
        return first_line[:max_chars].rstrip() + "…"
    return first_line


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
