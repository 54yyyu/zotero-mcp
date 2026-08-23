"""
Local Zotero database reader for semantic search.

Provides direct SQLite access to Zotero's local database for faster semantic search
when running in local mode.
"""

import json
import logging
import os
import platform
import re
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from . import fulltext_cache
from . import search_semantics as _semantics
from .config import load_config
from .extract import (
    categorize_attachment,
    extract_file,
    normalize_attachment_priority,
    pick_by_priority,
)
from .utils import _generate_search_variants, _normalize_for_search, is_local_mode

logger = logging.getLogger(__name__)

# Pages extracted per PDF when nothing else is configured. Overridable via
# the ``pdf_max_pages`` extraction config or ``ZOTERO_PDF_MAXPAGES``.
#
# Set for headroom, not for recall. What actually bounds indexed text is
# downstream: ~8k embedding tokens for a whole-document row, or
# ``chunking.max_chunks_per_item`` windows when chunking is on — both land
# near 7-8 pages of a typical paper (~3.9k chars/page). Raising this alone
# therefore does not widen what the index sees; it only stops extraction
# being the binding limit for anyone who raises those downstream settings.
#
# It is close to free either way: pdf-inspector computes font statistics
# across the whole document before emitting any page, so parse time is
# per-document, not per-page — measured at ~380 ms/doc whether the cap is
# 10 or 50.
DEFAULT_PDF_MAX_PAGES = 50

# Library identity used throughout zotero-mcp (ChromaDB metadata, the
# `zotero_switch_library` tool, etc.): 0 for the personal ("user") library,
# else the Zotero server-assigned groupID. This matches Zotero's own
# sentinel for the account-less personal library.
PERSONAL_LIBRARY_GROUP_ID = 0


class KeyGroupMap(NamedTuple):
    """Result of :meth:`LocalZoteroReader.get_key_group_map`.

    ``groups`` maps every non-deleted item key in the database to its
    library's group_id. ``excluded_keys`` holds keys from libraries with no
    group_id equivalent (feeds, "My Publications") — these are excluded from
    the semantic index and global search rather than mis-tagged as personal.
    """

    groups: dict[str, int]
    excluded_keys: set[str]


def _read_string_pref(prefs_path: Path, pref: str) -> str | None:
    """Read a string preference from a Zotero prefs.js file.

    Returns None if the file cannot be read or the preference is absent.
    """
    try:
        text = prefs_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(
        r'user_pref\("' + re.escape(pref) + r'",\s*"([^"]*)"\)',
        text,
    )
    if not m:
        return None
    raw = m.group(1)
    # prefs.js values are JavaScript string literals; unescape backslash
    # sequences so Windows paths like C:\\Users\\... resolve correctly.
    try:
        return json.loads(f'"{raw}"')
    except ValueError:
        return raw


def _zotero_profiles_dirs() -> list[Path]:
    """Return OS-specific directories that may contain Zotero profiles."""
    system = platform.system()
    home = Path.home()
    if system == "Darwin":
        return [home / "Library" / "Application Support" / "Zotero" / "Profiles"]
    if system == "Windows":
        appdata = os.getenv("APPDATA")
        return [Path(appdata) / "Zotero" / "Zotero" / "Profiles"] if appdata else []
    # Linux and others: profile folders live directly under ~/.zotero/zotero
    return [home / ".zotero" / "zotero"]


def _profile_prefs_files() -> list[Path]:
    """Return prefs.js files from all Zotero profiles found on this system."""
    prefs_files: list[Path] = []
    for profiles_dir in _zotero_profiles_dirs():
        if profiles_dir.is_dir():
            prefs_files.extend(sorted(profiles_dir.glob("*/prefs.js")))
    return prefs_files


def _data_dirs_from_profiles() -> list[Path]:
    """Collect custom data directories declared in Zotero profiles.

    A user-configured data directory is stored as the
    ``extensions.zotero.dataDir`` preference in the profile directory's
    prefs.js — not inside the data directory itself.
    """
    data_dirs: list[Path] = []
    for prefs_path in _profile_prefs_files():
        data_dir = _read_string_pref(prefs_path, "extensions.zotero.dataDir")
        if data_dir:
            data_dirs.append(Path(data_dir))
    return data_dirs


@dataclass
class ZoteroItem:
    """Represents a Zotero item with text content for semantic search."""
    item_id: int
    key: str
    item_type_id: int
    item_type: str | None = None
    doi: str | None = None
    title: str | None = None
    abstract: str | None = None
    creators: str | None = None
    fulltext: str | None = None
    fulltext_source: str | None = None  # 'pdf' or 'html'
    notes: str | None = None
    extra: str | None = None
    date_added: str | None = None
    date_modified: str | None = None

    def get_searchable_text(self) -> str:
        """
        Combine all text fields into a single searchable string.

        Returns:
            Combined text content for semantic search indexing.
        """
        parts = []

        if self.title:
            parts.append(f"Title: {self.title}")

        if self.creators:
            parts.append(f"Authors: {self.creators}")

        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")

        if self.extra:
            parts.append(f"Extra: {self.extra}")

        if self.notes:
            parts.append(f"Notes: {self.notes}")

        if self.fulltext:
            # Truncate very long fulltext for simple text search
            max_chars = 50000
            truncated_fulltext = self.fulltext[:max_chars] + "..." if len(self.fulltext) > max_chars else self.fulltext
            parts.append(f"Content: {truncated_fulltext}")

        return "\n\n".join(parts)


def _source_for_path(path: Path) -> str:
    """The ``fulltextSource`` tag recorded for text extracted from ``path``."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".html", ".htm"}:
        return "html"
    return "file"


def _init_extraction_worker() -> None:
    """Silence extraction warnings inside a pool worker.

    ``semantic_search`` raises the ``zotero_mcp.extract`` logger to CRITICAL
    for the duration of an indexing run, because a warning printed mid-scan
    corrupts the progress line. That setting lives in the parent interpreter
    and means nothing to a worker process, so without this initializer
    parallel extraction would print warnings that sequential extraction hides
    — and roughly 0.4% of real-world PDFs fail to parse, so it is not rare.
    """
    logging.getLogger("zotero_mcp.extract").setLevel(logging.CRITICAL)
    logging.getLogger("pdfminer").setLevel(logging.ERROR)


def _extract_worker(path_str: str, max_pages: int) -> str:
    """Parse one attachment. Runs in a pool worker, so it must be top-level.

    Returns "" rather than raising: a corrupt PDF is ordinary in a real
    library, and one bad file must not take down a pool worker.
    """
    try:
        doc = extract_file(Path(path_str), max_pages=max_pages)
        return doc.text if doc else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# #167 SQLite metadata search backend
#
# Direct-SQL equivalents of the pyzotero-based filtering in
# tools/search.py's search_items / advanced_search, active only when
# ZOTERO_SEARCH_BACKEND=sqlite (see utils.get_search_backend()). Each entry
# point returns None when it hits a condition/operator/itemType expression it
# doesn't support, so the caller can fall back to the existing pyzotero path
# — that path remains the correctness safety net.
# ---------------------------------------------------------------------------

# Operators supported by zotero_advanced_search's conditions today (must stay
# in sync with tools/search.py's own `valid_operations` set).
_VALID_CONDITION_OPERATIONS = frozenset({
    "is", "isNot", "contains", "doesNotContain", "beginsWith", "endsWith",
    "isGreaterThan", "isLessThan", "isBefore", "isAfter",
})

# field.lower() -> canonical field name, mirroring tools/search.py's
# advanced_search field_aliases (author/authors/creators/tags handled
# separately since they're multi-valued).
_CONDITION_FIELD_ALIASES = {
    "itemtype": "itemType",
    "dateadded": "dateAdded",
    "datemodified": "dateModified",
    "doi": "DOI",
}

# Single-valued fields resolvable to one scalar SQL expression correlated on
# the outer query's `i` (items) / `it` (itemTypes) aliases.
_SIMPLE_FIELD_SQL = {
    "title": (
        "(SELECT v.value FROM itemData d JOIN itemDataValues v ON d.valueID = v.valueID "
        "WHERE d.itemID = i.itemID AND d.fieldID = 1)"
    ),
    "abstractNote": (
        "(SELECT v.value FROM itemData d JOIN itemDataValues v ON d.valueID = v.valueID "
        "WHERE d.itemID = i.itemID AND d.fieldID = 2)"
    ),
    # "date" is handled separately below (_DATE_DISPLAY_SQL / _DATE_RANGE_SQL)
    # — it needs an operator-aware expression, unlike every other field here.
    "DOI": (
        "(SELECT v.value FROM itemData d JOIN itemDataValues v ON d.valueID = v.valueID "
        "JOIN fields f ON d.fieldID = f.fieldID "
        "WHERE d.itemID = i.itemID AND f.fieldName = 'DOI')"
    ),
    "publicationTitle": (
        "(SELECT v.value FROM itemData d JOIN itemDataValues v ON d.valueID = v.valueID "
        "JOIN fields f ON d.fieldID = f.fieldID "
        "WHERE d.itemID = i.itemID AND f.fieldName = 'publicationTitle')"
    ),
    "dateAdded": "i.dateAdded",
    "dateModified": "i.dateModified",
    "itemType": "it.typeName",
}

# Zotero stores the "date" field as a multipart string in itemDataValues.value:
# "<ISO YYYY-MM-DD, 00 for missing parts> <original user-typed text>" (e.g.
# "2016-10-01 October 1, 2016", "2021-07-00 07/2021", "0000-00-00 <unparseable
# text>"), confirmed against Zotero's own source (Zotero.Date.strToMultipart /
# item.js's setField). The pyzotero/web-API "date" field returns ONLY the
# display half (Zotero.Date.multipartToStr strips the ISO prefix before
# returning it) — so a condition needs a DIFFERENT substring depending on the
# operator, to match what each one is really comparing:
#   - is/isNot/contains/doesNotContain/beginsWith/endsWith: the display text
#     only (_DATE_DISPLAY_SQL) — matching what the API's "date" field is.
#   - isGreaterThan/isLessThan/isBefore/isAfter: the first 10 chars, i.e. the
#     ISO portion (_DATE_RANGE_SQL) — Zotero's own search.js does exactly this
#     (`SUBSTR(value, 1, 10)`) for date range queries, since lexicographic
#     comparison of "YYYY-MM-DD" strings is chronologically correct but
#     comparing arbitrary display text ("October 1, 2016" vs "2020") is not.
# "year" similarly mirrors Zotero's own item.js (`getField('date', true,
# true).substr(0, 4)`) and searchConditions.js (`SUBSTR(value, 1, 4)`): the
# first 4 characters of the RAW multipart value, which is always the ISO
# year regardless of display format. Note this deliberately diverges from
# tools/search.py's `_extract_values`, which takes `date[:4]` off the
# *display* string and is wrong whenever that doesn't start with a year
# (i.e. most non-ISO-typed dates) — a pre-existing bug, not something to
# replicate; see the plan's Phase D bug list.
_RANGE_OPERATIONS = frozenset({"isGreaterThan", "isLessThan", "isBefore", "isAfter"})

_DATE_DISPLAY_SQL = (
    "(SELECT SUBSTR(v.value, INSTR(v.value, ' ') + 1) FROM itemData d "
    "JOIN itemDataValues v ON d.valueID = v.valueID JOIN fields f ON d.fieldID = f.fieldID "
    "WHERE d.itemID = i.itemID AND f.fieldName = 'date')"
)
_DATE_RANGE_SQL = (
    "(SELECT SUBSTR(v.value, 1, 10) FROM itemData d JOIN itemDataValues v ON d.valueID = v.valueID "
    "JOIN fields f ON d.fieldID = f.fieldID WHERE d.itemID = i.itemID AND f.fieldName = 'date')"
)
_YEAR_FIELD_SQL = (
    "(SELECT SUBSTR(v.value, 1, 4) FROM itemData d JOIN itemDataValues v ON d.valueID = v.valueID "
    "JOIN fields f ON d.fieldID = f.fieldID "
    "WHERE d.itemID = i.itemID AND f.fieldName = 'date' AND LENGTH(v.value) >= 4)"
)

_CREATOR_NAME_EXPR = "TRIM(COALESCE(c.firstName, '') || ' ' || COALESCE(c.lastName, ''))"

# The shared item-metadata projection used by both search_items_sql and
# advanced_search_sql — everything row_to_api_item() needs to build a
# pyzotero-shaped item dict, correlated on `i` (items) via a leading
# `WHERE i.libraryID = ?`.
_ITEM_HYDRATION_SELECT = """
    SELECT i.itemID, i.key, it.typeName as itemType, i.dateAdded, i.dateModified,
           title_val.value as title, abstract_val.value as abstractNote,
           date_val.value as date, doi_val.value as DOI, pub_val.value as publicationTitle
    FROM items i
    JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
    LEFT JOIN itemData title_data ON i.itemID = title_data.itemID AND title_data.fieldID = 1
    LEFT JOIN itemDataValues title_val ON title_data.valueID = title_val.valueID
    LEFT JOIN itemData abstract_data ON i.itemID = abstract_data.itemID AND abstract_data.fieldID = 2
    LEFT JOIN itemDataValues abstract_val ON abstract_data.valueID = abstract_val.valueID
    LEFT JOIN fields date_f ON date_f.fieldName = 'date'
    LEFT JOIN itemData date_data ON i.itemID = date_data.itemID AND date_data.fieldID = date_f.fieldID
    LEFT JOIN itemDataValues date_val ON date_data.valueID = date_val.valueID
    LEFT JOIN fields doi_f ON doi_f.fieldName = 'DOI'
    LEFT JOIN itemData doi_data ON i.itemID = doi_data.itemID AND doi_data.fieldID = doi_f.fieldID
    LEFT JOIN itemDataValues doi_val ON doi_data.valueID = doi_val.valueID
    LEFT JOIN fields pub_f ON pub_f.fieldName = 'publicationTitle'
    LEFT JOIN itemData pub_data ON i.itemID = pub_data.itemID AND pub_data.fieldID = pub_f.fieldID
    LEFT JOIN itemDataValues pub_val ON pub_data.valueID = pub_val.valueID
    WHERE i.libraryID = ?
    AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
"""


def _like_or_eq(expr: str, operation: str, value: str) -> tuple[str, list]:
    """Build a single-value SQL comparison for one condition operator.

    ``isNot``/``doesNotContain`` are translated to their positive counterpart
    here (``is``/``contains``) — callers needing the true negated semantics
    (a value must be present AND not match) wrap the result themselves; see
    ``_scalar_condition`` and the creator/tag EXISTS builders below, which
    both replicate search_semantics.matches()'s rule that a missing/empty
    value never satisfies *any* operator, negated or not.

    Equality and pattern comparisons run against ``zsearch_norm(expr)`` rather
    than the raw column, so SQL folds the stored value exactly as
    ``search_semantics.compare`` folds it in Python. Without that the two
    backends disagree on every accented spelling. Range comparisons stay
    unwrapped — see ``search_semantics.sql_expression``.
    """
    positive = _semantics.POSITIVE_OF.get(operation, operation)
    target = _semantics.sql_expression(expr, positive)

    if positive == "is":
        return f"{target} = ?", [_semantics.normalize(value)]
    if positive in _semantics.PATTERN_OPS:
        pattern = _semantics.like_pattern(
            positive, _semantics.escape_like(_semantics.normalize(value))
        )
        return f"{target} LIKE ? ESCAPE '{_semantics.LIKE_ESCAPE}'", [pattern]
    if positive in ("isGreaterThan", "isAfter"):
        return f"{target} > ?", [value]
    if positive in ("isLessThan", "isBefore"):
        return f"{target} < ?", [value]
    raise AssertionError(f"unreachable: unvalidated operation {operation!r}")


def _scalar_condition(expr: str, operation: str, value: str) -> tuple[str, list]:
    """Condition SQL for a single-valued field (title, date, itemType, ...).

    A negated operator matches an item that simply *has no value* for the
    field: an item with no ``publicationTitle`` does not contain "Nature", so
    ``publicationTitle doesNotContain "Nature"`` returns it. That mirrors
    tools/search.py's ``_extract_values``, which yields ``[""]`` rather than
    ``[]`` for an absent scalar — the same choice it documents explicitly for
    ``collection``.

    The "empty satisfies nothing" rule applies only to the genuinely
    multi-valued fields, where ``_extract_values`` really does return an empty
    list; see ``_creator_condition`` and ``_tag_condition``, which keep their
    ``EXISTS`` guards for exactly that reason. Applying it here as well made
    the two backends disagree on every optional field.
    """
    if operation in ("isNot", "doesNotContain"):
        positive_sql, params = _like_or_eq(expr, operation, value)
        return f"(NOT ({positive_sql}))", params
    return _like_or_eq(expr, operation, value)


def _creator_condition(operation: str, value: str) -> tuple[str, list]:
    """Condition SQL for the multi-valued ``creator`` field.

    ``is``/``contains``/etc. match if ANY creator satisfies the comparison;
    ``isNot``/``doesNotContain`` match only if the item HAS creators and NONE
    of them satisfy the positive form — mirroring `_matches_condition`'s
    ``all(comparisons)`` rule for negated operators.
    """
    cmp_sql, params = _like_or_eq(_CREATOR_NAME_EXPR, operation, value)
    positive_exists = (
        "EXISTS (SELECT 1 FROM itemCreators ic JOIN creators c ON ic.creatorID = c.creatorID "
        f"WHERE ic.itemID = i.itemID AND {cmp_sql})"
    )
    if operation in ("isNot", "doesNotContain"):
        any_exists = "EXISTS (SELECT 1 FROM itemCreators ic2 WHERE ic2.itemID = i.itemID)"
        return f"({any_exists} AND NOT {positive_exists})", params
    return positive_exists, params


def _tag_condition(operation: str, value: str) -> tuple[str, list]:
    """Condition SQL for the multi-valued ``tag`` field (see _creator_condition)."""
    cmp_sql, params = _like_or_eq("t.name", operation, value)
    positive_exists = (
        "EXISTS (SELECT 1 FROM itemTags itg JOIN tags t ON itg.tagID = t.tagID "
        f"WHERE itg.itemID = i.itemID AND {cmp_sql})"
    )
    if operation in ("isNot", "doesNotContain"):
        any_exists = "EXISTS (SELECT 1 FROM itemTags itg2 WHERE itg2.itemID = i.itemID)"
        return f"({any_exists} AND NOT {positive_exists})", params
    return positive_exists, params


def row_to_api_item(row: sqlite3.Row, creators: list[dict], tags: list[dict]) -> dict:
    """Build a pyzotero-shaped item dict from a search-backend result row.

    Covers exactly the fields the two search tools consume downstream
    (``format_item_result``, the advanced-search sort keys, and — on the
    fallback path — ``_extract_values``) rather than a full item
    representation: no ``version``, ``collections``, ``relations``, etc.
    """
    data = {
        "key": row["key"],
        "itemType": row["itemType"],
        "title": row["title"] or "",
        "date": row["date"] or "",
        "dateAdded": row["dateAdded"] or "",
        "dateModified": row["dateModified"] or "",
        "abstractNote": row["abstractNote"] or "",
        "DOI": row["DOI"] or "",
        "publicationTitle": row["publicationTitle"] or "",
        "creators": creators,
        "tags": tags,
    }
    return {"key": row["key"], "data": data}


class LocalZoteroReader:
    """
    Direct SQLite reader for Zotero's local database.

    Provides fast access to item metadata and fulltext for semantic search
    without going through the Zotero API.
    """

    # Class-level fallbacks so subclasses that bypass __init__ (test stubs do
    # this) still work — with the transient fulltext cache OFF, so they can
    # never write into the user's real cache directory.
    extraction_workers: int = 1
    fulltext_cache_enabled: bool = False
    config_path: str | None = None

    def __init__(
        self,
        db_path: str | None = None,
        pdf_max_pages: int | None = None,
        attachment_priority=None,
        extraction_workers: int = 1,
        fulltext_cache_enabled: bool = False,
        config_path: str | None = None,
    ):
        """
        Initialize the local database reader.

        Args:
            db_path: Optional path to zotero.sqlite. If None, auto-detect.
            pdf_max_pages: Maximum pages to extract from PDFs.
            attachment_priority: Order in which attachment kinds are tried
                for an item with several readable files. None means the
                default (PDF > HTML > rest).
            extraction_workers: How many PDFs :meth:`extract_fulltext_for_items`
                may parse at once. 1 (the default) keeps the historical fully
                sequential behaviour. Higher values fan out over a process
                pool — processes rather than threads because pdf-inspector
                holds the GIL while parsing, so threads scale at ~1.1x while
                processes reach ~6x.
            fulltext_cache_enabled: Whether to read/write the transient
                plain-text cache (see the :mod:`.fulltext_cache` module).
                Off by default: callers that extract under a *different* page
                cap than indexing uses (``zotero_get_item_fulltext`` does)
                would otherwise poison the cache with truncated text.
            config_path: Semantic-search config path, used only to locate the
                fulltext cache directory next to it.
        """
        self.db_path = db_path or self._find_zotero_db()
        self._connection: sqlite3.Connection | None = None
        self.pdf_max_pages: int | None = pdf_max_pages
        self.attachment_priority: tuple[str, ...] = normalize_attachment_priority(
            attachment_priority
        )
        self.extraction_workers: int = max(1, int(extraction_workers or 1))
        self.fulltext_cache_enabled: bool = fulltext_cache_enabled
        self.config_path: str | None = config_path

    def _find_zotero_db(self) -> str:
        """
        Auto-detect the Zotero database location.

        Resolution order:
        1. The ``ZOTERO_DB_PATH`` environment variable.
        2. A custom data directory configured in Zotero's preferences
           (``extensions.zotero.dataDir`` in the profile's prefs.js).
        3. The default data directory (``~/Zotero``).

        Returns:
            Path to zotero.sqlite file.

        Raises:
            FileNotFoundError: If database cannot be located.
        """
        env_path = os.getenv("ZOTERO_DB_PATH")
        if env_path:
            db_path = Path(env_path).expanduser()
            if db_path.is_file():
                return str(db_path)
            raise FileNotFoundError(
                f"ZOTERO_DB_PATH is set to {db_path}, but no file exists there."
            )

        # A data directory configured in Zotero's own preferences is
        # authoritative; a leftover ~/Zotero from an old install is not.
        candidates = [
            data_dir / "zotero.sqlite" for data_dir in _data_dirs_from_profiles()
        ]
        candidates.append(Path.home() / "Zotero" / "zotero.sqlite")
        if platform.system() == "Windows":
            # Fallback to XP/2000 location
            candidates.append(
                Path(os.path.expanduser("~/Documents and Settings"))
                / os.getenv("USERNAME", "")
                / "Zotero"
                / "zotero.sqlite"
            )

        seen: set[str] = set()
        checked: list[Path] = []
        for db_path in candidates:
            if str(db_path) in seen:
                continue
            seen.add(str(db_path))
            checked.append(db_path)
            if db_path.is_file():
                return str(db_path)

        raise FileNotFoundError(
            "Could not locate the Zotero database (checked: "
            + ", ".join(str(c) for c in checked)
            + "). If Zotero stores its data in a custom location, set the "
            "ZOTERO_DB_PATH environment variable to your zotero.sqlite file "
            "or configure the path by running `zotero-mcp setup`."
        )

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection, creating if needed."""
        if self._connection is None:
            # Use immutable=1 to bypass locking entirely. Zotero uses rollback
            # journal mode and holds a write lock while running, which blocks
            # even read-only connections. immutable=1 skips all lock checks —
            # safe here since we only read and tolerate slightly stale data.
            uri = f"file:{self.db_path}?immutable=1"
            self._connection = sqlite3.connect(uri, uri=True)
            self._connection.row_factory = sqlite3.Row
            # Lets the search backend compare the same folded form of a string
            # that search_semantics.compare() produces in Python.
            _semantics.register_sqlite_functions(self._connection)
        return self._connection

    def _get_storage_dir(self) -> Path:
        """Return the Zotero storage directory path based on database location."""
        # Infer storage directory from database path (same parent directory)
        db_parent = Path(self.db_path).parent
        return db_parent / "storage"

    def _get_base_attachment_path(self) -> Path | None:
        """Read the linked attachment base directory from Zotero's prefs.js.

        Returns the configured ``extensions.zotero.baseAttachmentPath`` or
        ``None`` if the preference is not set or cannot be read. The
        preference lives in the profile directory's prefs.js; a prefs.js
        next to the database is also checked for unusual setups.
        """
        prefs_files = [Path(self.db_path).parent / "prefs.js"]
        prefs_files.extend(_profile_prefs_files())
        for prefs_path in prefs_files:
            if not prefs_path.exists():
                continue
            value = _read_string_pref(
                prefs_path, "extensions.zotero.baseAttachmentPath"
            )
            if value:
                return Path(value)
        return None

    def _iter_parent_attachments(self, parent_item_id: int):
        """Yield tuples (attachment_key, path, content_type) for a parent item.

        Explicitly trashed attachment rows (the attachment itself is in
        deletedItems; parent-trash inheritance is not checked) are excluded,
        and PDFs are yielded before other content types — otherwise a leftover
        HTML snapshot's .zotero-ft-cache would win over the PDF's in
        _extract_fulltext_for_item.
        """
        conn = self._get_connection()
        query = (
            """
            SELECT ia.itemID as attachmentItemID,
                   ia.parentItemID as parentItemID,
                   ia.path as path,
                   ia.contentType as contentType,
                   att.key as attachmentKey
            FROM itemAttachments ia
            JOIN items att ON att.itemID = ia.itemID
            LEFT JOIN deletedItems d ON d.itemID = ia.itemID
            WHERE ia.parentItemID = ? AND d.itemID IS NULL
            ORDER BY (ia.contentType = 'application/pdf') DESC, ia.itemID
            """
        )
        for row in conn.execute(query, (parent_item_id,)):
            yield row["attachmentKey"], row["path"], row["contentType"]

    def _resolve_attachment_path(self, attachment_key: str, zotero_path: str) -> Path | None:
        """Resolve a Zotero attachment path to a filesystem path.

        Handles four formats:
        - 'storage:filename.pdf' — Zotero-managed storage (most common)
        - 'file:///path/to/file.pdf' — linked file as URL
        - '/absolute/path/to/file.pdf' — linked file as absolute path
        - 'attachments:relative/path.pdf' — Zotero linked attachment base dir
        """
        if not zotero_path:
            return None

        storage_dir = self._get_storage_dir()

        # Zotero-managed storage: 'storage:filename.pdf'
        if zotero_path.startswith("storage:"):
            rel = zotero_path.split(":", 1)[1]
            parts = [p for p in rel.split("/") if p]
            return storage_dir / attachment_key / Path(*parts)

        # Linked file as URL: 'file:///path/to/file.pdf'
        if zotero_path.startswith("file://"):
            from urllib.parse import unquote, urlparse
            parsed = urlparse(zotero_path)
            decoded_path = unquote(parsed.path or "")
            # file:///C:/... on Windows
            if os.name == "nt" and decoded_path.startswith("/") and len(decoded_path) > 2 and decoded_path[2] == ":":
                decoded_path = decoded_path[1:]
            if not decoded_path:
                return None
            return Path(decoded_path)

        # Linked file as absolute path: '/Users/me/papers/file.pdf'
        if os.path.isabs(zotero_path):
            return Path(zotero_path)

        # Zotero 'attachments:' relative path — resolve against the linked
        # attachment base directory configured in Zotero preferences.
        if zotero_path.startswith("attachments:"):
            rel = zotero_path.split(":", 1)[1]
            parts = [p for p in rel.split("/") if p]
            base = self._get_base_attachment_path()
            if base and base.exists():
                return base / Path(*parts)
            # Fallback: cannot resolve without base path
            return None

        return None

    def _resolve_pdf_max_pages(self) -> int:
        """Page cap for PDF extraction.

        Indexing a whole library shouldn't pull every page of every
        thousand-page book into the embedding store, so an explicit cap
        always applies. A non-positive configured value falls through to the
        env override and then the default rather than meaning "unlimited".
        """
        if isinstance(self.pdf_max_pages, int) and self.pdf_max_pages > 0:
            return self.pdf_max_pages
        try:
            return int(os.getenv("ZOTERO_PDF_MAXPAGES") or DEFAULT_PDF_MAX_PAGES)
        except ValueError:
            return DEFAULT_PDF_MAX_PAGES

    def _extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from an attachment file, or "" if nothing readable."""
        doc = extract_file(file_path, max_pages=self._resolve_pdf_max_pages())
        return doc.text if doc else ""

    def _get_fulltext_meta_for_item(self, item_id: int):
        meta = []
        for key, path, ctype in self._iter_parent_attachments(item_id):
            meta.append([key, path, ctype])

        return meta

    def _read_zotero_ft_cache(self, attachment_key: str) -> str | None:
        """Return the text in Zotero's ``.zotero-ft-cache`` for an attachment.

        Zotero writes a plain-text full-text cache next to each indexed PDF /
        EPUB at ``storage/<attachment_key>/.zotero-ft-cache``. It is a
        fallback rather than the primary path: the text is flat pdftotext
        output with no heading structure, and most files carry no page
        separators, so chunks derived from it have no page provenance.

        What it still buys us is reach. It is keyed by attachment key rather
        than filename, so it survives Zotero file-naming drift / non-ASCII
        rewrites (#291), and it covers formats we don't parse ourselves
        (EPUB) as well as files that fail to parse.

        Returns ``None`` if the cache file is absent, empty, or unreadable.
        """
        try:
            cache_path = self._get_storage_dir() / attachment_key / ".zotero-ft-cache"
        except Exception:
            return None
        if not cache_path.exists():
            return None
        try:
            text = cache_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        return text or None

    def _scan_storage_for_attachment(
        self, attachment_key: str, ctype: str | None
    ) -> Path | None:
        """Fallback path resolver: find a likely attachment file on disk.

        ``itemAttachments.path`` in the Zotero sqlite is the filename Zotero
        recorded at import time, but the on-disk filename can drift (renames,
        non-ASCII normalization, external sync tools). When the recorded
        path no longer resolves, scan the attachment's own storage folder
        and pick the largest file whose extension is consistent with the
        recorded content type (#291).
        """
        try:
            attachment_dir = self._get_storage_dir() / attachment_key
        except Exception:
            return None
        if not attachment_dir.is_dir():
            return None

        if ctype == "application/pdf":
            wanted_suffixes = {".pdf"}
        elif (ctype or "").startswith("text/html"):
            wanted_suffixes = {".html", ".htm"}
        elif (ctype or "").startswith("application/epub"):
            wanted_suffixes = {".epub"}
        else:
            return None

        candidates: list[Path] = [
            child for child in attachment_dir.iterdir()
            if child.is_file() and child.suffix.lower() in wanted_suffixes
        ]
        if not candidates:
            return None
        # Largest file wins — for PDFs this is almost always the body content
        # rather than a stub or thumbnail.
        return max(candidates, key=lambda p: p.stat().st_size)

    def _resolve_extraction_target(self, item_id: int) -> tuple[Path, str] | None:
        """Pick the attachment to extract for an item, by configured priority.

        Returns ``(path, attachment_key)``, or None when the item has nothing
        readable. Deliberately cheap — sqlite plus one ``stat()`` per
        attachment, no parsing. That is what lets
        :meth:`extract_fulltext_for_items` resolve every target in the parent
        process (which owns the sqlite connection) and hand workers nothing
        but a path.
        """
        candidates = []
        keys: dict[Path, str] = {}
        for key, path, ctype in self._iter_parent_attachments(item_id):
            resolved = self._resolve_attachment_path(key, path or "")
            if not resolved or not resolved.exists():
                # Filename drift fallback: scan the storage folder.
                resolved = self._scan_storage_for_attachment(key, ctype)
                if not resolved or not resolved.exists():
                    continue
            category = categorize_attachment(resolved, ctype)
            if category is None:
                continue
            try:
                size = resolved.stat().st_size
            except OSError:
                size = 0
            candidates.append((category, size, resolved))
            keys.setdefault(resolved, key)

        target = pick_by_priority(candidates, self.attachment_priority)
        if target is None:
            return None
        return target, keys.get(target, "")

    def _cache_profile(self) -> str:
        """Extraction settings that change the text produced for one file.

        Only the page cap qualifies. ``attachment_priority`` does not: the
        cache is keyed by *attachment*, so changing the priority resolves to
        a different attachment and misses on its own.
        """
        return f"maxpages={self._resolve_pdf_max_pages()}"

    def _cache_lookup(self, target: Path, attachment_key: str) -> tuple[str, str] | None:
        """Transient-cache hit for an unchanged attachment, or None."""
        if not self.fulltext_cache_enabled or not attachment_key:
            return None
        try:
            st = target.stat()
        except OSError:
            return None
        return fulltext_cache.get_cached_text(
            attachment_key,
            st.st_mtime_ns,
            st.st_size,
            profile=self._cache_profile(),
            config_path=self.config_path,
        )

    def _cache_store(
        self,
        target: Path,
        attachment_key: str,
        item_key: str | None,
        text: str,
        source: str,
    ) -> None:
        """Persist freshly extracted text so a failed embed doesn't lose it."""
        if not self.fulltext_cache_enabled or not attachment_key:
            return
        try:
            st = target.stat()
        except OSError:
            return
        try:
            fulltext_cache.put_cached_text(
                attachment_key,
                st.st_mtime_ns,
                st.st_size,
                source=source,
                item_key=item_key,
                text=text,
                path=str(target),
                profile=self._cache_profile(),
                config_path=self.config_path,
            )
        except Exception as e:  # never let caching break extraction
            logger.debug(f"Could not cache fulltext for {attachment_key}: {e}")

    def _zotero_ft_cache_fallback(
        self,
        item_id: int,
        chosen: tuple[Path, str] | None = None,
        item_key: str | None = None,
    ) -> tuple[str, str] | None:
        """Read Zotero's own ``.zotero-ft-cache`` for whatever we could not parse.

        When ``chosen`` is given, the result is written to the transient cache
        against that attachment. This matters more than it looks: an
        image-only PDF parses to an empty string, so without it every run
        re-parses the file just to rediscover there is nothing in it — and
        the files that behave this way are scanned books, the slowest things
        in a library to parse. Keying on the chosen attachment keeps the
        entry invalidating normally when the file changes.
        """
        for key, _path, _ctype in self._iter_parent_attachments(item_id):
            cached = self._read_zotero_ft_cache(key)
            if cached:
                result = (cached, "zotero-cache")
                if chosen:
                    self._cache_store(chosen[0], chosen[1], item_key, cached, "zotero-cache")
                return result
        return None

    def _extract_fulltext_for_item(
        self, item_id: int, item_key: str | None = None
    ) -> tuple[str, str] | None:
        """Attempt to extract fulltext and source from the item's best attachment.

        Preference order:
        1. Our own extraction of the best attachment on disk, chosen by
           ``attachment_priority`` — source ``"pdf"``, ``"html"`` or
           ``"file"``.
        2. ``.zotero-ft-cache`` — source ``"zotero-cache"``.

        Our parser goes first because it is the only path that yields heading
        structure and the page separators chunk provenance needs; the cache is
        flat pdftotext output (see :meth:`_read_zotero_ft_cache`). The cache
        still covers everything the parser cannot reach: attachments whose
        file won't resolve, formats we don't read (EPUB), and files that fail
        to parse.

        If the sqlite-recorded filename doesn't resolve on disk, scan the
        attachment's storage folder for a content-type-matching file before
        giving up (#291, #265).
        """
        chosen = self._resolve_extraction_target(item_id)

        # 1. Best attachment by the configured priority.
        if chosen:
            target, attachment_key = chosen
            hit = self._cache_lookup(target, attachment_key)
            if hit:
                return hit
            text = self._extract_text_from_file(target)
            if text:
                source = _source_for_path(target)
                self._cache_store(target, attachment_key, item_key, text, source)
                return (text, source)

        # 2. Zotero's own cache, for whatever step 1 could not read.
        return self._zotero_ft_cache_fallback(item_id, chosen, item_key)

        return None

    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_libraries(self) -> list[dict[str, Any]]:
        """Get all libraries (user, group, feed) from the database."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT l.libraryID, l.type, l.editable,
                   g.groupID, g.name as groupName, g.description as groupDescription,
                   f.name as feedName, f.url as feedUrl,
                   f.lastCheck as feedLastCheck, f.lastUpdate as feedLastUpdate,
                   (SELECT COUNT(*) FROM items i
                    JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
                    WHERE i.libraryID = l.libraryID
                    AND it.typeName NOT IN ('attachment', 'note', 'annotation')) as itemCount
            FROM libraries l
            LEFT JOIN groups g ON l.libraryID = g.libraryID
            LEFT JOIN feeds f ON l.libraryID = f.libraryID
            ORDER BY l.type, l.libraryID
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def get_groups(self) -> list[dict[str, Any]]:
        """Get all group libraries with item counts."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT g.groupID, g.libraryID, g.name, g.description,
                   (SELECT COUNT(*) FROM items i
                    JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
                    WHERE i.libraryID = g.libraryID
                    AND it.typeName NOT IN ('attachment', 'note', 'annotation')) as itemCount
            FROM groups g
            ORDER BY g.name
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def get_feeds(self) -> list[dict[str, Any]]:
        """Get all RSS feed subscriptions with item counts."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT f.libraryID, f.name, f.url,
                   f.lastCheck, f.lastUpdate, f.lastCheckError,
                   f.refreshInterval,
                   (SELECT COUNT(*) FROM feedItems fi
                    JOIN items i ON fi.itemID = i.itemID
                    WHERE i.libraryID = f.libraryID) as itemCount
            FROM feeds f
            ORDER BY f.name
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def get_feed_items(
        self, library_id: int, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Get items from a specific RSS feed by its libraryID."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT i.itemID, i.key, it.typeName as itemType,
                   i.dateAdded,
                   fi.readTime, fi.translatedTime,
                   title_val.value as title,
                   abstract_val.value as abstract,
                   date_val.value as date,
                   doi_val.value as DOI,
                   url_val.value as url,
                   GROUP_CONCAT(
                       CASE
                           WHEN c.firstName IS NOT NULL AND c.lastName IS NOT NULL
                           THEN c.lastName || ', ' || c.firstName
                           WHEN c.lastName IS NOT NULL THEN c.lastName
                           ELSE NULL
                       END, '; '
                   ) as creators
            FROM feedItems fi
            JOIN items i ON fi.itemID = i.itemID
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            LEFT JOIN itemData title_data ON i.itemID = title_data.itemID AND title_data.fieldID = 1
            LEFT JOIN itemDataValues title_val ON title_data.valueID = title_val.valueID
            LEFT JOIN itemData abstract_data ON i.itemID = abstract_data.itemID AND abstract_data.fieldID = 2
            LEFT JOIN itemDataValues abstract_val ON abstract_data.valueID = abstract_val.valueID
            LEFT JOIN fields date_f ON date_f.fieldName = 'date'
            LEFT JOIN itemData date_data ON i.itemID = date_data.itemID AND date_data.fieldID = date_f.fieldID
            LEFT JOIN itemDataValues date_val ON date_data.valueID = date_val.valueID
            LEFT JOIN fields doi_f ON doi_f.fieldName = 'DOI'
            LEFT JOIN itemData doi_data ON i.itemID = doi_data.itemID AND doi_data.fieldID = doi_f.fieldID
            LEFT JOIN itemDataValues doi_val ON doi_data.valueID = doi_val.valueID
            LEFT JOIN fields url_f ON url_f.fieldName = 'url'
            LEFT JOIN itemData url_data ON i.itemID = url_data.itemID AND url_data.fieldID = url_f.fieldID
            LEFT JOIN itemDataValues url_val ON url_data.valueID = url_val.valueID
            LEFT JOIN itemCreators ic ON i.itemID = ic.itemID
            LEFT JOIN creators c ON ic.creatorID = c.creatorID
            WHERE i.libraryID = ?
            GROUP BY i.itemID
            ORDER BY i.dateAdded DESC
            LIMIT ?
            """,
            (library_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_item_count(self) -> int:
        """
        Get total count of non-attachment items.

        Returns:
            Number of items in the library.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT COUNT(*)
            FROM items i
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')
            AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
            """
        )
        return cursor.fetchone()[0]

    def get_all_item_keys(self) -> set[str]:
        """
        Get the keys of every item in the database, regardless of type.

        Used to verify that the sqlite snapshot is not lagging behind the
        Zotero API (an `immutable=1` read cannot see rows that are still
        in an un-checkpointed WAL file).
        """
        conn = self._get_connection()
        rows = conn.execute("SELECT key FROM items").fetchall()
        return {row[0] for row in rows}

    def get_key_group_map(self) -> KeyGroupMap:
        """Map every item key — including trashed items — to its library's group_id.

        Runs a single query joining ``items`` -> ``libraries`` -> ``groups``,
        translating each item's ``libraryID`` to the codebase-wide group_id
        (``0`` for the personal library, else the Zotero ``groupID``) in SQL.
        Items in libraries with no group_id equivalent (feed subscriptions,
        "My Publications") are returned in ``excluded_keys`` instead, so
        callers can drop them from the semantic index rather than silently
        mis-attribute them to the personal library.

        Trashed items are deliberately included: the group_id backfill needs
        their true library so each library's own scoped deletion pass cleans
        its trash from the index. Live-item consumers are unaffected — every
        item scan already excludes ``deletedItems`` at its own source, so
        trashed keys in this map are never looked up by them.
        """
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT i.key AS item_key, l.type AS lib_type, g.groupID AS group_id
            FROM items i
            JOIN libraries l ON i.libraryID = l.libraryID
            LEFT JOIN groups g ON l.libraryID = g.libraryID
            """
        ).fetchall()

        groups: dict[str, int] = {}
        excluded_keys: set[str] = set()
        for row in rows:
            key = row["item_key"]
            lib_type = row["lib_type"]
            group_id = row["group_id"]
            if lib_type == "user":
                groups[key] = PERSONAL_LIBRARY_GROUP_ID
            elif lib_type == "group" and group_id is not None:
                groups[key] = int(group_id)
            else:
                # Feeds, "My Publications", or any other non-group/user
                # library type have no group_id equivalent — exclude rather
                # than mis-attribute to the personal library.
                excluded_keys.add(key)

        return KeyGroupMap(groups, excluded_keys)

    def get_items_with_text(self, limit: int | None = None, include_fulltext: bool = False, key_filter: str | None = None, collection_keys: list[str] | None = None) -> list[ZoteroItem]:
        """
        Get all items with their text content for semantic search.

        Args:
            limit: Optional limit on number of items to return.
            collection_keys: Optional list of collection keys; when set, only
                items in those collections (or any of their subcollections)
                are returned.

        Returns:
            List of ZoteroItem objects with text content.
        """
        conn = self._get_connection()

        # Query to get items with their text content (simplified for now)
        query = """
        SELECT
            i.itemID,
            i.key,
            i.itemTypeID,
            it.typeName as item_type,
            i.dateAdded,
            i.dateModified,
            title_val.value as title,
            abstract_val.value as abstract,
            extra_val.value as extra,
            doi_val.value as doi,
            GROUP_CONCAT(n.note, ' ') as notes,
            GROUP_CONCAT(
                CASE
                    WHEN c.firstName IS NOT NULL AND c.lastName IS NOT NULL
                    THEN c.lastName || ', ' || c.firstName
                    WHEN c.lastName IS NOT NULL
                    THEN c.lastName
                    ELSE NULL
                END, '; '
            ) as creators
        FROM items i
        JOIN itemTypes it ON i.itemTypeID = it.itemTypeID

        -- Get title
        LEFT JOIN itemData title_data ON i.itemID = title_data.itemID AND title_data.fieldID = 1
        LEFT JOIN itemDataValues title_val ON title_data.valueID = title_val.valueID

        -- Get abstract
        LEFT JOIN itemData abstract_data ON i.itemID = abstract_data.itemID AND abstract_data.fieldID = 2
        LEFT JOIN itemDataValues abstract_val ON abstract_data.valueID = abstract_val.valueID

        -- Get extra field
        LEFT JOIN itemData extra_data ON i.itemID = extra_data.itemID AND extra_data.fieldID = 16
        LEFT JOIN itemDataValues extra_val ON extra_data.valueID = extra_val.valueID

        -- Get DOI field via fields table
        LEFT JOIN fields doi_f ON doi_f.fieldName = 'DOI'
        LEFT JOIN itemData doi_data ON i.itemID = doi_data.itemID AND doi_data.fieldID = doi_f.fieldID
        LEFT JOIN itemDataValues doi_val ON doi_data.valueID = doi_val.valueID

        -- Get notes
        LEFT JOIN itemNotes n ON i.itemID = n.parentItemID OR i.itemID = n.itemID

        -- Get creators
        LEFT JOIN itemCreators ic ON i.itemID = ic.itemID
        LEFT JOIN creators c ON ic.creatorID = c.creatorID

        WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')
        AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
        """

        params = []
        if collection_keys:
            # Restrict the corpus to the configured collections, including
            # all of their subcollections (resolved recursively).
            #
            # `seen` spans every configured key, not just one walk, so it does
            # two things. It terminates a parentCollection cycle, which the
            # walk would otherwise follow forever, appending as it went;
            # Zotero's own client cannot create one, but a corrupted,
            # partially synced or hand-edited database can. And it collapses
            # the overlap when the configured keys include both a collection
            # and one of its ancestors, which would otherwise bind the same
            # collectionID as several SQL parameters. Results are unchanged
            # either way — the generated query already selects DISTINCT
            # itemID — so this only bounds the walk and the parameter list.
            all_collection_ids = []
            seen_collection_ids: set[int] = set()
            for ckey in collection_keys:
                all_collection_ids.extend(
                    self._resolve_collection_ids(
                        conn, ckey, seen=seen_collection_ids
                    )
                )
            if all_collection_ids:
                placeholders = ','.join('?' * len(all_collection_ids))
                query += f" AND i.itemID IN (SELECT DISTINCT itemID FROM collectionItems WHERE collectionID IN ({placeholders}))"
                params.extend(all_collection_ids)

        if key_filter:
            query += " AND i.key = ?"
            params.append(key_filter)

        query += """
        GROUP BY i.itemID, i.key, i.itemTypeID, it.typeName, i.dateAdded, i.dateModified,
                 title_val.value, abstract_val.value, extra_val.value

        ORDER BY i.dateModified DESC
        """

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor = conn.execute(query, params)
        items = []

        for row in cursor:
            item = ZoteroItem(
                item_id=row['itemID'],
                key=row['key'],
                item_type_id=row['itemTypeID'],
                item_type=row['item_type'],
                doi=row['doi'],
                title=row['title'],
                abstract=row['abstract'],
                creators=row['creators'],
                fulltext=(res := (self._extract_fulltext_for_item(row['itemID']) if include_fulltext else None)) and res[0],
                fulltext_source=res[1] if include_fulltext and res else None,
                notes=row['notes'],
                extra=row['extra'],
                date_added=row['dateAdded'],
                date_modified=row['dateModified']
            )
            items.append(item)

        return items

    # Public helper to quickly check full text metadata for item.
    # Returns one [key, path, content_type] row per attachment of the item.
    def get_fulltext_meta_for_item(self, item_id: int) -> list[list[str | None]]:
        return self._get_fulltext_meta_for_item(item_id)

    # Public helper to extract fulltext on demand for a specific item
    def extract_fulltext_for_item(
        self, item_id: int, item_key: str | None = None
    ) -> tuple[str, str] | None:
        return self._extract_fulltext_for_item(item_id, item_key)

    def extract_fulltext_for_items(
        self, items: list[tuple[int, str | None]]
    ) -> Iterator[tuple[int, tuple[str, str] | None]]:
        """Extract fulltext for many items, fanning out over ``extraction_workers``.

        Yields ``(item_id, (text, source) | None)`` as results arrive, so a
        caller can report progress and evict cache entries incrementally
        rather than waiting for the whole batch. Order is *not* the input
        order once workers > 1.

        The division of labour is what makes this safe: everything touching
        sqlite, the filesystem index or the fulltext cache stays in this
        process, and workers receive only ``(path, max_pages)``. The sqlite
        connection is never shared, and nothing needs to be picklable except
        two strings and an int.

        With ``extraction_workers > 1`` this starts a process pool, so on
        platforms that spawn rather than fork (macOS, Windows) it must be
        called from inside a function or under an ``if __name__ ==
        "__main__"`` guard — never at module import time. Every shipped
        caller reaches it through a CLI command, so this only constrains
        embedding it in a script.
        """
        if self.extraction_workers <= 1:
            for item_id, item_key in items:
                yield item_id, self._extract_fulltext_for_item(item_id, item_key)
            return

        max_pages = self._resolve_pdf_max_pages()
        pending: dict[Any, tuple[int, str | None, Path, str]] = {}
        # Resolve targets and serve cache hits up front — both are cheap, and
        # doing them here keeps the pool busy with parsing alone.
        deferred: list[tuple[int, str | None, tuple[Path, str] | None]] = []
        # Outstanding work at the moment the pool died, to redo in-process.
        stranded: list[tuple[int, str | None]] = []
        with ProcessPoolExecutor(
            max_workers=self.extraction_workers, initializer=_init_extraction_worker
        ) as pool:
            for item_id, item_key in items:
                chosen = self._resolve_extraction_target(item_id)
                if not chosen:
                    deferred.append((item_id, item_key, None))
                    continue
                target, attachment_key = chosen
                hit = self._cache_lookup(target, attachment_key)
                if hit:
                    yield item_id, hit
                    continue
                future = pool.submit(_extract_worker, str(target), max_pages)
                pending[future] = (item_id, item_key, target, attachment_key)

            settled: set[Any] = set()
            for future in as_completed(pending):
                item_id, item_key, target, attachment_key = pending[future]
                try:
                    text = future.result()
                except BrokenProcessPool as e:
                    # A worker *process* died — an OOM on a pathological PDF,
                    # not the ordinary corrupt-file case ``_extract_worker``
                    # already absorbs. Every future still outstanding raises
                    # this same error, so letting the blanket handler below
                    # turn each one into empty text would mark the whole
                    # remainder of the batch has_fulltext="failed" — a sticky
                    # marker the skip logic then refuses to retry until the
                    # item changes or the collection is rebuilt. One transient
                    # death would poison a large run. Stop reaping futures and
                    # redo the outstanding work in this process instead.
                    logger.warning(
                        f"Extraction worker pool died ({e}); re-extracting the "
                        f"remaining {len(pending) - len(settled)} item(s) in-process"
                    )
                    stranded = [
                        (i, k)
                        for f, (i, k, _target, _att) in pending.items()
                        if f not in settled
                    ]
                    break
                except Exception as e:  # this item failed; fall back below
                    logger.debug(f"Extraction worker failed for item {item_id}: {e}")
                    text = ""
                settled.add(future)
                if text:
                    source = _source_for_path(target)
                    self._cache_store(target, attachment_key, item_key, text, source)
                    yield item_id, (text, source)
                else:
                    deferred.append((item_id, item_key, (target, attachment_key)))

        # Work the dead pool never finished, redone here. This is the exact
        # path ``extraction_workers <= 1`` takes, so target resolution, the
        # transient cache and the .zotero-ft-cache fallback all behave
        # identically — the blast radius of a dead worker is whichever single
        # file killed it, matching what the sequential path would have lost.
        for item_id, item_key in stranded:
            yield item_id, self._extract_fulltext_for_item(item_id, item_key)

        # Whatever the parser could not read falls back to Zotero's own
        # .zotero-ft-cache, exactly as the sequential path does. Cheap file
        # reads, so there is nothing to gain from parallelising them.
        for item_id, item_key, chosen in deferred:
            yield item_id, self._zotero_ft_cache_fallback(item_id, chosen, item_key)

    def get_attachment_paths(self, parent_key: str) -> list[dict]:
        """Return resolved filesystem paths for a parent item's attachments.

        Each entry has: ``key`` (attachment key), ``content_type``, ``zotero_path``
        (the raw stored path like ``storage:foo.pdf``), ``resolved_path`` (a
        ``Path`` or ``None`` if it could not be resolved), and ``exists`` (bool).
        """
        item = self.get_item_by_key(parent_key)
        if not item:
            return []
        out: list[dict] = []
        for att_key, zotero_path, ctype in self._iter_parent_attachments(item.item_id):
            resolved = self._resolve_attachment_path(att_key, zotero_path or "")
            out.append({
                "key": att_key,
                "content_type": ctype,
                "zotero_path": zotero_path,
                "resolved_path": resolved,
                "exists": bool(resolved and resolved.exists()),
            })
        return out

    def get_attachment_by_key(self, attachment_key: str) -> dict | None:
        """Return the attachment row addressed by its OWN key.

        ``get_item_by_key`` cannot see attachments: the query behind it
        excludes the 'attachment', 'note' and 'annotation' item types. So a
        key that names a PDF attachment directly (rather than its parent)
        needs its own lookup — without it, callers scan the attachment's
        (always empty) child list and conclude there is no PDF (#372).

        Each entry has: ``key``, ``content_type``, ``zotero_path`` (the raw
        stored path like ``storage:foo.pdf``), ``title`` and ``parent_key``.
        Returns ``None`` if the key does not name a live attachment.
        """
        conn = self._get_connection()
        row = conn.execute(
            """
            SELECT att.key as attachmentKey,
                   ia.path as path,
                   ia.contentType as contentType,
                   title_val.value as title,
                   parent.key as parentKey
            FROM itemAttachments ia
            JOIN items att ON att.itemID = ia.itemID
            LEFT JOIN items parent ON parent.itemID = ia.parentItemID
            LEFT JOIN itemData title_data
                ON title_data.itemID = att.itemID AND title_data.fieldID = 1
            LEFT JOIN itemDataValues title_val
                ON title_data.valueID = title_val.valueID
            WHERE att.key = ?
            AND att.itemID NOT IN (SELECT itemID FROM deletedItems)
            """,
            (attachment_key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "key": row["attachmentKey"],
            "content_type": row["contentType"],
            "zotero_path": row["path"],
            "title": row["title"],
            "parent_key": row["parentKey"],
        }

    def get_item_by_key(self, key: str) -> ZoteroItem | None:
        """
        Get a specific item by its Zotero key.

        Args:
            key: The Zotero item key.

        Returns:
            ZoteroItem if found, None otherwise.
        """
        items = self.get_items_with_text(key_filter=key)
        return items[0] if items else None

    def search_items_by_text(self, query: str, limit: int = 50) -> list[ZoteroItem]:
        """
        Simple text search through item content.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of matching ZoteroItem objects.
        """
        items = self.get_items_with_text()
        matching_items = []

        query_lower = _normalize_for_search(query).lower()

        for item in items:
            searchable_text = _normalize_for_search(item.get_searchable_text()).lower()
            if query_lower in searchable_text:
                matching_items.append(item)
                if len(matching_items) >= limit:
                    break

        return matching_items

    def search_notes_local(self, query: str, limit: int = 20) -> list[dict]:
        """Search notes in the local Zotero database by text content."""
        conn = self._get_connection()
        cursor = conn.cursor()
        pattern = f"%{query}%"
        cursor.execute("""
            SELECT i.key, n.note, n.title,
                   pi.key as parentKey,
                   pdv.value as parentTitle
            FROM itemNotes n
            JOIN items i ON n.itemID = i.itemID
            LEFT JOIN items pi ON n.parentItemID = pi.itemID
            LEFT JOIN itemData pd ON pi.itemID = pd.itemID AND pd.fieldID = 1
            LEFT JOIN itemDataValues pdv ON pd.valueID = pdv.valueID
            WHERE n.note LIKE ?
            AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
            LIMIT ?
        """, (pattern, limit))

        results = []
        for row in cursor.fetchall():
            note_html = row[1] or ""
            # Post-filter: skip if query only matches HTML tags, not content
            from zotero_mcp.utils import clean_html
            clean_text = clean_html(note_html)
            if query.lower() not in clean_text.lower():
                continue
            results.append({
                "type": "note",
                "key": row[0],
                "text": note_html,
                "parent_key": row[3],
                "parent_title": row[4] or ("Unknown" if row[3] else None),
                "tags": [],  # Tags require a separate query; omitted for speed
            })
        return results

    def search_annotations_local(self, query: str, limit: int = 20) -> list[dict]:
        """Search annotations in the local Zotero database by text or comment."""
        conn = self._get_connection()
        cursor = conn.cursor()
        pattern = f"%{query}%"
        # Two-hop join: annotation -> attachment -> grandparent item (for title)
        cursor.execute("""
            SELECT i.key, ia.text, ia.comment, ia.type, ia.color, ia.pageLabel,
                   att.key as attachmentKey,
                   gpi.key as parentKey,
                   gpdv.value as parentTitle
            FROM itemAnnotations ia
            JOIN items i ON ia.itemID = i.itemID
            LEFT JOIN items att ON ia.parentItemID = att.itemID
            LEFT JOIN itemAttachments iatt ON ia.parentItemID = iatt.itemID
            LEFT JOIN items gpi ON iatt.parentItemID = gpi.itemID
            LEFT JOIN itemData gpd ON gpi.itemID = gpd.itemID AND gpd.fieldID = 1
            LEFT JOIN itemDataValues gpdv ON gpd.valueID = gpdv.valueID
            WHERE (ia.text LIKE ? OR ia.comment LIKE ?)
            AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
            LIMIT ?
        """, (pattern, pattern, limit))

        # Map integer annotation types to names
        type_map = {1: "highlight", 2: "note", 3: "image", 4: "ink", 5: "underline"}

        results = []
        for row in cursor.fetchall():
            results.append({
                "type": "annotation",
                "key": row[0],
                "text": row[1] or "",
                "comment": row[2] or "",
                "annotation_type": type_map.get(row[3], "unknown"),
                "color": row[4] or "",
                "page_label": row[5] or None,
                "attachment_key": row[6],
                "parent_key": row[7],
                "parent_title": row[8] or ("Unknown" if row[7] else None),
            })
        return results

    # -----------------------------------------------------------------
    # #167 SQLite metadata search backend
    # -----------------------------------------------------------------

    def _resolve_collection_ids(
        self,
        conn: sqlite3.Connection,
        collection_key: str,
        seen: set[int] | None = None,
    ) -> list[int]:
        """Expand one collection key to its own collectionID plus every
        descendant subcollection's, recursively. Returns [] for an unknown key.

        The walk records where it has been, which bounds it on a
        ``parentCollectionID`` cycle (#451). Zotero's own client cannot nest a
        collection inside itself, so reaching one takes a corrupted, partially
        synced or hand-edited ``zotero.sqlite`` — but the alternative is a loop
        that appends forever until the process runs out of memory, and the
        guard costs a set.

        Pass *seen* to share that record across several calls. Callers
        expanding a list of configured keys do, so a collection reachable from
        two of them is bound once rather than once per key; results are
        unchanged either way, since every query built on these ids selects
        DISTINCT.
        """
        root = conn.execute(
            "SELECT collectionID FROM collections WHERE key = ?", (collection_key,)
        ).fetchone()
        if not root:
            return []
        if seen is None:
            seen = set()
        all_ids: list[int] = []
        to_process = [root[0]]
        while to_process:
            cid = to_process.pop()
            if cid in seen:
                continue
            seen.add(cid)
            all_ids.append(cid)
            for sub in conn.execute(
                "SELECT collectionID FROM collections WHERE parentCollectionID = ?", (cid,)
            ).fetchall():
                to_process.append(sub[0])
        return all_ids

    def _resolve_collection_id(self, conn: sqlite3.Connection, collection_key: str) -> int | None:
        """Direct (non-recursive) lookup: collection key -> its own
        collectionID, or None for an unknown key. Companion to
        `_resolve_collection_ids`, which additionally walks descendants."""
        row = conn.execute(
            "SELECT collectionID FROM collections WHERE key = ?", (collection_key,)
        ).fetchone()
        return row[0] if row else None

    def _resolve_scope_library_id(self, group_id: int) -> int | None:
        """Translate a codebase-wide group_id (0 = personal) to this
        database's local ``libraryID``, or None if no such library is
        present (e.g. a group that hasn't synced to this machine)."""
        conn = self._get_connection()
        if group_id == PERSONAL_LIBRARY_GROUP_ID:
            row = conn.execute("SELECT libraryID FROM libraries WHERE type = 'user'").fetchone()
        else:
            row = conn.execute("SELECT libraryID FROM groups WHERE groupID = ?", (group_id,)).fetchone()
        return row[0] if row is not None else None

    def _collection_condition(
        self, conn: sqlite3.Connection, operation: str, value: str, *, recursive: bool = False
    ) -> tuple[str, list] | None:
        """Condition SQL for the ``collection`` field.

        ``value`` is a collection key (8-char), matching the convention
        ``get_items_with_text``'s own ``collection_keys`` param already uses.
        Only membership checks make sense here, so only ``is``/``isNot`` are
        supported — anything else (and an unknown key) returns None so the
        caller falls back to the existing path.

        Defaults to DIRECT membership only (an item must be filed in this
        exact collection) — matching both the pyzotero/API advanced_search
        path (``_extract_values`` in tools/search.py, which only ever sees an
        item's own "collections" list) and Zotero's own "Collection is X" UI
        checkbox with subcollections unchecked. This keeps the two
        advanced_search backends in agreement regardless of
        ZOTERO_SEARCH_BACKEND.

        Recursive (subcollection-inclusive) resolution is fully implemented —
        pass ``recursive=True`` to get it, reusing the same tree walk
        ``get_items_with_text``'s ``collection_keys`` scoping already relies
        on (``_resolve_collection_ids``). It is not wired to any public
        parameter yet; it's kept here, tested, and reachable so it can be
        exposed later (e.g. an ``includeSubcollections`` option) without
        re-deriving this SQL.
        """
        if operation not in ("is", "isNot"):
            return None
        if recursive:
            ids = self._resolve_collection_ids(conn, value)
        else:
            cid = self._resolve_collection_id(conn, value)
            ids = [cid] if cid is not None else []
        if not ids:
            return None
        placeholders = ",".join("?" * len(ids))
        membership = (
            f"i.itemID IN (SELECT DISTINCT itemID FROM collectionItems "
            f"WHERE collectionID IN ({placeholders}))"
        )
        if operation == "isNot":
            return f"NOT {membership}", list(ids)
        return membership, list(ids)

    def _condition_sql(self, field: str, operation: str, value: str) -> tuple[str, list] | None:
        """Translate one advanced-search condition to a SQL fragment + params.

        Mirrors tools/search.py's ``_extract_values``/``_matches_condition``
        field-by-field; returns None for any field/operator this v1 backend
        doesn't cover, so ``advanced_search_sql`` can bail out to the
        existing pyzotero-based path.
        """
        if operation not in _VALID_CONDITION_OPERATIONS:
            return None
        field_lower = field.lower()
        if field_lower in ("author", "authors", "creator", "creators"):
            return _creator_condition(operation, value)
        if field_lower in ("tag", "tags"):
            return _tag_condition(operation, value)
        if field_lower == "year":
            return _scalar_condition(_YEAR_FIELD_SQL, operation, value)
        if field_lower == "date":
            date_expr = _DATE_RANGE_SQL if operation in _RANGE_OPERATIONS else _DATE_DISPLAY_SQL
            return _scalar_condition(date_expr, operation, value)
        if field_lower == "collection":
            return self._collection_condition(self._get_connection(), operation, value)
        resolved = _CONDITION_FIELD_ALIASES.get(field_lower, field)
        if resolved in _SIMPLE_FIELD_SQL:
            return _scalar_condition(_SIMPLE_FIELD_SQL[resolved], operation, value)
        return None

    def _fetch_creators(self, conn: sqlite3.Connection, item_ids: list[int]) -> dict[int, list[dict]]:
        if not item_ids:
            return {}
        placeholders = ",".join("?" * len(item_ids))
        rows = conn.execute(
            f"""
            SELECT ic.itemID, c.firstName, c.lastName, ct.creatorType
            FROM itemCreators ic
            JOIN creators c ON ic.creatorID = c.creatorID
            LEFT JOIN creatorTypes ct ON ic.creatorTypeID = ct.creatorTypeID
            WHERE ic.itemID IN ({placeholders})
            ORDER BY ic.itemID, ic.orderIndex
            """,
            item_ids,
        ).fetchall()
        result: dict[int, list[dict]] = {}
        for row in rows:
            creator: dict[str, str] = {"creatorType": row["creatorType"] or "author"}
            if row["firstName"]:
                creator["firstName"] = row["firstName"]
                creator["lastName"] = row["lastName"] or ""
            else:
                creator["name"] = row["lastName"] or ""
            result.setdefault(row["itemID"], []).append(creator)
        return result

    def _fetch_tags(self, conn: sqlite3.Connection, item_ids: list[int]) -> dict[int, list[dict]]:
        if not item_ids:
            return {}
        placeholders = ",".join("?" * len(item_ids))
        rows = conn.execute(
            f"""
            SELECT itg.itemID, t.name
            FROM itemTags itg
            JOIN tags t ON itg.tagID = t.tagID
            WHERE itg.itemID IN ({placeholders})
            """,
            item_ids,
        ).fetchall()
        result: dict[int, list[dict]] = {}
        for row in rows:
            result.setdefault(row["itemID"], []).append({"tag": row["name"]})
        return result

    def _hydrate_rows(self, conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> list[dict]:
        item_ids = [row["itemID"] for row in rows]
        creators_by_item = self._fetch_creators(conn, item_ids)
        tags_by_item = self._fetch_tags(conn, item_ids)
        return [
            row_to_api_item(
                row, creators_by_item.get(row["itemID"], []), tags_by_item.get(row["itemID"], [])
            )
            for row in rows
        ]

    def search_items_sql(
        self,
        query: str,
        qmode: str = "titleCreatorYear",
        item_type: str = "-attachment",
        tag: list[str] | None = None,
        limit: int = 10,
        group_id: int = PERSONAL_LIBRARY_GROUP_ID,
    ) -> list[dict] | None:
        """#167 SQLite metadata search backend for zotero_search_items.

        Substring-matches every variant `_generate_search_variants(query)`
        produces against title/creator/year (plus abstract/tags/notes in
        'everything' mode), OR'd together in one query, scoped to the
        library identified by `group_id`. Returns None — signalling "fall
        back to the pyzotero path" — when a `tag` filter is given (the
        boolean OR/exclusion tag DSL stays pyzotero's job) or `item_type`
        is anything other than a bare type name or a single "-type"
        exclusion, or when `group_id` has no matching library in this
        database.
        """
        if tag:
            return None
        if qmode not in ("titleCreatorYear", "everything"):
            return None

        conn = self._get_connection()
        lib_id = self._resolve_scope_library_id(group_id)
        if lib_id is None:
            return None

        type_filter_sql = ""
        type_params: list = []
        if item_type:
            if item_type.startswith("-") and item_type.count("-") == 1 and "||" not in item_type:
                type_filter_sql = "AND it.typeName != ?"
                type_params = [item_type[1:]]
            elif re.fullmatch(r"[A-Za-z]+", item_type):
                type_filter_sql = "AND it.typeName = ?"
                type_params = [item_type]
            else:
                return None  # boolean itemType expressions ("a || b") unsupported

        variants = _generate_search_variants(query)
        if not variants:
            return None
        like_clauses: list[str] = []
        like_params: list = []
        for variant in variants:
            # Escaped, but deliberately not zsearch_norm-folded: this free-text
            # path matches by OR-ing _generate_search_variants, which also
            # covers dash/space and umlaut *expansion* (Müller -> Mueller) that
            # normalize() does not do. Folding here as well would over-match
            # relative to the pyzotero path.
            pattern = f"%{_semantics.escape_like(variant)}%"
            like_clauses.append("title_val.value LIKE ? ESCAPE '\\'")
            like_params.append(pattern)
            like_clauses.append("date_val.value LIKE ? ESCAPE '\\'")
            like_params.append(pattern)
            like_clauses.append(
                f"EXISTS (SELECT 1 FROM itemCreators ic JOIN creators c ON ic.creatorID = c.creatorID "
                f"WHERE ic.itemID = i.itemID AND {_CREATOR_NAME_EXPR} LIKE ? ESCAPE '\\')"
            )
            like_params.append(pattern)
            if qmode == "everything":
                like_clauses.append("abstract_val.value LIKE ? ESCAPE '\\'")
                like_params.append(pattern)
                like_clauses.append(
                    "EXISTS (SELECT 1 FROM itemTags itg JOIN tags t ON itg.tagID = t.tagID "
                    "WHERE itg.itemID = i.itemID AND t.name LIKE ? ESCAPE '\\')"
                )
                like_params.append(pattern)
                like_clauses.append(
                    "EXISTS (SELECT 1 FROM itemNotes n WHERE "
                    "(n.parentItemID = i.itemID OR n.itemID = i.itemID) AND n.note LIKE ? ESCAPE '\\')"
                )
                like_params.append(pattern)

        query_sql = (
            _ITEM_HYDRATION_SELECT
            + f" {type_filter_sql} AND ({' OR '.join(like_clauses)})"
            + " ORDER BY i.dateModified DESC LIMIT ?"
        )
        params = [lib_id] + type_params + like_params + [limit]
        rows = conn.execute(query_sql, params).fetchall()
        return self._hydrate_rows(conn, rows)

    def advanced_search_sql(
        self,
        conditions: list[dict[str, str]],
        join_mode: str = "all",
        group_id: int = PERSONAL_LIBRARY_GROUP_ID,
    ) -> list[dict] | None:
        """#167 SQLite metadata search backend for zotero_advanced_search.

        Always excludes attachments/notes/annotations and trashed items,
        matching the existing pyzotero-based paging loop it replaces.
        Returns None when any single condition uses a field/operator this
        v1 backend doesn't cover, or when `group_id` has no matching
        library in this database — the caller falls back to the existing
        client-side path. Sorting and the result limit are left to the
        caller, exactly as with that existing path.
        """
        conn = self._get_connection()
        lib_id = self._resolve_scope_library_id(group_id)
        if lib_id is None:
            return None

        clauses: list[str] = []
        params: list = []
        for condition in conditions:
            built = self._condition_sql(condition["field"], condition["operation"], condition["value"])
            if built is None:
                return None
            clause_sql, clause_params = built
            clauses.append(clause_sql)
            params.extend(clause_params)

        if not clauses:
            return None

        joiner = " AND " if join_mode == "all" else " OR "
        where_sql = joiner.join(f"({c})" for c in clauses)

        query_sql = (
            _ITEM_HYDRATION_SELECT
            + " AND it.typeName NOT IN ('attachment', 'note', 'annotation')"
            + f" AND ({where_sql})"
        )
        all_params = [lib_id] + params
        rows = conn.execute(query_sql, all_params).fetchall()
        return self._hydrate_rows(conn, rows)


def get_local_zotero_reader() -> LocalZoteroReader | None:
    """
    Get a LocalZoteroReader instance if in local mode.

    Returns:
        LocalZoteroReader instance if in local mode and database exists,
        None otherwise.
    """
    if not is_local_mode():
        return None

    try:
        return LocalZoteroReader(db_path=load_config().resolve_zotero_db_path())
    except FileNotFoundError:
        return None


def is_local_db_available() -> bool:
    """
    Check if local Zotero database is available.

    Returns:
        True if local database can be accessed, False otherwise.
    """
    reader = get_local_zotero_reader()
    if reader:
        reader.close()
        return True
    return False
