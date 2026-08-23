"""One definition of search-condition semantics, shared by both backends.

``zotero_advanced_search`` can run two ways: the pyzotero path in
``tools/search.py``, which fetches items and filters them in Python, and the
direct-SQL path in ``local_db.py`` used when ``ZOTERO_SEARCH_BACKEND=sqlite``.
Both have to answer a condition like ``creator contains "muller"`` identically,
or the same tool call returns different results depending on one environment
variable.

Keeping them identical by hand did not work. Three divergences appeared while
#417 was in review — collection recursion, unescaped ``LIKE`` metacharacters,
and, most consequentially, normalization: the Python path folded diacritics on
both sides while the SQL path folded neither, so a creator search for
``muller`` returned 4 items through SQL and 15 through the API on a real
57,000-item library. Every accented spelling was silently missing.

So the semantics live here once, and each backend consumes them rather than
restating them:

* :func:`compare` / :func:`matches` are the Python path's comparator.
* :func:`normalize` is *also* registered on the SQLite connection as
  ``zsearch_norm`` (see :func:`register_sqlite_functions`), so SQL compares
  the same folded form of the same strings via the same Python function.

What is deliberately *not* shared is which column or expression a field maps
to. The SQL path reads Zotero's raw multipart ``date`` value and slices the ISO
prefix, where the pyzotero path can only see the display half; that difference
is a fix, not a divergence, and belongs to each backend.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable, Sequence

from .utils import _normalize_for_search

# The ten operators the advanced-search tool accepts. Frozen: they are part of
# the public tool schema and appear in stored saved-search definitions.
OPERATORS: frozenset[str] = frozenset(
    {
        "is",
        "isNot",
        "contains",
        "doesNotContain",
        "beginsWith",
        "endsWith",
        "isGreaterThan",
        "isLessThan",
        "isBefore",
        "isAfter",
    }
)

#: Operators that assert the *absence* of a match.
NEGATED: frozenset[str] = frozenset({"isNot", "doesNotContain"})

#: Each negated operator's positive counterpart. Both backends evaluate the
#: positive form and invert, rather than implementing negation twice.
POSITIVE_OF: dict[str, str] = {"isNot": "is", "doesNotContain": "contains"}

#: Ordering comparisons. These are never normalized — see :func:`sql_expression`.
RANGE_OPS: frozenset[str] = frozenset(
    {"isGreaterThan", "isLessThan", "isBefore", "isAfter"}
)

#: Operators whose SQL form is a ``LIKE`` pattern rather than an equality or
#: an inequality.
PATTERN_OPS: frozenset[str] = frozenset({"contains", "beginsWith", "endsWith"})

#: Condition-field spellings accepted from callers, mapped to the canonical
#: name. Callers lowercase the incoming field before looking it up.
FIELD_ALIASES: dict[str, str] = {
    "author": "creator",
    "authors": "creator",
    "creator": "creator",
    "creators": "creator",
    "tag": "tag",
    "tags": "tag",
    "collection": "collection",
    "collections": "collection",
    "itemtype": "itemType",
    "dateadded": "dateAdded",
    "datemodified": "dateModified",
    "doi": "DOI",
}

#: Name under which :func:`normalize` is registered on a SQLite connection.
SQLITE_NORM_FUNCTION = "zsearch_norm"

#: Escape character used with ``LIKE ... ESCAPE``. Backslash rather than a
#: rarer character because the values being escaped are bibliographic text,
#: where a literal backslash is far less common than ``%`` or ``_``.
LIKE_ESCAPE = "\\"


def canonical_field(field: str) -> str:
    """Resolve a caller-supplied condition field to its canonical spelling."""
    return FIELD_ALIASES.get(field.lower(), field)


def normalize(text: str | None) -> str:
    """Fold *text* into the form both backends compare against.

    ASCII transliteration (via ``unidecode``, so ``Müller`` and ``Muller``
    agree), dash unification, then case folding. Registered on SQLite as
    ``zsearch_norm`` so the stored side is folded the same way as the query
    side — folding only the query would still miss a stored ``Müller``.
    """
    return _normalize_for_search(text or "").lower()


def escape_like(value: str) -> str:
    """Neutralise ``LIKE`` metacharacters in a user-supplied value.

    Without this, a search for a title containing ``%`` or ``_`` matches as a
    wildcard under SQL while matching literally through the Python path. The
    escape character itself is escaped first, or escaping would corrupt values
    that already contain a backslash.
    """
    return (
        value.replace(LIKE_ESCAPE, LIKE_ESCAPE * 2)
        .replace("%", LIKE_ESCAPE + "%")
        .replace("_", LIKE_ESCAPE + "_")
    )


def like_pattern(operation: str, value: str) -> str:
    """Build the ``LIKE`` pattern for *operation* from an already-escaped value."""
    positive = POSITIVE_OF.get(operation, operation)
    if positive == "contains":
        return f"%{value}%"
    if positive == "beginsWith":
        return f"{value}%"
    if positive == "endsWith":
        return f"%{value}"
    raise ValueError(f"{operation!r} is not a pattern operator")


def sql_expression(column_expr: str, operation: str) -> str:
    """Wrap a column expression so SQL compares the normalized form.

    Range operators are returned unwrapped. :func:`compare` normalizes before
    its numeric parse, but the values that reach an ordering comparison are
    ASCII digits and ISO date prefixes, where normalization is the identity —
    so skipping it is behaviour-preserving and avoids a Python callback per
    row. ``tests/test_search_parity_offline.py`` pins that equivalence.
    """
    if operation in RANGE_OPS:
        return column_expr
    return f"{SQLITE_NORM_FUNCTION}({column_expr})"


def register_sqlite_functions(conn: sqlite3.Connection) -> None:
    """Register :func:`normalize` on *conn* as ``zsearch_norm``.

    ``deterministic=True`` lets SQLite reuse results within a statement, but it
    raises ``NotSupportedError`` against SQLite older than 3.8.3; fall back to
    a plain registration there rather than failing to open the database.
    """
    try:
        conn.create_function(SQLITE_NORM_FUNCTION, 1, normalize, deterministic=True)
    except sqlite3.NotSupportedError:  # pragma: no cover - very old SQLite
        conn.create_function(SQLITE_NORM_FUNCTION, 1, normalize)


def _as_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def compare(candidate: str, expected: str, operation: str) -> bool:
    """Evaluate one operator against one candidate value.

    Both sides are normalized first. Ordering operators compare numerically
    when both sides parse as numbers and lexically otherwise, so ``year
    isGreaterThan 2010`` orders by magnitude while a string field still
    orders sensibly.
    """
    left = normalize(candidate)
    right = normalize(expected)

    if operation == "is":
        return left == right
    if operation == "isNot":
        return left != right
    if operation == "contains":
        return right in left
    if operation == "doesNotContain":
        return right not in left
    if operation == "beginsWith":
        return left.startswith(right)
    if operation == "endsWith":
        return left.endswith(right)

    left_num = _as_float(left)
    right_num = _as_float(right)
    if operation in RANGE_OPS and left_num is not None and right_num is not None:
        if operation in {"isGreaterThan", "isAfter"}:
            return left_num > right_num
        return left_num < right_num

    if operation in {"isGreaterThan", "isAfter"}:
        return left > right
    return left < right


def matches(values: Sequence[str] | Iterable[str], expected: str, operation: str) -> bool:
    """Evaluate an operator against a field that may hold several values.

    An item with no value for the field satisfies *nothing* — not even a
    negated operator. That rule is what keeps ``creator isNot "X"`` from
    sweeping in every item that has no creators at all, and the SQL builders
    reproduce it with an ``EXISTS`` guard alongside the negation.
    """
    values = list(values)
    if not values:
        return False

    comparisons = [compare(value, expected, operation) for value in values]
    if operation in NEGATED:
        return all(comparisons)
    return any(comparisons)
