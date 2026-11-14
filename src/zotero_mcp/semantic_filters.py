"""Semantic-search filters, split between Chroma and live Zotero data (#826).

Filters keep the contract of 0.17: ChromaDB's ``where`` syntax, the same
accepted inputs and errors, the same value formats and the same operator
rules. What changed is where a bibliographic field is read: records no
longer store metadata, so a filter on one is evaluated against the hit's
owner as Zotero has it now. Legacy records still carry copied fields, but
those go stale and are not consulted, except for the few fields no live
source has (``fulltext_source``, ``attachment_keys``, ``attachment_priority``).

ChromaDB-free on purpose, so the tool layer and tests can import it cheaply;
only ``check_where`` imports ChromaDB, to reuse its own validator.
"""

from __future__ import annotations

from typing import Any

from .utils import format_creators

#: Stored in every record and never stale: filtered by Chroma.
CHROMA_FIELDS = frozenset({"group_id", "section", "page", "chunk_index", "n_chunks", "char_start", "char_end"})

#: Evaluated against the hit's owner item (or the hit's stored legacy value).
LIVE_FIELDS = frozenset(
    {
        "item_key", "item_type", "title", "date", "date_added", "date_modified", "creators", "publication",
        "url", "doi", "tags", "citation_key", "parent_item_key", "has_fulltext",
        "fulltext_source", "attachment_keys", "attachment_priority",
    }
)

#: Live fields that zotero.sqlite can answer for an owner alone, and whose
#: values read the same from there as from a resolved hit. Everything else
#: depends on the hit (``has_fulltext``, ``parent_item_key`` and the legacy
#: fields) or on data the owner query does not read.
OWNER_FIELDS = frozenset({"item_key", "item_type", "title", "doi", "citation_key", "publication", "tags"})

_MISSING = object()


def _conjuncts(where: dict[str, Any]) -> list[dict[str, Any]]:
    """The clauses ANDed together at the top of ``where``: ``$and`` lists, flattened.

    A dict with several keys is not an implicit ``$and``, as Chroma does not
    treat it as one: it stays one clause, which ``check_where`` rejects.
    """
    if len(where) == 1 and isinstance(where.get("$and"), list):
        out: list[dict[str, Any]] = []
        for sub in where["$and"]:
            out.extend(_conjuncts(sub) if isinstance(sub, dict) else [sub])
        return out
    return [where]


def referenced_fields(where: dict[str, Any] | None) -> set[str]:
    """Every field name ``where`` mentions, at any depth."""
    if not where:
        return set()
    fields: set[str] = set()
    for key, value in where.items():
        if key in ("$and", "$or"):
            for sub in value:
                fields |= referenced_fields(sub)
        else:
            fields.add(key)
    return fields


def combine(parts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """AND ``parts`` the way Chroma accepts: ``$and`` needs two or more."""
    parts = [p for p in parts if p]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"$and": parts}


_RANGE_OPERATORS = ("$gt", "$gte", "$lt", "$lte")


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _rejected_by_the_engine(where: Any) -> bool:
    """Whether ChromaDB's query engine refuses ``where`` after its validator passed it.

    The engine answers "Invalid where clause" for: a ``$``-prefixed key other
    than ``$and``/``$or``; a range operator with a boolean or NaN operand;
    ``$eq``/``$ne`` with a list; ``$in``/``$nin`` with a list holding a
    non-scalar. (Checked against chromadb 1.5.9.)
    """
    for key, value in where.items():
        if key in ("$and", "$or"):
            if any(_rejected_by_the_engine(sub) for sub in value):
                return True
            continue
        if str(key).startswith("$"):
            return True
        if not isinstance(value, dict):
            continue
        for operator, operand in value.items():
            if operator in _RANGE_OPERATORS and (isinstance(operand, bool) or operand != operand):
                return True
            if operator in ("$eq", "$ne") and isinstance(operand, (list, tuple)):
                return True
            if operator in ("$in", "$nin") and not all(_is_scalar(o) for o in operand):
                return True
    return False


def check_where(filters: Any, group_id: int | None = None) -> None:
    """Raise ValueError for a filter 0.17 would have failed on, with its message.

    0.17 passed ``filters`` to ChromaDB as the ``where`` of its query (ANDed
    with the library clause when a library was named) and surfaced whatever
    ChromaDB said, which is ChromaDB's validator's text followed by ``in
    query.``. The same validator runs here on the same ``where``, so a
    several-key dict, a one-element ``$and``, an unknown operator or a bad
    operand fail exactly as before, before any hit is looked at.
    """
    where = filters
    if group_id is not None:
        clause = {"group_id": int(group_id)}
        where = {"$and": [filters, clause]} if filters else clause
    if where is None:
        return
    try:
        from chromadb.api.types import validate_where
    except ImportError:  # no ChromaDB to ask: the evaluator's own checks remain
        return
    try:
        validate_where(where)
    except ValueError as e:
        raise ValueError(f"{e} in query.") from None
    if _rejected_by_the_engine(where):
        raise ValueError("Invalid where clause")


def split_filters(where: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Split ``where`` into ``(chroma_where, live_where)``.

    A clause naming only stored fields goes to Chroma; one naming a live
    field is evaluated live. A ``$or`` that mixes both cannot be split, so it
    is evaluated live as a whole, with the stored fields read from the hit's
    metadata. Unknown field names stay on the Chroma side, as before.
    """
    if not where:
        return None, None
    chroma: list[dict[str, Any]] = []
    live: list[dict[str, Any]] = []
    for clause in _conjuncts(where):
        (live if referenced_fields(clause) & LIVE_FIELDS else chroma).append(clause)
    live_where = combine(live)
    # Checked up front: evaluation only meets a bad condition when a hit
    # reaches it, so a search with no hits would hide the mistake.
    validate(live_where)
    return combine(chroma), live_where


def matches(where: dict[str, Any] | None, fields: dict[str, Any]) -> bool:
    """Evaluate a Chroma-style ``where`` over ``fields``, as ChromaDB evaluates metadata.

    Supports ``$eq``, ``$ne``, ``$in``, ``$nin``, ``$gt``, ``$gte``, ``$lt``,
    ``$lte``, ``$contains``, ``$not_contains``, ``$and``, ``$or`` and the
    plain-value shorthand for ``$eq``. As in ChromaDB: a field that is absent
    matches ``$ne`` and ``$nin`` and nothing else; types are strict (``True``
    is not ``1``, ``"0"`` is not ``0``, ``0`` equals ``0.0``); range operators
    compare numbers only; a stored value is a scalar, so ``$contains`` matches
    nothing and ``$not_contains`` everything. ``tags`` is the space-joined tag
    string, compared as a string.
    """
    if not where:
        return True
    for key, cond in where.items():
        if key == "$and":
            ok = all(matches(c, fields) for c in cond)
        elif key == "$or":
            ok = any(matches(c, fields) for c in cond)
        else:
            ok = _field_matches(fields.get(key, _MISSING), cond)
        if not ok:
            return False
    return True


_OPERATORS = ("$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte", "$contains", "$not_contains")


def _condition(cond: Any) -> tuple[str, Any]:
    """``(operator, operand)`` of one field condition, validated as Chroma would."""
    if isinstance(cond, dict) and any(str(k).startswith("$") for k in cond):
        if len(cond) != 1:
            raise ValueError(f"A filter condition takes one operator, got {sorted(cond)}; combine them with $and")
        ((op, operand),) = cond.items()
    else:
        op, operand = "$eq", cond
    if op not in _OPERATORS:
        raise ValueError(f"Unsupported filter operator {op!r}")
    if op in ("$in", "$nin") and not isinstance(operand, (list, tuple)):
        raise ValueError(f"{op} takes a list, got {operand!r}")
    return op, operand


def validate(where: dict[str, Any] | None) -> None:
    """Raise ValueError for a condition ``matches`` cannot evaluate, at any depth."""
    for key, cond in (where or {}).items():
        if key in ("$and", "$or"):
            if not isinstance(cond, list):
                raise ValueError(f"{key} takes a list of conditions, got {cond!r}")
            for sub in cond:
                validate(sub)
        else:
            _condition(cond)


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "str"
    return "other"


def _same(value: Any, operand: Any) -> bool:
    """ChromaDB's equality: same type (booleans apart from numbers) and equal."""
    return _kind(value) == _kind(operand) and value == operand


def _field_matches(value: Any, cond: Any) -> bool:
    op, operand = _condition(cond)
    absent = value is _MISSING or value is None
    if op == "$ne":
        return absent or not _same(value, operand)
    if op == "$nin":
        return absent or not any(_same(value, o) for o in operand)
    if op == "$not_contains":
        return True
    if absent or op == "$contains":
        return False
    if op == "$eq":
        return _same(value, operand)
    if op == "$in":
        return any(_same(value, o) for o in operand)
    if _kind(value) != "number" or _kind(operand) != "number":
        return False
    if op == "$gt":
        return value > operand
    if op == "$gte":
        return value >= operand
    if op == "$lt":
        return value < operand
    return value <= operand


def _citation_key(data: dict[str, Any]) -> str:
    """The Better BibTeX key from ``extra``, as 0.17 stored it: the first ``citation key:`` line."""
    for line in (data.get("extra") or "").split("\n"):
        if line.lower().startswith(("citation key:", "citationkey:")):
            return line.split(":", 1)[1].strip()
    return ""


def live_fields(item: dict[str, Any]) -> dict[str, Any]:
    """The owner's filterable fields from a pyzotero-shaped item, in the formats 0.17 stored."""
    data = item.get("data") or {}
    return {
        "item_key": item.get("key") or data.get("key", ""),
        "item_type": data.get("itemType", ""),
        "title": data.get("title", ""),
        "date": data.get("date", ""),
        "date_added": data.get("dateAdded", ""),
        "date_modified": data.get("dateModified", ""),
        "creators": format_creators(data.get("creators") or []),
        "publication": data.get("publicationTitle", ""),
        "url": data.get("url", ""),
        "doi": data.get("DOI", ""),
        "citation_key": _citation_key(data),
        "tags": " ".join(t.get("tag", "") for t in data.get("tags") or []),
    }


def prefilter_safe(where: dict[str, Any] | None) -> bool:
    """Whether evaluating ``where`` over owners alone gives a superset of what hits satisfy.

    The owner fields read the same from zotero.sqlite as from a resolved
    hit, except ``tags``: it is one string whose order of tags may differ
    between the two sources. An operand with no space can equal it only when
    the item has that one tag (or none), whatever the order; an operand with
    a space might equal one order and not the other, so it is not safe.
    """
    if not where:
        return False
    if referenced_fields(where) - OWNER_FIELDS:
        return False
    return not any(" " in operand for operand in _tag_operands(where))


def _tag_operands(where: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key, cond in where.items():
        if key in ("$and", "$or"):
            for sub in cond:
                out.extend(_tag_operands(sub))
        elif key == "tags":
            try:
                _, operand = _condition(cond)
            except ValueError:
                continue
            for value in operand if isinstance(operand, (list, tuple)) else [operand]:
                if isinstance(value, str):
                    out.append(value)
    return out


def web_query_plan(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """What the Zotero web API can answer of a live filter, or None if it cannot answer it (#826).

    Only a conjunction (a single-key dict, or ``$and``) of conditions on three
    fields: ``item_type`` (``$eq``, a plain value or ``$in``), ``tags``
    (a plain value, ``$eq`` or ``$in``) and ``item_key`` (plain, ``$eq`` or
    ``$in``). Anything else (a dict of several keys, which is an error, ``$or``,
    ``$ne``, ``$nin``, a range, any other field) returns None, and the caller
    filters live.

    The plan is a superset: it names the items that have the tag, which
    includes items that have others too, and the live check keeps results
    exact. A tag value that is empty or holds a space, or any value that
    holds `` || ``, could not be sent faithfully, so it has no plan.

    The plan is ``{"itemType": "a || b" or None, "tag": [one "x || y" string
    per tag clause], "item_keys": set or None}``. The API ANDs repeated
    ``tag`` parameters and ORs the values joined by `` || ``. Several
    ``item_type`` or ``item_key`` clauses are intersected.
    """
    if not where:
        return None
    item_types: set[str] | None = None
    keys: set[str] | None = None
    tags: list[list[str]] = []
    for clause in _conjuncts(where):
        if not isinstance(clause, dict) or len(clause) != 1:
            return None
        ((field, cond),) = clause.items()
        if field not in ("item_type", "tags", "item_key"):
            return None
        try:
            op, operand = _condition(cond)
        except ValueError:
            return None
        if op == "$eq":
            values = [operand]
        elif op == "$in":
            values = list(operand)
        else:
            return None
        if not values or not all(isinstance(v, str) and v and "||" not in v for v in values):
            return None
        if field == "tags" and any(" " in v for v in values):
            return None
        if field == "item_type":
            item_types = set(values) if item_types is None else item_types & set(values)
        elif field == "item_key":
            keys = set(values) if keys is None else keys & set(values)
        else:
            tags.append(values)
    return {
        "itemType": " || ".join(sorted(item_types)) if item_types else None,
        "tag": [" || ".join(_api_tag(t) for t in values) for values in tags],
        "item_keys": keys,
        # An empty intersection matches nothing, with no request to make.
        "empty": item_types == set() or keys == set(),
    }


def _api_tag(tag: str) -> str:
    """A tag as the API's ``tag`` parameter reads it: a leading ``-`` would negate, so it is escaped."""
    return "\\" + tag if tag.startswith("-") else tag
