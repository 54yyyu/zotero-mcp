"""Semantic-search filters, split between Chroma and live Zotero data (#826).

Chroma records hold only their own state (keys, library, passage location),
so a filter on a bibliographic field is evaluated against the hit's owner as
Zotero has it now, never against metadata copied into the index. Legacy
records still carry copied fields, but those go stale and are not consulted.

ChromaDB-free on purpose, so the tool layer and tests can import it cheaply.
"""

from __future__ import annotations

from typing import Any

from zotero_mcp.utils import format_creators

#: Stored in every record and never stale: filtered by Chroma.
CHROMA_FIELDS = frozenset({"group_id", "section", "page", "chunk_index", "n_chunks", "char_start", "char_end"})

#: Evaluated against the owner item's live data.
LIVE_FIELDS = frozenset(
    {"item_type", "item_key", "doi", "citation_key", "publication", "tags", "title", "has_fulltext",
     "date", "date_added", "creators", "url"}
)

#: Live fields that zotero.sqlite can answer for an owner alone. has_fulltext
#: depends on which records were hit, so it is known only at hit time.
OWNER_FIELDS = LIVE_FIELDS - {"has_fulltext"}

#: Copied into legacy records only; a filter on one would silently match nothing but them.
LEGACY_ONLY_FIELDS = frozenset(
    {"parent_item_key", "fulltext_source", "attachment_keys", "attachment_priority"}
)

_MISSING = object()


def _conjuncts(where: dict[str, Any]) -> list[dict[str, Any]]:
    """The clauses ANDed together at the top of ``where``.

    ``$and`` lists are flattened, and a dict with several keys counts as an
    implicit ``$and`` (Chroma itself rejects that form).
    """
    out: list[dict[str, Any]] = []
    for key, value in where.items():
        if key == "$and":
            for sub in value:
                out.extend(_conjuncts(sub))
        else:
            out.append({key: value})
    return out


def _normalise(where: dict[str, Any]) -> dict[str, Any]:
    """``where`` with every multi-key dict inside ``$or`` rewritten as ``$and`` of its keys.

    ``matches`` ANDs the keys of such a dict, but Chroma rejects it, so both
    sides must see the explicit form.
    """
    out: dict[str, Any] = {}
    for key, value in where.items():
        if key in ("$and", "$or") and isinstance(value, list):
            subs = []
            for sub in value:
                if isinstance(sub, dict):
                    sub = _normalise(sub)
                    if key == "$or" and len(sub) > 1:
                        sub = {"$and": [{k: v} for k, v in sub.items()]}
                subs.append(sub)
            out[key] = subs
        else:
            out[key] = value
    return out


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


def split_filters(where: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Split ``where`` into ``(chroma_where, live_where)``.

    A clause naming only stored fields goes to Chroma; one naming only live
    fields is evaluated live. A ``$or`` that mixes both cannot be split, so it
    is evaluated live as a whole, with the stored fields read from the hit's
    metadata. A field name outside the supported list raises ValueError. A
    dict with several keys inside ``$or`` is rewritten as ``$and`` of its keys
    (what the live evaluator already does), because Chroma rejects it.
    """
    if not where:
        return None, None
    validate(where)
    where = _normalise(where)
    referenced = referenced_fields(where)
    supported = ", ".join(sorted(CHROMA_FIELDS | LIVE_FIELDS))
    unknown = referenced - CHROMA_FIELDS - LIVE_FIELDS - LEGACY_ONLY_FIELDS
    if unknown:
        raise ValueError(
            f"Cannot filter on {', '.join(sorted(unknown))}: not a filterable field, so the filter would "
            f"match nothing. Supported fields: {supported}."
        )
    unsupported = referenced & LEGACY_ONLY_FIELDS
    if unsupported:
        raise ValueError(
            f"Cannot filter on {', '.join(sorted(unsupported))}: records written by current versions do not "
            f"store it, so the filter would match only records from earlier versions. Supported fields: {supported}."
        )
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
    """Evaluate a Chroma-style ``where`` over ``fields``.

    Supports ``$eq``, ``$ne``, ``$in``, ``$nin``, ``$gt``, ``$gte``, ``$lt``,
    ``$lte``, ``$and``, ``$or`` and the plain-value shorthand for ``$eq``. A
    field that is absent matches nothing, as in Chroma. ``tags`` is a set and
    means membership: ``{"tags": "x"}`` holds when the owner has tag x.
    """
    if not where:
        return True
    for key, cond in where.items():
        if key == "$and":
            ok = all(matches(c, fields) for c in cond)
        elif key == "$or":
            ok = any(matches(c, fields) for c in cond)
        else:
            ok = _field_matches(fields.get(key, _MISSING), cond, key)
        if not ok:
            return False
    return True


_OPERATORS = ("$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte")
_SCALARS = (str, int, float, bool)


def _scalar(value: Any) -> bool:
    return isinstance(value, _SCALARS)


def _condition(cond: Any, field: str = "") -> tuple[str, Any]:
    """``(operator, operand)`` of one field condition, validated as Chroma would.

    A plain value means ``$eq``. ``$eq``, ``$ne`` and the range operators take
    one string, number or boolean; ``$in`` and ``$nin`` take a non-empty list
    of them. Anything else raises ValueError naming the field and operator.
    """
    where = f"filter on {field!r}: " if field else "filter: "
    if isinstance(cond, dict):
        if not cond:
            raise ValueError(f"{where}an empty condition; give a value or one operator such as $eq")
        if not any(str(k).startswith("$") for k in cond):
            raise ValueError(
                f"{where}a nested object {cond!r} is not a condition; use a value or one operator such as $eq"
            )
        if len(cond) != 1:
            raise ValueError(f"{where}a condition takes one operator, got {sorted(cond)}; combine them with $and")
        ((op, operand),) = cond.items()
    else:
        op, operand = "$eq", cond
    if op not in _OPERATORS:
        raise ValueError(f"{where}unsupported operator {op!r}; supported: {', '.join(_OPERATORS)}")
    if op in ("$in", "$nin"):
        if not isinstance(operand, (list, tuple)):
            raise ValueError(f"{where}{op} takes a list of values, got {operand!r}")
        if not operand:
            raise ValueError(f"{where}{op} takes a non-empty list")
        if not all(_scalar(v) for v in operand):
            raise ValueError(f"{where}{op} takes a list of strings, numbers or booleans, got {list(operand)!r}")
    elif not _scalar(operand):
        raise ValueError(f"{where}{op} takes a single string, number or boolean, got {operand!r}")
    return op, operand


def validate(where: dict[str, Any] | None) -> None:
    """Raise ValueError for a condition ``matches`` cannot evaluate, at any depth.

    The shapes are the ones ChromaDB accepts: a dict of field conditions and
    ``$and`` / ``$or`` lists of non-empty dicts, and per field a value or one
    operator with a correctly shaped operand (see ``_condition``).
    """
    if where is None:
        return
    if not isinstance(where, dict):
        raise ValueError(f"filters must be an object, got {where!r}")
    for key, cond in where.items():
        if key in ("$and", "$or"):
            if not isinstance(cond, list) or not cond:
                raise ValueError(f"{key} takes a non-empty list of conditions, got {cond!r}")
            for sub in cond:
                if not isinstance(sub, dict) or not sub:
                    raise ValueError(f"{key} takes a list of non-empty objects, got {sub!r}")
                validate(sub)
        elif str(key).startswith("$"):
            raise ValueError(f"unsupported operator {key!r} at the top of a filter; use $and or $or")
        else:
            _condition(cond, key)


def _field_matches(value: Any, cond: Any, field: str = "") -> bool:
    op, operand = _condition(cond, field)
    if value is _MISSING or value is None:
        return False
    if isinstance(value, (set, frozenset)):
        if op == "$eq":
            return operand in value
        if op == "$ne":
            return operand not in value
        if op == "$in":
            return any(o in value for o in operand)
        if op == "$nin":
            return not any(o in value for o in operand)
        return False
    if op == "$eq":
        return value == operand
    if op == "$ne":
        return value != operand
    if op == "$in":
        return value in operand
    if op == "$nin":
        return value not in operand
    try:
        if op == "$gt":
            return value > operand
        if op == "$gte":
            return value >= operand
        if op == "$lt":
            return value < operand
        return value <= operand
    except TypeError:
        return False


def _citation_key(data: dict[str, Any]) -> str:
    if key := data.get("citationKey"):
        return key
    for line in (data.get("extra") or "").split("\n"):
        if line.lower().startswith(("citation key:", "citationkey:")):
            return line.split(":", 1)[1].strip()
    return ""


def live_fields(item: dict[str, Any]) -> dict[str, Any]:
    """The owner's filterable fields from a pyzotero-shaped item."""
    data = item.get("data") or {}
    return {
        "item_key": item.get("key") or data.get("key", ""),
        "item_type": data.get("itemType", ""),
        "title": data.get("title", ""),
        "doi": data.get("DOI", ""),
        "publication": data.get("publicationTitle", ""),
        "citation_key": _citation_key(data),
        "tags": frozenset(t.get("tag", "") for t in data.get("tags") or []),
        # As 0.17 stored them: Zotero's date text, the ISO 8601 UTC timestamp
        # the item was added, "Last, First; Last, First" (what
        # ``format_creators`` makes, "No authors listed" for none), the URL.
        "date": data.get("date", ""),
        "date_added": data.get("dateAdded", ""),
        "creators": format_creators(data.get("creators") or []),
        "url": data.get("url", ""),
    }


def equality_clauses(where: dict[str, Any] | None, fields: tuple[str, ...]) -> dict[str, list[list[str]]]:
    """The ``$eq`` / ``$in`` conditions on ``fields`` that ``where`` ANDs at its top level.

    ``{field: [values, ...]}``, one list of alternatives per clause, so
    repeated clauses on a field all apply. A clause is left out unless it is
    a single-field condition whose operand is non-empty strings: the caller
    narrows candidates with what is returned and still evaluates ``where``
    in full, so a clause left out only means less narrowing.
    """
    out: dict[str, list[list[str]]] = {}
    for clause in _conjuncts(where) if where else []:
        if not isinstance(clause, dict) or len(clause) != 1:
            continue
        ((field, cond),) = clause.items()
        if field not in fields:
            continue
        try:
            op, operand = _condition(cond)
        except ValueError:
            continue
        values = [operand] if op == "$eq" else list(operand) if op == "$in" else []
        if values and all(isinstance(v, str) and v for v in values):
            out.setdefault(field, []).append(values)
    return out


def web_query_plan(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """What the Zotero web API can answer of a live filter, or None if it cannot answer it (#826).

    Only a conjunction (top-level keys and ``$and``) of conditions on three
    fields: ``item_type`` (``$eq``, a plain value or ``$in``), ``tags``
    (membership: a plain value, ``$eq`` or ``$in``) and ``item_key`` (plain,
    ``$eq`` or ``$in``). Anything else (``$or``, ``$ne``, ``$nin``, a range,
    any other field) returns None, and the caller filters live.

    The plan is ``{"itemType": "a || b" or None, "tag": [one "x || y" string
    per tag clause], "item_keys": set or None}``. The API ANDs repeated
    ``tag`` parameters and ORs the values joined by `` || ``. Several
    ``item_type`` or ``item_key`` clauses are intersected. A value that holds
    `` || `` could not be told from the separator, so it is not expressible.
    """
    if not where:
        return None
    item_types: set[str] | None = None
    keys: set[str] | None = None
    tags: list[list[str]] = []
    for clause in _conjuncts(where):
        ((field, cond),) = clause.items()
        if field not in ("item_type", "tags", "item_key"):
            return None
        try:
            op, operand = _condition(cond, field)
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
