"""Semantic-search filters, split between Chroma and live Zotero data (#826).

Chroma records hold only their own state (keys, library, passage location),
so a filter on a bibliographic field is evaluated against the hit's owner as
Zotero has it now, never against metadata copied into the index. Legacy
records still carry copied fields, but those go stale and are not consulted.

ChromaDB-free on purpose, so the tool layer and tests can import it cheaply.
"""

from __future__ import annotations

from typing import Any

#: Stored in every record and never stale: filtered by Chroma.
CHROMA_FIELDS = frozenset({"group_id", "section", "page", "chunk_index", "n_chunks", "char_start", "char_end"})

#: Evaluated against the owner item's live data.
LIVE_FIELDS = frozenset(
    {"item_type", "item_key", "doi", "citation_key", "publication", "tags", "title", "has_fulltext"}
)

#: Live fields that zotero.sqlite can answer for an owner alone. has_fulltext
#: depends on which records were hit, so it is known only at hit time.
OWNER_FIELDS = LIVE_FIELDS - {"has_fulltext"}

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
    metadata. Unknown field names stay on the Chroma side, as before.
    """
    if not where:
        return None, None
    chroma: list[dict[str, Any]] = []
    live: list[dict[str, Any]] = []
    for clause in _conjuncts(where):
        (live if referenced_fields(clause) & LIVE_FIELDS else chroma).append(clause)
    return combine(chroma), combine(live)


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
            ok = _field_matches(fields.get(key, _MISSING), cond)
        if not ok:
            return False
    return True


def _field_matches(value: Any, cond: Any) -> bool:
    if isinstance(cond, dict) and len(cond) == 1 and next(iter(cond)).startswith("$"):
        ((op, operand),) = cond.items()
    else:
        op, operand = "$eq", cond
    if op not in ("$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte"):
        raise ValueError(f"Unsupported filter operator {op!r}")
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
    }
