"""Make a tool's advertised input schema say only what an agent needs.

Every tool's parameter schema is sent to the client on every request. The
signatures here are deliberately lenient (``limit: int | str | None``,
``tags: list[str] | str | None``) so that clients which serialize arguments
oddly still work, but advertising every accepted shape costs tokens and
invites the odd shapes. :func:`advertise` rewrites what the schema *shows*
while the function, and so argument validation, is untouched:

* a ``null`` branch is dropped when the default is already ``None``; leaving
  the argument out says the same thing;
* ``"additionalProperties": true`` is dropped; it is the JSON Schema default;
* for list and count parameters named in :data:`LIST_PARAMS` and
  :data:`COUNT_PARAMS`, only the canonical shape (an array of strings, an
  integer) is shown. The string forms they also accept (a bare value, a JSON
  list, ``"10"``, or ``null`` for "use the default") are fallbacks for
  misbehaving clients, not a second API.

Parameters whose string form means something different (a page range, a JSON
filter, a single key) are not in those lists and keep every branch.
"""

from __future__ import annotations

import inspect
import typing
from typing import Annotated, Any

#: List-valued parameters whose string form is only a fallback encoding.
LIST_PARAMS = frozenset({
    "tag", "tags", "add_tags", "remove_tags",
    "collections", "collection_names", "add_to", "remove_from",
    "item_keys", "duplicate_keys", "remove_keys",
})

#: Integer parameters that also accept a numeric string.
COUNT_PARAMS = frozenset({"limit", "offset", "max_groups"})

_NULL = {"type": "null"}
_STR_ARRAY = {"type": "array", "items": {"type": "string"}}
_INT = {"type": "integer"}


def _drop_open_objects(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {
            k: _drop_open_objects(v)
            for k, v in schema.items()
            if not (k == "additionalProperties" and v is True)
        }
    if isinstance(schema, list):
        return [_drop_open_objects(v) for v in schema]
    return schema


def compact(name: str, schema: dict, default: Any) -> dict:
    """The schema to advertise for one parameter (without its default)."""
    schema = _drop_open_objects(schema)
    branches = schema.get("anyOf")
    if not branches:
        return schema
    if default is None:
        branches = [b for b in branches if b != _NULL]
    if name in LIST_PARAMS and _STR_ARRAY in branches:
        branches = [_STR_ARRAY]
    elif name in COUNT_PARAMS and _INT in branches:
        branches = [_INT]
    rest = {k: v for k, v in schema.items() if k != "anyOf"}
    if len(branches) == 1:
        return {**rest, **branches[0]}
    return {**rest, "anyOf": branches}


def advertise(wrapper, original) -> None:
    """Point ``wrapper``'s signature at compacted schemas, in place.

    ``original`` is the undecorated tool function, whose module resolves any
    string annotations. Each changed parameter becomes
    ``Annotated[<same type>, WithJsonSchema(<compact>)]``: pydantic validates
    exactly as before and only the generated JSON schema differs. On anything
    unexpected the signature is left alone; a larger schema is the worst case.
    """
    try:
        from pydantic import TypeAdapter, WithJsonSchema

        hints = typing.get_type_hints(inspect.unwrap(original), include_extras=True)
        sig = inspect.signature(original)
        params = []
        changed = False
        for p in sig.parameters.values():
            ann = hints.get(p.name, p.annotation)
            if p.name != "ctx" and ann is not inspect.Parameter.empty:
                full = TypeAdapter(ann).json_schema()
                small = compact(p.name, full, p.default)
                if small != full and "$defs" not in full and "$ref" not in str(full):
                    p = p.replace(annotation=Annotated[ann, WithJsonSchema(small)])
                    changed = True
                else:
                    p = p.replace(annotation=ann)
            elif ann is not inspect.Parameter.empty:
                p = p.replace(annotation=ann)
            params.append(p)
        if not changed:
            return
        ret = hints.get("return", sig.return_annotation)
        wrapper.__signature__ = sig.replace(parameters=params, return_annotation=ret)
        wrapper.__annotations__ = {
            p.name: p.annotation
            for p in params
            if p.annotation is not inspect.Parameter.empty
        }
        if ret is not inspect.Signature.empty:
            wrapper.__annotations__["return"] = ret
    except Exception:  # pragma: no cover - defensive; schema stays as generated
        return
