"""Bounded, structured reads of known items in the active library."""

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints, TypeAdapter, ValidationError

from zotero_mcp import library
from zotero_mcp._app import mcp
from zotero_mcp._context import Context, ToolError
from zotero_mcp.client import with_zotero_api_lock
from zotero_mcp.tools import _helpers

_Key = Annotated[str, Field(strict=True, pattern=r"^[A-Z0-9]{8}$")]
_CitationKey = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]


class _Read(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Metadata(_Read):
    op: Literal["item_metadata"]
    item_key: _Key
    include: dict[Literal["abstract", "tags", "collections", "relations"], StrictBool] = Field(default_factory=dict)


class _Children(_Read):
    op: Literal["item_children"]
    item_key: _Key
    child_types: list[Literal["note", "attachment", "annotation"]] | None = None


class _Notes(_Read):
    op: Literal["notes"]
    parent_item_key: _Key


class _Citation(_Read):
    op: Literal["by_citation_key"]
    citation_key: _CitationKey


_Operation = _Metadata | _Children | _Notes | _Citation
_OPERATION = TypeAdapter(Annotated[_Operation, Field(discriminator="op")])
_VERBS = {"item_metadata", "item_children", "notes", "by_citation_key"}


def _result(index, op, error=None, **payload):
    return {"index": index, "ok": error is None, "op": op, "error": error, **payload}


def _key(operation):
    return operation.parent_item_key if isinstance(operation, _Notes) else operation.item_key


def _load(backend, operations):
    """One plural read per kind, without hiding a failed backend as empty data."""
    keyed = [op for op in operations if isinstance(op, _Metadata | _Children | _Notes)]
    keys = list(dict.fromkeys(_key(op) for op in keyed))
    items, children = {}, {}
    if not keys:
        return items, children
    try:
        items = backend.get_items(keys)
    except Exception as exc:
        return exc, children
    child_keys = list(dict.fromkeys(_key(op) for op in keyed if not isinstance(op, _Metadata) and _key(op) in items))
    if child_keys:
        try:
            children = backend.get_children(child_keys)
        except Exception as exc:
            children = exc
    return items, children


def _payload(operation, backend, items, children):
    if isinstance(operation, _Citation):
        found = backend.find_by_citation_key(operation.citation_key)
        return {"item_key": found.get("key") if found else None, "metadata": found}
    if isinstance(items, Exception):
        raise items
    key = _key(operation)
    if key not in items:
        return None
    if isinstance(operation, _Metadata):
        # A selection must not mutate cached records used by later operations.
        record = {**items[key], "data": dict(items[key].get("data", {}))}
        for name, enabled in operation.include.items():
            if not enabled:
                record["data"].pop("abstractNote" if name == "abstract" else name, None)
        return {"metadata": record}
    if isinstance(children, Exception):
        raise children
    if key not in children:
        return None
    if isinstance(operation, _Notes):
        return {
            "notes": [
                {
                    "note_key": note.get("key"),
                    "content": note.get("data", {}).get("note", ""),
                    "tags": [t["tag"] for t in _helpers._normalize_item_tags(note.get("data", {}).get("tags"))],
                }
                for note in children[key]
                if note.get("data", {}).get("itemType") == "note"
            ]
        }
    return {
        "children": [
            child
            for child in children[key]
            if operation.child_types is None or child.get("data", {}).get("itemType") in operation.child_types
        ]
    }


@mcp.tool(
    name="zotero_bulk_get",
    description=(
        "Structured mixed reads, up to 50 operations in the active library. "
        "Each operation has op: item_metadata or item_children (item_key), "
        "notes (parent_item_key), or by_citation_key (citation_key). "
        "item_metadata accepts include: {abstract,tags,collections,relations: bool}; "
        "false omits a field. item_children accepts child_types: an array of "
        "note/attachment/annotation. Returns JSON {results:[{index,ok,op,error,...payload}]}, "
        "in input order. continue_on_error=False stops reading on the first error "
        "and marks the remaining results aborted. No writes, fulltext, PDF or "
        "connector operations. Existing single-item tools remain available."
    ),
)
@with_zotero_api_lock
def bulk_get(operations: list[dict[str, Any]] | str, continue_on_error: bool = True, *, ctx: Context) -> str:
    """Return metadata, children, notes or citation-key matches as index-aligned JSON.

    Default mode batches item and child lookups through the existing library
    backend. Stop-on-error mode reads sequentially, never prefetching later
    operations. This is read-only batching, not a transactional snapshot.
    """
    if isinstance(operations, str):
        try:
            operations = json.loads(operations)
        except json.JSONDecodeError as exc:
            raise ToolError("operations must be an array of objects or its JSON encoding.") from exc
    if not isinstance(operations, list) or any(not isinstance(op, dict) for op in operations):
        raise ToolError("operations must be an array of objects or its JSON encoding.")
    if len(operations) > 50:
        raise ToolError("A batch accepts at most 50 operations; split larger requests.")
    parsed = []
    for index, raw in enumerate(operations):
        op = raw.get("op")
        if not isinstance(op, str) or op not in _VERBS:
            parsed.append(_result(index, op, "unknown_op"))
            continue
        try:
            parsed.append(_OPERATION.validate_python(raw))
        except ValidationError as exc:
            parsed.append(_result(index, op, "invalid_arguments", detail=exc.errors()[0]["msg"]))

    backend = None
    items, children = {}, {}
    setup_error = None
    valid = [op for op in parsed if isinstance(op, _Read)]
    if valid and continue_on_error:
        try:
            backend = library.get_library_backend()
            items, children = _load(backend, valid)
        except Exception as exc:
            setup_error = exc
    results = []
    stopped = False
    for index, operation in enumerate(parsed):
        op = operation["op"] if isinstance(operation, dict) else operation.op
        if stopped:
            result = _result(index, op, "aborted")
        elif isinstance(operation, dict):
            result = operation
        else:
            try:
                if setup_error is not None:
                    raise setup_error
                if not continue_on_error:
                    if backend is None:
                        backend = library.get_library_backend()
                    items, children = _load(backend, [operation])
                payload = _payload(operation, backend, items, children)
                result = (
                    _result(index, op, **payload)
                    if payload is not None
                    else _result(index, op, "not_found_or_unavailable")
                )
            except Exception as exc:
                result = _result(index, op, "read_failed", detail=str(exc))
        results.append(result)
        stopped = not continue_on_error and not result["ok"]
    return json.dumps({"results": results}, ensure_ascii=False)
