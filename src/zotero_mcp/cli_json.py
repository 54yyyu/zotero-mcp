"""Machine-readable output for ``zotero-cli --json``.

The CLI's default output is markdown written for a person reading a terminal.
That is the wrong shape for the other half of its audience: shell pipelines,
cron jobs, and agents with shell access, all of which have to regex prose to
get an item key back out.

``--json`` gives those callers a stable envelope instead. Two rules hold
everywhere:

* Every response is a single JSON object with ``ok`` (bool) and ``command``
  (str). Success carries ``data``; failure carries ``error`` with ``message``
  and a machine-comparable ``code``. A caller can branch on ``ok`` without
  knowing which command ran.
* Item projections are *stable subsets*, never reformatted prose. Anything
  listing items yields the same item shape (:func:`project_item`), so a
  pipeline written against ``search`` also works against ``get recent`` or
  ``get collection-items``.

Where a command's real answer is a human-facing status line and nothing more
-- a write that succeeded, a config dump -- ``data`` carries ``{"text": ...}``
rather than inventing structure that isn't there. That is honest about what
the underlying tool returns, and it keeps `ok`/`error` useful regardless.
"""

from __future__ import annotations

import json
import sys
from typing import Any

#: Bumped when the envelope's own shape changes in a way that could break a
#: caller. Individual commands may add fields to `data` without a bump --
#: additive changes are always allowed, removals are not.
SCHEMA_VERSION = 1


class CliError(Exception):
    """An error with a stable machine-readable code.

    The code is the part a script should branch on; the message is for the
    human reading the log.
    """

    def __init__(self, message: str, code: str = "error"):
        super().__init__(message)
        self.code = code


def envelope(command: str, data: Any = None, *, ok: bool = True,
             error: str | None = None, code: str | None = None) -> dict:
    """Build the response object. Shape is identical for every command."""
    out: dict[str, Any] = {"ok": ok, "command": command, "schema": SCHEMA_VERSION}
    if ok:
        out["data"] = data if data is not None else {}
    else:
        out["error"] = {"message": error or "unknown error", "code": code or "error"}
    return out


def emit(command: str, data: Any = None, *, stream=None) -> None:
    """Print a success envelope to stdout."""
    print(json.dumps(envelope(command, data), ensure_ascii=False, default=str),
          file=stream or sys.stdout)


def emit_error(command: str, message: str, code: str = "error", *, stream=None) -> None:
    """Print a failure envelope.

    Deliberately to *stdout*, not stderr: a caller reading one stream should
    not have to merge two to find out that the call failed. The human-readable
    path keeps writing errors to stderr, which is right for a terminal.
    """
    print(
        json.dumps(envelope(command, ok=False, error=message, code=code),
                   ensure_ascii=False, default=str),
        file=stream or sys.stdout,
    )


# ---------------------------------------------------------------------------
# Item projection
# ---------------------------------------------------------------------------

def _creators(data: dict) -> list[dict]:
    out = []
    for creator in data.get("creators", []) or []:
        if not isinstance(creator, dict):
            continue
        entry = {"type": creator.get("creatorType", "author")}
        if creator.get("name"):
            entry["name"] = creator["name"]
        else:
            entry["first"] = creator.get("firstName", "")
            entry["last"] = creator.get("lastName", "")
        out.append(entry)
    return out


def project_item(item: dict, detail: str = "summary") -> dict:
    """Project a raw Zotero item to the CLI's stable item shape.

    ``detail`` selects how much comes along:

    * ``keys_only`` -- key, type, title, date. Enough to decide what to fetch
      next, and small enough to list a whole library.
    * ``summary`` (default) -- adds creators, DOI, publication, tags.
    * ``full`` -- adds the abstract and the complete raw ``data`` dict under
      ``raw``, for callers that need a field this projection doesn't name.

    Every level carries ``key`` and ``itemType``, so a consumer can always
    identify what it is holding.
    """
    from zotero_mcp.utils import item_display_title

    data = item.get("data", {}) or {}
    out: dict[str, Any] = {
        "key": item.get("key") or data.get("key", ""),
        "itemType": data.get("itemType", ""),
        "title": item_display_title(data),
        "date": data.get("date", ""),
    }
    if data.get("deleted"):
        # A trashed item looks live in every other respect; a caller acting on
        # one without knowing is the failure mode this exists to prevent.
        out["deleted"] = True

    if detail == "keys_only":
        return out

    out["creators"] = _creators(data)
    for field, key in (("DOI", "doi"), ("publicationTitle", "publication"),
                       ("url", "url"), ("parentItem", "parentItem")):
        if value := data.get(field):
            out[key] = value
    out["tags"] = [
        t.get("tag") for t in data.get("tags", []) or [] if isinstance(t, dict) and t.get("tag")
    ]
    out["collections"] = list(data.get("collections", []) or [])

    if detail == "full":
        if abstract := data.get("abstractNote"):
            out["abstract"] = abstract
        out["raw"] = data
    return out


def project_items(items: list[dict], detail: str = "summary") -> list[dict]:
    return [project_item(i, detail) for i in items or []]


def project_collection(collection: dict) -> dict:
    data = collection.get("data", {}) or {}
    meta = collection.get("meta", {}) or {}
    out = {
        "key": collection.get("key") or data.get("key", ""),
        "name": data.get("name", ""),
        "parentCollection": data.get("parentCollection") or None,
    }
    for field, key in (("numItems", "numItems"), ("numCollections", "numCollections")):
        if field in meta:
            out[key] = meta[field]
    return out


def project_tag(tag: Any) -> dict:
    """Zotero returns tags as bare strings from some endpoints, dicts from
    others. Normalise so a caller never has to check."""
    if isinstance(tag, str):
        return {"tag": tag}
    if isinstance(tag, dict):
        out = {"tag": tag.get("tag", "")}
        if (meta := tag.get("meta")) and isinstance(meta, dict):
            if "numItems" in meta:
                out["numItems"] = meta["numItems"]
        return out
    return {"tag": str(tag)}


def project_annotation(item: dict) -> dict:
    data = item.get("data", {}) or {}
    out = {
        "key": item.get("key") or data.get("key", ""),
        "type": data.get("annotationType", ""),
        "text": data.get("annotationText", ""),
        "comment": data.get("annotationComment", ""),
        "color": data.get("annotationColor", ""),
        "pageLabel": data.get("annotationPageLabel", ""),
        "parentItem": data.get("parentItem", ""),
    }
    out["tags"] = [
        t.get("tag") for t in data.get("tags", []) or [] if isinstance(t, dict) and t.get("tag")
    ]
    return out


def project_note(item: dict) -> dict:
    from zotero_mcp.utils import html_to_text, note_title

    data = item.get("data", {}) or {}
    body = data.get("note", "") or ""
    return {
        "key": item.get("key") or data.get("key", ""),
        "title": note_title(body),
        "text": html_to_text(body),
        "html": body,
        "parentItem": data.get("parentItem", ""),
        "tags": [
            t.get("tag") for t in data.get("tags", []) or []
            if isinstance(t, dict) and t.get("tag")
        ],
    }
