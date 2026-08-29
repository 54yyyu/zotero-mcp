"""Backend-neutral read port over a Zotero library.

Every read tool used to reach pyzotero directly, which had two consequences.
Closing Zotero desktop broke reading a library sitting readable on disk; and
the access patterns those tools inherited were shaped by HTTP — paged at 100
rows, one round trip per item — so they stayed expensive even when the data
was local. Measured against a real 36k-item / 2.3 GB library, listing 17 302
tags through ``_paginate`` cost 14.9 s against 221 ms for a single SELECT, and
fetching 100 items' children one key at a time cost 5.7 s against 31 ms.

This module defines the read operations those tools actually need, shaped for
bulk access rather than for paging, with two implementations:

``SqliteBackend``
    Answers from ``zotero.sqlite`` in a fixed number of queries per call, with
    no HTTP at all. Raises :class:`UnsupportedByBackend` for reads it cannot
    serve faithfully, rather than quietly answering a narrower question.

``ApiBackend``
    The pyzotero calls the tools used to make, preserved as they were.

The port is **read-only**. Writes keep going through
:func:`zotero_mcp.client.get_zotero_client`, which is unchanged and still
serves every write call site, ``zot._retrieve_data`` and the
``library_id``/``library_type`` attributes.

Both backends return **pyzotero-shaped dicts** (``{"key", "version", "data"}``)
rather than a bespoke record type. That is deliberate: ``format_item_metadata``,
``generate_bibtex``, ``item_display_title`` and the formatting body of every
ported tool already consume that shape. What changes here is how rows are
*fetched*, not what they look like afterwards.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from zotero_mcp import utils as _utils


class UnsupportedByBackend(Exception):
    """This backend cannot serve this read faithfully.

    Raised rather than returning a partial or narrowed answer — the same
    principle the global-search path already follows. Callers that have a
    working Zotero API available may catch it and retry that one call
    through :func:`zotero_mcp.client.get_zotero_client`; callers that do
    not should surface the message, which says what is missing and why.
    """


@runtime_checkable
class LibraryBackend(Protocol):
    """The read surface every tool in this codebase actually needs.

    Plural by default. ``get_children`` takes a list of parents and
    ``get_items`` a list of keys because the single-key versions are what
    made the API-shaped code slow, and a backend that can answer a batch in
    one query should never be asked N times.
    """

    name: str

    # -- items ------------------------------------------------------------
    def get_item(self, key: str) -> dict | None: ...
    def get_items(self, keys: list[str]) -> dict[str, dict]: ...
    def get_children(self, keys: list[str], *, item_type: str | None = None) -> dict[str, list[dict]]: ...
    """Children per parent key.

    A parent with no children maps to an empty list; a parent whose lookup
    *failed* is absent from the mapping entirely. Callers rely on that
    distinction to report a bad key differently from an empty one.
    """

    def get_related(self, key: str) -> list[dict]: ...
    def find_by_citation_key(self, citekey: str) -> dict | None: ...
    def recent_items(self, *, limit: int = 10, collection_key: str | None = None) -> list[dict]: ...
    def top_items(self, *, limit: int = 100) -> list[dict]: ...
    def list_items(
        self,
        item_type: str | None = None,
        *,
        limit: int = 100,
        tag: list[str] | None = None,
    ) -> list[dict]: ...

    # -- collections ------------------------------------------------------
    def list_collections(self, *, include_trashed: bool = False) -> list[dict]: ...
    def get_collection(self, key: str) -> dict | None: ...
    def collection_items(
        self, key: str, *, include_subcollections: bool = False
    ) -> list[dict] | None: ...

    # -- tags -------------------------------------------------------------
    def list_tags(self, *, limit: int | None = None) -> list[str]: ...

    # -- search -----------------------------------------------------------
    def search_items(
        self,
        query: str,
        *,
        qmode: str = "titleCreatorYear",
        item_type: str = "-attachment",
        tag: list[str] | None = None,
        limit: int = 10,
        collection_keys: list[str] | None = None,
        include_subcollections: bool = False,
    ) -> list[dict]: ...

    # -- files ------------------------------------------------------------
    def attachment_paths(self, key: str) -> list[dict]: ...


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

# One LocalZoteroReader per thread. sqlite3 connections are bound to the
# thread that opened them (`check_same_thread` defaults to True) and MCP tool
# calls arrive on a pool, so a single process-wide reader would raise as soon
# as a second thread touched it. Per-thread also means the database path is
# probed once per thread rather than once per tool call.
_thread_state = threading.local()


def _sqlite_reader():
    """The calling thread's LocalZoteroReader, or None if there is no DB."""
    reader = getattr(_thread_state, "reader", None)
    if reader is None:
        from zotero_mcp.local_db import get_local_zotero_reader

        reader = get_local_zotero_reader()
        _thread_state.reader = reader
    return reader


def reset_sqlite_reader() -> None:
    """Drop this thread's cached reader (tests, and after a DB path change)."""
    reader = getattr(_thread_state, "reader", None)
    if reader is not None:
        try:
            reader.close()
        except Exception:
            pass
    _thread_state.reader = None


class SqliteBackend:
    """Reads served directly from ``zotero.sqlite``. Never makes an HTTP call."""

    name = "sqlite"

    def __init__(self, reader, group_id: int):
        self._reader = reader
        self._group_id = group_id

    # -- items ------------------------------------------------------------

    def get_item(self, key: str) -> dict | None:
        return self._reader.get_full_items([key], group_id=self._group_id).get(key)

    def get_items(self, keys: list[str]) -> dict[str, dict]:
        return self._reader.get_full_items(list(keys), group_id=self._group_id)

    def get_children(self, keys: list[str], *, item_type: str | None = None) -> dict[str, list[dict]]:
        return self._reader.get_children_of(
            list(keys), item_type=item_type, group_id=self._group_id
        )

    def get_related(self, key: str) -> list[dict]:
        return self._reader.get_related_items(key, group_id=self._group_id)

    def find_by_citation_key(self, citekey: str) -> dict | None:
        return self._reader.find_by_citation_key(citekey, group_id=self._group_id)

    def recent_items(self, *, limit: int = 10, collection_key: str | None = None) -> list[dict]:
        result = self._reader.get_recent_items(
            limit=limit, collection_key=collection_key, group_id=self._group_id
        )
        if result is None:
            raise UnsupportedByBackend("that sort order is not available from SQLite")
        return result

    def top_items(self, *, limit: int = 100) -> list[dict]:
        result = self._reader.get_recent_items(limit=limit, group_id=self._group_id)
        if result is None:
            raise UnsupportedByBackend("top-level listing is not available from SQLite")
        return result

    def list_items(
        self,
        item_type: str | None = None,
        *,
        limit: int = 100,
        tag: list[str] | None = None,
    ) -> list[dict]:
        result = self._reader.list_items_of_type(
            item_type, limit=limit, group_id=self._group_id, tag=tag
        )
        if result is None:
            raise UnsupportedByBackend(
                "that itemType or tag filter is not expressible in SQL"
            )
        return result

    # -- collections ------------------------------------------------------

    def list_collections(self, *, include_trashed: bool = False) -> list[dict]:
        return self._reader.list_collections(
            group_id=self._group_id, include_trashed=include_trashed
        )

    def get_collection(self, key: str) -> dict | None:
        return self._reader.get_collection(key, group_id=self._group_id)

    def collection_items(
        self, key: str, *, include_subcollections: bool = False
    ) -> list[dict] | None:
        return self._reader.get_collection_items(
            key, include_subcollections=include_subcollections, group_id=self._group_id
        )

    # -- tags -------------------------------------------------------------

    def list_tags(self, *, limit: int | None = None) -> list[str]:
        return self._reader.list_tags(group_id=self._group_id, limit=limit)

    # -- search -----------------------------------------------------------

    def search_items(
        self,
        query: str,
        *,
        qmode: str = "titleCreatorYear",
        item_type: str = "-attachment",
        tag: list[str] | None = None,
        limit: int = 10,
        collection_keys: list[str] | None = None,
        include_subcollections: bool = False,
    ) -> list[dict]:
        result = self._reader.search_items_sql(
            query, qmode=qmode, item_type=item_type, tag=tag,
            limit=limit, group_id=self._group_id, collection_keys=collection_keys,
            include_subcollections=include_subcollections,
        )
        if result is None:
            raise UnsupportedByBackend(
                "the SQLite backend cannot express this query — a wildcard tag "
                "filter, or a boolean itemType expression like 'book || journalArticle'"
            )
        return result

    # -- files ------------------------------------------------------------

    def attachment_paths(self, key: str) -> list[dict]:
        return self._reader.get_attachment_paths(key)


# ---------------------------------------------------------------------------
# API backend
# ---------------------------------------------------------------------------


class ApiBackend:
    """The pyzotero reads the tools used to make, unchanged.

    Paging stays here — over HTTP it is not optional — which is why
    ``_paginate`` lives on this side of the port and never sees the SQLite
    backend. Handing an unpaged backend to ``_paginate`` would loop forever:
    it stops when a batch comes back shorter than the page size, and a
    backend that returns everything on the first call never does.
    """

    name = "api"

    def __init__(self, zot):
        self._zot = zot

    # -- items ------------------------------------------------------------

    def get_item(self, key: str) -> dict | None:
        try:
            return self._zot.item(key)
        except Exception:
            return None

    #: Zotero's cap on keys per ``itemKey`` parameter.
    _ITEMKEY_BATCH = 50

    def get_items(self, keys: list[str]) -> dict[str, dict]:
        found: dict[str, dict] = {}
        keys = list(keys)
        for start in range(0, len(keys), self._ITEMKEY_BATCH):
            batch = keys[start:start + self._ITEMKEY_BATCH]
            wanted = set(batch)
            try:
                for item in self._zot.items(itemKey=",".join(batch)) or []:
                    # Zotero's API answers an itemKey filter with the matching
                    # items *and their child notes* — verified against the
                    # local server at the HTTP level, so it is not a pyzotero
                    # artifact. Returning those would break this method's
                    # contract, which is "the keys you asked for".
                    if item.get("key") in wanted:
                        found[item["key"]] = item
            except Exception:
                continue
        # `items(itemKey=...)` omits trashed items, which `item()` returns.
        # Resolving the stragglers individually keeps this equivalent to the
        # per-key loops it replaced, while still costing one call in the
        # common case where nothing is missing.
        for key in keys:
            if key not in found:
                try:
                    if (item := self._zot.item(key)) is not None:
                        found[key] = item
                except Exception:
                    continue
        return found

    def get_children(self, keys: list[str], *, item_type: str | None = None) -> dict[str, list[dict]]:
        params: dict[str, Any] = {"itemType": item_type} if item_type else {}
        result: dict[str, list[dict]] = {}
        for key in keys:
            try:
                result[key] = _utils._paginate(self._zot.children, key, **params)
            except Exception:
                # Left absent, not empty: the caller reports a key that could
                # not be fetched differently from one with no children.
                continue
        return result

    def get_related(self, key: str) -> list[dict]:
        import re

        item = self.get_item(key)
        if not item:
            return []
        related: list[dict] = []
        seen: set[str] = set()
        for objects in (item.get("data", {}).get("relations") or {}).values():
            for uri in objects if isinstance(objects, list) else [objects]:
                match = re.search(r"/items/([A-Z0-9]{8})$", uri) if isinstance(uri, str) else None
                if match and match.group(1) not in seen:
                    seen.add(match.group(1))
                    if (fetched := self.get_item(match.group(1))) is not None:
                        related.append(fetched)
        return related

    def find_by_citation_key(self, citekey: str) -> dict | None:
        # No indexed field to query server-side, so this stays the ranked
        # substring search it always was: fetch candidates, verify locally.
        from zotero_mcp.tools import _helpers

        self._zot.add_parameters(
            q=citekey, qmode="everything", itemType="-attachment", limit=25
        )
        for item in self._zot.items() or []:
            data = item.get("data", {})
            if data.get("citationKey") == citekey or _helpers._extra_has_citekey(
                data.get("extra", ""), citekey
            ):
                return item
        return None

    def recent_items(self, *, limit: int = 10, collection_key: str | None = None) -> list[dict]:
        if collection_key:
            return _utils._paginate(
                self._zot.collection_items, collection_key, max_items=limit
            )[:limit]
        return _utils._paginate(
            self._zot.items, sort="dateAdded", direction="desc", max_items=limit
        )[:limit]

    def top_items(self, *, limit: int = 100) -> list[dict]:
        return _utils._paginate(self._zot.top, max_items=limit)[:limit]

    def list_items(
        self,
        item_type: str | None = None,
        *,
        limit: int = 100,
        tag: list[str] | None = None,
    ) -> list[dict]:
        params: dict[str, Any] = {}
        if item_type:
            params["itemType"] = item_type
        if tag:
            params["tag"] = tag
        return _utils._paginate(self._zot.items, max_items=limit, **params)

    # -- collections ------------------------------------------------------

    def list_collections(self, *, include_trashed: bool = False) -> list[dict]:
        collections = _utils._paginate(self._zot.collections)
        if include_trashed:
            from zotero_mcp.tools import _helpers

            existing = {c.get("key") for c in collections}
            for coll in _helpers.fetch_trashed_collections(self._zot):
                if coll.get("key") and coll["key"] not in existing:
                    coll.setdefault("data", {})["deleted"] = 1
                    collections.append(coll)
        return collections

    def get_collection(self, key: str) -> dict | None:
        try:
            return self._zot.collection(key)
        except Exception:
            return None

    def collection_items(
        self, key: str, *, include_subcollections: bool = False
    ) -> list[dict] | None:
        if self.get_collection(key) is None:
            return None
        keys = [key]
        if include_subcollections:
            from zotero_mcp.tools import _helpers

            keys = _helpers.expand_collection_scope(self._zot, key, True) or [key]
        items: list[dict] = []
        seen: set[str] = set()
        for scope_key in keys:
            for item in _utils._paginate(self._zot.collection_items, scope_key):
                if item.get("key") and item["key"] not in seen:
                    seen.add(item["key"])
                    items.append(item)
        return items

    # -- tags -------------------------------------------------------------

    def list_tags(self, *, limit: int | None = None) -> list[str]:
        tags = _utils._paginate(self._zot.tags, max_items=limit)
        return sorted(tags)

    # -- search -----------------------------------------------------------

    def search_items(
        self,
        query: str,
        *,
        qmode: str = "titleCreatorYear",
        item_type: str = "-attachment",
        tag: list[str] | None = None,
        limit: int = 10,
        collection_keys: list[str] | None = None,
        include_subcollections: bool = False,
    ) -> list[dict]:
        if collection_keys:
            # No server-side equivalent of "this tag, within these
            # collections" — page each collection with the filters applied
            # and merge, exactly as the tool used to do inline.
            results: list[dict] = []
            seen: set[str] = set()
            extra: dict[str, Any] = {}
            if tag:
                extra["tag"] = tag
            if item_type:
                extra["itemType"] = item_type
            from zotero_mcp.tools import _helpers

            scope_keys = list(collection_keys)
            if include_subcollections:
                scope_keys = [
                    expanded
                    for key in collection_keys
                    for expanded in _helpers.expand_collection_scope(self._zot, key, True)
                ]
            for scope_key in scope_keys:
                for item in _utils._paginate(
                    self._zot.collection_items, scope_key, max_items=limit, **extra
                ):
                    if item.get("key") and item["key"] not in seen:
                        seen.add(item["key"])
                        results.append(item)
            return results[:limit]
        params: dict[str, Any] = {"q": query, "qmode": qmode, "limit": limit}
        if item_type:
            params["itemType"] = item_type
        if tag:
            params["tag"] = tag
        self._zot.add_parameters(**params)
        return self._zot.items() or []

    # -- files ------------------------------------------------------------

    def attachment_paths(self, key: str) -> list[dict]:
        raise UnsupportedByBackend(
            "local attachment paths need the SQLite backend (set ZOTERO_LOCAL=true)"
        )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def configured_backend() -> str:
    """``"sqlite"`` or ``"api"``, from configuration alone.

    ``ZOTERO_BACKEND`` is the setting; ``ZOTERO_SEARCH_BACKEND`` is honoured
    as an alias because it selected the SQLite path back when only search
    used it, and existing deployments still set it. Neither one asserts the
    database is actually readable — :func:`get_library_backend` checks that.
    """
    return _utils.get_search_backend()


def get_library_backend(zot=None) -> LibraryBackend:
    """The backend to read through, honouring the active library scope.

    Falls back to the API backend whenever SQLite is configured but
    unusable (no local mode, no readable ``zotero.sqlite``) — the same
    query-site fallback the search path has always done.

    ``zot`` is an already-built pyzotero client to reuse for the API
    backend; omitted, one is created on demand.
    """
    if configured_backend() == "sqlite":
        reader = _sqlite_reader()
        if reader is not None:
            from zotero_mcp.client import get_active_group_id

            return SqliteBackend(reader, get_active_group_id())

    from zotero_mcp import client as _client

    return ApiBackend(zot if zot is not None else _client.get_zotero_client())


def attachment_path_for(key: str) -> Path | None:
    """Best local path for an attachment key, or None.

    Reads ``zotero.sqlite`` directly regardless of the configured backend:
    the file lives on this machine either way, and resolving it locally is
    both faster than downloading and the only option with Zotero closed.
    """
    reader = _sqlite_reader()
    if reader is None:
        return None
    for entry in reader.get_attachment_paths(key):
        resolved = entry.get("resolved_path")
        if resolved and entry.get("exists"):
            return Path(resolved)
    return None
