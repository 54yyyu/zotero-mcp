"""Zotero base-field resolution.

Several Zotero item types store a *base* field under a type-specific key: a
statute's title is ``nameOfAct``, a case's is ``caseName``, an email's is
``subject``; likewise ``date`` -> ``dateEnacted`` for a statute, and so on.
The Web API writes the raw key, so a generic ``title=`` update has to be routed
to the type's actual field, and a field's validity has to be judged against the
type's declared field set rather than its presence on a fetched item.

The mapping lives in Zotero's global schema (``/schema``, the same document the
desktop client uses); pyzotero does not surface the ``baseField`` attribute, so
we carry a trimmed slice of it:

    {"version": N, "itemTypes": {type: {actualField: baseField, ...}, ...}}

``itemTypes[type]`` doubles as the type's valid-field set (its keys) and its
base->actual inverse (invert the mapping).

Freshness is layered so resolution is always correct offline:

* a **vendored** copy (``data/zotero_basefields.json``) ships with the package
  and is the floor — regenerate with ``scripts/gen_basefield_map.py``;
* a **runtime refresh** (:func:`refresh`) does a TTL-gated conditional GET and
  caches the result on disk, picking up new item types between releases.

Zotero has no user- or library-defined item types, so the global schema is
authoritative for every library.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

SCHEMA_URL = "https://api.zotero.org/schema"
DATA_FILE = Path(__file__).parent / "data" / "zotero_basefields.json"
REFRESH_TTL_SECONDS = 7 * 24 * 3600  # weekly; renames are structurally frozen

_table_cache: dict | None = None  # process memo for the active table


# ---------------------------------------------------------------------------
# Table loading
# ---------------------------------------------------------------------------

def _cache_path() -> Path:
    override = os.environ.get("ZOTERO_MCP_SCHEMA_CACHE")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "zotero-mcp" / "schema.json"


def _load_vendored() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def _read_cache() -> dict | None:
    try:
        return json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_cache(table: dict) -> bool:
    """Persist ``table`` to the cache. Returns False if it couldn't be written
    (read-only cache dir, disk full) — the caller decides whether that matters;
    resolution keeps working from the in-memory table regardless."""
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table), encoding="utf-8")
        return True
    except OSError:
        return False


def get_table() -> dict:
    """Return the active table (on-disk cache if present, else vendored)."""
    global _table_cache
    if _table_cache is None:
        _table_cache = _read_cache() or _load_vendored()
    return _table_cache


def _fields(item_type: str) -> dict[str, str]:
    return get_table()["itemTypes"].get(item_type, {})


# ---------------------------------------------------------------------------
# Resolution / validation
# ---------------------------------------------------------------------------

def resolve_field(item_type: str, field: str) -> str:
    """Route a generic/base ``field`` to ``item_type``'s actual field key.

    ``title`` -> ``nameOfAct`` for a statute. A field that is already the
    type's actual key, or is unknown, is returned unchanged (the caller then
    validates it with :func:`valid_fields`).
    """
    for actual, base in _fields(item_type).items():
        if base == field:
            return actual
    return field


def valid_fields(item_type: str) -> set[str]:
    """The set of field keys valid for ``item_type`` (empty if unknown)."""
    return set(_fields(item_type).keys())


def base_field_of(item_type: str, field: str) -> str:
    """The Zotero base field for ``item_type``'s ``field`` — the inverse of
    :func:`resolve_field` (``nameOfAct`` -> ``title`` for a statute; a native
    field is its own base).

    Not used within this module; provided for the deferred read-side migration
    that will render renamed-title items by their real name.
    """
    return _fields(item_type).get(field, field)


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------

def _build_table(schema_doc: dict) -> dict:
    item_types = {
        it["itemType"]: {
            f["field"]: f.get("baseField", f["field"])
            for f in it.get("fields", [])
        }
        for it in schema_doc["itemTypes"]
    }
    return {"version": schema_doc["version"], "itemTypes": item_types}


def _http_get(url: str, headers: dict):  # seam for tests
    import requests

    return requests.get(url, headers=headers, timeout=30)


def refresh(force: bool = False) -> str:
    """Refresh the cached table from ``/schema`` if the TTL has lapsed.

    Returns an outcome — ``"unchanged"``, ``"refreshed"``, or ``"offline"`` —
    so a human-facing caller can report accurately, and updates the in-process
    memo so a running server sees the new table without a reload. Uses a
    conditional GET (``If-None-Match``), so an unchanged schema costs a bodyless
    ``304``. Never raises: on any failure resolution keeps working from the
    existing cache or the vendored floor, and the outcome is ``"offline"``.
    """
    global _table_cache
    cached = _read_cache()
    if not force and cached is not None:
        if (time.time() - cached.get("_checked", 0)) < REFRESH_TTL_SECONDS:
            return "unchanged"

    headers = {}
    if cached and cached.get("_etag"):
        headers["If-None-Match"] = cached["_etag"]

    try:
        resp = _http_get(SCHEMA_URL, headers)
        if resp.status_code == 304 and cached is not None:
            cached["_checked"] = time.time()
            _write_cache(cached)
            _table_cache = cached
            return "unchanged"
        if resp.status_code == 200:
            table = _build_table(resp.json())
            table["_etag"] = resp.headers.get("ETag")
            table["_checked"] = time.time()
            if not _write_cache(table):
                sys.stderr.write(
                    "Warning: refreshed the Zotero schema in memory but could "
                    f"not write the cache at {_cache_path()}\n"
                )
            _table_cache = table
            return "refreshed"
    except Exception:
        # Connection error, non-JSON 200 (captive portal), malformed schema —
        # all keep resolution working from the cache or vendored floor.
        return "offline"
    return "offline"  # unexpected status (4xx / 5xx)
