"""Shared private helpers used across tool modules."""

import contextlib
import hashlib
import json
import os
import re
import socket
import tempfile
import threading
from ipaddress import ip_address
from urllib.parse import urljoin, urlparse

import requests
from pyzotero.zotero_errors import (
    PreConditionFailedError,
    TooManyRequestsError,
    TooManyRetriesError,
)

from zotero_mcp import client as _client
from zotero_mcp import utils as _utils
from zotero_mcp.identifiers import normalize_doi
from zotero_mcp.local_db import get_local_zotero_reader
from zotero_mcp.utils import _paginate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Crossref's own type vocabulary -> Zotero item types.
#
# An unmapped type does not merely get a wrong label: the caller writes
# fields into an item template, and ``document`` has no publicationTitle,
# volume, issue or pages, so everything type-specific is dropped on the
# floor. Seventeen of Crossref's thirty types used to be missing here — a
# journal article registered by its publisher as ``journal-issue`` arrived
# as a ``document`` with the journal name, volume, issue and page range gone
# and nothing said. So the table covers the whole vocabulary, and anything
# genuinely outside it is reported rather than quietly defaulted (see
# ``crossref_type_note``).
CROSSREF_TYPE_MAP = {
    "journal-article": "journalArticle",
    # Container types. Zotero has no record for "a whole issue" or "a whole
    # volume", and publishers do mis-register ordinary articles as these —
    # journalArticle keeps the journal, volume, issue and pages that the
    # metadata actually carries, where document would discard all four.
    "journal-issue": "journalArticle",
    "journal-volume": "journalArticle",
    "journal": "journalArticle",
    "book": "book",
    "monograph": "book",
    "edited-book": "book",
    "reference-book": "book",
    "book-set": "book",
    "book-series": "book",
    "book-chapter": "bookSection",
    "book-part": "bookSection",
    "book-section": "bookSection",
    "book-track": "bookSection",
    "proceedings-article": "conferencePaper",
    # A proceedings volume is a book; the paper inside it is the
    # conferencePaper above.
    "proceedings": "book",
    "proceedings-series": "book",
    "report": "report",
    "report-series": "report",
    "report-component": "report",
    "dissertation": "thesis",
    "posted-content": "preprint",
    "reference-entry": "encyclopediaArticle",
    # Zotero grew native dataset and standard types; both used to land on
    # document and lose their type-specific fields.
    "dataset": "dataset",
    "database": "dataset",
    "standard": "standard",
    # No Zotero equivalent, and document is the honest answer rather than a
    # lossy guess: a component is a figure or supplement, not a work; a
    # grant is funding; peer-review is a review of something else; "other"
    # is Crossref saying it does not know either.
    "component": "document",
    "grant": "document",
    "peer-review": "document",
    "other": "document",
}


def crossref_type_note(cr_type: str) -> str:
    """A one-line warning when *cr_type* is outside the mapped vocabulary.

    Returns "" for a mapped type. Crossref adds types over time, and the
    failure mode of a lookup miss here is silent data loss — the item is
    created as a ``document`` and every type-specific field is dropped — so
    a miss is surfaced to the caller instead of being absorbed.
    """
    if not cr_type or cr_type in CROSSREF_TYPE_MAP:
        return ""
    return (
        f"\nNote: CrossRef type '{cr_type}' is not one this version maps to a "
        "Zotero type, so the item was created as 'document' and any journal, "
        "volume, issue or page values were dropped. Set the right type with "
        "zotero_update_item(item_type=...)."
    )


# ---------------------------------------------------------------------------
# Write-operation helpers
# ---------------------------------------------------------------------------

def apply_library_override(zot, override: dict | None) -> None:
    """Apply an active-library override to *zot* in place.

    pyzotero uses ``library_type`` as a URL path segment and expects the
    plural form (``users`` / ``groups``), but the runtime override stores
    the singular form (``user`` / ``group``) as used by Zotero's switch-
    library tool. Without the normalization below, writes against a group
    library hit ``/group/{id}/items`` and 404.
    """
    if not override:
        return
    zot.library_id = override.get("library_id", zot.library_id)
    raw_type = override.get("library_type")
    if raw_type:
        zot.library_type = raw_type if raw_type.endswith("s") else raw_type + "s"


def _get_write_client(ctx):
    """Return (read_client, write_client) for hybrid-mode operations.

    In web-only mode: both are the web client.
    In local mode with web credentials: read from local, write to web.
    In local-only mode: raises ValueError with clear message.
    """
    read_zot = _client.get_zotero_client()
    if not _utils.is_local_mode():
        return read_zot, read_zot
    web_zot = _client.get_web_zotero_client()
    if web_zot is not None:
        apply_library_override(web_zot, _client.get_active_library())
        return read_zot, web_zot
    raise ValueError(
        "Cannot perform write operations in local-only mode. "
        "Add ZOTERO_API_KEY and ZOTERO_LIBRARY_ID to enable hybrid mode."
    )


def _get_bibliography_client(ctx=None):
    """Return a client able to render CSL bibliographies/citations.

    The local API *does* have a citation engine; what it lacks is Atom. The
    original diagnosis of #371 confused the two, because the only rendering
    request we made was ``content=bib``/``citation``/``bibtex``, and ``content``
    implies ``format=atom`` — which the local API answers with 501 "Local API
    does not support Atom output". That was read as "no citation engine", and
    rendering was routed through the web API, which locked local-only users out
    of a feature their own Zotero could serve.

    Asking the JSON way instead (``include=bib``/``citation`` with ``style``,
    or the top-level ``format=bibtex`` export) works against the local API with
    no credentials at all, so every mode can now use its normal client.
    """
    return _client.get_zotero_client()


def fetch_trashed_collections(zot) -> list[dict]:
    """Return collections in the active library's trash, or [] on failure.

    Zotero's REST API exposes trashed collections at
    ``/{users|groups}/{id}/collections/trash``. pyzotero doesn't have a
    dedicated method for it (only ``trash()``, which returns items), so
    fall back to ``_retrieve_data``. Non-fatal — callers should treat
    failures as "no trash data available" rather than raising.
    """
    try:
        resp = zot._retrieve_data(
            f"/{zot.library_type}/{zot.library_id}/collections/trash"
        )
    except Exception:
        return []
    try:
        data = resp.json()
    except Exception:
        return []
    return data if isinstance(data, list) else []


def is_collection_trashed(zot, collection_key: str) -> bool | None:
    """Return True if a collection is in the trash, False if live, None on error.

    Reads a single collection by key and inspects ``data.deleted``. Used to
    pre-validate ``zotero_set_item_collections`` calls so the tool returns a
    clear error instead of silently filing items into trashed parents.
    """
    try:
        coll = zot.collection(collection_key)
    except Exception:
        return None
    return bool(coll.get("data", {}).get("deleted"))


# Fields the Zotero reader/server sets on items but pyzotero's check_items()
# whitelist (pyzotero/_client.py: check_items) does not include. Any fetched
# item that carries one of these will be rejected client-side with
# "Invalid keys present in item N: <field>" when passed back to update_item().
# The canonical fetch→mutate→update flow then breaks on attachments that have
# been opened in the Zotero PDF reader (which writes lastRead).
_UNWRITABLE_ITEM_FIELDS = frozenset({"lastRead"})


def _strip_unwritable_fields(item: dict) -> dict:
    """Remove fields that pyzotero's check_items() rejects from a fetched item.

    Mutates ``item["data"]`` in place and returns the same dict so the caller
    can chain. Safe to call on any item type — fields not present are ignored.
    """
    data = item.get("data")
    if isinstance(data, dict):
        for field in _UNWRITABLE_ITEM_FIELDS:
            data.pop(field, None)
    return item


def _handle_write_response(response, ctx=None):
    """Check if a pyzotero write operation succeeded."""
    if hasattr(response, "status_code"):
        ok = response.status_code in (200, 204)
        if not ok and ctx is not None:
            ctx.error(f"Write failed ({response.status_code}): {response.text[:500]}")
        return ok
    if isinstance(response, dict):
        return bool(response.get("success"))
    return bool(response)


_MAX_VERSION_CONFLICT_RETRIES = 3


def _update_item_with_version_retry(write_zot, item_key, mutate_fn, ctx=None):
    """Fetch *item_key*, apply *mutate_fn* to it, and write it back —
    retrying on HTTP 412 (stale version) by re-fetching and re-applying.

    Callers already re-fetch the item once before writing, to pick up the
    web API's version number before mutating. That closes the common case
    but not the race: another writer (a concurrent MCP call, Zotero
    Desktop, or sync) can update the item again between that re-fetch and
    this write, and pyzotero raises PreConditionFailedError for exactly
    that window. A bounded retry — re-fetch, re-apply, re-send — closes it;
    any other exception propagates immediately, unretried.

    *mutate_fn* receives the freshly-fetched item dict and mutates it (or
    its ``data``) in place. Returns the raw pyzotero response from
    ``update_item``.
    """
    last_error = None
    for attempt in range(_MAX_VERSION_CONFLICT_RETRIES):
        item = write_zot.item(item_key)
        mutate_fn(item)
        try:
            return write_zot.update_item(item)
        except PreConditionFailedError as e:
            last_error = e
            if ctx is not None:
                ctx.info(
                    f"Version conflict updating item {item_key} (attempt "
                    f"{attempt + 1}/{_MAX_VERSION_CONFLICT_RETRIES}); "
                    "re-fetching and retrying."
                )
    raise last_error


def ensure_collection_membership(write_zot, item_key: str, coll_keys: list[str], ctx=None) -> list[str]:
    """Force *item_key* into each collection in *coll_keys*; return keys we couldn't file.

    Setting ``item["collections"]`` on ``create_items`` is supposed to atomically
    file the new item, but reports show it intermittently no-ops — the item
    lands in My Library root despite the request (#235). This is the
    deterministic backstop: read the item back, diff against the requested
    set, and ``addto_collection`` for any that didn't take.
    """
    if not coll_keys:
        return []
    try:
        item = write_zot.item(item_key)
    except Exception as e:
        if ctx is not None:
            ctx.warning(f"Could not re-fetch item {item_key} to verify collection membership: {e}")
        return list(coll_keys)
    actual = set(item.get("data", {}).get("collections") or [])
    failed: list[str] = []
    for coll_key in coll_keys:
        if coll_key in actual:
            continue
        try:
            write_zot.addto_collection(coll_key, item)
            actual.add(coll_key)
        except Exception as e:
            failed.append(coll_key)
            if ctx is not None:
                ctx.warning(f"Could not file {item_key} in collection {coll_key}: {e}")
    return failed


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

def _normalize_limit(limit: int | str | None, default: int = 10, max_val: int = 100) -> int:
    """Coerce *limit* to a bounded int.

    A limit of zero or below is meaningless rather than minimal, so it falls
    back to *default*. Clamping it to 1 (the previous behaviour) answered
    `limit=0` with a single item and no indication why, which reads as a
    one-item collection (#453).
    """
    if limit is None:
        return default
    if isinstance(limit, str):
        limit = limit.strip()
        if not limit:
            return default
        limit = int(limit)
    if limit <= 0:
        return default
    return min(limit, max_val)


def _normalize_offset(offset: int | str | None, default: int = 0) -> int:
    """Coerce *offset* to a non-negative int.

    Companion to :func:`_normalize_limit` for tools that page through a
    result set. Unlike the limit there is no ceiling: an offset past the end
    is not an error, it just yields an empty page, and the caller is expected
    to report the total so it can tell the difference.
    """
    if offset is None:
        return default
    if isinstance(offset, str):
        offset = offset.strip()
        if not offset:
            return default
        offset = int(offset)
    return max(0, int(offset))


def global_search_error() -> str | None:
    """None when a global search can run here, else why it cannot (#163).

    Global search is served exclusively by direct SQL over ``zotero.sqlite``:
    one query covering every library at once. The Zotero API has no
    equivalent — the best it could do is replay a single-library search
    against each library in turn, which is a different (and far slower)
    operation than the one the caller asked for. Refusing is therefore the
    honest answer, and the message says what to change.
    """
    if _utils.get_search_backend() != "sqlite":
        return (
            "Error: global search requires the SQLite backend. The Zotero API "
            "cannot search across libraries in one query, so this is refused "
            "rather than emulated by searching each library in turn. Set "
            "ZOTERO_SEARCH_BACKEND=sqlite (with ZOTERO_LOCAL=true) and restart "
            "the server, or search one library at a time with "
            "zotero_switch_library."
        )
    reader = get_local_zotero_reader()
    if reader is None:
        return (
            "Error: global search needs to read zotero.sqlite directly, but the "
            "local database is not available. Set ZOTERO_LOCAL=true (and "
            "ZOTERO_DB_PATH if your Zotero data directory is in a custom "
            "location), or search one library at a time with "
            "zotero_switch_library."
        )
    reader.close()
    return None


def _parse_library_id_param(value: int | str | None) -> int | None:
    """Parse a `library_id` filter param into a group_id (0=personal library).

    Accepts an int, a numeric string (the Zotero groupID), "0"/"user" for
    the personal library, or None (no filter — search all indexed
    libraries). This is the single-parameter convention `zotero_semantic_search`
    exposes; `zotero_switch_library` instead takes separate library_id +
    library_type args since it must also validate library_type ("feed" has
    no meaning here — feed libraries are never semantically indexed).
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.lower() == "user":
            return 0
        try:
            return int(stripped)
        except ValueError:
            raise ValueError(
                f"Invalid library_id: {value!r}. Use an integer groupID, 0, or 'user'."
            ) from None
    return int(value)


def _normalize_float_list_input(value, length, field_name="value"):
    """Normalize a fixed-length numeric list that MCP clients may stringify.

    Mirrors ``_normalize_str_list_input``: some MCP transports stringify
    untyped/loosely-typed arguments before dispatch, so a client-side
    ``[x, y, w, h]`` list can arrive here as the JSON text ``"[x, y, w, h]"``
    instead. Returns ``None`` (never raises) when ``value`` is not a JSON
    array of exactly ``length`` numbers, so callers keep their own
    user-facing error message for the invalid-shape case.
    """
    if isinstance(value, (list, tuple)):
        candidate = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, list):
            return None
        candidate = parsed
    else:
        return None

    if len(candidate) != length:
        return None
    try:
        return [float(v) for v in candidate]
    except (TypeError, ValueError):
        return None


def _normalize_str_list_input(value, field_name="value"):
    """Normalize list-like user input into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
            if isinstance(parsed, str):
                s = parsed.strip()
                return [s] if s else []
            if isinstance(parsed, dict):
                raise ValueError(
                    f"{field_name} must be a list of strings or a string, "
                    f"got JSON {type(parsed).__name__}"
                )
            # A bare JSON scalar (int/float/bool/null) — e.g. an all-digit
            # ISBN parses as a JSON number. That's plain text, not structured
            # input; fall through to the raw-string handling below instead
            # of rejecting it.
        except json.JSONDecodeError:
            pass
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) > 1:
            return parts
        return [raw]
    raise ValueError(f"{field_name} must be a list of strings or a string")


def _normalize_tag_filter(value):
    """Normalize a tag-filter argument into a list[str] for pyzotero.

    Accepts every shape we've seen clients produce:
    - None / empty                 → []
    - ["a", "b"]                   → ["a", "b"]   (canonical)
    - [{"tag": "a"}, {"tag": "b"}] → ["a", "b"]   (common LLM mis-shape)
    - "a"                          → ["a"]
    - '["a", "b"]'                 → ["a", "b"]   (JSON list of strings)
    - '[{"tag": "a"}]'             → ["a"]        (JSON list of dicts, #237)

    MCP runtimes sometimes stringify array arguments before they reach the
    pydantic validator, and agents sometimes pass the dict-shape that Zotero
    uses INSIDE an item (``{"tag": "X"}``) rather than the bare-string form
    pyzotero's ``tag=`` parameter expects. Either path ended up rejected
    upstream of the search logic. This normalizer collapses them all.
    """
    def _extract(v):
        if isinstance(v, dict):
            for key in ("tag", "name", "value"):
                if key in v and str(v[key]).strip():
                    return str(v[key]).strip()
            return ""
        return str(v).strip()

    if value is None:
        return []
    if isinstance(value, list):
        return [s for s in (_extract(v) for v in value) if s]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]
        if isinstance(parsed, list):
            return [s for s in (_extract(v) for v in parsed) if s]
        if isinstance(parsed, dict):
            s = _extract(parsed)
            return [s] if s else []
        if isinstance(parsed, str):
            s = parsed.strip()
            return [s] if s else []
        return []
    return []


def _normalize_item_tags(value):
    """Normalize tags read off an item/annotation into Zotero's dict shape.

    Zotero stores tags as ``[{"tag": "name", "type": 1}, ...]`` and the
    rendering layers index them with ``t["tag"]``. Annotation sources other
    than the web API hand tags back in looser shapes — Better BibTeX's
    JSON-RPC returns bare strings, pdfannots2json omits the field entirely —
    so normalize to the dict shape (preserving ``type`` when present) and
    drop empties rather than letting a renderer KeyError (#377).
    """
    if not value:
        return []
    if isinstance(value, (str, dict)):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []

    normalized: list[dict] = []
    for entry in value:
        if isinstance(entry, dict):
            name = next(
                (
                    str(entry[key]).strip()
                    for key in ("tag", "name", "value")
                    if entry.get(key) is not None and str(entry[key]).strip()
                ),
                "",
            )
            if not name:
                continue
            tag = {"tag": name}
            if entry.get("type") is not None:
                tag["type"] = entry["type"]
            normalized.append(tag)
            continue
        name = str(entry).strip()
        if name:
            normalized.append({"tag": name})
    return normalized


def _resolve_collection_names(zot, names, ctx=None):
    """Resolve collection names to keys (case-insensitive)."""
    if not names:
        return []
    all_collections = _paginate(zot.collections)
    results = []
    for name in names:
        name_lower = name.lower()
        matches = [
            c["key"] for c in all_collections
            if c.get("data", {}).get("name", "").lower() == name_lower
        ]
        if not matches:
            raise ValueError(f"No collection found matching name '{name}'")
        if len(matches) > 1 and ctx is not None:
            ctx.warning(
                f"Multiple collections match '{name}': {matches}. "
                "Using all. Pass collection keys directly to disambiguate."
            )
        results.extend(matches)
    return results


_COLLECTION_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")


def build_collection_paths(collections) -> dict[str, list[str]]:
    """Map collection key → full path segments ``[root, ..., name]``.

    Built from ``data.parentCollection`` links. A parent that isn't in the
    fetched set (e.g. trashed) or a parent cycle degrades to a shorter path
    rather than failing.
    """
    by_key = {c["key"]: c for c in collections if c.get("key")}
    paths: dict[str, list[str]] = {}

    def _segments(key: str, seen: set[str]) -> list[str]:
        if key in paths:
            return paths[key]
        coll = by_key[key]
        name = coll.get("data", {}).get("name") or key
        parent = coll.get("data", {}).get("parentCollection")
        if parent in by_key and parent not in seen:
            seen.add(key)
            segs = _segments(parent, seen) + [name]
        else:
            segs = [name]
        paths[key] = segs
        return segs

    for key in by_key:
        _segments(key, {key})
    return paths


def collection_descendants(collections, collection_key: str) -> list[str]:
    """``collection_key`` plus every collection nested beneath it, breadth-first.

    Pure: takes an already-fetched collection list so it can be tested without
    a client. Built from the same ``data.parentCollection`` links
    :func:`build_collection_paths` uses.

    An unknown key returns ``[collection_key]`` unchanged rather than an empty
    list — the caller's existing "collection not found" handling should decide
    what that means, not this function. A parent cycle terminates instead of
    looping: Zotero's own schema should not produce one, but a partially
    synced or hand-edited database can, and this walk is cheap to make safe.
    """
    children: dict[str, list[str]] = {}
    for coll in collections:
        key = coll.get("key")
        if not key:
            continue
        parent = coll.get("data", {}).get("parentCollection")
        if parent:  # False/None/"" all mean top level
            children.setdefault(parent, []).append(key)

    ordered = [collection_key]
    seen = {collection_key}
    queue = [collection_key]
    while queue:
        current = queue.pop(0)
        for child in children.get(current, []):
            if child in seen:
                continue
            seen.add(child)
            ordered.append(child)
            queue.append(child)
    return ordered


def expand_collection_scope(zot, collection_key: str, include_subcollections: bool) -> list[str]:
    """Collection keys a scoped query should cover.

    Returns ``[collection_key]`` unless subcollections were asked for, so the
    default path costs nothing. Fetching the collection list is one extra API
    round-trip, paid only when the caller opts in.
    """
    if not include_subcollections:
        return [collection_key]
    collections = _utils._paginate(zot.collections)
    return collection_descendants(collections, collection_key)


def resolve_collection_specs(
    zot,
    specs,
    *,
    create_missing: bool = False,
    write_zot=None,
    ctx=None,
) -> list[str]:
    """Resolve collection *specs* — keys, names, or '/'-paths — to live keys.

    Resolution order per spec:

    1. **Key**: 8-char uppercase-alphanumeric AND currently a live collection
       key → used as-is. Existence is checked, not just shape, so trashed or
       bogus keys fail loudly here instead of producing an invisibly-filed or
       unfiled item after creation (#233/#235).
    2. **Name/path**: the spec is split on '/' and matched case-insensitively
       against the *trailing* path segments of every collection, so a bare
       name matches anywhere in the tree and 'parent/name' disambiguates
       same-named leaves.

    An ambiguous spec raises ValueError listing every candidate. An unknown
    spec raises ValueError with near-miss suggestions — unless
    ``create_missing`` is True, in which case the collection (including any
    missing intermediate path segments) is created via ``write_zot``.

    Returns resolved keys in input order, deduplicated.
    """
    cleaned = [str(s).strip() for s in (specs or []) if str(s).strip()]
    if not cleaned:
        return []

    paths = build_collection_paths(_paginate(zot.collections))

    resolved: list[str] = []
    for spec in cleaned:
        if _COLLECTION_KEY_RE.match(spec) and spec in paths:
            resolved.append(spec)
            continue

        wanted = [seg.strip().lower() for seg in spec.split("/") if seg.strip()]
        if not wanted:
            raise ValueError(f"Collection spec '{spec}' is empty.")

        matches = [
            key for key, segs in paths.items()
            if len(segs) >= len(wanted)
            and [s.lower() for s in segs[-len(wanted):]] == wanted
        ]

        if len(matches) > 1:
            candidates = "; ".join(
                f"'{'/'.join(paths[k])}' ({k})"
                for k in sorted(matches, key=lambda k: paths[k])
            )
            raise ValueError(
                f"Collection spec '{spec}' is ambiguous — it matches: "
                f"{candidates}. Disambiguate with a longer path or the "
                "8-character collection key."
            )
        if matches:
            resolved.append(matches[0])
            continue

        if create_missing:
            if write_zot is None:
                raise ValueError(
                    f"Collection '{spec}' not found and no writable client "
                    "is available to create it."
                )
            resolved.append(
                _create_collection_path(write_zot, paths, spec, ctx=ctx)
            )
            continue

        raise ValueError(_collection_not_found_message(zot, spec, paths))

    seen: set[str] = set()
    return [k for k in resolved if not (k in seen or seen.add(k))]


def _create_collection_path(write_zot, paths, spec, ctx=None) -> str:
    """Create the collections needed to satisfy *spec*; return the leaf key.

    The longest prefix of the path that already resolves (unique trailing-
    segment match) anchors the chain; remaining segments are created beneath
    it, or at the library root when nothing resolves. Mutates *paths* with
    the created entries so later specs in the same call see them.
    """
    names = [seg.strip() for seg in spec.split("/") if seg.strip()]

    parent_key = None
    start = 0
    for i in range(len(names) - 1, 0, -1):
        prefix = [s.lower() for s in names[:i]]
        matches = [
            key for key, segs in paths.items()
            if len(segs) >= len(prefix)
            and [s.lower() for s in segs[-len(prefix):]] == prefix
        ]
        if len(matches) > 1:
            candidates = "; ".join(f"'{'/'.join(paths[k])}' ({k})" for k in matches)
            raise ValueError(
                f"Cannot create '{spec}': parent path "
                f"'{'/'.join(names[:i])}' is ambiguous — it matches: "
                f"{candidates}."
            )
        if matches:
            parent_key = matches[0]
            start = i
            break

    for name in names[start:]:
        payload = {"name": name, "parentCollection": parent_key or False}
        result = write_zot.create_collections([payload])
        if not (isinstance(result, dict) and result.get("success")):
            raise ValueError(f"Failed to create collection '{name}': {result}")
        new_key = next(iter(result["success"].values()))
        paths[new_key] = (paths[parent_key] if parent_key else []) + [name]
        if ctx is not None:
            ctx.info(f"Created collection '{'/'.join(paths[new_key])}' ({new_key})")
        parent_key = new_key
    return parent_key


def find_existing_items(zot, *, doi=None, arxiv_id=None, isbn=None, url=None,
                        ctx=None) -> list[dict]:
    """Find non-attachment items already in the library by a normalized id.

    Exactly one of doi / arxiv_id / isbn / url should be given (already
    normalized via the corresponding ``_normalize_*`` helper, except url).
    A server-side quick search (``q=<id>, qmode='everything',
    itemType='-attachment'``) narrows candidates cheaply; a client-side
    normalized comparison confirms real matches. Searching with the BARE
    identifier means the substring quick-search also catches values stored
    with prefixes ('https://doi.org/10...', 'arXiv:...'). The items endpoint
    excludes the Trash, so a trashed copy never blocks a re-add.

    Returns full item dicts (with ``key``/``version``/``data``) so callers
    can update them without re-fetching. Returns [] on search failure —
    callers treat that as "nothing found" and proceed to create — except when
    Zotero rate-limited the search, which propagates rather than masquerading
    as "nothing found" and duplicating an item that exists.
    """
    if doi:
        query = doi
        def _matches(data):
            return _normalize_doi(data.get("DOI") or "") == doi
    elif arxiv_id:
        # Compare on the version-independent identity, and search on it too:
        # quick-search is a substring match, so the bare id finds a stored
        # 'arXiv:2401.00001v2' while the versioned form would miss a stored
        # bare one.
        ident = _arxiv_identity(arxiv_id) or arxiv_id
        query = ident
        def _matches(data):
            # Zotero stores an arXiv identity in up to four places depending
            # on how the item arrived (connector, DOI add, arXiv add, manual).
            # Checking only url+extra misses connector- and DOI-sourced items,
            # which is how a re-add duplicates a paper already in the library.
            for field in ("url", "archiveID", "DOI"):
                if _arxiv_identity(data.get(field) or "") == ident:
                    return True
            return f"arxiv:{ident}".lower() in (data.get("extra") or "").lower()
    elif isbn:
        query = isbn
        def _matches(data):
            # Zotero's ISBN field may hold several space-separated values,
            # in 10- or 13-digit form; compare each normalized to ISBN-13.
            raw = data.get("ISBN") or ""
            for token in re.split(r"[,;\s]+", raw):
                if token and _normalize_isbn(token) == isbn:
                    return True
            return False
    elif url:
        query = url
        def _matches(data):
            return (data.get("url") or "").rstrip("/") == url.rstrip("/")
    else:
        return []

    try:
        candidates = zot.items(
            q=query, qmode="everything", itemType="-attachment", limit=50
        )
    except (TooManyRetriesError, TooManyRequestsError):
        # A rate-limited search is deliberately not swallowed. Every other
        # failure here degrades to "no match" and the caller creates the item,
        # which is the right trade for a genuinely failed search — but a
        # throttled search hasn't answered the question, and reading it as "not
        # present" silently creates duplicates of items that are. pyzotero
        # >=1.13.5 has already retried and waited out the server's backoff by
        # the time it raises, so there is nothing left to do but propagate.
        raise
    except Exception as e:
        if ctx is not None:
            ctx.warning(f"Existing-item search failed (treating as no match): {e}")
        return []

    matches = []
    for item in candidates or []:
        # Skip anything that isn't a well-formed item dict. The try above only
        # wraps the call, not this iteration, so a malformed entry would raise
        # here and abort the whole import instead of costing one dedup match.
        # The known cause of that is fixed in pyzotero >=1.13.5, which this
        # package now requires, but this stays as a backstop: nothing about the
        # contract of a search result guarantees every entry is a dict.
        if not isinstance(item, dict):
            continue
        data = item.get("data")
        if not isinstance(data, dict):
            continue
        if data.get("itemType") in ("attachment", "note", "annotation"):
            continue
        if _matches(data):
            matches.append(item)
    return matches


def _collection_not_found_message(zot, spec, paths) -> str:
    """Build the error message for an unresolvable collection spec."""
    if _COLLECTION_KEY_RE.match(spec):
        try:
            trashed = {c.get("key") for c in fetch_trashed_collections(zot)}
        except Exception:
            trashed = set()
        if spec in trashed:
            return (
                f"Collection '{spec}' is in the Zotero Trash. Restore it in "
                "Zotero (or use another collection) before filing items into it."
            )
    msg = (
        f"Collection '{spec}' not found in the active library "
        "(tried key, name, and path matching)."
    )
    words = [w for w in spec.lower().replace("/", " ").split() if w]
    suggestions = [
        key for key, segs in paths.items()
        if all(w in "/".join(segs).lower() for w in words)
    ]
    if suggestions:
        shown = ", ".join(
            f"'{'/'.join(paths[k])}' ({k})"
            for k in sorted(suggestions, key=lambda k: paths[k])[:5]
        )
        msg += f" Close matches: {shown}."
    else:
        msg += (
            " Use zotero_search_collections (or `zotero-cli collections "
            "search`) to list available collections."
        )
    return msg


#: Compatibility alias. The implementation moved to the public,
#: stdlib-only :mod:`zotero_mcp.identifiers` so consumers can import it
#: without pulling in the tool layer. Existing callers keep working.
_normalize_doi = normalize_doi


# ---------------------------------------------------------------------------
# Per-identifier serialization of adds (#486)
#
# The Zotero API lock exists to protect the single-threaded local API on port
# 23119. It never promised to make check-then-create atomic; that was a side
# effect of how wide it used to be. Narrowing it — correctly, to keep CrossRef
# and page fetches out of a held lock — removed that side effect wherever a
# fetch now sits between the dedup check and the create, so two callers can
# both pass the check and both create. The version-checked retry cannot close
# it: two ``create_items()`` POSTs produce two new keys and no version to
# conflict on.
#
# These locks are keyed on the *normalized* identifier, so ISBN-10 and ISBN-13
# of one book take the same lock, and held across a re-check and the create
# only — never across third-party network work, which is the whole point of
# the narrowing.
#
# IN-PROCESS ONLY, and worth saying plainly: this serializes parallel tool
# calls within one server, which is the reported case. Two servers against one
# library still race, and nothing here changes that.
# ---------------------------------------------------------------------------

#: key -> [lock, waiter_count]. Entries are dropped once the last holder
#: leaves, so a long indexing run does not accumulate one lock per identifier
#: it has ever seen.
_identifier_locks = {}
_identifier_locks_guard = threading.Lock()


def identifier_lock_key(kind, raw):
    """Canonical lock key for an identifier, or ``None`` if it has none.

    ``None`` means "take no lock": an identifier that cannot be normalized
    cannot dedup-match anything either, so serializing on it buys nothing and
    would make a batch of junk tokens queue behind each other. Kinds are part
    of the key so a DOI and a URL that happen to stringify alike stay apart.
    """
    if raw is None:
        return None
    if kind == "doi":
        normalized = normalize_doi(raw)
    elif kind == "isbn":
        normalized = _normalize_isbn(raw)
    elif kind == "arxiv":
        normalized = _normalize_arxiv_id(raw)
    elif kind == "url":
        # URLs have no normalizer, so this is exact-after-strip, matching what
        # #443 settled on for collapsing repeats. Deliberately no case or
        # trailing-slash folding: either can be a genuinely different page.
        normalized = str(raw).strip() or None
    else:
        raise ValueError(f"unknown identifier kind {kind!r}")
    return None if normalized is None else f"{kind}:{normalized}"


@contextlib.contextmanager
def identifier_lock(kind, raw):
    """Serialize check-then-create for one identifier within this process.

    Yields the lock key, or ``None`` when the identifier is unnormalizable and
    no lock was taken. Re-entrant, because the add paths call into helpers
    that may take the same identifier again — a plain ``Lock`` there would
    wedge the process rather than fail loudly.
    """
    key = identifier_lock_key(kind, raw)
    if key is None:
        yield None
        return

    with _identifier_locks_guard:
        entry = _identifier_locks.get(key)
        if entry is None:
            entry = _identifier_locks[key] = [threading.RLock(), 0]
        entry[1] += 1

    entry[0].acquire()
    try:
        yield key
    finally:
        entry[0].release()
        with _identifier_locks_guard:
            entry[1] -= 1
            if entry[1] == 0:
                # Only drop it if nobody re-created it in the meantime.
                if _identifier_locks.get(key) is entry:
                    del _identifier_locks[key]


def _normalize_isbn(raw):
    """Normalize an ISBN string and validate the checksum.

    Accepts ISBN-10, ISBN-13, and prefixed/URL forms (isbn:, https://isbndb.com/...).
    Strips hyphens, spaces, and any prefix. Returns the canonical digits-only
    form (13-digit preferred — ISBN-10 inputs are converted to ISBN-13).
    Returns None on invalid input or failing checksum.
    """
    if not raw:
        return None
    s = str(raw).strip()
    if s.lower().startswith("isbn:"):
        s = s[5:].strip()
    if s.lower().startswith("isbn-") or s.lower().startswith("isbn "):
        s = s[5:].strip()
    if s.lower().startswith("http://") or s.lower().startswith("https://"):
        m = re.search(r"/(97[89][\- ]?\d[\- ]?\d{3}[\- ]?\d{5}[\- ]?\d|\d{9}[\dX])",
                      s, flags=re.IGNORECASE)
        if not m:
            return None
        s = m.group(1)
    digits = re.sub(r"[\s\-]", "", s)
    if re.match(r"^\d{9}[\dXx]$", digits):
        if not _isbn10_checksum_valid(digits):
            return None
        return _isbn10_to_isbn13(digits)
    if re.match(r"^97[89]\d{10}$", digits):
        if not _isbn13_checksum_valid(digits):
            return None
        return digits
    return None


def _isbn10_checksum_valid(s):
    total = 0
    for i, ch in enumerate(s):
        v = 10 if ch in ("X", "x") else int(ch)
        total += v * (10 - i)
    return total % 11 == 0


def _isbn13_checksum_valid(s):
    total = 0
    for i, ch in enumerate(s):
        v = int(ch)
        total += v if i % 2 == 0 else v * 3
    return total % 10 == 0


def _isbn10_to_isbn13(isbn10):
    core = "978" + isbn10[:9]
    total = 0
    for i, ch in enumerate(core):
        total += int(ch) * (1 if i % 2 == 0 else 3)
    check = (10 - total % 10) % 10
    return core + str(check)


def _normalize_arxiv_id(raw):
    """Normalize an arXiv ID from various input formats."""
    if not raw:
        return None
    s = raw.strip()
    if s.lower().startswith("arxiv:"):
        s = s[6:].strip()
    if s.lower().startswith("http://") or s.lower().startswith("https://"):
        m = re.search(
            r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z\-]+/\d{7}(?:v\d+)?)(?:\.pdf)?",
            s, flags=re.IGNORECASE,
        )
        if not m:
            return None
        s = m.group(1)
    if re.match(r"^[0-9]{4}\.[0-9]{4,5}(?:v\d+)?$", s):
        return s
    if re.match(r"^[a-z\-]+/\d{7}(?:v\d+)?$", s, flags=re.IGNORECASE):
        return s
    return None


# arXiv's DataCite DOIs are minted as 10.48550/arXiv.<id>, which is what
# Zotero puts in the DOI field for a preprint imported from arXiv.
_ARXIV_DOI_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/)?10\.48550/arxiv\.(.+)$",
                           re.IGNORECASE)
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)


def _arxiv_identity(raw):
    """The version-independent arXiv identity of an ID, URL, DOI or archiveID.

    ``_normalize_arxiv_id`` deliberately keeps the ``v2`` suffix: callers use
    its result to fetch a specific version from arXiv. Deduplication wants the
    opposite — 2401.00001v1 and 2401.00001v2 are the same paper and must not
    become two library items — so identity comparison goes through here
    instead. This also accepts arXiv's DataCite DOI form, so an item added by
    DOI is recognized by a later add of the same paper's arXiv ID.

    Returns the bare, unversioned ID, or None if ``raw`` isn't an arXiv
    identifier in any of those forms.
    """
    if not raw:
        return None
    s = str(raw).strip()
    m = _ARXIV_DOI_RE.match(s)
    if m:
        s = m.group(1)
    ident = _normalize_arxiv_id(s)
    if not ident:
        return None
    return _ARXIV_VERSION_RE.sub("", ident)


# ---------------------------------------------------------------------------
# PDF / open-access helpers
# ---------------------------------------------------------------------------

_MAX_PDF_REDIRECTS = 5
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def _url_resolves_to_public_host(url: str) -> bool:
    """Return ``True`` only if ``url`` is http(s) and its host resolves
    entirely to globally-routable IP addresses.

    SSRF guard for the open-access PDF download path: the candidate URL comes
    from third-party metadata APIs (Unpaywall / Semantic Scholar) and is
    therefore attacker-influenceable (a hostile paper record, or prompt
    injection steering ``zotero_add_item``). We reject non-http(s) schemes
    and any host that resolves to a private, loopback, link-local, reserved,
    or otherwise non-global address — including the 169.254.169.254
    cloud-metadata endpoint, which matters for HTTP/SSE-transport deployments.

    Note: a determined DNS-rebinding attacker could still flip the record
    between this check and the socket connect. Re-validating every redirect
    hop (see ``_guarded_pdf_get``) and rejecting on the first non-global
    result narrows that window to a non-practical vector for this tool's
    threat model; full pinning would require a custom connection adapter.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or None)
    except (socket.gaierror, UnicodeError, ValueError):
        return False
    if not infos:
        return False
    for info in infos:
        sockaddr = info[4]
        try:
            ip = ip_address(sockaddr[0])
        except ValueError:
            return False
        if not ip.is_global or ip.is_reserved or ip.is_multicast:
            return False
    return True


def _guarded_pdf_get(pdf_url, ctx):
    """GET ``pdf_url`` with SSRF protection.

    Validates that the host resolves to public IPs, follows redirects
    manually (re-validating each hop), and returns the final ``requests``
    response, or ``None`` if any URL in the chain is rejected or there are
    too many redirects.
    """
    current = pdf_url
    for _ in range(_MAX_PDF_REDIRECTS + 1):
        if not _url_resolves_to_public_host(current):
            ctx.info(f"PDF URL rejected by SSRF guard: {current}")
            return None
        resp = requests.get(current, timeout=30, stream=True, allow_redirects=False)
        if resp.status_code in _REDIRECT_STATUSES:
            location = resp.headers.get("Location")
            try:
                resp.close()
            except Exception:
                pass
            if not location:
                return None
            current = urljoin(current, location)
            continue
        return resp
    ctx.info("Too many redirects while fetching PDF")
    return None


def _download_and_attach_pdf(write_zot, item_key, pdf_url, doi, ctx):
    """Download a PDF from a URL and attach it to a Zotero item.

    The URL is fetched through ``_guarded_pdf_get`` (SSRF guard + manual
    redirect re-validation), since it originates from third-party metadata
    APIs rather than the user.

    Returns the WebDAV-status suffix string on success (``""`` when WebDAV
    is not configured, otherwise something like ``" (uploaded to WebDAV
    as <key>.zip)"`` or a warning if the PUT failed). Returns ``None``
    on failure so callers can branch with ``if suffix is not None``.
    """
    try:
        pdf_resp = _guarded_pdf_get(pdf_url, ctx)
        if pdf_resp is None:
            return None
        pdf_resp.raise_for_status()

        content_type = pdf_resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and "octet-stream" not in content_type:
            ctx.info(f"URL did not return a PDF (Content-Type: {content_type})")
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f"{doi.replace('/', '_')}.pdf"
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "wb") as f:
                for chunk in pdf_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            if os.path.getsize(filepath) < 1000:
                ctx.info("Downloaded file too small, likely not a real PDF")
                return None

            suffix = _webdav_first_attach(
                write_zot,
                filename,
                filepath,
                item_key,
                ctx,
                content_type="application/pdf",
            )
            if suffix is not None:
                return suffix
            ok, suffix, _key = _attach_and_verify(
                write_zot,
                filename,
                filepath,
                item_key,
                ctx,
                content_type="application/pdf",
            )
            if not ok:
                ctx.info(f"PDF attach failed: {suffix}")
                return None
            return suffix
    except Exception as e:
        ctx.info(f"PDF download/attach failed: {e}")
        return None


def _maybe_upload_to_webdav(attach_result, file_path, ctx, write_zot=None):
    """Suffix to append to a user-facing 'file attached' message.

    PR #279 added WebDAV-aware upload to ``zotero_add_item``. The same
    treatment is needed everywhere else ``attachment_both`` is called: the
    Web API's file upload lands bytes in Zotero Storage, which a desktop
    client with File Syncing set to WebDAV never consults.

    Returns ``""`` when WebDAV is not configured, when the attachment key
    cannot be extracted, or after a successful PUT with logging via ``ctx``
    (callers that don't surface the suffix can ignore the return value).
    On a successful PUT returns ``" (uploaded to WebDAV as <key>.zip)"``;
    on PUT failure returns ``" (WARNING: WebDAV upload failed — <err>; ...)"``
    so callers can keep the user-visible signal without re-implementing the
    branch.
    """
    from zotero_mcp import webdav as _webdav

    if not _webdav.is_webdav_configured():
        return ""

    attachment_key = _extract_attachment_key(attach_result)
    if not attachment_key:
        return ""

    try:
        _webdav.upload_attachment_to_webdav(
            attachment_key=attachment_key,
            file_path=file_path,
        )
        ctx.info(f"WebDAV PUT: {attachment_key}.zip uploaded")
        return f" (uploaded to WebDAV as {attachment_key}.zip)"
    except Exception as e:
        ctx.info(f"WebDAV PUT failed for {attachment_key}: {e}")
        # A failed PUT leaves the attachment item with no file bytes — an
        # orphan that confuses the Zotero UI and breaks sync. Clean it up,
        # and only fall back to the "no file bytes" warning if the delete
        # itself fails.
        try:
            attachment_version = next(
                (
                    entry.get("version")
                    for status in ("success", "unchanged")
                    for entry in (attach_result.get(status, []) or [])
                    if isinstance(entry, dict) and entry.get("key") == attachment_key
                ),
                None,
            )
            if write_zot is not None:
                if attachment_version is None:
                    attachment_version = write_zot.item(attachment_key)["version"]
                write_zot.delete_item({"key": attachment_key, "version": attachment_version})
                ctx.info(f"Cleaned up orphan attachment {attachment_key}")
                return f" (WARNING: WebDAV upload failed — {e}; attachment {attachment_key} was deleted)"
            raise RuntimeError("no writable client available for cleanup")
        except Exception as del_err:
            ctx.info(f"Cleanup of orphan attachment {attachment_key} failed: {del_err}")
            return (
                f" (WARNING: WebDAV upload failed — {e}; "
                f"attachment {attachment_key} exists but has no file bytes on WebDAV "
                f"and could not be deleted: {del_err})"
            )


def _guess_content_type(filename):
    """Guess a Zotero ``contentType`` from a filename's extension.

    Covers the file types ``add_from_file`` accepts (PDF, EPUB, DJVU, plus a
    few common extras). Returns ``None`` when there is no useful guess so the
    caller can leave the field unset and let Zotero fall back.
    """
    if not filename:
        return None
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    return {
        "pdf": "application/pdf",
        "epub": "application/epub+zip",
        "djvu": "image/vnd.djvu",
        "html": "text/html",
        "txt": "text/plain",
        "doc": "application/msword",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "rtf": "application/rtf",
        "odt": "application/vnd.oasis.opendocument.text",
    }.get(ext)


def _webdav_first_attach(write_zot, filename, file_path, parent_key, ctx, content_type=None):
    """Create attachment shell + WebDAV upload when WebDAV is configured; else return None.

    Returns a user-facing suffix or None (caller falls back to attachment_both).
    ``content_type``, when given, is written to the attachment shell's
    ``contentType`` field so Zotero renders and opens the file correctly
    (e.g. ``application/pdf``).
    """
    from zotero_mcp import webdav as _webdav

    if not _webdav.is_webdav_configured():
        return None

    template = write_zot.item_template("attachment", linkmode="imported_file")
    template["title"] = filename
    template["filename"] = filename
    template["parentItem"] = parent_key
    if content_type:
        template["contentType"] = content_type
    result = write_zot.create_items([template])
    if not (isinstance(result, dict) and result.get("success")):
        return " (WARNING: could not create attachment shell)"
    attachment_key = next(iter(result["success"].values()))
    # successVersions is keyed in parallel to success; delete_item() needs the
    # version for its If-Unmodified-Since-Version header. Older pyzotero may
    # omit the field, so fall back to a fetch.
    success_versions = result.get("successVersions") or {}
    attachment_version = next(iter(success_versions.values()), None)

    try:
        _webdav.upload_attachment_to_webdav(attachment_key=attachment_key, file_path=file_path)
        ctx.info(f"WebDAV PUT: {attachment_key}.zip uploaded")
        return f" (uploaded to WebDAV as {attachment_key}.zip)"
    except Exception as e:
        ctx.info(f"WebDAV PUT failed for {attachment_key}: {e}")
        # A failed PUT leaves the shell with no file bytes — an orphan that
        # confuses the Zotero UI and breaks sync. Clean it up, and only fall
        # back to the "no file bytes" warning if the delete itself fails.
        try:
            if attachment_version is None:
                attachment_version = write_zot.item(attachment_key)["version"]
            write_zot.delete_item({"key": attachment_key, "version": attachment_version})
            ctx.info(f"Cleaned up orphan attachment shell {attachment_key}")
            return f" (WARNING: WebDAV upload failed — {e}; attachment shell {attachment_key} was deleted)"
        except Exception as del_err:
            ctx.info(f"Cleanup of orphan shell {attachment_key} failed: {del_err}")
            return (
                f" (WARNING: WebDAV upload failed — {e}; "
                f"attachment {attachment_key} exists but has no file bytes on WebDAV "
                f"and could not be deleted: {del_err})"
            )


def _extract_attachment_key(attach_result):
    """First attachment key in a pyzotero upload result, or ``None``.

    ``Zupload.upload()`` returns ``{"success": [...], "failure": [...],
    "unchanged": [...]}`` with the registered key on each payload entry.
    """
    if not isinstance(attach_result, dict):
        return None
    for status in ("success", "unchanged"):
        for entry in attach_result.get(status, []) or []:
            if isinstance(entry, dict) and entry.get("key"):
                return entry["key"]
    return None


def _describe_attach_failure(attach_result):
    """Short reason string for a pyzotero upload result that landed no file.

    ``attachment_both()`` reports client-side rejections by returning the
    payload in ``failure`` rather than raising (#403), so a caller that
    only checks for exceptions reports success for a file that never
    landed. Returns ``None`` when the result did register an attachment.
    """
    if _extract_attachment_key(attach_result) is not None:
        return None
    if not isinstance(attach_result, dict):
        return f"unexpected upload result: {attach_result!r}"
    failures = attach_result.get("failure") or []
    if failures:
        return f"upload rejected by pyzotero: {failures}"
    return "upload returned no attachment key"


def _assert_upload_capable(write_zot):
    """Raise ValueError if *write_zot* cannot upload file bytes.

    Zotero's local HTTP API has no attachment/upload endpoints — the
    template fetch that starts ``attachment_both()`` 404s against
    ``localhost:23119`` (#403). Failing fast here beats a confusing
    "No endpoint found" deep inside pyzotero.
    """
    if getattr(write_zot, "local", False):
        raise ValueError(
            "Cannot upload file bytes through Zotero's local API — it has no "
            "attachment endpoints. Add ZOTERO_API_KEY and ZOTERO_LIBRARY_ID "
            "to enable hybrid mode (local reads, web-API writes)."
        )


def _two_step_attach(write_zot, filename, file_path, parent_key, ctx, content_type=None):
    """Create the attachment item, then upload its bytes; verify md5 landed.

    Fallback for the case in #403 where ``attachment_both()`` fails
    client-side (it puts the full filesystem path in ``filename`` and
    leaves ``md5`` unset). Creating the item with the basename and only
    then uploading with the full path succeeds where the combined call
    does not. Returns ``(attachment_key, None)`` on success or
    ``(None, reason)`` on failure, cleaning up the orphaned shell so a
    failed upload doesn't leave a fileless attachment behind.
    """
    template = write_zot.item_template("attachment", linkmode="imported_file")
    template["title"] = filename
    template["filename"] = filename
    template["parentItem"] = parent_key
    if content_type:
        template["contentType"] = content_type

    result = write_zot.create_items([template])
    if not (isinstance(result, dict) and result.get("success")):
        return None, f"could not create attachment item: {result}"
    attachment_key = next(iter(result["success"].values()))
    success_versions = result.get("successVersions") or {}
    attachment_version = next(iter(success_versions.values()), None)

    try:
        attachment = write_zot.item(attachment_key)["data"]
        # The full path is only correct for the upload step; the stored
        # filename stays the basename set on the template above.
        attachment["filename"] = file_path
        upload = write_zot.upload_attachments([attachment])
        if isinstance(upload, dict) and upload.get("failure"):
            raise RuntimeError(f"upload rejected: {upload['failure']}")
        # Only md5 on the stored item proves the bytes actually landed.
        if not write_zot.item(attachment_key)["data"].get("md5"):
            raise RuntimeError("upload reported success but no md5 was stored")
        return attachment_key, None
    except Exception as e:
        try:
            if attachment_version is None:
                attachment_version = write_zot.item(attachment_key)["version"]
            write_zot.delete_item(
                {"key": attachment_key, "version": attachment_version}
            )
            ctx.info(f"Cleaned up orphan attachment shell {attachment_key}")
        except Exception as del_err:
            ctx.info(f"Cleanup of orphan shell {attachment_key} failed: {del_err}")
            return None, (
                f"{e}; attachment {attachment_key} exists but has no file "
                f"bytes and could not be deleted: {del_err}"
            )
        return None, str(e)


def _attach_and_verify(
    write_zot, filename, file_path, parent_key, ctx, content_type=None
):
    """Upload *file_path* onto *parent_key*, confirming the file landed.

    Returns ``(ok, suffix, attachment_key)``. ``suffix`` is the user-facing
    tail to append to a "file attached" message when ``ok``; when not
    ``ok`` it is the reason the attach failed, and the caller must NOT
    claim success (#403, the root cause behind #278 / #306 / #399).
    """
    _assert_upload_capable(write_zot)

    attach_result = write_zot.attachment_both(
        [(filename, file_path)],
        parentid=parent_key,
    )
    reason = _describe_attach_failure(attach_result)
    if reason is None:
        suffix = _maybe_upload_to_webdav(
            attach_result, file_path, ctx, write_zot=write_zot
        )
        return True, suffix, _extract_attachment_key(attach_result)

    ctx.info(f"attachment_both failed ({reason}); retrying as create + upload")
    attachment_key, fallback_reason = _two_step_attach(
        write_zot, filename, file_path, parent_key, ctx, content_type=content_type
    )
    if attachment_key is None:
        return False, f"{reason}; two-step retry also failed: {fallback_reason}", None

    suffix = _maybe_upload_to_webdav(
        {"success": [{"key": attachment_key}]}, file_path, ctx, write_zot=write_zot
    )
    return True, suffix, attachment_key


def _file_md5(path):
    """MD5 hex digest of ``path``, or ``None`` if unreadable.

    Non-fatal so content dedupe degrades to filename-only rather than
    failing the attach.
    """
    try:
        digest = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _find_child_attachment(write_zot, parent_key, filename=None, file_md5=None):
    """First child of ``parent_key`` stored as ``filename`` or hashing to ``file_md5``.

    Either criterion matching returns the child dict; ``None`` criteria are
    skipped (a child without an ``md5`` field never matches ``file_md5=None``).
    Paginates past the API's default page size and ignores trashed children
    (``deleted`` flag), so a match beyond the first page isn't missed and a
    trashed attachment doesn't count as "already attached".
    Non-fatal: errors while listing children count as "no match", so attach
    flows degrade to re-uploading rather than failing outright.
    """
    try:
        kids = _paginate(write_zot.children, parent_key)
    except Exception:
        return None
    for kid in kids:
        data = kid.get("data", {}) or {}
        if data.get("deleted"):
            continue
        if filename is not None and data.get("filename") == filename:
            return kid
        if file_md5 is not None and data.get("md5") == file_md5:
            return kid
    return None


def _attachment_filename_exists(write_zot, parent_key, filename):
    """True if ``parent_key`` already has a child attachment stored as ``filename``."""
    return _find_child_attachment(write_zot, parent_key, filename=filename) is not None


def _attach_pdf_linked_url(write_zot, pdf_url, parent_key, ctx):
    """Create a linked-URL attachment (bookmarks the PDF URL without downloading)."""
    try:
        template = write_zot.item_template("attachment", "linked_url")
        template["url"] = pdf_url
        template["title"] = "PDF (linked URL)"
        template["contentType"] = "application/pdf"
        template["parentItem"] = parent_key
        result = write_zot.create_items([template])
        if result.get("success"):
            ctx.info(f"Linked URL attachment created for {pdf_url}")
            return True
        return False
    except Exception as e:
        ctx.info(f"Linked URL attachment failed: {e}")
        return False


def _try_unpaywall(doi, ctx):
    """Try Unpaywall API for open-access PDF URLs."""
    try:
        resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": "zotero-mcp@users.noreply.github.com"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None

        oa_data = resp.json()

        best = oa_data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf")
        if pdf_url:
            ctx.info("Unpaywall: found PDF via best_oa_location")
            return pdf_url

        for loc in oa_data.get("oa_locations", []):
            pdf_url = loc.get("url_for_pdf")
            if pdf_url:
                ctx.info("Unpaywall: found PDF via alternate oa_location")
                return pdf_url

        landing = best.get("url")
        if landing:
            ctx.info("Unpaywall: no direct PDF URL, trying landing page")
            return landing

        return None
    except Exception as e:
        ctx.info(f"Unpaywall lookup failed: {e}")
        return None


def _try_arxiv_from_crossref(crossref_metadata, ctx):
    """Check CrossRef metadata for an arXiv ID and return a PDF URL."""
    if not crossref_metadata:
        return None
    try:
        relations = crossref_metadata.get("relation", {})
        for rel_type in ("has-preprint", "is-preprint-of", "is-identical-to",
                         "is-version-of", "has-version"):
            for rel in relations.get(rel_type, []):
                rel_id = rel.get("id", "")
                if rel.get("id-type") == "arxiv" and rel_id:
                    ctx.info(f"CrossRef relation contains arXiv ID: {rel_id}")
                    return f"https://arxiv.org/pdf/{rel_id}.pdf"
                if rel.get("id-type") == "doi" and "arxiv" in rel_id.lower():
                    m = re.search(r"arXiv\.(\d{4}\.\d{4,5}(?:v\d+)?)", rel_id, re.IGNORECASE)
                    if m:
                        arxiv_id = m.group(1)
                        ctx.info(f"CrossRef relation contains arXiv DOI: {rel_id} -> {arxiv_id}")
                        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        for alt_id in crossref_metadata.get("alternative-id", []):
            if re.match(r"\d{4}\.\d{4,5}", str(alt_id)):
                ctx.info(f"CrossRef alternative-id looks like arXiv: {alt_id}")
                return f"https://arxiv.org/pdf/{alt_id}.pdf"

        for link in crossref_metadata.get("link", []):
            url = link.get("URL", "")
            if "arxiv.org" in url:
                m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", url)
                if m:
                    ctx.info("CrossRef link contains arXiv URL")
                    return f"https://arxiv.org/pdf/{m.group(1)}.pdf"

        return None
    except Exception as e:
        ctx.info(f"arXiv-from-CrossRef check failed: {e}")
        return None


def _try_semantic_scholar(doi, ctx):
    """Try Semantic Scholar API for an open-access PDF URL."""
    try:
        resp = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
            params={"fields": "openAccessPdf"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        oa_pdf = data.get("openAccessPdf") or {}
        pdf_url = oa_pdf.get("url")
        if pdf_url:
            ctx.info("Semantic Scholar: found OA PDF")
            return pdf_url
        return None
    except Exception as e:
        ctx.info(f"Semantic Scholar lookup failed: {e}")
        return None


def _try_pmc(doi, ctx):
    """Try PubMed Central for a free PDF via DOI-to-PMCID conversion."""
    try:
        conv_resp = requests.get(
            "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/",
            params={"ids": doi, "format": "json", "tool": "zotero-mcp",
                    "email": "zotero-mcp@users.noreply.github.com"},
            timeout=10,
        )
        if conv_resp.status_code != 200:
            return None

        records = conv_resp.json().get("records", [])
        if not records:
            return None

        pmcid = records[0].get("pmcid")
        if not pmcid:
            return None

        ctx.info(f"PMC: found PMCID {pmcid}")
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"

    except Exception as e:
        ctx.info(f"PMC lookup failed: {e}")
        return None


class OaPdfRequiredError(Exception):
    """Raised by _try_attach_oa_pdf when attach_mode='required' finds no PDF.

    Signals that the caller should fail (or flag) the entry rather than
    report silent success — the item may already be created in Zotero, but
    without the PDF the caller promised.
    """


def _try_attach_oa_pdf(write_zot, item_key, doi, ctx, crossref_metadata=None,
                       attach_mode="auto"):
    """Attempt to find and attach an open-access PDF for a DOI.

    attach_mode: 'auto' downloads and uploads the first working OA PDF;
    'linked_url' bookmarks the PDF URL instead of uploading the binary;
    'none' skips the OA lookup entirely; 'required' behaves like 'auto' but
    raises OaPdfRequiredError instead of returning a status string when no
    OA PDF could be attached.
    """
    if attach_mode == "none":
        return "skipped (attach_mode=none)"

    sources = [
        ("Unpaywall", lambda: _try_unpaywall(doi, ctx)),
        ("arXiv (via CrossRef)", lambda: _try_arxiv_from_crossref(crossref_metadata, ctx)),
        ("Semantic Scholar", lambda: _try_semantic_scholar(doi, ctx)),
        ("PubMed Central", lambda: _try_pmc(doi, ctx)),
    ]

    found_urls = []  # Track URLs found but not downloadable

    for source_name, find_url in sources:
        try:
            pdf_url = find_url()
            if pdf_url:
                ctx.info(f"Trying PDF from {source_name}: {pdf_url}")
                found_urls.append((source_name, pdf_url))

                if attach_mode == "linked_url":
                    if _attach_pdf_linked_url(write_zot, pdf_url, item_key, ctx):
                        return f"PDF linked (source: {source_name})"
                else:  # "auto" or "required" — try download only
                    webdav_suffix = _download_and_attach_pdf(
                        write_zot, item_key, pdf_url, doi, ctx
                    )
                    if webdav_suffix is not None:
                        return f"PDF attached (source: {source_name}){webdav_suffix}"

                ctx.info(f"{source_name} URL didn't yield a valid PDF, trying next source")
        except Exception as e:
            ctx.info(f"{source_name} failed: {e}")

    if found_urls:
        # URLs were found but couldn't be downloaded — report them so the user
        # can access the paper through their university library
        url_info = found_urls[0][1]  # Best URL found
        message = (
            f"no open-access PDF could be downloaded, but a URL was found: {url_info} — "
            "you may be able to access it through your university library or VPN"
        )
    else:
        message = "no open-access PDF found (checked Unpaywall, arXiv, Semantic Scholar, PMC)"

    if attach_mode == "required":
        raise OaPdfRequiredError(message)
    return message


# ---------------------------------------------------------------------------
# Citation key helpers
# ---------------------------------------------------------------------------

def _extra_has_citekey(extra: str, citekey: str) -> bool:
    """Check if the Extra field contains the given citation key."""
    for line in extra.splitlines():
        lower = line.lower().strip()
        if lower.startswith("citation key:") or lower.startswith("citationkey:"):
            value = line.split(":", 1)[1].strip()
            if value == citekey:
                return True
    return False


def _format_citekey_result(item: dict, citekey: str) -> str:
    """Format a Zotero item found by citation key as markdown."""
    extra = {"Citation Key": citekey}
    if doi := item.get("data", {}).get("DOI"):
        extra["DOI"] = doi
    lines = [f"# Citation Key: {citekey}", ""]
    lines.extend(_utils.format_item_result(item, extra_fields=extra))
    return "\n".join(lines)


def _format_bbt_result(bbt_item: dict, citekey: str) -> str:
    """Format a BetterBibTeX search result."""
    title = bbt_item.get("title", "Untitled")
    year = bbt_item.get("year", "N/A")
    creators_str = _utils.format_creators(bbt_item.get("creators", []))

    output = [
        f"# Citation Key: {citekey}",
        "",
        f"## {title}",
        f"**Citation Key:** {citekey}",
        f"**Year:** {year}",
        f"**Authors:** {creators_str}",
        "",
        "*Note: Item found via BetterBibTeX. Use the citation key with other tools for full details.*",
        "",
    ]
    return "\n".join(output)


# ---------------------------------------------------------------------------
# Token estimation helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate at ~4 characters per token."""
    return len(text) // 4


def _prepend_size_warning(text: str, suggestions: str = "") -> str:
    """If text exceeds ~5K tokens, prepend a size warning header."""
    est = _estimate_tokens(text)
    if est < 5000:
        return text
    suggestion_text = f" {suggestions}" if suggestions else ""
    warning = f"*Response size: ~{est // 1000}K tokens.{suggestion_text}*\n\n"
    return warning + text
