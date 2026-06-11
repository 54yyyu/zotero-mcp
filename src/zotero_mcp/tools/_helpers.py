"""Shared private helpers used across tool modules."""

import json
import os
import re
import socket
import tempfile
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from zotero_mcp import client as _client
from zotero_mcp import utils as _utils

# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------

ZOTERO_MCP_CONFIG_PATH = Path.home() / ".config" / "zotero-mcp" / "config.json"


def _load_zotero_mcp_config() -> dict:
    """Return the parsed ``~/.config/zotero-mcp/config.json``, or ``{}``.

    Missing file or parse errors yield an empty dict so callers can use
    ``.get(...)`` chains without guarding.
    """
    if not ZOTERO_MCP_CONFIG_PATH.exists():
        return {}
    try:
        with open(ZOTERO_MCP_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Pagination helper
# ---------------------------------------------------------------------------

def _paginate(zot_method, *args, max_items=None, **kwargs):
    """Fetch all results from a pyzotero method using manual pagination.

    Avoids zot.everything() which can cause RLock pickling in MCP contexts.
    Accepts the same positional and keyword arguments as the wrapped method,
    plus an optional max_items to cap the total results.
    """
    items = []
    start = 0
    page_size = 100
    while True:
        batch = zot_method(*args, start=start, limit=page_size, **kwargs)
        if not batch:
            break
        items.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
        if max_items and len(items) >= max_items:
            items = items[:max_items]
            break
    return items


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CROSSREF_TYPE_MAP = {
    "journal-article": "journalArticle",
    "book": "book",
    "book-chapter": "bookSection",
    "proceedings-article": "conferencePaper",
    "report": "report",
    "dissertation": "thesis",
    "posted-content": "preprint",
    "monograph": "book",
    "reference-entry": "encyclopediaArticle",
    "dataset": "document",
    "peer-review": "document",
    "edited-book": "book",
    "standard": "document",
}


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
    pre-validate ``zotero_manage_collections`` calls so the tool returns a
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
    """Coerce *limit* to a bounded int."""
    if limit is None:
        return default
    if isinstance(limit, str):
        limit = int(limit)
    return max(1, min(limit, max_val))


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
            raise ValueError(
                f"{field_name} must be a list of strings or a string, "
                f"got JSON {type(parsed).__name__}"
            )
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
    callers treat that as "nothing found" and proceed to create.
    """
    if doi:
        query = doi
        def _matches(data):
            return _normalize_doi(data.get("DOI") or "") == doi
    elif arxiv_id:
        query = arxiv_id
        def _matches(data):
            if _normalize_arxiv_id(data.get("url") or "") == arxiv_id:
                return True
            return f"arxiv:{arxiv_id}".lower() in (data.get("extra") or "").lower()
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
    except Exception as e:
        if ctx is not None:
            ctx.warning(f"Existing-item search failed (treating as no match): {e}")
        return []

    matches = []
    for item in candidates or []:
        data = item.get("data", {})
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


def _normalize_doi(raw):
    """Normalize a DOI string from various input formats."""
    if not raw:
        return None
    s = raw.strip()
    if s.lower().startswith("doi:"):
        s = s[4:].strip()
    if s.lower().startswith("http://") or s.lower().startswith("https://"):
        m = re.search(r"doi\.org/(10\.\d{4,9}/[^\s?#]+)", s, flags=re.IGNORECASE)
        if not m:
            return None
        s = m.group(1)
    s = s.rstrip(".,);]")
    if re.match(r"^10\.\d{4,9}/\S+$", s):
        return s
    return None


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
    injection steering ``zotero_add_by_doi``). We reject non-http(s) schemes
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

            attach_result = write_zot.attachment_both(
                [(filename, filepath)],
                parentid=item_key,
            )
            # Must run inside the with-block — temp file disappears on exit.
            return _maybe_upload_to_webdav(attach_result, filepath, ctx)
    except Exception as e:
        ctx.info(f"PDF download/attach failed: {e}")
        return None


def _maybe_upload_to_webdav(attach_result, file_path, ctx):
    """Suffix to append to a user-facing 'file attached' message.

    PR #279 added WebDAV-aware upload to ``zotero_add_from_file``. The same
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

    attachment_key = None
    if isinstance(attach_result, dict):
        for status in ("success", "unchanged"):
            for entry in attach_result.get(status, []) or []:
                if isinstance(entry, dict) and entry.get("key"):
                    attachment_key = entry["key"]
                    break
            if attachment_key:
                break

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
        return (
            f" (WARNING: WebDAV upload failed — {e}; "
            f"attachment {attachment_key} exists but has no file bytes on WebDAV)"
        )


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


def _try_attach_oa_pdf(write_zot, item_key, doi, ctx, crossref_metadata=None,
                       attach_mode="auto"):
    """Attempt to find and attach an open-access PDF for a DOI."""
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
                else:  # "auto" or "import_file" — try download only
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
        return (
            f"no open-access PDF could be downloaded, but a URL was found: {url_info} — "
            "you may be able to access it through your university library or VPN"
        )

    return "no open-access PDF found (checked Unpaywall, arXiv, Semantic Scholar, PMC)"


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
