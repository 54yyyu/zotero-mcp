"""Search-related tool functions for the Zotero MCP server."""

import json
import logging as _logging
import re
import threading as _threading
import time as _time
from pathlib import Path
from typing import Literal

from zotero_mcp import client as _client
from zotero_mcp import search_semantics as _semantics
from zotero_mcp import utils as _utils
from zotero_mcp._app import mcp
from zotero_mcp._context import Context
from zotero_mcp.client import with_zotero_api_lock
from zotero_mcp.local_db import get_local_zotero_reader
from zotero_mcp.tools import _helpers

_search_logger = _logging.getLogger("zotero_mcp.search")

CASCADE_TIMEOUT = 60  # seconds — total budget for the entire fallback cascade

# Pre-search background sync debounce: at most one fire-and-forget sync per
# this many seconds, shared across all semantic_search tool invocations.
_PRESEARCH_SYNC_MIN_INTERVAL = 60.0
_last_presearch_sync_ts: float = 0.0
_presearch_sync_lock = _threading.Lock()


def _maybe_fire_presearch_sync(search) -> None:
    """Schedule a background semantic-search DB update if auto-update is due.

    Runs in a daemon thread so the current tool call returns immediately.
    Intentionally swallows exceptions — a failed background sync must never
    surface as a search-tool error to the user.
    """
    global _last_presearch_sync_ts
    try:
        if not search.should_update_database():
            return
    except Exception:
        return
    now = _time.monotonic()
    with _presearch_sync_lock:
        if now - _last_presearch_sync_ts < _PRESEARCH_SYNC_MIN_INTERVAL:
            return
        _last_presearch_sync_ts = now

    def _run():
        try:
            search.update_database(extract_fulltext=_utils.is_local_mode())
        except Exception as e:
            _search_logger.debug(f"Background pre-search sync failed: {e}")

    _threading.Thread(target=_run, daemon=True, name="zmcp-presearch-sync").start()


#: How long the client-side advanced-search walk may run before it returns a
#: partial answer. Kept well under the Zotero API lock's 45s wait bound so a
#: broad search cannot cascade into "Zotero API busy" on every other tool
#: (#456), and far under the 300s MCP idle timeout it used to hit.
_ADVANCED_SCAN_BUDGET_SECONDS = 20


def _server_side_filters(
    parsed_conditions: list[dict[str, str]], join_mode: str
) -> dict[str, str]:
    """Conditions the Zotero API can apply itself, as ``zot.items()`` kwargs.

    The client-side walk re-checks every condition regardless, so this is a
    pure narrowing of what has to be fetched — never a change of meaning. Two
    rules keep it that way, and both matter:

    Only ``join_mode="all"`` qualifies. Under ``any`` a server-side filter
    would drop items that another condition would have matched, turning an OR
    into an AND.

    Only ``itemType is <known type>`` and ``tag is <value>`` are pushed. The
    API compares these exactly while :mod:`search_semantics` compares
    case- and diacritic-insensitively, so ``itemType`` is canonicalised
    against the schema first and skipped when it does not resolve — a user
    who typed ``blogpost`` still gets the client-side match rather than an
    empty result. ``tag`` is passed through as typed, which is what Zotero's
    own tag filter does.
    """
    if join_mode != "all":
        return {}

    filters: dict[str, str] = {}
    for condition in parsed_conditions:
        if condition["operation"] != "is":
            continue
        field = condition["field"].lower()
        value = condition["value"]
        if field == "itemtype" and "itemType" not in filters:
            canonical = _canonical_item_type(value)
            if canonical:
                filters["itemType"] = canonical
        elif field in {"tag", "tags"} and "tag" not in filters:
            if value:
                filters["tag"] = value
    return filters


def _canonical_item_type(value: str) -> str | None:
    """Map a user-supplied item type to the schema's exact spelling.

    Returns None for anything the schema doesn't know, which is the signal not
    to filter server-side — the API would answer an unknown type with nothing
    at all, and a typo should not silently become "no results".
    """
    if not value:
        return None
    try:
        from zotero_mcp import schema as _schema

        known = _schema.get_table().get("itemTypes", {})
    except Exception:  # schema unavailable — decline to filter
        return None
    if value in known:
        return value
    lowered = value.lower()
    for name in known:
        if name.lower() == lowered:
            return name
    return None


def _exclude_note_content_matches(items: list[dict], qmode: str) -> list[dict]:
    """Drop standalone notes from a `titleCreatorYear` result set.

    Notes have no title/creator/year field, so Zotero's own server-side
    quicksearch matches a note's *content* instead when it appears in
    `titleCreatorYear` results — the note's content stands in for its
    missing title. That contradicts the mode's own name/semantics and
    diverges from the #167 SQL backend's `search_items_sql`, whose
    titleCreatorYear query only ever inspects title/creator/date itemData
    rows a note doesn't have, so it never matches note content there.
    Filtering here brings the pyzotero path in line. `everything` mode is
    untouched — content matching is exactly what it's for.
    """
    if qmode != "titleCreatorYear":
        return items
    return [item for item in items if item.get("data", {}).get("itemType") != "note"]


@with_zotero_api_lock
def _search_with_variants(zot, query: str, qmode: str, limit: int,
                          item_type: str = "-attachment",
                          tag: list[str] | None = None,
                          cascade_start: float | None = None,
                          cascade_timeout: float | None = None) -> list:
    """Search using multiple query variants, deduplicate by key.

    Generates ASCII, dash-to-space, and umlaut-expanded variants of the query
    and searches for each one.  Results are deduplicated by item key.

    All params (including item_type and tag) are explicitly set on every
    add_parameters call to avoid stale accumulated params in pyzotero.

    If cascade_start and cascade_timeout are provided, checks the budget
    before each API call and bails out if exceeded.
    """
    variants = _utils._generate_search_variants(query)
    _search_logger.debug(f"[SEARCH] query='{query}' variants={variants}")

    all_items: list[dict] = []
    seen_keys: set[str] = set()
    for variant in variants:
        # Check cascade timeout before each API call
        if cascade_start is not None and cascade_timeout is not None:
            if _time.monotonic() - cascade_start > cascade_timeout:
                _search_logger.debug("[SEARCH] Cascade timeout reached, skipping remaining variants")
                break

        params: dict = {
            "q": variant, "qmode": qmode, "limit": limit, "itemType": item_type,
        }
        if tag:
            params["tag"] = tag
        zot.add_parameters(**params)
        try:
            t0 = _time.monotonic()
            batch = zot.items()
            elapsed = _time.monotonic() - t0
            _search_logger.debug(f"[SEARCH] variant='{variant}' qmode={qmode}: {len(batch)} results in {elapsed:.2f}s")
            for item in batch:
                key = item.get("key", "")
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    all_items.append(item)
        except Exception as e:
            _search_logger.debug(f"[SEARCH] variant='{variant}' failed: {e}")
            continue  # Skip failed variant, try next

    return _exclude_note_content_matches(all_items, qmode)


def _search_items_via_backend(zot, query: str, qmode: str, limit: int,
                              item_type: str = "-attachment",
                              tag: list[str] | None = None,
                              cascade_start: float | None = None,
                              cascade_timeout: float | None = None) -> list:
    """Try the #167 SQLite metadata backend first; fall back to the
    pyzotero-based `_search_with_variants` on any unsupported condition or
    error. Mirrors `_search_with_variants`'s signature and return shape so
    every call site in the fallback cascade can swap it in unchanged.
    """
    if _utils.get_search_backend() == "sqlite":
        try:
            reader = get_local_zotero_reader()
            if reader is not None:
                try:
                    result = reader.search_items_sql(
                        query, qmode=qmode, item_type=item_type, tag=tag,
                        limit=limit, group_id=_client.get_active_group_id(),
                    )
                finally:
                    reader.close()
                if result is not None:
                    return result
        except Exception as e:
            _search_logger.debug(f"[SEARCH] sqlite backend failed, falling back: {e}")
    return _search_with_variants(zot, query, qmode, limit, item_type=item_type, tag=tag,
                                 cascade_start=cascade_start, cascade_timeout=cascade_timeout)


@mcp.tool(
    name="zotero_search_items",
    description=(
        "Search Zotero items by substring match against metadata (title, "
        "creators, year, and — in 'everything' mode — abstract). Returns "
        "metadata + abstracts as markdown. "
        "IMPORTANT: keep queries SHORT and SIMPLE — 'Author Year' "
        "(e.g. 'Brewer 2011') or just an author name ('Cladder-Micus'). "
        "This is substring matching, not web search: each extra word "
        "NARROWS the match, so adding topic words usually returns fewer "
        "results, not more. For topic discovery, use zotero_semantic_search "
        "instead; for tag filtering use zotero_search_by_tag. "
        "If a query finds nothing, this tool automatically falls back to "
        "simplified queries and then semantic search. "
        "query: required substring. qmode: 'titleCreatorYear' (default) "
        "matches only title/authors/year; 'everything' also searches "
        "abstract. item_type: '-attachment' (default) excludes attachments; "
        "pass 'journalArticle', 'book', etc. to filter. tag: optional list "
        "of tag conditions (ANDed). limit: max results (default 10). "
        "collection_key: 8-char key to restrict to a collection (bypasses "
        "the fallback cascade). include_subcollections: also search "
        "collections nested beneath it (default False). "
        "Example: zotero_search_items(query='Cladder-Micus') or "
        "zotero_search_items(query='Brewer 2011', limit=5)."
    )
)
@with_zotero_api_lock
def search_items(
    query: str,
    qmode: Literal["titleCreatorYear", "everything"] = "titleCreatorYear",
    item_type: str = "-attachment",  # Exclude attachments by default
    limit: int | str | None = 10,
    tag: list[str] | list[dict] | str | None = None,
    collection_key: str | None = None,
    include_subcollections: bool = False,
    *,
    ctx: Context
) -> str:
    """
    Search for items in your Zotero library.

    Args:
        query: Search query string
        qmode: Query mode (titleCreatorYear or everything)
        item_type: Type of items to search for. Use "-attachment" to exclude attachments.
        limit: Maximum number of results to return
        tag: Tag filter. Accepts ["tagA", "tagB"] (preferred), a bare string
            "tagA", a JSON-string list '["tagA", "tagB"]', or the dict-shape
            [{"tag": "tagA"}] sometimes emitted by clients that confuse the
            filter form with Zotero's stored-tag form. All are normalized
            internally to the list[str] form pyzotero expects.
        collection_key: Optional collection key to scope the search to a specific collection.
            When provided, bypasses the fallback cascade and searches the collection directly.
        include_subcollections: Also search collections nested beneath
            collection_key. Ignored when collection_key is not given. Defaults
            to False, matching Zotero's own "Search subcollections" checkbox.
        ctx: MCP context

    Returns:
        Markdown-formatted search results
    """
    try:
        if not query.strip():
            return "Error: Search query cannot be empty"

        # Normalize tag across every wire shape clients produce (#237).
        tag = _helpers._normalize_tag_filter(tag)

        tag_condition_str = ""
        if tag:
            tag_condition_str = f" with tags: '{', '.join(tag)}'"

        ctx.info(f"Searching Zotero for '{query}'{tag_condition_str}")
        zot = _client.get_zotero_client()

        limit = _helpers._normalize_limit(limit, default=10)

        if collection_key:
            # Collection-scoped search — query the collection directly, no cascade needed
            try:
                _col = zot.collection(collection_key)
            except Exception:
                _col = None
            if not _col or _col.get("key") != collection_key:
                return f"Collection not found: '{collection_key}'. Use zotero_get_collections or zotero_search_collections to find valid collection keys."
            scope_keys = _helpers.expand_collection_scope(
                zot, collection_key, include_subcollections
            )
            items = []
            _seen: set[str] = set()
            for _scope_key in scope_keys:
                # limit applies to the merged result, so each subcollection may
                # still contribute up to it before deduplication.
                for _item in _helpers._paginate(
                    zot.collection_items, _scope_key,
                    q=query, qmode=qmode, itemType=item_type,
                    max_items=limit, **({"tag": tag} if tag else {}),
                ):
                    _key = _item.get("key")
                    if _key and _key in _seen:
                        continue
                    if _key:
                        _seen.add(_key)
                    items.append(_item)
            # Ahead of the slice, so a dropped note never costs a result slot
            # that a real match could have filled.
            items = _exclude_note_content_matches(items, qmode)
            items = items[:limit]
            fallback_strategy = None
        else:
            # --- Initial search with variant generation ---
            _cascade_start = _time.monotonic()
            items = _search_items_via_backend(zot, query, qmode, limit,
                                              item_type=item_type, tag=tag,
                                              cascade_start=_cascade_start,
                                              cascade_timeout=CASCADE_TIMEOUT)
            _search_logger.debug(f"[CASCADE] initial: {len(items)} results in {_time.monotonic() - _cascade_start:.2f}s")

            # --- Fallback cascade (only if initial search returned nothing) ---
            fallback_strategy = None
            _timed_out = False

            def _check_cascade_timeout():
                nonlocal _timed_out
                if _time.monotonic() - _cascade_start > CASCADE_TIMEOUT:
                    _timed_out = True
                    _search_logger.debug("[CASCADE] Timeout — stopping cascade")
                    ctx.info("Search took too long — returning best results found so far")
                return _timed_out

            if not items and query.strip():
                ctx.info("No results with original query, trying fallback strategies...")
                words = query.strip().split()

                # Strategy 1: Simplify to author + year (P2 fix)
                if not _check_cascade_timeout() and not items and len(words) > 2:
                    # Extract year-like token (4 digits between 1800-2099)
                    year_token = next((w for w in words if re.match(r'^(1[89]\d{2}|20\d{2})$', w)), None)
                    # Extract author (first non-numeric word)
                    author_token = next((w for w in words if not re.match(r'^\d+$', w)), None)

                    if author_token and year_token:
                        simple_query = f"{author_token} {year_token}"
                    elif author_token:
                        simple_query = author_token
                    else:
                        simple_query = words[0]

                    t0 = _time.monotonic()
                    ctx.info(f"Retry with simplified query: '{simple_query}'")
                    items = _search_items_via_backend(zot, simple_query, qmode, limit,
                                                      item_type=item_type, tag=tag,
                                                      cascade_start=_cascade_start,
                                                      cascade_timeout=CASCADE_TIMEOUT)
                    _search_logger.debug(f"[CASCADE] strategy 1 (author+year): {len(items)} results in {_time.monotonic() - t0:.2f}s")
                    if items:
                        fallback_strategy = f"simplified to '{simple_query}'"

                # Strategy 2: Author surname only (first non-numeric word)
                if not _check_cascade_timeout() and not items and len(words) >= 2:
                    author_only = next((w for w in words if not re.match(r'^\d+$', w)), words[0])
                    t0 = _time.monotonic()
                    ctx.info(f"Retry with author only: '{author_only}'")
                    items = _search_items_via_backend(zot, author_only, qmode, limit,
                                                      item_type=item_type, tag=tag,
                                                      cascade_start=_cascade_start,
                                                      cascade_timeout=CASCADE_TIMEOUT)
                    _search_logger.debug(f"[CASCADE] strategy 2 (author only): {len(items)} results in {_time.monotonic() - t0:.2f}s")
                    if items:
                        fallback_strategy = f"author only '{author_only}'"

                # Strategy 3: qmode="everything" (searches full text on Zotero's side)
                # Safe — no tokens consumed, only metadata returned
                if not _check_cascade_timeout() and not items and qmode != "everything":
                    t0 = _time.monotonic()
                    ctx.info(f"Retry with qmode='everything': '{query}'")
                    items = _search_items_via_backend(zot, query, "everything", limit,
                                                      item_type=item_type, tag=tag,
                                                      cascade_start=_cascade_start,
                                                      cascade_timeout=CASCADE_TIMEOUT)
                    _search_logger.debug(f"[CASCADE] strategy 3 (everything): {len(items)} results in {_time.monotonic() - t0:.2f}s")
                    if items:
                        fallback_strategy = "full-text search"

                # Strategy 4: Semantic search (if database exists)
                if not _check_cascade_timeout() and not items:
                    try:
                        from zotero_mcp.semantic_search import create_semantic_search
                        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
                        if config_path.exists():
                            ctx.info(f"Retry with semantic search: '{query}'")
                            t0 = _time.monotonic()
                            sem_search = create_semantic_search(str(config_path))
                            _search_logger.debug(f"[CASCADE] semantic init: {_time.monotonic() - t0:.2f}s")
                            t0 = _time.monotonic()
                            # The semantic index can span multiple libraries
                            # (#163) while this tool's own results come from
                            # exactly one (whichever zot is scoped to) — scope
                            # the fallback the same way so it never surfaces a
                            # group-library hit for what looks like a
                            # single-library search.
                            sem_results = sem_search.search(
                                query=query, limit=limit or 10, group_id=_client.get_active_group_id()
                            )
                            _search_logger.debug(f"[CASCADE] semantic query: {_time.monotonic() - t0:.2f}s")
                            if sem_results and sem_results.get("results"):
                                seen_keys: set[str] = set()
                                for sr in sem_results["results"]:
                                    zot_item = sr.get("zotero_item", {})
                                    key = sr.get("item_key", zot_item.get("key", ""))
                                    if key and key not in seen_keys:
                                        seen_keys.add(key)
                                        if "key" not in zot_item:
                                            zot_item["key"] = key
                                        items.append(zot_item)
                                if items:
                                    fallback_strategy = "semantic search"
                    except Exception as e:
                        _search_logger.debug(f"[CASCADE] semantic failed: {e}")
                        ctx.info(f"Semantic search fallback failed: {e}")

            _search_logger.debug(f"[CASCADE] total: {_time.monotonic() - _cascade_start:.2f}s, fallback={fallback_strategy}")

        # --- No results after all strategies ---
        if not items:
            return f"No items found matching query: '{query}'{tag_condition_str}"

        # --- Format results as markdown ---
        output = [f"# Search Results for '{query}'", f"{tag_condition_str}", ""]

        for i, item in enumerate(items, 1):
            output.extend(_utils.format_item_result(item, index=i))

        # Prepend fallback verification note (AFTER output is built)
        if fallback_strategy:
            if fallback_strategy == "semantic search":
                note_text = (
                    f"*Note: Original search for '{query}' returned no results. "
                    f"The following {len(items)} item(s) are semantically related papers found "
                    f"via AI-powered search — they may be ABOUT the same topic but may NOT be "
                    f"the exact paper you're looking for. The target paper may not be in your "
                    f"library. Verify carefully by checking title, authors, and journal.*"
                )
            else:
                note_text = (
                    f"*Note: Original search for '{query}' returned no results. "
                    f"Found {len(items)} item(s) via {fallback_strategy} — verify the correct one "
                    f"by checking title, authors, journal, and year match your original query.*"
                )
            output.insert(1, "")
            output.insert(2, note_text)
            output.insert(3, "")

        return _helpers._prepend_size_warning("\n".join(output))

    except Exception as e:
        ctx.error(f"Error searching Zotero: {str(e)}")
        return f"Error searching Zotero: {str(e)}"

@mcp.tool(
    name="zotero_search_by_tag",
    description=(
        "Find items carrying one or more tags, with boolean syntax "
        "support. tag: list of tag strings; each entry is a condition ANDed "
        "with the others, and within an entry you can use ' OR ' for "
        "disjunction and a leading '-' for exclusion. "
        "Example: tag=['methods OR methodology', '-draft'] matches items "
        "tagged 'methods' OR 'methodology' AND NOT tagged 'draft'. "
        "item_type: '-attachment' (default) excludes attachments; pass "
        "'journalArticle', 'book', etc. to filter. "
        "limit: max results (default 10). "
        "collection_key: optional 8-char key to scope to a collection. "
        "include_subcollections: also search collections nested beneath it "
        "(default False). "
        "Use zotero_get_tags to discover available tag names first. For "
        "free-text content search, use zotero_search_items or "
        "zotero_semantic_search instead. "
        "Example: zotero_search_by_tag(tag=['to-read'], limit=20)."
    )
)
@with_zotero_api_lock
def search_by_tag(
    tag: list[str] | list[dict] | str,
    item_type: str = "-attachment",
    limit: int | str | None = 10,
    collection_key: str | None = None,
    include_subcollections: bool = False,
    *,
    ctx: Context
) -> str:
    """
    Search for items in your Zotero library by tag.
    Conditions are ANDed, each term supports disjunction (`OR`) and exclusion (`-`).

    Args:
        tag: List of tag conditions. Items are returned only if they satisfy
            ALL conditions in the list. Each tag condition can be expressed
            in two ways:
                As alternatives: tag1 OR tag2 (matches items with either tag1 OR tag2)
                As exclusions: -tag (matches items that do NOT have this tag)
            For example, a tag field with ["research OR important", "-draft"] would
            return items that:
                Have either "research" OR "important" tags, AND
                Do NOT have the "draft" tag
        item_type: Type of items to search for. Use "-attachment" to exclude attachments.
        limit: Maximum number of results to return
        collection_key: Optional collection key to scope the search to a specific collection
        include_subcollections: Also search collections nested beneath
            collection_key. Ignored when collection_key is not given.
        ctx: MCP context

    Returns:
        Markdown-formatted search results
    """
    try:
        # Normalize tag across every wire shape clients produce (#237).
        tag = _helpers._normalize_tag_filter(tag)
        if not tag:
            return "Error: Tag cannot be empty"

        ctx.info(f"Searching Zotero for tag '{tag}'")
        zot = _client.get_zotero_client()

        limit = _helpers._normalize_limit(limit, default=10)

        # Search library-wide or scoped to a collection
        if collection_key:
            try:
                _col = zot.collection(collection_key)
            except Exception:
                _col = None
            if not _col or _col.get("key") != collection_key:
                return f"Collection not found: '{collection_key}'. Use zotero_get_collections or zotero_search_collections to find valid collection keys."
            scope_keys = _helpers.expand_collection_scope(
                zot, collection_key, include_subcollections
            )
            results = []
            _seen: set[str] = set()
            for _scope_key in scope_keys:
                for _item in _helpers._paginate(
                    zot.collection_items, _scope_key,
                    tag=tag, itemType=item_type, max_items=limit,
                ):
                    _key = _item.get("key")
                    if _key and _key in _seen:
                        continue
                    if _key:
                        _seen.add(_key)
                    results.append(_item)
            results = results[:limit]
        else:
            zot.add_parameters(q="", tag=tag, itemType=item_type, limit=limit)
            results = zot.items()

        if not results:
            if collection_key:
                # Name the scope that was applied. The bare message read as
                # "this tag matches nothing", which invites a retry without
                # collection_key — a library-wide search whose results look
                # like scoped ones (#418).
                return (
                    f"No items found with tag: '{tag}' in collection {collection_key}. "
                    f"The collection was searched and no item in it carries that tag. "
                    f"Items elsewhere in the library may still carry it; re-running "
                    f"without collection_key searches the whole library, not this collection."
                )
            return f"No items found with tag: '{tag}'"

        # Format results as markdown. State the scope in both directions, so a
        # library-wide result is never mistaken for a collection-scoped one.
        scope = (
            f" in Collection {collection_key}" if collection_key
            else " (entire library — no collection scope applied)"
        )
        output = [f"# Search Results for Tag: '{tag}'{scope}", ""]

        for i, item in enumerate(results, 1):
            output.extend(_utils.format_item_result(item, index=i))

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error searching Zotero: {str(e)}")
        return f"Error searching Zotero: {str(e)}"


@mcp.tool(
    name="zotero_search_by_citation_key",
    description=(
        "Look up a single Zotero item by its BetterBibTeX citation key "
        "(e.g. 'Smith2024' or 'cladderMicus2018'). Returns that one item's "
        "metadata, or a not-found message if no item has that key. "
        "citekey: the citation key exactly as assigned by BetterBibTeX "
        "(case-sensitive). "
        "In local mode: queries the running Better BibTeX plugin via its "
        "HTTP API (Zotero desktop must be running and have BBT installed). "
        "In web mode: scans the 'Extra' field of items for 'Citation Key:' "
        "lines — slower, and may miss items whose keys aren't persisted to "
        "Extra. "
        "Requires the Better BibTeX plugin in the user's Zotero install. "
        "For partial-key or free-text lookup, use zotero_search_items. "
        "Example: zotero_search_by_citation_key(citekey='hasan2026mcp') → "
        "metadata for that single item."
    )
)
@with_zotero_api_lock
def search_by_citation_key(
    citekey: str,
    *,
    ctx: Context
) -> str:
    """
    Look up a Zotero item by its BetterBibTeX citation key.

    Args:
        citekey: The BetterBibTeX citation key to search for (e.g., 'Smith2024')
        ctx: MCP context

    Returns:
        Formatted item details or error message
    """
    try:
        if not citekey.strip():
            return "Error: Citation key cannot be empty"

        citekey = citekey.strip()
        ctx.info(f"Looking up citation key: {citekey}")

        # Strategy A: pyzotero search across all fields, then verify via Extra.
        # Note: the previous BetterBibTeX ``item.search`` JSON-RPC call was
        # removed in #293 — that BBT method does not exist in current versions
        # (always returned -32601 Method not found) and the exception handler
        # silently fell through to the same Extra-field search, so the BBT
        # branch only added noise.
        zot = _client.get_zotero_client()
        zot.add_parameters(q=citekey, qmode="everything", itemType="-attachment", limit=25)
        results = zot.items()

        for item in results:
            data = item.get("data", {})
            extra = data.get("extra", "")
            if data.get("citationKey") == citekey or _helpers._extra_has_citekey(extra, citekey):
                return _helpers._format_citekey_result(item, citekey)

        return f"No item found with citation key: '{citekey}'"

    except Exception as e:
        ctx.error(f"Error looking up citation key: {str(e)}")
        return f"Error looking up citation key: {str(e)}"


@mcp.tool(
    name="zotero_advanced_search",
    description=(
        "Advanced item search with multiple structured-field conditions "
        "joined by AND or OR. Use this when you need to filter by fields "
        "that zotero_search_items and zotero_search_by_tag can't express "
        "(date ranges, specific itemTypes, etc.). "
        "For plain text use zotero_search_items; for tags use "
        "zotero_search_by_tag; for topic discovery use "
        "zotero_semantic_search. "
        "conditions: list of {field, operation, value} dicts (also accepts "
        "a JSON string). "
        "  Common fields: title, creator, date, dateAdded, dateModified, "
        "tag, itemType, publicationTitle, abstractNote, collection. "
        "  Supported operations (exhaustive): is, isNot, contains, "
        "doesNotContain, beginsWith, endsWith, isGreaterThan, isLessThan, "
        "isBefore, isAfter. "
        "For 'added in the last N days', use field='dateAdded' with "
        "operation='isAfter' and an ISO date value (e.g. '2026-03-22'). "
        "join_mode: 'all' (AND, default) or 'any' (OR). "
        "sort_by: dateAdded, dateModified, title, creator, etc. "
        "sort_direction: 'asc' (default) or 'desc'. "
        "limit: max results (default 50, max 500). "
        "include_subcollections: make a 'collection' condition match items "
        "anywhere in that collection's subtree, for the is/isNot operations "
        "(default False). "
        "Example: zotero_advanced_search(conditions=[{'field': 'itemType', "
        "'operation': 'is', 'value': 'preprint'}, {'field': 'dateAdded', "
        "'operation': 'isAfter', 'value': '2026-03-22'}], "
        "join_mode='all')."
    )
)
@with_zotero_api_lock
def advanced_search(
    conditions: list[dict[str, str]] | str,
    join_mode: Literal["all", "any"] = "all",
    sort_by: str | None = None,
    sort_direction: Literal["asc", "desc"] = "asc",
    limit: int | str = 50,
    include_subcollections: bool = False,
    *,
    ctx: Context
) -> str:
    """
    Perform an advanced search with multiple criteria.

    Args:
        conditions: List of search condition dictionaries, each containing:
                   - field: The field to search (title, creator, date, tag, etc.)
                   - operation: The operation to perform (is, isNot, contains, etc.)
                   - value: The value to search for
        join_mode: Whether all conditions must match ("all") or any condition can match ("any")
        sort_by: Field to sort by (dateAdded, dateModified, title, creator, etc.)
        sort_direction: Direction to sort (asc or desc)
        limit: Maximum number of results to return
        include_subcollections: Make a `collection` condition match items filed
            anywhere in that collection's subtree rather than in it directly.
            Applies to the `is` and `isNot` operations, which are the
            membership questions; other operators keep comparing keys as
            before. Defaults to False, matching Zotero's own "Search
            subcollections" checkbox.
        ctx: MCP context

    Returns:
        Markdown-formatted search results
    """
    try:
        # `| str` on the annotation is load-bearing, not documentation. Pydantic
        # validates against the published schema before this body runs, so with
        # `list[dict[str, str]]` alone a client that stringifies untyped
        # arguments was rejected at the boundary and this branch was dead code
        # — while the tool's own description promised "also accepts a JSON
        # string". Same failure as #459's `rect`, found by sweeping every tool
        # for list/dict params without string tolerance.
        if isinstance(conditions, str):
            try:
                conditions = json.loads(conditions)
            except json.JSONDecodeError as parse_error:
                return (
                    "Error: conditions must be valid JSON when provided as a string "
                    f"({parse_error})"
                )

        if not isinstance(conditions, list) or not conditions:
            return "Error: No search conditions provided"

        if join_mode not in {"all", "any"}:
            return "Error: join_mode must be either 'all' or 'any'"

        limit = _helpers._normalize_limit(limit, default=50, max_val=500)

        ctx.info(f"Performing advanced search with {len(conditions)} conditions")
        zot = _client.get_zotero_client()

        valid_operations = {
            "is",
            "isNot",
            "contains",
            "doesNotContain",
            "beginsWith",
            "endsWith",
            "isGreaterThan",
            "isLessThan",
            "isBefore",
            "isAfter",
        }

        parsed_conditions: list[dict[str, str]] = []
        for i, condition in enumerate(conditions, 1):
            if not isinstance(condition, dict):
                return f"Error: Condition {i} must be an object"
            if "field" not in condition or "operation" not in condition or "value" not in condition:
                return (
                    f"Error: Condition {i} is missing required fields "
                    "(field, operation, value)"
                )

            field = str(condition["field"]).strip()
            operation = str(condition["operation"]).strip()
            value = str(condition["value"]).strip()

            if operation not in valid_operations:
                return (
                    f"Error: Unsupported operation '{operation}' in condition {i}. "
                    f"Supported: {', '.join(sorted(valid_operations))}"
                )
            if not field:
                return f"Error: Condition {i} has an empty field"

            parsed_conditions.append(
                {"field": field, "operation": operation, "value": value}
            )

        # With subcollections requested, a `collection` condition stops being a
        # per-value comparison and becomes set membership: the item matches if
        # any collection it is filed in lies anywhere in the requested subtree.
        # Expanding the *condition value* once here costs one API round-trip
        # for the whole search, where expanding per item would cost one each.
        collection_scopes: dict[str, set[str]] = {}
        if include_subcollections:
            _coll_values = {
                c["value"] for c in parsed_conditions
                if c["field"].lower() in {"collection", "collections"}
            }
            if _coll_values:
                # One fetch for the whole search, however many collection
                # conditions it carries.
                _all_collections = _utils._paginate(zot.collections)
                for _value in _coll_values:
                    collection_scopes[_value] = set(
                        _helpers.collection_descendants(_all_collections, _value)
                    )

        def _extract_values(data: dict[str, object], field: str) -> list[str]:
            field_lower = field.lower()

            if field_lower in {"author", "authors", "creator", "creators"}:
                creators = data.get("creators", []) or []
                values: list[str] = []
                for creator in creators:
                    if not isinstance(creator, dict):
                        continue
                    if creator.get("firstName") or creator.get("lastName"):
                        full_name = " ".join(
                            [
                                str(creator.get("firstName", "")).strip(),
                                str(creator.get("lastName", "")).strip(),
                            ]
                        ).strip()
                        if full_name:
                            values.append(full_name)
                    if creator.get("name"):
                        values.append(str(creator.get("name", "")).strip())
                return values

            if field_lower in {"tag", "tags"}:
                tags = data.get("tags", []) or []
                values = []
                for tag in tags:
                    if isinstance(tag, dict) and tag.get("tag"):
                        values.append(str(tag.get("tag", "")).strip())
                return values

            if field_lower in {"collection", "collections"}:
                # Membership lives in data["collections"] (a list of keys);
                # data["collection"] does not exist, so the generic branch
                # below used to extract [""] and no collection condition could
                # ever match (#418). Direct membership only, matching Zotero's
                # own "Collection is X" with subcollections not included.
                collections = data.get("collections", []) or []
                keys = [str(k).strip() for k in collections if str(k).strip()]
                # An item in no collection must still satisfy `isNot`, so fall
                # back to a single empty value rather than an empty list (which
                # _matches_condition rejects outright).
                return keys or [""]

            if field_lower == "year":
                date_value = str(data.get("date", "")).strip()
                return [date_value[:4]] if len(date_value) >= 4 else []

            source_field = _semantics.FIELD_ALIASES.get(field_lower, field)
            raw_value = data.get(source_field, "")
            if raw_value is None:
                return []
            return [str(raw_value).strip()]

        def _matches_condition(data: dict[str, object], condition: dict[str, str]) -> bool:
            values = _extract_values(data, condition["field"])
            operation = condition["operation"]
            target = condition["value"]

            # Subtree membership, when asked for. Only is/isNot are membership
            # questions; the string operators keep comparing raw keys, which is
            # what they did before and is unaffected by nesting.
            is_collection_field = condition["field"].lower() in {"collection", "collections"}
            scope = collection_scopes.get(target) if is_collection_field else None
            if scope is not None and operation in {"is", "isNot"}:
                in_subtree = bool(set(values) & scope)
                return in_subtree if operation == "is" else not in_subtree

            # Everything else: the comparison lives in search_semantics so the
            # SQLite backend evaluates the identical rules — see that module's
            # docstring for what went wrong when they were stated twice.
            return _semantics.matches(values, target, operation)

        # #167: try the SQLite metadata backend first — it replaces the
        # client-side paging loop below entirely when it can serve the
        # query. None means "unsupported/unavailable"; fall back.
        #
        # A subtree-scoped collection condition is one of the things it
        # cannot serve: `advanced_search_sql` compares collection keys
        # directly, with no notion of include_subcollections, so it would
        # quietly answer a narrower question than the caller asked. Keep
        # that on the client-side path until the SQL translator grows
        # subtree membership of its own.
        results = None
        if _utils.get_search_backend() == "sqlite" and not collection_scopes:
            try:
                reader = get_local_zotero_reader()
                if reader is not None:
                    try:
                        results = reader.advanced_search_sql(
                            parsed_conditions, join_mode=join_mode,
                            group_id=_client.get_active_group_id(),
                        )
                    finally:
                        reader.close()
            except Exception as e:
                _search_logger.debug(f"[ADVANCED SEARCH] sqlite backend failed, falling back: {e}")
                results = None

        scan_warning: str | None = None
        if results is None:
            # Execute advanced search by iterating items and filtering
            # client-side. Three things bound that walk (#456). Without them a
            # single condition over a large library paged the *entire* library
            # 100 items at a time, holding the process-global Zotero API lock
            # the whole way: the call hit the client's 300s idle timeout and
            # every other tool queued behind it reported "Zotero API busy".
            #
            #  1. Conditions the Zotero API can evaluate itself are sent to it
            #     instead of being re-checked here, so the walk starts from a
            #     filtered set rather than from everything.
            #  2. With no sort requested, the walk stops as soon as it has
            #     `limit` matches -- the results are in library order either
            #     way, so nothing later in the library can displace them. A
            #     sort has to see every match before it can order them, so it
            #     does not get the early exit.
            #  3. Whatever is left is bounded by a deadline. Returning a
            #     partial answer that says it is partial beats returning
            #     nothing after five minutes.
            server_filters = _server_side_filters(parsed_conditions, join_mode)
            deadline = _time.monotonic() + _ADVANCED_SCAN_BUDGET_SECONDS
            can_stop_early = not sort_by

            results = []
            batch_size = 100
            start = 0
            while True:
                batch = zot.items(start=start, limit=batch_size, **server_filters)
                if not batch:
                    break

                for item in batch:
                    data = item.get("data", {})
                    if data.get("itemType") in {"attachment", "note", "annotation"}:
                        continue

                    checks = [_matches_condition(data, c) for c in parsed_conditions]
                    matched = all(checks) if join_mode == "all" else any(checks)
                    if matched:
                        results.append(item)

                if can_stop_early and len(results) >= limit:
                    break
                if len(batch) < batch_size:
                    break
                start += batch_size

                if _time.monotonic() > deadline:
                    scan_warning = (
                        f"Search stopped after {_ADVANCED_SCAN_BUDGET_SECONDS}s having "
                        f"examined {start} items; results below are partial. This "
                        f"backend filters client-side, so a broad condition over a "
                        f"large library has to read the library. Narrow the "
                        f"conditions, or set ZOTERO_SEARCH_BACKEND=sqlite (local "
                        f"mode) to evaluate the search in SQL instead."
                    )
                    break

        sort_warning: str | None = None
        if sort_by:
            sort_field = sort_by.strip()
            reverse = sort_direction == "desc"

            def _sort_key(item: dict[str, object]) -> str:
                data = item.get("data", {}) if isinstance(item, dict) else {}
                if sort_field in {"creator", "author"}:
                    return _utils.format_creators(data.get("creators", []))
                if sort_field in {"date", "year", "publicationDate"}:
                    # data["date"] is Zotero's *display* string ("October 1,
                    # 2016"), so sorting it lexically orders by month name.
                    # meta.parsedDate is the normalized ISO form the API
                    # computes for exactly this purpose.
                    meta = item.get("meta", {}) if isinstance(item, dict) else {}
                    parsed = str(meta.get("parsedDate", "") or "").strip()
                    if parsed:
                        return parsed
                    # No parsedDate (local API, sparse records): fall back to
                    # the first 4-digit year in the display string, which at
                    # least sorts by year instead of by month name.
                    match = re.search(r"\b(\d{4})\b", str(data.get("date", "")))
                    return match.group(1) if match else ""
                return str(data.get(sort_field, "")).lower()

            # A sort field absent from every result (misspelled, or simply not
            # present in this backend's item shape) used to sort every key to
            # "" and silently return library order as though it had been
            # honored. Say so instead (#418).
            if results and not any(_sort_key(item) for item in results):
                sort_warning = (
                    f"Requested sort by `{sort_field}` was not applied: no result carries "
                    f"that field. Results are in library order. Sortable fields include "
                    f"dateAdded, dateModified, title, date, creator."
                )
            else:
                results.sort(key=_sort_key, reverse=reverse)

        if not results:
            if scan_warning:
                # An empty result after a truncated scan is not the same claim
                # as an empty result after a complete one, and a caller that
                # cannot tell them apart will conclude the library has no such
                # items (#456).
                return (
                    "No items found matching the search criteria *in the portion "
                    f"of the library that was searched*.\n\n> **Note:** {scan_warning}"
                )
            return "No items found matching the search criteria."

        results = results[:limit]

        output = ["# Advanced Search Results", ""]
        output.append(f"Found {len(results)} items matching the search criteria:")
        output.append("")
        output.append("## Search Criteria")
        output.append(f"Join mode: {join_mode.upper()}")
        for i, condition in enumerate(parsed_conditions, 1):
            output.append(
                f"{i}. {condition['field']} {condition['operation']} \"{condition['value']}\""
            )
        if sort_warning:
            output.append("")
            output.append(f"> **Note:** {sort_warning}")
        if scan_warning:
            output.append("")
            output.append(f"> **Note:** {scan_warning}")
        output.append("")
        output.append("## Results")

        for i, item in enumerate(results, 1):
            output.extend(_utils.format_item_result(item, index=i))

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error in advanced search: {str(e)}")
        return f"Error in advanced search: {str(e)}"


@mcp.tool(
    name="zotero_semantic_search",
    description=(
        "Prioritized topic-search tool. Find papers by semantic similarity "
        "to a query using AI embeddings — the BEST tool for finding papers "
        "on a topic (e.g. 'papers about mindfulness-based therapy'), far "
        "more efficient than scanning collection items or reading "
        "abstracts. Searches across every indexed library by default (your "
        "personal library plus any group libraries that have been synced); "
        "use library_id to scope to one. "
        "query: the topic or concept; natural-language phrases work well. "
        "limit: max results (default 10). "
        "filters: optional metadata filters as a dict (e.g. "
        "{'itemType': 'journalArticle', 'year': '2023'}); also accepts a "
        "JSON string. "
        "library_id: optional — restrict results to one library. 0 or "
        "'user' for your personal library, or a group's numeric groupID "
        "(see zotero_list_libraries). Omit to search all indexed libraries. "
        "Requires the semantic search database to be POPULATED — run "
        "zotero_update_search_database first if you just installed the "
        "server or added new items; check readiness with "
        "zotero_get_search_database_status. "
        "Available only when the [semantic] optional dependency is "
        "installed. "
        "Example: zotero_semantic_search(query='mindfulness-based "
        "cognitive therapy for depression', limit=5)."
    )
)
@with_zotero_api_lock
def semantic_search(
    query: str,
    limit: int = 10,
    filters: dict[str, str] | str | None = None,
    library_id: int | str | None = None,
    *,
    ctx: Context
) -> str:
    """
    Perform semantic search over your Zotero library.

    Args:
        query: Search query text - can be concepts, topics, or natural language descriptions
        limit: Maximum number of results to return (default: 10)
        filters: Optional metadata filters as dict or JSON string. Example: {"item_type": "note"}
        library_id: Optional library scope — 0/"user" for the personal library, a
            groupID for a group library, or None (default) to search every
            indexed library.
        ctx: MCP context

    Returns:
        Markdown-formatted search results with similarity scores
    """
    try:
        if not query.strip():
            return "Error: Search query cannot be empty"

        try:
            group_id = _helpers._parse_library_id_param(library_id)
        except ValueError as e:
            return f"Error: {e}"

        # Parse and validate filters parameter
        if filters is not None:
            # Handle JSON string input
            if isinstance(filters, str):
                try:
                    filters = json.loads(filters)
                    ctx.info(f"Parsed JSON string filters: {filters}")
                except json.JSONDecodeError as e:
                    return f"Error: Invalid JSON in filters parameter: {str(e)}"

            # Validate it's a dictionary
            if not isinstance(filters, dict):
                return "Error: filters parameter must be a dictionary or JSON string. Example: {\"item_type\": \"note\"}"

            # Automatically translate common field names
            if "itemType" in filters:
                filters["item_type"] = filters.pop("itemType")
                ctx.info(f"Automatically translated 'itemType' to 'item_type': {filters}")

            # Additional field name translations can be added here
            # Example: if "creatorType" in filters:
            #     filters["creator_type"] = filters.pop("creatorType")

        ctx.info(f"Performing semantic search for: '{query}'")

        # Import semantic search module
        try:
            from zotero_mcp.semantic_search import create_semantic_search
        except ImportError:
            return (
                "Semantic search is not available.\n"
                f"{_utils.install_hint('semantic')}\n\n"
                "This installs chromadb, sentence-transformers, and related dependencies."
            )

        # Determine config path
        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"

        # Create semantic search instance
        search = create_semantic_search(str(config_path))

        # Fire-and-forget: if auto-update is due, kick off a background sync
        # so subsequent searches see fresh library state. Never blocks here.
        _maybe_fire_presearch_sync(search)

        # Perform search
        results = search.search(query=query, limit=limit, filters=filters, group_id=group_id)

        if results.get("error"):
            return f"Semantic search error: {results['error']}"

        search_results = results.get("results", [])

        if not search_results:
            return f"No semantically similar items found for query: '{query}'"

        # Format results as markdown
        output = [f"# Semantic Search Results for '{query}'", ""]
        output.append(f"Found {len(search_results)} similar items:")
        output.append("")

        for i, result in enumerate(search_results, 1):
            similarity_score = result.get("similarity_score", 0)
            zotero_item = result.get("zotero_item", {})

            # Prefer the grounded passage — the window of the document that
            # actually overlaps the query — over a blind head-truncation, so
            # the agent gets a citable quote rather than the abstract's opening.
            passage = result.get("matched_passage") or result.get("matched_text", "")
            snippet = passage[:400] + "..." if len(passage) > 400 else passage

            # Provenance for citing: page (when the index carries page breaks),
            # else which passage of how many, else an approximate char offset.
            loc_bits = []
            if (page := result.get("page")) is not None:
                loc_bits.append(f"p. {page}")
            if (ci := result.get("chunk_index")) is not None and (nc := result.get("n_chunks")):
                loc_bits.append(f"passage {ci + 1}/{nc}")
            elif off := result.get("char_start", result.get("passage_offset")):
                loc_bits.append(f"char ~{off}")

            if zotero_item:
                extra = {"Relevance": f"{similarity_score:.3f}"}
                if loc_bits:
                    extra["Location"] = ", ".join(loc_bits)
                if snippet:
                    extra["Matched Passage"] = snippet
                # Override key from result since it may differ from item["key"]
                zotero_item.setdefault("key", result.get("item_key", ""))
                output.extend(_utils.format_item_result(zotero_item, index=i, extra_fields=extra))
            else:
                # Fallback if full Zotero item not available
                output.append(f"## {i}. Item {result.get('item_key', 'Unknown')}")
                output.append(f"**Relevance:** {similarity_score:.3f}")
                if loc_bits:
                    output.append(f"**Location:** {', '.join(loc_bits)}")
                if snippet:
                    output.append(f"**Matched Passage:** {snippet}")
                if error := result.get("error"):
                    output.append(f"**Error:** {error}")
                output.append("")

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error in semantic search: {str(e)}")
        return f"Error in semantic search: {str(e)}"


@mcp.tool(
    name="zotero_update_search_database",
    description=(
        "Build or refresh the semantic search embedding database from "
        "Zotero items. Run this: (a) after first install, (b) after adding "
        "items via zotero_add_item, or "
        "(c) when the user has added items directly in Zotero desktop "
        "since the last update. "
        "By default the update is INCREMENTAL — only new or changed items "
        "are re-embedded, so repeated calls are cheap. "
        "force_rebuild=True re-embeds ALL items from scratch (slow; use "
        "when changing the embedding model or recovering from corruption). "
        "limit: optional cap on items processed (useful for smoke-testing). "
        "Progress is reported via the MCP context; on large libraries an "
        "incremental update is seconds, a full rebuild can take minutes. "
        "Requires the [semantic] optional dependency and a configured "
        "embedding provider (see config.json). Check status with "
        "zotero_get_search_database_status. "
        "Example: zotero_update_search_database() after adding a batch of "
        "papers."
    )
)
@with_zotero_api_lock
def update_search_database(
    force_rebuild: bool = False,
    limit: int | None = None,
    *,
    ctx: Context
) -> str:
    """
    Update the semantic search database.

    Args:
        force_rebuild: Whether to rebuild the entire database from scratch
        limit: Limit number of items to process (useful for testing)
        ctx: MCP context

    Returns:
        Update status and statistics
    """
    try:
        ctx.info("Starting semantic search database update...")

        # Import semantic search module
        try:
            from zotero_mcp.semantic_search import create_semantic_search
        except ImportError:
            return (
                "Semantic search is not available.\n"
                f"{_utils.install_hint('semantic')}\n\n"
                "This installs chromadb, sentence-transformers, and related dependencies."
            )

        # Determine config path
        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"

        # Create semantic search instance
        search = create_semantic_search(str(config_path))

        # Use fulltext extraction when in local mode (has access to PDFs)
        stats = search.update_database(
            force_full_rebuild=force_rebuild,
            limit=limit,
            extract_fulltext=_utils.is_local_mode()
        )

        # Format results
        output = ["# Database Update Results", ""]

        if stats.get("error"):
            output.append(f"**Error:** {stats['error']}")
        else:
            output.append(f"**Total items:** {stats.get('total_items', 0)}")
            output.append(f"**Processed:** {stats.get('processed_items', 0)}")
            output.append(f"**Added:** {stats.get('added_items', 0)}")
            output.append(f"**Updated:** {stats.get('updated_items', 0)}")
            output.append(f"**Skipped:** {stats.get('skipped_items', 0)}")
            output.append(f"**Errors:** {stats.get('errors', 0)}")
            output.append(f"**Duration:** {stats.get('duration', 'Unknown')}")

            if stats.get('start_time'):
                output.append(f"**Started:** {stats['start_time']}")
            if stats.get('end_time'):
                output.append(f"**Completed:** {stats['end_time']}")

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error updating search database: {str(e)}")
        return f"Error updating search database: {str(e)}"


@mcp.tool(
    name="zotero_get_search_database_status",
    description=(
        "Report the semantic search database's readiness and stats: item "
        "count, last update time, embedding provider / model, and whether "
        "the [semantic] optional dependency is installed. "
        "Use this to decide whether zotero_semantic_search will return "
        "useful results, or whether the user should run "
        "zotero_update_search_database first. "
        "Takes no parameters; no side effects. "
        "Returns a human-readable status block. If the [semantic] extras "
        "are not installed, returns an install hint instead of stats. "
        "Example: zotero_get_search_database_status() → count, last sync, "
        "provider summary."
    )
)
def get_search_database_status(*, ctx: Context) -> str:
    """
    Get semantic search database status.

    Deliberately NOT wrapped in ``@with_zotero_api_lock``: this is a read-only
    ChromaDB query that never touches the Zotero API, and holding the shared
    lock here would make a slow status read block every other tool. The read
    path below also avoids constructing the embedding function, which for the
    default backend downloads an ONNX model on first use and could otherwise
    hang this call for minutes.

    Args:
        ctx: MCP context

    Returns:
        Database status information
    """
    try:
        ctx.info("Getting semantic search database status...")

        # Import the lightweight, model-free status readers. These live in the
        # semantic-search modules so they share the [semantic] extra's import
        # guard, but neither loads an embedding model or a Zotero client.
        try:
            from zotero_mcp.chroma_client import read_collection_status
            from zotero_mcp.semantic_search import load_update_config, should_update
        except ImportError:
            return (
                "Semantic search is not available.\n"
                f"{_utils.install_hint('semantic')}\n\n"
                "This installs chromadb, sentence-transformers, and related dependencies."
            )

        # Determine config path
        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"

        # Read status without loading any embedding model (fast, no network).
        collection_info = read_collection_status(str(config_path))
        update_config = load_update_config(str(config_path))

        # Format results
        output = ["# Semantic Search Database Status", ""]

        output.append("## Collection Information")
        output.append(f"**Name:** {collection_info.get('name', 'Unknown')}")
        output.append(f"**Document Count:** {collection_info.get('count', 0)}")
        output.append(f"**Embedding Model:** {collection_info.get('embedding_model', 'Unknown')}")
        output.append(f"**Database Path:** {collection_info.get('persist_directory', 'Unknown')}")

        if collection_info.get("initialized") is False and not collection_info.get("error"):
            output.append("**Status:** Not initialized — run zotero_update_search_database first.")
        if collection_info.get('error'):
            output.append(f"**Error:** {collection_info['error']}")

        output.append("")

        output.append("## Update Configuration")
        output.append(f"**Auto Update:** {update_config.get('auto_update', False)}")
        output.append(f"**Frequency:** {update_config.get('update_frequency', 'manual')}")
        output.append(f"**Last Update:** {update_config.get('last_update', 'Never')}")
        output.append(f"**Should Update Now:** {should_update(update_config)}")

        frequency = update_config.get('update_frequency', 'manual')
        if frequency.startswith('every_') and update_config.get('update_days'):
            output.append(f"**Update Interval:** Every {update_config['update_days']} days")

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error getting database status: {str(e)}")
        return f"Error getting database status: {str(e)}"
