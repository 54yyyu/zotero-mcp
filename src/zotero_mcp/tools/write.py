"""Write / mutation tool functions for the Zotero MCP server."""

import json
import os
import re
import tempfile
import time as _time
import xml.etree.ElementTree as ET
from typing import Annotated, Literal

import requests
from pydantic import Field

from zotero_mcp import citation_import as _citation_import
from zotero_mcp import client as _client
from zotero_mcp import utils as _utils
from zotero_mcp._app import mcp
from zotero_mcp._context import Context
from zotero_mcp.client import with_zotero_api_lock
from zotero_mcp.tools import _helpers

# Accessed as _helpers.X so that monkeypatch/mock on the module attribute works.
CROSSREF_TYPE_MAP = _helpers.CROSSREF_TYPE_MAP


def _resolve_collections_arg(
    read_zot,
    collections,
    ctx,
    *,
    create_missing: bool = False,
    write_zot=None,
) -> list[str]:
    """Normalize the caller's ``collections`` argument and resolve every spec
    (key, name, or '/'-path) to a live collection key.

    Raises ValueError with a user-facing message on unknown or ambiguous
    specs — callers should fail the add *before* creating an item, so a typo
    can't produce an unfiled or invisibly-filed item.
    """
    specs = _helpers._normalize_str_list_input(collections, "collections")
    if not specs:
        return []
    return _helpers.resolve_collection_specs(
        read_zot, specs,
        create_missing=create_missing, write_zot=write_zot, ctx=ctx,
    )


def _collections_status(coll_keys: list[str], missing: list[str]) -> str:
    """Render the post-create collection-membership state for tool output."""
    if not coll_keys:
        return "My Library (no collection)"
    if missing:
        return (
            f"Filed in {sorted(set(coll_keys) - set(missing))}; "
            f"FAILED to file in {missing}"
        )
    return f"Filed in {coll_keys}"


_IF_EXISTS_VALUES = ("duplicate", "file", "skip")


def _converge_existing_item(write_zot, item, coll_keys, tags, ctx) -> dict:
    """Additively converge an existing item to the requested state.

    Adds the item to any of *coll_keys* it isn't in yet and attaches any of
    *tags* it doesn't carry yet. Never removes anything. Returns a summary
    dict: ``{"key", "title", "colls_added", "colls_already", "colls_failed",
    "tags_added", "tags_failed"}``.
    """
    item_key = item.get("key")
    data = item.get("data", {})
    title = data.get("title") or "(untitled)"

    current_colls = set(data.get("collections") or [])
    to_add = [k for k in coll_keys if k not in current_colls]
    already = [k for k in coll_keys if k in current_colls]

    tag_list = _helpers._normalize_str_list_input(tags, "tags")
    current_tags = {t.get("tag") for t in data.get("tags") or []}
    tags_to_add = [t for t in tag_list if t not in current_tags]

    tags_failed = False
    if tags_to_add:
        # Update tags first, on our fetched copy (current version); the
        # collection backstop below re-fetches, so it sees the new version.
        item["data"]["tags"] = (data.get("tags") or []) + [
            {"tag": t} for t in tags_to_add
        ]
        try:
            resp = write_zot.update_item(item)
            tags_failed = not _helpers._handle_write_response(resp, ctx)
        except Exception as e:
            tags_failed = True
            if ctx is not None:
                ctx.warning(f"Could not add tags to {item_key}: {e}")

    colls_failed = _helpers.ensure_collection_membership(
        write_zot, item_key, to_add, ctx=ctx
    )
    colls_added = [k for k in to_add if k not in colls_failed]

    return {
        "key": item_key,
        "title": title,
        "colls_added": colls_added,
        "colls_already": already,
        "colls_failed": colls_failed,
        "tags_added": [] if tags_failed else tags_to_add,
        "tags_failed": tags_failed,
    }


def _handle_existing_item(write_zot, existing, coll_keys, tags, if_exists,
                          matched_by, ctx) -> str:
    """Render the if_exists='file'/'skip' outcome for a single-item add tool.

    The report keeps the ``Item key: `KEY``` line that callers (and
    add_from_file's key extraction) rely on.
    """
    item = existing[0]
    item_key = item.get("key")
    title = item.get("data", {}).get("title") or "(untitled)"

    note = ""
    if len(existing) > 1:
        other = [i.get("key") for i in existing[1:]]
        note = (
            f"\n\nNote: {len(existing)} items match ({other} besides the one "
            "used); consider zotero_find_duplicates / zotero_merge_duplicates."
        )

    header = (
        f"Already in library: **{title}** (`{item_key}`, matched by {matched_by})\n\n"
        f"Item key: `{item_key}`\n"
    )

    if if_exists == "skip":
        return header + "No changes made (if_exists='skip')." + note

    summary = _converge_existing_item(write_zot, item, coll_keys, tags, ctx)

    lines = []
    coll_bits = []
    if summary["colls_added"]:
        coll_bits.append(f"added to {summary['colls_added']}")
    if summary["colls_failed"]:
        coll_bits.append(f"FAILED to add to {summary['colls_failed']}")
    if summary["colls_already"]:
        coll_bits.append(f"already in {summary['colls_already']}")
    if coll_keys:
        lines.append("Collections: " + "; ".join(coll_bits))
    if summary["tags_added"]:
        lines.append(f"Tags: added {summary['tags_added']}")
    elif summary["tags_failed"]:
        lines.append("Tags: FAILED to update")

    if not lines:
        lines.append("Nothing to change — item already in the requested state.")

    return header + "\n".join(lines) + note


@mcp.tool(
    name="zotero_batch_update_tags",
    description=(
        "Add and/or remove tags across multiple items in one call, selecting "
        "items by a text query, an existing tag, or both. "
        "Must supply at least one selector (query or tag) AND at least one "
        "action (add_tags or remove_tags) — otherwise returns an error. "
        "query: free-text matched against item metadata (title, creators, "
        "abstract, etc.) — same search as zotero_search_items. "
        "tag: filter to items already bearing this tag. When both are "
        "given, they are ANDed; pass tag as a list to OR multiple tags. "
        "add_tags, remove_tags: list of tag strings (or a JSON-encoded list "
        "string). Existing tags are preserved; this is not a replace-all. "
        "limit: max items to process (default 50). Attachments are "
        "auto-skipped. "
        "Requires a writable library (web API key or hybrid mode) — fails "
        "in local-only mode. Use zotero_get_tags to discover existing tag "
        "names first. "
        "Example: zotero_batch_update_tags(tag='to-read', "
        "add_tags=['reviewed'], remove_tags=['to-read'], limit=100) — "
        "mark everything tagged 'to-read' as 'reviewed'."
    )
)
@with_zotero_api_lock
def batch_update_tags(
    query: str = "",
    add_tags: list[str] | str | None = None,
    remove_tags: list[str] | str | None = None,
    tag: str | list[str] | None = None,
    limit: int | str = 50,
    *,
    ctx: Context
) -> str:
    """
    Batch update tags across multiple items matching a search query or tag filter.

    Args:
        query: Search query to find items to update (text search)
        add_tags: List of tags to add to matched items (can be list or JSON string)
        remove_tags: List of tags to remove from matched items (can be list or JSON string)
        tag: Filter by existing tag name (e.g., "test" finds items with that exact tag).
             When provided alongside query, both filters are applied (AND).
        limit: Maximum number of items to process
        ctx: MCP context

    Returns:
        Summary of the batch update
    """
    try:
        if not query and not tag:
            return "Error: Must provide a search query and/or tag filter"

        if not add_tags and not remove_tags:
            return "Error: You must specify either tags to add or tags to remove"

        try:
            add_tags = _helpers._normalize_str_list_input(add_tags, "add_tags")
            remove_tags = _helpers._normalize_str_list_input(remove_tags, "remove_tags")
        except ValueError as validation_error:
            return f"Error: {validation_error}"

        if not add_tags and not remove_tags:
            return "Error: After parsing, no valid tags were provided to add or remove"

        ctx.info(f"Batch updating tags for items matching '{query}'")
        zot = _client.get_zotero_client()

        # Use shared hybrid-mode helper for correct library override propagation
        try:
            _, write_zot = _helpers._get_write_client(ctx)
        except ValueError as e:
            return str(e)

        limit = _helpers._normalize_limit(limit, default=50)

        # Normalize tag parameter: accept string, list, or JSON string
        if tag is not None:
            if isinstance(tag, list):
                # Pyzotero expects comma-separated tags for AND filtering
                tag = " || ".join(str(t).strip() for t in tag if str(t).strip())
            elif isinstance(tag, str):
                tag = tag.strip()
                # Handle JSON string like '["test"]'
                try:
                    import json
                    parsed = json.loads(tag)
                    if isinstance(parsed, list):
                        tag = " || ".join(str(t).strip() for t in parsed if str(t).strip())
                    elif isinstance(parsed, str):
                        tag = parsed.strip()
                except (json.JSONDecodeError, ValueError):
                    pass  # Use as-is
            if not tag:
                tag = None

        # Search for items matching the query and/or tag filter
        params = {"limit": limit}
        if query:
            params["q"] = query
        if tag:
            params["tag"] = tag
        zot.add_parameters(**params)
        items = zot.items()

        if not items:
            filter_desc = []
            if query:
                filter_desc.append(f"query '{query}'")
            if tag:
                filter_desc.append(f"tag '{tag}'")
            return f"No items found matching {' and '.join(filter_desc) or 'the given filters'}"

        # Initialize counters
        updated_count = 0
        skipped_count = 0
        added_tag_counts = {tag: 0 for tag in (add_tags or [])}
        removed_tag_counts = {tag: 0 for tag in (remove_tags or [])}

        # Process each item
        for item in items:
            # Skip attachments if they were included in the results
            if item["data"].get("itemType") == "attachment":
                skipped_count += 1
                continue

            # Get current tags
            current_tags = item["data"].get("tags", [])
            current_tag_values = {t["tag"] for t in current_tags}

            # Track if this item needs to be updated
            needs_update = False

            # Process tags to remove
            if remove_tags:
                new_tags = []
                for tag_obj in current_tags:
                    tag = tag_obj["tag"]
                    if tag in remove_tags:
                        removed_tag_counts[tag] += 1
                        needs_update = True
                    else:
                        new_tags.append(tag_obj)
                current_tags = new_tags
                # Refresh the set of current tag values after removal
                current_tag_values = {t["tag"] for t in current_tags}

            # Process tags to add
            if add_tags:
                for tag in add_tags:
                    if tag and tag not in current_tag_values:
                        current_tags.append({"tag": tag})
                        added_tag_counts[tag] += 1
                        needs_update = True

            # Update the item if needed
            if needs_update:
                try:
                    item_key = item.get("key", "unknown")

                    # If writing via web API, re-fetch the item from web to get
                    # the correct version number for the update
                    if write_zot is not zot:
                        try:
                            web_item = write_zot.item(item_key)
                            web_item["data"]["tags"] = current_tags
                            ctx.info(f"Updating item {item_key} via web API with tags: {current_tags}")
                            result = write_zot.update_item(web_item)
                        except Exception as e:
                            ctx.error(f"Failed to fetch/update item {item_key} via web API: {str(e)}")
                            skipped_count += 1
                            continue
                    else:
                        item["data"]["tags"] = current_tags
                        ctx.info(f"Updating item {item_key} with tags: {current_tags}")
                        result = write_zot.update_item(item)

                    if _helpers._handle_write_response(result, ctx):
                        updated_count += 1
                    else:
                        ctx.error(f"Update may have failed for item {item_key}: {result}")
                        skipped_count += 1
                except Exception as e:
                    ctx.error(f"Failed to update item {item.get('key', 'unknown')}: {str(e)}")
                    # Continue with other items instead of failing completely
                    skipped_count += 1
            else:
                skipped_count += 1

        # Format the response
        response = ["# Batch Tag Update Results", ""]
        response.append(f"Query: '{query}'")
        response.append(f"Items processed: {len(items)}")
        response.append(f"Items updated: {updated_count}")
        response.append(f"Items skipped: {skipped_count}")

        if add_tags:
            response.append("\n## Tags Added")
            for tag, count in added_tag_counts.items():
                response.append(f"- `{tag}`: {count} items")

        if remove_tags:
            response.append("\n## Tags Removed")
            for tag, count in removed_tag_counts.items():
                response.append(f"- `{tag}`: {count} items")

        return "\n".join(response)

    except Exception as e:
        ctx.error(f"Error in batch tag update: {str(e)}")
        return f"Error in batch tag update: {str(e)}"


def _apply_extra_edits(
    extra: str,
    set_keys: dict[str, str],
    remove_keys: list[str],
    replace: bool,
) -> tuple[str, bool]:
    """Apply `Key: value` line edits to an Extra field value.

    Extra is treated as newline-separated lines; lines of the form
    "Key: value" are matched by the text before the first colon,
    case-insensitively. Free-form lines (no colon) are never touched.

    Returns:
        (new_extra, changed)
    """
    def line_key(line: str) -> str | None:
        head, sep, _ = line.partition(":")
        return head.strip().lower() if sep else None

    original = extra or ""

    if replace:
        new_extra = "\n".join(f"{k}: {v}" for k, v in set_keys.items())
        return new_extra, new_extra != original

    lines = original.splitlines()

    if remove_keys:
        remove = {k.strip().lower() for k in remove_keys if k.strip()}
        lines = [ln for ln in lines if line_key(ln) not in remove]

    for key, value in (set_keys or {}).items():
        target = key.strip().lower()
        new_line = f"{key}: {value}"
        out = []
        replaced = False
        for ln in lines:
            if line_key(ln) == target:
                # Replace the first matching line in place; drop duplicates.
                if not replaced:
                    out.append(new_line)
                    replaced = True
            else:
                out.append(ln)
        if not replaced:
            out.append(new_line)
        lines = out

    new_extra = "\n".join(lines)
    return new_extra, new_extra != original


@mcp.tool(
    name="zotero_batch_update_extra",
    description=(
        "Upsert and/or remove `Key: value` lines in the Extra field across "
        "multiple items in one call — the batch counterpart of "
        "zotero_update_item for Extra-field metadata (Better BibTeX "
        "citation keys, tex.* fields, CSL variables). "
        "item_keys: list of item keys to edit (or a JSON-encoded list "
        "string). set_keys: mapping of key→value lines to upsert (or a "
        "JSON object string); an existing line with the same key is "
        "replaced in place, otherwise the line is appended. "
        "remove_keys: list of key names whose lines are deleted. "
        "replace: when true, rebuild Extra from set_keys only, dropping "
        "every other line (incompatible with remove_keys). "
        "Keys are matched by their `key:` prefix, case-insensitively; "
        "free-form lines without a colon are preserved. Items needing no "
        "change, attachments/notes/annotations, and unknown keys are "
        "skipped (counted in the summary). "
        "Requires a writable library (web API key or hybrid mode) — fails "
        "in local-only mode. "
        "Example: zotero_batch_update_extra(item_keys=['ABCD1234', "
        "'EFGH5678'], set_keys={'tex.otscore': '2'}, "
        "remove_keys=['tex.draft'])."
    )
)
@with_zotero_api_lock
def batch_update_extra(
    item_keys: list[str] | str | None = None,
    set_keys: dict[str, str] | str | None = None,
    remove_keys: list[str] | str | None = None,
    replace: bool | str = False,
    *,
    ctx: Context
) -> str:
    """
    Batch update Extra-field key lines across multiple items.

    Args:
        item_keys: Item keys to edit (list or JSON-encoded list string)
        set_keys: Mapping of key→value lines to upsert (dict or JSON object string)
        remove_keys: Key names whose lines are deleted (list or JSON string)
        replace: When true, rebuild Extra from set_keys only
        ctx: MCP context

    Returns:
        Summary of the batch update
    """
    try:
        try:
            item_keys = _helpers._normalize_str_list_input(item_keys, "item_keys")
            remove_keys = _helpers._normalize_str_list_input(remove_keys, "remove_keys")
        except ValueError as validation_error:
            return f"Error: {validation_error}"

        if not item_keys:
            return "Error: Must provide item_keys to update"

        if isinstance(set_keys, str):
            try:
                set_keys = json.loads(set_keys)
            except json.JSONDecodeError:
                return "Error: set_keys must be a mapping of key→value strings"
        if set_keys is None:
            set_keys = {}
        if not isinstance(set_keys, dict):
            return "Error: set_keys must be a mapping of key→value strings"
        set_keys = {
            str(k).strip(): str(v).strip()
            for k, v in set_keys.items()
            if str(k).strip()
        }

        if isinstance(replace, str):
            replace = replace.strip().lower() in ("true", "1", "yes")
        replace = bool(replace)

        if not set_keys and not remove_keys and not replace:
            return "Error: Must specify set_keys, remove_keys, or replace"
        if replace and remove_keys:
            return "Error: replace=True is incompatible with remove_keys"

        ctx.info(f"Batch updating Extra field for {len(item_keys)} item(s)")
        zot = _client.get_zotero_client()

        try:
            _, write_zot = _helpers._get_write_client(ctx)
        except ValueError as e:
            return str(e)

        updated_count = 0
        skipped_count = 0

        for item_key in item_keys:
            try:
                item = zot.item(item_key)
            except Exception as e:
                ctx.error(f"Failed to fetch item {item_key}: {str(e)}")
                skipped_count += 1
                continue
            if not item:
                skipped_count += 1
                continue

            if item["data"].get("itemType") in ("attachment", "note", "annotation"):
                skipped_count += 1
                continue

            extra = item["data"].get("extra", "") or ""
            new_extra, changed = _apply_extra_edits(
                extra, set_keys, remove_keys, replace
            )
            if not changed:
                skipped_count += 1
                continue

            try:
                # If writing via web API, re-fetch the item from web to get
                # the correct version number for the update
                if write_zot is not zot:
                    web_item = write_zot.item(item_key)
                    web_item["data"]["extra"] = new_extra
                    result = write_zot.update_item(web_item)
                else:
                    item["data"]["extra"] = new_extra
                    result = write_zot.update_item(item)

                if _helpers._handle_write_response(result, ctx):
                    updated_count += 1
                else:
                    ctx.error(f"Update may have failed for item {item_key}: {result}")
                    skipped_count += 1
            except Exception as e:
                ctx.error(f"Failed to update item {item_key}: {str(e)}")
                skipped_count += 1

        response = ["# Batch Extra Update Results", ""]
        response.append(f"Items processed: {len(item_keys)}")
        response.append(f"Items updated: {updated_count}")
        response.append(f"Items skipped: {skipped_count}")

        if set_keys:
            response.append("\n## Keys Set")
            for key, value in set_keys.items():
                response.append(f"- `{key}: {value}`")
        if remove_keys:
            response.append("\n## Keys Removed")
            for key in remove_keys:
                response.append(f"- `{key}`")
        if replace:
            response.append("\nExtra field fully replaced from set_keys.")

        return "\n".join(response)

    except Exception as e:
        ctx.error(f"Error in batch extra update: {str(e)}")
        return f"Error in batch extra update: {str(e)}"


@mcp.tool(
    name="zotero_create_collection",
    description=(
        "Create a new collection (project/folder) in your Zotero library. "
        "To create a subcollection, pass parent_collection (not parent_key) as either "
        "a collection key (8-character string like 'KMMQDFQ4') or a collection name. "
        "Use zotero_search_collections to find collection keys."
    )
)
@with_zotero_api_lock
def create_collection(
    name: str,
    parent_collection: str | None = None,
    *,
    ctx: Context
) -> str:
    try:
        read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        ctx.info(f"Creating collection '{name}'")

        # Resolve parent_collection name if it doesn't look like a key
        parent_key = parent_collection
        if parent_collection and not re.match(r'^[A-Z0-9]{8}$', parent_collection):
            try:
                keys = _helpers._resolve_collection_names(read_zot, [parent_collection], ctx=ctx)
                parent_key = keys[0] if keys else None
            except ValueError as e:
                return f"Error resolving parent collection: {e}"

        coll_data = {"name": name}
        if parent_key:
            coll_data["parentCollection"] = parent_key
        else:
            coll_data["parentCollection"] = False

        result = write_zot.create_collections([coll_data])

        if isinstance(result, dict) and result.get("success"):
            coll_key = next(iter(result["success"].values()))
            parent_info = f" under parent '{parent_collection}'" if parent_collection else ""
            return (
                f"Successfully created collection \"{name}\"{parent_info}\n\n"
                f"Collection key: `{coll_key}`"
            )
        return f"Failed to create collection: {result}"

    except Exception as e:
        ctx.error(f"Error creating collection: {e}")
        return f"Error creating collection: {e}"


@mcp.tool(
    name="zotero_delete_collection",
    description=(
        "Delete a collection (folder) from your Zotero library by its "
        "8-character key. Items inside the collection are NOT deleted — they "
        "remain in the library (and in any other collections they belong to). "
        "Subcollections ARE deleted along with the parent. "
        "This is a hard delete — Zotero's API does not trash collections, so "
        "the operation cannot be undone via the API. Use "
        "zotero_search_collections to find the key first. "
        'Example: zotero_delete_collection(collection_key="KMMQDFQ4").'
    )
)
def delete_collection(
    collection_key: str,
    *,
    ctx: Context
) -> str:
    try:
        _read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        ctx.info(f"Deleting collection {collection_key}")

        try:
            coll = write_zot.collection(collection_key)
        except Exception as e:
            return f"Collection not found: `{collection_key}` ({e})"

        name = coll.get("data", {}).get("name", collection_key)
        resp = write_zot.delete_collection(coll)
        if _helpers._handle_write_response(resp, ctx):
            return f"Deleted collection \"{name}\" (`{collection_key}`)"
        return f"Failed to delete collection `{collection_key}`: {resp}"

    except Exception as e:
        ctx.error(f"Error deleting collection: {e}")
        return f"Error deleting collection: {e}"


@mcp.tool(
    name="zotero_search_collections",
    description=(
        "Search collections by name in the active library and return their "
        "8-character keys. Matching is case-insensitive substring and applies "
        "ONLY to the collection's own name — not to parent names, "
        "descriptions, or items inside the collection. "
        "Multi-word queries are ANDed across words (NOT OR-ed): query "
        "'reading list' matches only collections whose name contains both "
        "'reading' AND 'list'. To match either word, issue two separate "
        "searches. Leading/trailing whitespace is ignored and empty words "
        "are dropped. "
        "Returns the collection's key plus its parent (if any). "
        "include_trashed: when True, also match collections currently in "
        "the Zotero Trash (results annotated as such). Default False — "
        "trashed collections are otherwise invisible to automated clients. "
        "Performance: scans all collections in the active library (O(n)); "
        "for very large libraries expect a full-list pagination under the "
        "hood. "
        'Example: zotero_search_collections(query="orals") → keys for every '
        'collection with "orals" in its name.'
    )
)
@with_zotero_api_lock
def search_collections(
    query: str,
    include_trashed: bool = False,
    *,
    ctx: Context
) -> str:
    try:
        zot = _client.get_zotero_client()
        ctx.info(f"Searching collections for '{query}'")

        collections = _helpers._paginate(zot.collections)
        trashed_keys: set[str] = set()
        if include_trashed:
            trashed = _helpers.fetch_trashed_collections(zot)
            existing_keys = {c.get("key") for c in collections}
            for coll in trashed:
                key = coll.get("key")
                if key and key not in existing_keys:
                    trashed_keys.add(key)
                    collections.append(coll)
        if not collections:
            return "No collections found in your Zotero library."

        words = query.lower().split()
        matching = [
            c for c in collections
            if all(w in c.get("data", {}).get("name", "").lower() for w in words)
        ]

        if not matching:
            return f"No collections found matching '{query}'"

        lines = [f"# Collections matching '{query}'", ""]
        for i, coll in enumerate(matching, 1):
            name = coll["data"].get("name", "Unnamed")
            key = coll["key"]
            parent_key = coll["data"].get("parentCollection")
            trash_marker = " *[trashed]*" if key in trashed_keys else ""
            lines.append(f"## {i}. {name}{trash_marker}")
            lines.append(f"**Key:** `{key}`")
            if parent_key:
                try:
                    parent = zot.collection(parent_key)
                    lines.append(f"**Parent:** {parent['data'].get('name', parent_key)}")
                except Exception:
                    lines.append(f"**Parent key:** {parent_key}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        ctx.error(f"Error searching collections: {e}")
        return f"Error searching collections: {e}"


@mcp.tool(
    name="zotero_manage_collections",
    description=(
        "Add or remove one or more items from collections. "
        "item_keys must be an ARRAY of item keys, e.g. [\"KEY1\", \"KEY2\"] — not a single string. "
        "add_to and remove_from accept arrays of collection keys, names, or "
        "'/'-separated paths (resolved and validated automatically; unknown, "
        "trashed, or ambiguous specs fail before anything is changed). "
        "Use zotero_search_items to find item keys and zotero_search_collections to find collection keys."
    )
)
@with_zotero_api_lock
def manage_collections(
    item_keys: list[str] | str,
    add_to: list[str] | str | None = None,
    remove_from: list[str] | str | None = None,
    *,
    ctx: Context
) -> str:
    try:
        read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        keys = _helpers._normalize_str_list_input(item_keys, "item_keys")
        add_specs = _helpers._normalize_str_list_input(add_to, "add_to")
        remove_specs = _helpers._normalize_str_list_input(remove_from, "remove_from")

        if not keys:
            return "Error: No item keys provided."
        if not add_specs and not remove_specs:
            return "Error: Must specify add_to and/or remove_from."

        # Resolve specs (keys, names, or '/'-paths) to live collection keys
        # before doing any work. Resolution also validates existence — Zotero
        # will happily accept add/remove against a trashed collection, leaving
        # items parented under an invisible bucket so the caller sees
        # "success" but nothing renders in the desktop client (#233).
        try:
            add_colls = _helpers.resolve_collection_specs(
                read_zot, add_specs, ctx=ctx
            )
            remove_colls = _helpers.resolve_collection_specs(
                read_zot, remove_specs, ctx=ctx
            )
        except ValueError as e:
            return f"Error: {e}"

        results = []

        # Cache item fetches to avoid repeated API calls for the same key
        item_cache = {}
        def _get_item(key):
            if key not in item_cache:
                item_cache[key] = write_zot.item(key)
            return item_cache[key]

        for coll_key in add_colls:
            for item_key in keys:
                item_dict = _get_item(item_key)
                resp = write_zot.addto_collection(coll_key, item_dict)
                if _helpers._handle_write_response(resp, ctx):
                    results.append(f"Added {item_key} to {coll_key}")
                    # Invalidate cache — version changed after addto_collection
                    item_cache.pop(item_key, None)
                else:
                    results.append(f"Failed to add {item_key} to {coll_key}")

        for coll_key in remove_colls:
            for item_key in keys:
                item_dict = _get_item(item_key)
                resp = write_zot.deletefrom_collection(coll_key, item_dict)
                if _helpers._handle_write_response(resp, ctx):
                    results.append(f"Removed {item_key} from {coll_key}")
                    item_cache.pop(item_key, None)
                else:
                    results.append(f"Failed to remove {item_key} from {coll_key}")

        return "\n".join(results)

    except ValueError as e:
        return f"Input error: {e}"
    except Exception as e:
        ctx.error(f"Error managing collections: {e}")
        return f"Error managing collections: {e}"


@mcp.tool(
    name="zotero_add_by_doi",
    description=(
        "Add an item to the active Zotero library by DOI, resolving rich "
        "metadata (title, creators, journal, year, abstract) from "
        "CrossRef. "
        "Use this as the FIRST choice when the user gives you a DOI — "
        "cleaner metadata than zotero_add_by_url. For arXiv IDs or raw "
        "URLs use zotero_add_by_url; for a local PDF use "
        "zotero_add_from_file. "
        "doi: the DOI string (with or without the '10.' prefix, with or "
        "without a leading 'https://doi.org/'). "
        "collections: optional list of collection keys, names, or "
        "'/'-separated paths (e.g. '_project/topic') — resolved and "
        "validated before the item is created; unknown or ambiguous "
        "specs fail the call with suggestions instead of producing an "
        "unfiled item. "
        "tags: optional list of tag strings to attach. "
        "if_exists: 'duplicate' (default) always creates a new item; "
        "'file' makes the call idempotent — when an item with this DOI "
        "already exists it is reused, filed into any missing collections "
        "and given any missing tags (nothing is ever removed); 'skip' "
        "leaves an existing match untouched. "
        "create_missing_collections: when True, collection specs that "
        "don't resolve are created (including path chains) instead of "
        "failing. "
        "attach_mode: 'auto' (default) downloads a PDF if CrossRef links "
        "one and storage is available; 'none' skips PDF download; "
        "'required' fails if no PDF can be attached. PDF uploads may fail "
        "on the Zotero cloud free-tier 300MB quota — metadata still lands "
        "even when the upload fails. "
        "Requires a writable library (web API key or hybrid mode); fails "
        "in local-only mode. Remember to run zotero_update_search_database "
        "afterwards to make the new item searchable semantically. "
        "Example: zotero_add_by_doi(doi='10.1145/3708319', "
        "collections=['9SU943GB'], tags=['MCP'])."
    )
)
@with_zotero_api_lock
def add_by_doi(
    doi: str,
    collections: list[str] | str | None = None,
    tags: list[str] | str | None = None,
    attach_mode: str = "auto",
    if_exists: Literal["duplicate", "file", "skip"] = "duplicate",
    create_missing_collections: bool = False,
    *,
    ctx: Context
) -> str:
    try:
        read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        if if_exists not in _IF_EXISTS_VALUES:
            return f"Error: if_exists must be one of {_IF_EXISTS_VALUES}."
        normalized = _helpers._normalize_doi(doi)
        if not normalized:
            return f"Error: '{doi}' does not appear to be a valid DOI."

        # Resolve collection specs (keys/names/paths) BEFORE any network or
        # write work — a bad spec must not produce an unfiled item.
        try:
            coll_keys = _resolve_collections_arg(
                read_zot, collections, ctx,
                create_missing=create_missing_collections, write_zot=write_zot,
            )
        except ValueError as e:
            return f"Error: {e}"

        if if_exists != "duplicate":
            existing = _helpers.find_existing_items(read_zot, doi=normalized, ctx=ctx)
            if existing:
                return _handle_existing_item(
                    write_zot, existing, coll_keys, tags, if_exists,
                    matched_by=f"DOI {normalized}", ctx=ctx,
                )

        ctx.info(f"Fetching metadata for DOI: {normalized}")

        # CrossRef "polite pool": identifying via mailto gives higher rate limits
        # and priority routing. See https://api.crossref.org/swagger-ui/index.html
        crossref_url = f"https://api.crossref.org/works/{normalized}"
        contact_email = os.environ.get("ZOTERO_MCP_CONTACT_EMAIL", "").strip()
        if contact_email:
            crossref_url += f"?mailto={contact_email}"

        resp = requests.get(
            crossref_url,
            headers={
                "User-Agent": "zotero-mcp/1.0 (https://github.com/54yyyu/zotero-mcp)",
                "Accept": "application/json",
            },
            timeout=15,
        )

        if resp.status_code == 404:
            return f"DOI not found on CrossRef: {normalized}"
        resp.raise_for_status()

        cr = resp.json().get("message", {})

        # Determine Zotero item type
        cr_type = cr.get("type", "")
        zot_type = CROSSREF_TYPE_MAP.get(cr_type, "document")

        # Get valid fields from item template
        template = write_zot.item_template(zot_type)
        item_data = dict(template)

        # Map fields
        title_list = cr.get("title", [])
        if title_list and "title" in item_data:
            item_data["title"] = title_list[0]

        # Creators
        creators = []
        for author in cr.get("author", []):
            if "family" in author:
                creators.append({
                    "creatorType": "author",
                    "firstName": author.get("given", ""),
                    "lastName": author["family"],
                })
            elif "name" in author:
                creators.append({
                    "creatorType": "author",
                    "name": author["name"],
                })
        for editor in cr.get("editor", []):
            if "family" in editor:
                creators.append({
                    "creatorType": "editor",
                    "firstName": editor.get("given", ""),
                    "lastName": editor["family"],
                })
            elif "name" in editor:
                creators.append({
                    "creatorType": "editor",
                    "name": editor["name"],
                })
        if creators:
            item_data["creators"] = creators

        # Date
        date_parts = cr.get("published", cr.get("created", {})).get("date-parts", [[]])
        if date_parts and date_parts[0]:
            parts = date_parts[0]
            item_data["date"] = "-".join(str(p) for p in parts)

        # Simple string fields
        field_map = {
            "DOI": normalized,
            "url": cr.get("URL", ""),
            "volume": cr.get("volume", ""),
            "issue": cr.get("issue", ""),
            "pages": cr.get("page", ""),
            "publisher": cr.get("publisher", ""),
            "ISSN": (cr.get("ISSN") or [""])[0],
        }

        container = (cr.get("container-title") or [""])[0]
        if container:
            field_map["publicationTitle"] = container

        abstract = _utils.clean_html(cr.get("abstract", ""), collapse_whitespace=True)
        if abstract:
            field_map["abstractNote"] = abstract

        for field, value in field_map.items():
            if field in item_data and value:
                item_data[field] = value

        # Tags
        tag_list = _helpers._normalize_str_list_input(tags, "tags")
        if tag_list:
            item_data["tags"] = [{"tag": t} for t in tag_list]

        # Collections (resolved to live keys above, before the CrossRef fetch)
        if coll_keys:
            item_data["collections"] = coll_keys

        # Create item
        result = write_zot.create_items([item_data])

        if isinstance(result, dict) and result.get("success"):
            item_key = next(iter(result["success"].values()))
            title = item_data.get("title", normalized)

            # Defensive: pyzotero's atomic ``item["collections"]`` filing is
            # intermittent (#235) — reconcile membership before reporting success
            # so the caller sees the real routing state.
            missing = _helpers.ensure_collection_membership(
                write_zot, item_key, coll_keys, ctx=ctx
            )
            collections_status = _collections_status(coll_keys, missing)

            # Attempt open-access PDF attachment (pass CrossRef metadata for arXiv fallback)
            pdf_status = _helpers._try_attach_oa_pdf(write_zot, item_key, normalized, ctx,
                                            crossref_metadata=cr,
                                            attach_mode=attach_mode)

            return (
                f"Successfully added: **{title}**\n\n"
                f"Item key: `{item_key}`\n"
                f"Type: {zot_type}\n"
                f"DOI: {normalized}\n"
                f"Collections: {collections_status}\n"
                f"PDF: {pdf_status}\n\n"
                "_Note: To include this item in semantic search, run "
                "zotero_update_search_database._"
            )
        return f"Failed to create item: {result}"

    except requests.Timeout:
        return "Error: CrossRef API request timed out. Please try again."
    except requests.RequestException as e:
        return f"Error fetching from CrossRef: {e}"
    except Exception as e:
        ctx.error(f"Error adding by DOI: {e}")
        return f"Error adding by DOI: {e}"


@mcp.tool(
    name="zotero_add_by_url",
    description=(
        "Add an item to the active Zotero library from a URL. Routes by "
        "URL shape: doi.org/... → CrossRef metadata (same path as "
        "zotero_add_by_doi); arxiv.org/abs/... → arXiv metadata + PDF; "
        "anything else → webpage item (title + URL, minimal metadata). "
        "Prefer zotero_add_by_doi when you have a clean DOI — it skips "
        "the routing and is more robust. For a local file use "
        "zotero_add_from_file. "
        "url: the URL to import. "
        "collections: optional list of collection keys, names, or "
        "'/'-separated paths — resolved and validated before the item is "
        "created; unknown or ambiguous specs fail the call. "
        "tags: optional list of tag strings to attach. "
        "if_exists: 'duplicate' (default) always creates; 'file' reuses "
        "an existing item matching the arXiv ID / DOI / URL, filing it "
        "into missing collections and adding missing tags; 'skip' leaves "
        "a match untouched. create_missing_collections: create unknown "
        "collection specs instead of failing. "
        "attach_mode: 'auto' (default) attaches a PDF if one is "
        "available; 'none' skips; 'required' fails if no PDF can be "
        "attached. PDF uploads may fail on the Zotero cloud free-tier "
        "300MB quota — metadata still lands even when the upload fails. "
        "WARNING: for bibliography use, a general web-page URL produces "
        "a 'webpage' itemType that often isn't acceptable as a citation; "
        "resolve to a DOI and use zotero_add_by_doi instead when "
        "possible. "
        "Requires a writable library (fails in local-only mode). Run "
        "zotero_update_search_database afterwards for semantic search. "
        "Example: zotero_add_by_url(url='https://arxiv.org/abs/2602.14878', "
        "collections=['9SU943GB'])."
    )
)
@with_zotero_api_lock
def add_by_url(
    url: str,
    collections: list[str] | str | None = None,
    tags: list[str] | str | None = None,
    attach_mode: str = "auto",
    if_exists: Literal["duplicate", "file", "skip"] = "duplicate",
    create_missing_collections: bool = False,
    *,
    ctx: Context
) -> str:
    try:
        read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        if if_exists not in _IF_EXISTS_VALUES:
            return f"Error: if_exists must be one of {_IF_EXISTS_VALUES}."
        url = (url or "").strip()
        if not url:
            return "Error: No URL provided."

        # DOI URL routing
        doi = _helpers._normalize_doi(url)
        if doi:
            return add_by_doi(doi=url, collections=collections, tags=tags,
                              attach_mode=attach_mode, if_exists=if_exists,
                              create_missing_collections=create_missing_collections,
                              ctx=ctx)

        # arXiv URL routing
        arxiv_id = _helpers._normalize_arxiv_id(url)
        if arxiv_id:
            return _add_by_arxiv(arxiv_id, collections, tags, write_zot, ctx,
                                 attach_mode=attach_mode, read_zot=read_zot,
                                 if_exists=if_exists,
                                 create_missing_collections=create_missing_collections)

        # Generic webpage
        try:
            coll_keys = _resolve_collections_arg(
                read_zot, collections, ctx,
                create_missing=create_missing_collections, write_zot=write_zot,
            )
        except ValueError as e:
            return f"Error: {e}"

        if if_exists != "duplicate":
            existing = _helpers.find_existing_items(read_zot, url=url, ctx=ctx)
            if existing:
                return _handle_existing_item(
                    write_zot, existing, coll_keys, tags, if_exists,
                    matched_by=f"URL {url}", ctx=ctx,
                )

        ctx.info(f"Creating webpage item for: {url}")
        template = write_zot.item_template("webpage")
        template["url"] = url
        template["title"] = url
        template["accessDate"] = ""

        tag_list = _helpers._normalize_str_list_input(tags, "tags")
        if tag_list:
            template["tags"] = [{"tag": t} for t in tag_list]
        if coll_keys:
            template["collections"] = coll_keys

        result = write_zot.create_items([template])
        if isinstance(result, dict) and result.get("success"):
            item_key = next(iter(result["success"].values()))
            missing = _helpers.ensure_collection_membership(
                write_zot, item_key, coll_keys, ctx=ctx
            )
            return (
                f"Created webpage item for: {url}\n\nItem key: `{item_key}`\n"
                f"Collections: {_collections_status(coll_keys, missing)}\n\n"
                "_Note: To include this item in semantic search, run "
                "zotero_update_search_database._"
            )
        return f"Failed to create item: {result}"

    except Exception as e:
        ctx.error(f"Error adding by URL: {e}")
        return f"Error adding by URL: {e}"


@with_zotero_api_lock
def _add_by_arxiv(arxiv_id, collections, tags, write_zot, ctx, attach_mode="auto",
                  read_zot=None, if_exists="duplicate",
                  create_missing_collections=False):
    """Add an arXiv paper by ID. Internal helper for add_by_url.

    arXiv (export.arxiv.org) periodically sheds load — rate-limiting (429),
    returning 5xx, or timing out outright. This helper degrades gracefully:
    it retries transient failures with backoff, and if arXiv stays
    unreachable it falls back to CrossRef via the arXiv DOI
    (10.48550/arXiv.{id}), which serves the same metadata from independent
    infrastructure. The fallback is best-effort — CrossRef may also lack a
    very recent preprint — so a clear, actionable message is returned when
    both routes fail, never a bare timeout.
    """
    try:
        coll_keys = _resolve_collections_arg(
            read_zot or write_zot, collections, ctx,
            create_missing=create_missing_collections, write_zot=write_zot,
        )
    except ValueError as e:
        return f"Error: {e}"

    if if_exists != "duplicate":
        existing = _helpers.find_existing_items(
            read_zot or write_zot, arxiv_id=arxiv_id, ctx=ctx
        )
        if existing:
            return _handle_existing_item(
                write_zot, existing, coll_keys, tags, if_exists,
                matched_by=f"arXiv ID {arxiv_id}", ctx=ctx,
            )

    ctx.info(f"Fetching arXiv metadata for: {arxiv_id}")

    resp = None
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.get(
                f"https://export.arxiv.org/api/query?id_list={arxiv_id}",
                timeout=20,
            )
        except requests.RequestException as e:
            # Timeout / connection error — the classic "arXiv is overloaded"
            # symptom. Retry with backoff rather than failing on the first miss.
            last_error = e
            resp = None
            if attempt < 2:
                wait = 3 * (2 ** attempt)  # 3s, 6s
                ctx.info(
                    f"arXiv API unreachable ({e}); retrying in {wait}s "
                    f"({attempt + 1}/3)..."
                )
                _time.sleep(wait)
            continue
        # Retry rate-limits and server-side errors; 4xx (except 429) won't heal.
        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = f"HTTP {resp.status_code}"
            if attempt < 2:
                wait = 5 * (2 ** attempt)  # 5s, 10s
                ctx.info(
                    f"arXiv API returned {resp.status_code}; retrying in {wait}s "
                    f"({attempt + 1}/3)..."
                )
                _time.sleep(wait)
            continue
        break

    # arXiv exhausted its retries — fall back to CrossRef (independent infra).
    if resp is None or resp.status_code == 429 or resp.status_code >= 500:
        ctx.info(
            f"arXiv unreachable after retries ({last_error}); "
            f"falling back to CrossRef via the arXiv DOI."
        )
        arxiv_doi = f"10.48550/arXiv.{arxiv_id}"
        try:
            result = add_by_doi(
                doi=arxiv_doi,
                collections=coll_keys,
                tags=tags,
                attach_mode=attach_mode,
                if_exists=if_exists,
                ctx=ctx,
            )
        except Exception as e:  # noqa: BLE001 — fallback must not raise
            result = None
            ctx.info(f"CrossRef fallback errored: {e}")
        # add_by_doi returns a human string; treat "not found"/"Error" as a miss.
        if result and not result.startswith(("DOI not found", "Error")):
            return result
        return (
            f"arXiv is currently unreachable (last error: {last_error}) and the "
            f"CrossRef fallback (DOI {arxiv_doi}) did not resolve it — this is "
            f"often a transient arXiv overload. Please retry shortly. "
            f"(arXiv ID: {arxiv_id})"
        )
    try:
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"arXiv API error for {arxiv_id}: {e}"

    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    entries = root.findall("atom:entry", ns)
    if not entries:
        return f"No arXiv paper found for ID: {arxiv_id}"

    entry = entries[0]

    # Check for error response
    id_elem = entry.find("atom:id", ns)
    if id_elem is not None and "api/errors" in (id_elem.text or ""):
        return f"arXiv API error for ID: {arxiv_id}"

    title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
    abstract = (entry.findtext("atom:summary", "", ns) or "").strip()
    published = (entry.findtext("atom:published", "", ns) or "")[:10]

    authors = []
    for author_elem in entry.findall("atom:author", ns):
        name = (author_elem.findtext("atom:name", "", ns) or "").strip()
        if name:
            parts = name.rsplit(" ", 1)
            if len(parts) == 2:
                authors.append({
                    "creatorType": "author",
                    "firstName": parts[0],
                    "lastName": parts[1],
                })
            else:
                authors.append({"creatorType": "author", "name": name})

    template = write_zot.item_template("preprint")
    template["title"] = title
    if authors:
        template["creators"] = authors
    if abstract and "abstractNote" in template:
        template["abstractNote"] = abstract
    if published and "date" in template:
        template["date"] = published
    template["url"] = f"https://arxiv.org/abs/{arxiv_id}"
    if "extra" in template:
        template["extra"] = f"arXiv:{arxiv_id}"

    tag_list = _helpers._normalize_str_list_input(tags, "tags")
    if tag_list:
        template["tags"] = [{"tag": t} for t in tag_list]
    if coll_keys:
        template["collections"] = coll_keys

    result = write_zot.create_items([template])
    if isinstance(result, dict) and result.get("success"):
        item_key = next(iter(result["success"].values()))
        missing = _helpers.ensure_collection_membership(
            write_zot, item_key, coll_keys, ctx=ctx
        )

        # arXiv always has a free PDF — try to attach it
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_status = "no PDF attached"
        if attach_mode == "none":
            # Honour the caller's explicit opt-out: skip the PDF download/upload
            # entirely. Without this, the arXiv path always fetched + uploaded
            # the PDF regardless of attach_mode (only "linked_url" was special-
            # cased), so attach_mode="none" did far more network/cloud work than
            # asked — a slow upload here is a prime candidate for wedging the
            # process under the global API lock.
            pdf_status = "skipped (attach_mode=none)"
        elif attach_mode == "linked_url":
            # Bookmark the PDF URL only — no binary upload. Useful for users who
            # sync attachment files outside of Zotero's official storage (e.g. WebDAV).
            try:
                if _helpers._attach_pdf_linked_url(write_zot, pdf_url, item_key, ctx):
                    pdf_status = "PDF linked (URL only, no upload)"
                else:
                    pdf_status = "linked URL attachment failed"
            except Exception as e:
                ctx.info(f"arXiv linked URL attachment failed (non-fatal): {e}")
                pdf_status = f"no PDF attached ({e})"
        else:
            try:
                pdf_resp = requests.get(pdf_url, timeout=30, stream=True)
                pdf_resp.raise_for_status()
                with tempfile.TemporaryDirectory() as tmpdir:
                    filename = f"arxiv_{arxiv_id.replace('/', '_')}.pdf"
                    filepath = os.path.join(tmpdir, filename)
                    with open(filepath, "wb") as f:
                        for chunk in pdf_resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    attach_result = write_zot.attachment_both(
                        [(filename, filepath)],
                        parentid=item_key,
                    )
                    # Must run inside the with-block — temp file disappears on exit.
                    webdav_suffix = _helpers._maybe_upload_to_webdav(
                        attach_result, filepath, ctx
                    )
                pdf_status = "PDF attached" + webdav_suffix
            except Exception as e:
                ctx.info(f"arXiv PDF attachment failed (non-fatal): {e}")
                pdf_status = f"no PDF attached ({e})"

        return (
            f"Successfully added arXiv paper: **{title}**\n\n"
            f"Item key: `{item_key}`\n"
            f"arXiv ID: {arxiv_id}\n"
            f"Collections: {_collections_status(coll_keys, missing)}\n"
            f"PDF: {pdf_status}\n\n"
            "_Note: To include this item in semantic search, run "
            "zotero_update_search_database._"
        )
    return f"Failed to create arXiv item: {result}"


# ---------------------------------------------------------------------------
# ISBN lookup — Open Library (primary) + Google Books (fallback) (#226)
# ---------------------------------------------------------------------------

def _lookup_isbn_openlibrary(isbn, ctx):
    """Look up book metadata by ISBN on Open Library. Returns a dict of
    normalized fields, or None on miss / error. Network errors are logged
    and surfaced as None so the caller can fall through to Google Books.
    """
    try:
        url = (
            f"https://openlibrary.org/api/books"
            f"?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
        )
        resp = requests.get(
            url,
            headers={"User-Agent": "zotero-mcp/1.0 (https://github.com/54yyyu/zotero-mcp)"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json() or {}
        record = payload.get(f"ISBN:{isbn}") or {}
        if not record:
            return None

        title = record.get("title", "")
        if record.get("subtitle"):
            title = f"{title}: {record['subtitle']}"

        creators = []
        for author in record.get("authors", []) or []:
            name = (author.get("name") or "").strip()
            if not name:
                continue
            parts = name.rsplit(" ", 1)
            if len(parts) == 2:
                creators.append({
                    "creatorType": "author",
                    "firstName": parts[0],
                    "lastName": parts[1],
                })
            else:
                creators.append({"creatorType": "author", "name": name})

        publisher = ""
        publishers = record.get("publishers") or []
        if publishers:
            publisher = (publishers[0].get("name") or "").strip()

        place = ""
        places = record.get("publish_places") or []
        if places:
            place = (places[0].get("name") or "").strip()

        return {
            "source": "Open Library",
            "title": title,
            "creators": creators,
            "date": (record.get("publish_date") or "").strip(),
            "publisher": publisher,
            "place": place,
            "num_pages": str(record.get("number_of_pages", "") or "").strip(),
            "url": (record.get("url") or "").strip(),
        }
    except requests.RequestException as e:
        ctx.info(f"Open Library lookup failed (non-fatal): {e}")
        return None
    except Exception as e:
        ctx.info(f"Open Library parse failed (non-fatal): {e}")
        return None


def _lookup_isbn_google_books(isbn, ctx):
    """Look up book metadata by ISBN on Google Books. Returns a dict of
    normalized fields, or None on miss / error."""
    try:
        url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
        resp = requests.get(
            url,
            headers={"User-Agent": "zotero-mcp/1.0 (https://github.com/54yyyu/zotero-mcp)"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        payload = resp.json() or {}
        items = payload.get("items") or []
        if not items:
            return None
        info = items[0].get("volumeInfo") or {}

        title = info.get("title", "")
        if info.get("subtitle"):
            title = f"{title}: {info['subtitle']}"

        creators = []
        for name in info.get("authors", []) or []:
            name = (name or "").strip()
            if not name:
                continue
            parts = name.rsplit(" ", 1)
            if len(parts) == 2:
                creators.append({
                    "creatorType": "author",
                    "firstName": parts[0],
                    "lastName": parts[1],
                })
            else:
                creators.append({"creatorType": "author", "name": name})

        return {
            "source": "Google Books",
            "title": title,
            "creators": creators,
            "date": (info.get("publishedDate") or "").strip(),
            "publisher": (info.get("publisher") or "").strip(),
            "place": "",  # Google Books doesn't expose publication place
            "num_pages": str(info.get("pageCount", "") or "").strip(),
            "url": (info.get("infoLink") or info.get("canonicalVolumeLink") or "").strip(),
        }
    except requests.RequestException as e:
        ctx.info(f"Google Books lookup failed (non-fatal): {e}")
        return None
    except Exception as e:
        ctx.info(f"Google Books parse failed (non-fatal): {e}")
        return None


@mcp.tool(
    name="zotero_add_by_isbn",
    description=(
        "Add a book to your Zotero library by ISBN. Resolves metadata via "
        "Open Library (primary) and Google Books (fallback). Accepts ISBN-10, "
        "ISBN-13, with or without hyphens, or a URL/isbn: prefix. Response "
        "includes the resolver source so you can audit metadata quality. "
        "collections accepts keys, names, or '/'-paths (validated before "
        "create). if_exists: 'duplicate' (default) | 'file' (reuse an "
        "existing item with this ISBN — add missing collections/tags) | "
        "'skip'. create_missing_collections: create unknown collection "
        "specs instead of failing."
    )
)
def add_by_isbn(
    isbn: str,
    collections: list[str] | str | None = None,
    tags: list[str] | str | None = None,
    if_exists: Literal["duplicate", "file", "skip"] = "duplicate",
    create_missing_collections: bool = False,
    *,
    ctx: Context
) -> str:
    try:
        read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        if if_exists not in _IF_EXISTS_VALUES:
            return f"Error: if_exists must be one of {_IF_EXISTS_VALUES}."
        normalized = _helpers._normalize_isbn(isbn)
        if not normalized:
            return (
                f"Error: '{isbn}' does not appear to be a valid ISBN "
                "(checksum failed or wrong length)."
            )

        try:
            coll_keys = _resolve_collections_arg(
                read_zot, collections, ctx,
                create_missing=create_missing_collections, write_zot=write_zot,
            )
        except ValueError as e:
            return f"Error: {e}"

        if if_exists != "duplicate":
            existing = _helpers.find_existing_items(read_zot, isbn=normalized, ctx=ctx)
            if existing:
                return _handle_existing_item(
                    write_zot, existing, coll_keys, tags, if_exists,
                    matched_by=f"ISBN {normalized}", ctx=ctx,
                )

        ctx.info(f"Resolving ISBN {normalized} via Open Library...")
        meta = _lookup_isbn_openlibrary(normalized, ctx)
        if not meta:
            ctx.info("Open Library miss — falling back to Google Books...")
            meta = _lookup_isbn_google_books(normalized, ctx)
        if not meta:
            return (
                f"ISBN not found on Open Library or Google Books: {normalized}"
            )

        # Build Zotero book item
        template = write_zot.item_template("book")
        item_data = dict(template)
        if meta.get("title"):
            item_data["title"] = meta["title"]
        if meta.get("creators"):
            item_data["creators"] = meta["creators"]
        if meta.get("date") and "date" in item_data:
            item_data["date"] = meta["date"]
        if meta.get("publisher") and "publisher" in item_data:
            item_data["publisher"] = meta["publisher"]
        if meta.get("place") and "place" in item_data:
            item_data["place"] = meta["place"]
        if meta.get("num_pages") and "numPages" in item_data:
            item_data["numPages"] = meta["num_pages"]
        if meta.get("url") and "url" in item_data:
            item_data["url"] = meta["url"]
        if "ISBN" in item_data:
            item_data["ISBN"] = normalized

        tag_list = _helpers._normalize_str_list_input(tags, "tags")
        if tag_list:
            item_data["tags"] = [{"tag": t} for t in tag_list]
        if coll_keys:
            item_data["collections"] = coll_keys

        result = write_zot.create_items([item_data])
        if isinstance(result, dict) and result.get("success"):
            item_key = next(iter(result["success"].values()))
            missing = _helpers.ensure_collection_membership(
                write_zot, item_key, coll_keys, ctx=ctx
            )
            return (
                f"Successfully added: **{item_data.get('title', normalized)}**\n\n"
                f"Item key: `{item_key}`\n"
                f"Type: book\n"
                f"ISBN: {normalized}\n"
                f"Collections: {_collections_status(coll_keys, missing)}\n"
                f"Source: {meta['source']}\n\n"
                "_Note: Open Library and Google Books metadata can be noisy "
                "(publisher-as-author, concatenated places, off-by-one dates). "
                "Verify via `zotero_get_item_metadata` after creation. "
                "Run `zotero_update_search_database` to include this item "
                "in semantic search._"
            )
        return f"Failed to create item: {result}"

    except Exception as e:
        ctx.error(f"Error adding by ISBN: {e}")
        return f"Error adding by ISBN: {e}"


# Maps Zotero API field names to tool parameter names for user-facing messages
_UPDATE_ITEM_API_TO_PARAM = {
    "title": "title",
    "date": "date",
    "accessDate": "access_date",
    "publicationTitle": "publication_title",
    "abstractNote": "abstract",
    "DOI": "doi",
    "url": "url",
    "extra": "extra",
    "volume": "volume",
    "issue": "issue",
    "pages": "pages",
    "publisher": "publisher",
    "place": "place",
    "ISSN": "issn",
    "language": "language",
    "shortTitle": "short_title",
    "edition": "edition",
    "ISBN": "isbn",
    "bookTitle": "book_title",
    "citationKey": "citation_key",
}


@mcp.tool(
    name="zotero_update_item",
    description=(
        "Update metadata on an existing Zotero item by key. Only fields "
        "you pass are modified; unspecified fields are left alone. "
        "TAG SEMANTICS (easy to get wrong): `tags` REPLACES the entire "
        "tag list. To add tags without touching existing ones, use "
        "`add_tags`. To remove specific tags, use `remove_tags`. These "
        "three are mutually exclusive — prefer `add_tags`/`remove_tags` "
        "for incremental edits. "
        "Similarly, collections/collection_names REPLACE the item's "
        "collection memberships (pass collections=[] to clear all "
        "memberships); for incremental moves use "
        "zotero_manage_collections instead. "
        "item_key: 8-character Zotero item key of the item to update. "
        "Editable fields include: title, creators, date, publisher, place, "
        "publication_title, volume, issue, pages, DOI, ISBN, ISSN, url, "
        "language, abstract, short_title, edition, book_title, extra, "
        "citation_key, item_type. "
        "To migrate an item across types (e.g., journalArticle → book), pass item_type "
        "with a valid Zotero item-type vocabulary value; overlapping fields are preserved "
        "and type-specific fields that do not map to the target type are dropped. "
        "Requires a writable library (web API key or hybrid mode); fails "
        "in local-only mode. To edit notes use zotero_update_note, not "
        "this. "
        "Example: zotero_update_item(item_key='RTKZQI8E', "
        "add_tags=['reviewed'], doi='10.1145/3708319')."
    )
)
@with_zotero_api_lock
def update_item(
    item_key: str,
    title: str | None = None,
    creators: list[dict] | str | None = None,
    date: str | None = None,
    access_date: str | None = None,
    publication_title: str | None = None,
    abstract: str | None = None,
    tags: list[str] | str | None = None,
    add_tags: list[str] | str | None = None,
    remove_tags: list[str] | str | None = None,
    collections: list[str] | str | None = None,
    collection_names: list[str] | str | None = None,
    doi: str | None = None,
    url: str | None = None,
    extra: str | None = None,
    volume: str | None = None,
    issue: str | None = None,
    pages: str | None = None,
    publisher: str | None = None,
    place: Annotated[
        str | None,
        Field(description="Publication place (city), e.g., 'New York' or 'Cambridge, MA'."),
    ] = None,
    issn: str | None = None,
    language: str | None = None,
    short_title: str | None = None,
    edition: str | None = None,
    isbn: str | None = None,
    book_title: str | None = None,
    citation_key: Annotated[
        str | None,
        Field(description="BetterBibTeX / Zotero native citation key. Writes to data.citationKey. Useful when BBT auto-pinned the key from incomplete metadata and the programmatic refresh path is blocked (see https://github.com/retorquere/zotero-better-bibtex/issues/3522)."),
    ] = None,
    item_type: str | None = None,
    *,
    ctx: Context
) -> str:
    """
    Update metadata fields on an existing Zotero item.

    Only fields you pass are modified; unspecified fields are left
    untouched. Fields whose API key does not exist on the item's
    itemType (e.g. ``place`` on a ``journalArticle``) are reported as
    skipped rather than written.

    Args:
        item_key: 8-character Zotero item key of the item to update.
        title, creators, date, publication_title, abstract, doi, url,
        extra, volume, issue, pages, publisher, place, issn, language,
        short_title, edition, isbn, book_title, citation_key: per-field
        overrides; ``place`` is the publication city (e.g. ``"New York"``
        or ``"Cambridge, MA"``) and is valid on book, bookSection,
        thesis, manuscript, report, and conferencePaper item types.
        ``citation_key`` writes Zotero's native ``data.citationKey``
        (the BetterBibTeX citation key); BBT auto-pins from metadata on
        creation and provides no programmatic refresh path in 9.x, so
        direct write here is the only programmatic remediation for
        malformed pinned keys.
        tags / add_tags / remove_tags: mutually exclusive; ``tags``
        REPLACES the full tag list, ``add_tags`` / ``remove_tags`` are
        incremental. Prefer the incremental forms.
        collections / collection_names: REPLACE collection memberships;
        for incremental moves use zotero_manage_collections instead.
        ctx: MCP context.

    Returns:
        A markdown-formatted summary of what changed (or a skip
        warning for fields not valid on the item type).
    """
    try:
        read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        # Mutual exclusivity check
        if tags is not None and (add_tags is not None or remove_tags is not None):
            return (
                "Error: Cannot use 'tags' (replace all) together with "
                "'add_tags'/'remove_tags' (incremental). Use one approach or the other."
            )

        ctx.info(f"Updating item {item_key}")

        # Fetch current item from write client for correct version
        item = _helpers._strip_unwritable_fields(write_zot.item(item_key))
        data = item.get("data", {})
        changes = []

        # Handle item_type migration first so subsequent field updates are
        # validated against the NEW type's schema. Reshape by merging old
        # data into the new type's template: overlapping typed fields are
        # preserved; type-specific fields not present in the new template
        # are dropped; internal bookkeeping fields (key, version, tags,
        # collections, relations, creators, dateAdded, dateModified) are
        # always preserved regardless of type.
        if item_type is not None:
            old_item_type = data.get("itemType", "")
            if old_item_type != item_type:
                try:
                    new_template = write_zot.item_template(item_type)
                except Exception as e:
                    return f"Error: invalid item_type '{item_type}': {e}"

                preserved = {"key", "version", "tags", "collections",
                             "relations", "creators", "dateAdded",
                             "dateModified"}
                reshaped = dict(new_template)
                for k, v in data.items():
                    if k in preserved or k in new_template:
                        reshaped[k] = v
                reshaped["itemType"] = item_type
                data = reshaped
                item["data"] = data
                changes.append(
                    f"- **item_type**: '{old_item_type}' -> '{item_type}'"
                )

        # Apply field updates
        field_updates = {}
        if title is not None:
            field_updates["title"] = title
        if date is not None:
            field_updates["date"] = date
        if access_date is not None:
            field_updates["accessDate"] = access_date
        if publication_title is not None:
            field_updates["publicationTitle"] = publication_title
        if abstract is not None:
            field_updates["abstractNote"] = abstract
        if doi is not None:
            field_updates["DOI"] = doi
        if url is not None:
            field_updates["url"] = url
        if extra is not None:
            field_updates["extra"] = extra
        if volume is not None:
            field_updates["volume"] = volume
        if issue is not None:
            field_updates["issue"] = issue
        if pages is not None:
            field_updates["pages"] = pages
        if publisher is not None:
            field_updates["publisher"] = publisher
        if place is not None:
            field_updates["place"] = place
        if issn is not None:
            field_updates["ISSN"] = issn
        if language is not None:
            field_updates["language"] = language
        if short_title is not None:
            field_updates["shortTitle"] = short_title
        if edition is not None:
            field_updates["edition"] = edition
        if isbn is not None:
            field_updates["ISBN"] = isbn
        if book_title is not None:
            field_updates["bookTitle"] = book_title
        if citation_key is not None:
            field_updates["citationKey"] = citation_key

        skipped = []
        for field, value in field_updates.items():
            param_name = _UPDATE_ITEM_API_TO_PARAM.get(field, field)
            if field in data:
                old = data[field]
                if old != value:
                    changes.append(f"- **{param_name}**: '{old}' -> '{value}'")
                data[field] = value
            elif field == "citationKey":
                # citationKey is universally valid; absence on the fetched
                # item just means BBT has not yet auto-pinned a key, so we
                # add rather than skip-as-invalid-for-item-type.
                changes.append(f"- **{param_name}**: (none) -> '{value}'")
                data[field] = value
            else:
                skipped.append(param_name)

        # Creators
        if creators is not None:
            if isinstance(creators, str):
                creators = json.loads(creators)
            data["creators"] = creators
            changes.append("- **creators**: updated")

        # Tags
        if tags is not None:
            tag_list = _helpers._normalize_str_list_input(tags, "tags")
            data["tags"] = [{"tag": t} for t in tag_list]
            changes.append(f"- **tags**: replaced with {tag_list}")
        elif add_tags is not None or remove_tags is not None:
            existing = {t["tag"] for t in data.get("tags", [])}
            if add_tags is not None:
                to_add = _helpers._normalize_str_list_input(add_tags, "add_tags")
                existing.update(to_add)
                changes.append(f"- **tags**: added {to_add}")
            if remove_tags is not None:
                to_remove = set(_helpers._normalize_str_list_input(remove_tags, "remove_tags"))
                existing -= to_remove
                changes.append(f"- **tags**: removed {list(to_remove)}")
            data["tags"] = [{"tag": t} for t in sorted(existing)]

        # Collections — REPLACE membership (matches tags semantics and the
        # docstring contract). For incremental moves use
        # zotero_manage_collections. Passing collections=[] clears all
        # memberships. ``collections`` and ``collection_names`` may both be
        # supplied; the union of their resolved keys is the new membership.
        if collections is not None or collection_names is not None:
            new_collections: list[str] = []
            if collections is not None:
                new_collections.extend(
                    _helpers._normalize_str_list_input(collections, "collections")
                )
            if collection_names is not None:
                names = _helpers._normalize_str_list_input(
                    collection_names, "collection_names"
                )
                new_collections.extend(
                    _helpers._resolve_collection_names(read_zot, names, ctx=ctx)
                )
            # Preserve order while deduplicating.
            seen: set[str] = set()
            deduped = [
                k for k in new_collections if not (k in seen or seen.add(k))
            ]
            old_collections = list(data.get("collections") or [])
            if old_collections != deduped:
                data["collections"] = deduped
                changes.append(
                    f"- **collections**: replaced {old_collections} -> {deduped}"
                )

        skip_warning = ""
        if skipped:
            item_type = data.get("itemType", "unknown")
            skip_warning = (
                f"\n\nSkipped (not valid for item type "
                f"'{item_type}'): {', '.join(skipped)}"
            )

        if not changes:
            return "No changes to apply." + skip_warning

        resp = write_zot.update_item(item)
        if _helpers._handle_write_response(resp, ctx):
            result = (
                f"Successfully updated item `{item_key}`:\n\n"
                + "\n".join(changes)
            )
            return result + skip_warning
        return "Failed to update item: write operation returned failure"

    except ValueError as e:
        return f"Input error: {e}"
    except Exception as e:
        ctx.error(f"Error updating item: {e}")
        return f"Error updating item: {e}"


@mcp.tool(
    name="zotero_delete_item",
    description=(
        "Move a Zotero item to the Trash. Works for any item type (book, "
        "journalArticle, webpage, attachment, etc.). For notes, use "
        "zotero_delete_note — identical mechanism, constrained to notes "
        "for safety. Trashed items are recoverable from Zotero's Trash — "
        "empty the Trash in the Zotero UI for permanent deletion. "
        "By default refuses to trash notes; set allow_note=True to override."
    )
)
def delete_item(
    item_key: str,
    allow_note: bool = False,
    *,
    ctx: Context
) -> str:
    """
    Move a Zotero item to the Trash.

    Args:
        item_key: Zotero item key/ID to trash
        allow_note: If True, permits trashing note items. Default False
            directs callers to zotero_delete_note for notes (which has the
            same mechanism but is explicit about what it affects).
        ctx: MCP context

    Returns:
        Confirmation message, or an error if the item cannot be trashed.
    """
    try:
        _, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        ctx.info(f"Trashing item {item_key}")

        try:
            item = write_zot.item(item_key)
        except Exception:
            return f"Error: No item found with key: {item_key}"

        data = item.get("data", {})
        item_type = data.get("itemType", "unknown")

        if item_type == "note" and not allow_note:
            return (
                f"Error: Item {item_key} is a note. Use zotero_delete_note "
                "for notes, or pass allow_note=True to override."
            )

        # pyzotero's delete_item() permanently destroys items, and update_item()
        # strips the "deleted" field. Send a direct PATCH with {"deleted": 1}
        # to move the item to Zotero's Trash (recoverable by the user).
        from pyzotero.zotero import build_url
        url = build_url(
            write_zot.endpoint,
            f"/{write_zot.library_type}/{write_zot.library_id}/items/{item_key}",
        )
        resp = write_zot.client.patch(
            url=url,
            headers={"If-Unmodified-Since-Version": str(item["version"])},
            content=json.dumps({"deleted": 1}),
        )
        if resp.status_code in (200, 204):
            return (
                f"Successfully trashed item {item_key} "
                f"(type={item_type}, recoverable from Zotero's Trash)"
            )
        return (
            f"Failed to trash item {item_key} (HTTP {resp.status_code}): "
            f"{resp.text[:200]}"
        )

    except Exception as e:
        ctx.error(f"Error trashing item: {str(e)}")
        return f"Error trashing item: {str(e)}"


@mcp.tool(
    name="zotero_find_duplicates",
    description=(
        "Scan the active library (or a single collection) for duplicate "
        "items and return candidate groups for review. This tool only "
        "IDENTIFIES duplicates — it doesn't merge them. Call "
        "zotero_merge_duplicates to actually merge a group. "
        "method: 'both' (default) — match on title OR DOI; 'title' — "
        "normalized-title match only (lowercase, punctuation-stripped); "
        "'doi' — exact DOI match only (safest for automation). Prefer "
        "'doi' when the user intends to run merge_duplicates "
        "unattended. "
        "collection_key: optional 8-character key to restrict scanning "
        "to one collection; otherwise scans the whole active library. "
        "LIBRARY SIZE CAP: refuses to scan a library with > 5,000 items "
        "(the whole-library scan is O(n²) on titles) — on larger "
        "libraries you MUST pass collection_key to narrow the scope. "
        "limit: max groups to return (default 50). "
        "Returns a markdown block per group with keys, titles, DOIs, "
        "and dateAdded — use this to decide which item to KEEP before "
        "calling zotero_merge_duplicates(keeper_key=..., "
        "duplicate_keys=[...]). "
        "Read-only; works in local or web mode. "
        "Example: zotero_find_duplicates(method='doi', limit=20)."
    )
)
@with_zotero_api_lock
def find_duplicates(
    method: Literal["title", "doi", "both"] = "both",
    collection_key: str | None = None,
    limit: int | str | None = 50,
    *,
    ctx: Context
) -> str:
    try:
        zot = _client.get_zotero_client()
        limit = _helpers._normalize_limit(limit, default=50)
        ctx.info(f"Searching for duplicates (method={method})")

        # Paginate manually instead of using zot.everything() which can
        # cause "cannot pickle '_thread.RLock' object" in MCP contexts.
        items = []
        start = 0
        page_size = 100
        while True:
            if collection_key:
                batch = zot.collection_items(collection_key, start=start, limit=page_size)
            else:
                batch = zot.items(start=start, limit=page_size)
            if not batch:
                break
            items.extend(batch)
            if len(batch) < page_size:
                break
            start += page_size
            if len(items) > 5000:
                break

        if len(items) > 5000:
            return (
                f"Library has {len(items)} items — too large for duplicate scan. "
                "Please scope by collection_key to reduce the search."
            )

        # Normalize and group
        def normalize_title(t):
            t = (t or "").lower().strip()
            t = re.sub(r'[^\w\s]', '', t)
            t = re.sub(r'\s+', ' ', t).strip()
            for article in ("a ", "an ", "the "):
                if t.startswith(article):
                    t = t[len(article):]
            return t

        groups = {}
        for item in items:
            data = item.get("data", {})
            if data.get("itemType") in ("attachment", "note", "annotation"):
                continue

            keys_to_check = []
            if method in ("title", "both"):
                nt = normalize_title(data.get("title", ""))
                if nt:
                    keys_to_check.append(("title", nt))
            if method in ("doi", "both"):
                doi_val = (data.get("DOI") or "").strip().lower()
                if doi_val:
                    keys_to_check.append(("doi", doi_val))

            for group_type, group_key in keys_to_check:
                full_key = f"{group_type}:{group_key}"
                if full_key not in groups:
                    groups[full_key] = []
                groups[full_key].append(item)

        # Filter to groups with duplicates
        dups = {k: v for k, v in groups.items() if len(v) >= 2}

        if not dups:
            return "No duplicates found."

        lines = [f"# Found {len(dups)} duplicate groups", ""]
        shown = 0
        for group_key, group_items in sorted(dups.items()):
            if shown >= limit:
                lines.append(f"\n... and {len(dups) - shown} more groups")
                break
            shown += 1
            lines.append(f"## Group: {group_key}")
            for item in group_items:
                d = item.get("data", {})
                key = item.get("key", "?")
                t = d.get("title", "Untitled")
                dt = d.get("date", "")
                doi_val = d.get("DOI", "")
                lines.append(f"- `{key}` — {t} ({dt}) {f'DOI:{doi_val}' if doi_val else ''}")
            lines.append("")

        lines.append(
            "\nTo merge, call `zotero_merge_duplicates` with the key you want to keep "
            "and the keys to merge into it."
        )
        return "\n".join(lines)

    except Exception as e:
        ctx.error(f"Error finding duplicates: {e}")
        return f"Error finding duplicates: {e}"


@mcp.tool(
    name="zotero_merge_duplicates",
    description=(
        "Merge one or more duplicate items INTO a keeper: consolidates "
        "tags, collections, notes, annotations, and all child items onto "
        "the keeper, then moves the duplicates to Trash (recoverable "
        "from Zotero desktop's Trash view). "
        "SAFETY: dry-run by DEFAULT — prints what would happen without "
        "changing anything. Pass confirm=True to actually execute. Always "
        "run dry-first at least once to verify the keeper choice. "
        "Discover groups first with zotero_find_duplicates. "
        "keeper_key: 8-character key of the item to KEEP. All metadata "
        "gaps on the keeper are filled from duplicates where possible; "
        "conflicting fields keep the keeper's value. "
        "duplicate_keys: ARRAY of 8-character item keys to merge into "
        "the keeper and trash (also accepts a JSON-encoded list "
        "string) — pass as an array, not a single concatenated string. "
        "The keeper itself must NOT appear in this list. "
        "confirm: False (default) runs dry; True executes the merge. "
        "Requires a writable library (web API key or hybrid mode); fails "
        "in local-only mode. "
        "Example dry-run: zotero_merge_duplicates("
        "keeper_key='ABC12345', duplicate_keys=['XYZ98765']). "
        "Example execute: same, plus confirm=True."
    )
)
@with_zotero_api_lock
def merge_duplicates(
    keeper_key: str,
    duplicate_keys: list[str] | str,
    confirm: bool = False,
    *,
    ctx: Context
) -> str:
    try:
        read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        dup_keys = _helpers._normalize_str_list_input(duplicate_keys, "duplicate_keys")

        # Safety: remove keeper from duplicates
        if keeper_key in dup_keys:
            dup_keys.remove(keeper_key)
            ctx.warning(f"Keeper key '{keeper_key}' was in duplicate list — removed.")

        if not dup_keys:
            return "Error: No duplicate keys to merge (after removing keeper if present)."

        # Fetch all items and children
        keeper = write_zot.item(keeper_key)
        keeper_children = write_zot.children(keeper_key)
        duplicates = []
        for dk in dup_keys:
            dup_item = write_zot.item(dk)
            dup_children = write_zot.children(dk)
            duplicates.append({"item": dup_item, "children": dup_children})

        # Compute what will be merged
        all_tags = set()
        for t in keeper.get("data", {}).get("tags", []):
            all_tags.add(t.get("tag", ""))
        all_collections = set(keeper.get("data", {}).get("collections", []))
        total_children_to_move = 0

        for dup in duplicates:
            for t in dup["item"].get("data", {}).get("tags", []):
                all_tags.add(t.get("tag", ""))
            all_collections.update(dup["item"].get("data", {}).get("collections", []))
            total_children_to_move += len(dup["children"])

        all_tags.discard("")
        new_tags = all_tags - {t.get("tag", "") for t in keeper.get("data", {}).get("tags", [])}
        new_collections = all_collections - set(keeper.get("data", {}).get("collections", []))

        # Build keeper's attachment signatures for deduplication
        keeper_attachment_sigs = set()
        for kc in keeper_children:
            kd = kc.get("data", {})
            if kd.get("itemType") == "attachment":
                sig = (
                    kd.get("contentType", ""),
                    kd.get("filename", ""),
                    kd.get("md5", ""),
                    kd.get("url", ""),
                )
                keeper_attachment_sigs.add(sig)

        # Count duplicate attachments that would be skipped
        skipped_attachment_count = 0
        for dup in duplicates:
            for child in dup["children"]:
                cd = child.get("data", {})
                if cd.get("itemType") == "attachment":
                    sig = (
                        cd.get("contentType", ""),
                        cd.get("filename", ""),
                        cd.get("md5", ""),
                        cd.get("url", ""),
                    )
                    if sig in keeper_attachment_sigs:
                        skipped_attachment_count += 1

        # DRY RUN
        if not confirm:
            lines = [
                "# Merge Preview (dry run)",
                "",
                f"**Keeper:** `{keeper_key}` — {keeper.get('data', {}).get('title', 'Untitled')}",
                f"**Duplicates to merge:** {', '.join(f'`{k}`' for k in dup_keys)}",
                "",
                f"**Tags to add:** {sorted(new_tags) if new_tags else 'none'}",
                f"**Collections to add:** {sorted(new_collections) if new_collections else 'none'}",
                f"**Child items to re-parent:** {total_children_to_move - skipped_attachment_count}",
                f"  ({skipped_attachment_count} duplicate attachment(s) will be skipped)" if skipped_attachment_count else "  (notes, PDFs, annotations, highlights, etc.)",
                "",
                "Duplicates will be moved to **Trash** (recoverable in Zotero).",
                "",
                "**Call again with `confirm=True` to execute.**",
            ]
            return "\n".join(lines)

        # EXECUTE MERGE
        ctx.info(f"Merging {len(dup_keys)} duplicates into {keeper_key}")

        # Step 3: Consolidate tags
        if new_tags:
            keeper_data = keeper.get("data", {})
            existing_tags = [t.get("tag", "") for t in keeper_data.get("tags", [])]
            keeper_data["tags"] = [{"tag": t} for t in sorted(set(existing_tags) | all_tags)]
            _helpers._strip_unwritable_fields(keeper)
            resp = write_zot.update_item(keeper)
            if not _helpers._handle_write_response(resp, ctx):
                return "Error: Failed to merge tags into keeper."
            keeper = write_zot.item(keeper_key)  # re-fetch for version

        # Step 4: Consolidate collections
        for coll_key in new_collections:
            resp = write_zot.addto_collection(coll_key, keeper)
            if not _helpers._handle_write_response(resp, ctx):
                ctx.warning(f"Failed to add keeper to collection {coll_key}")
            keeper = write_zot.item(keeper_key)  # re-fetch for version

        # Step 5: Re-parent children (skip duplicate attachments)
        moved = []
        failed = []
        skipped_dupes = []
        for dup in duplicates:
            for child in dup["children"]:
                child_key = child.get("key", "?")
                try:
                    fresh_child = write_zot.item(child_key)
                    # Skip duplicate attachments — keeper already has this one
                    child_data = fresh_child.get("data", {})
                    if child_data.get("itemType") == "attachment":
                        child_sig = (
                            child_data.get("contentType", ""),
                            child_data.get("filename", ""),
                            child_data.get("md5", ""),
                            child_data.get("url", ""),
                        )
                        if child_sig in keeper_attachment_sigs:
                            skipped_dupes.append(child_key)
                            continue  # Skip — keeper already has this attachment
                    fresh_child.get("data", {})["parentItem"] = keeper_key
                    _helpers._strip_unwritable_fields(fresh_child)
                    resp = write_zot.update_item(fresh_child)
                    if _helpers._handle_write_response(resp, ctx):
                        moved.append(child_key)
                    else:
                        failed.append(child_key)
                except Exception as e:
                    failed.append(f"{child_key} ({e})")

        if failed:
            return (
                f"Merge partially completed. Moved {len(moved)} children, "
                f"but {len(failed)} failed: {failed}\n\n"
                "Duplicates were NOT trashed. Fix the failures and retry."
            )

        # Step 6: Trash duplicates (move to Zotero Trash, NOT permanent delete)
        # pyzotero's update_item() strips "deleted" and delete_item() permanently
        # destroys items. We send a direct PATCH with {"deleted": 1} which moves
        # items to Zotero's Trash — recoverable by the user.
        trashed = []
        for dup in duplicates:
            dup_key = dup["item"]["key"]
            try:
                dup_item = write_zot.item(dup_key)
                version = dup_item["version"]
                from pyzotero.zotero import build_url
                url = build_url(
                    write_zot.endpoint,
                    f"/{write_zot.library_type}/{write_zot.library_id}/items/{dup_key}",
                )
                headers = {"If-Unmodified-Since-Version": str(version)}
                resp = write_zot.client.patch(
                    url=url,
                    headers=headers,
                    content=json.dumps({"deleted": 1}),
                )
                if resp.status_code in (200, 204):
                    trashed.append(dup_key)
                else:
                    ctx.warning(f"Failed to trash {dup_key}: HTTP {resp.status_code}")
            except Exception as e:
                ctx.warning(f"Failed to trash {dup_key}: {e}")

        skip_info = f" ({len(skipped_dupes)} duplicate attachments skipped)" if skipped_dupes else ""
        return (
            f"Merge complete.\n\n"
            f"- Tags merged: {len(new_tags)} new\n"
            f"- Collections added: {len(new_collections)} new\n"
            f"- Children re-parented: {len(moved)}{skip_info}\n"
            f"- Duplicates trashed: {', '.join(f'`{k}`' for k in trashed)}\n\n"
            "Trashed items can be restored from Zotero's Trash."
        )

    except ValueError as e:
        return f"Input error: {e}"
    except Exception as e:
        ctx.error(f"Error merging duplicates: {e}")
        return f"Error merging duplicates: {e}"


@mcp.tool(
    name="zotero_get_pdf_outline",
    description=(
        "Extract the table of contents (outline/bookmarks) from a PDF "
        "attachment, returned as a hierarchical markdown list with each "
        "entry's page number. "
        "Use this to orient in a paper before calling "
        "zotero_get_item_fulltext — the outline is typically < 200 "
        "tokens versus 10K+ for the full text. If the PDF has no "
        "embedded outline, returns a short 'no outline' message rather "
        "than failing. "
        "item_key: the PDF ATTACHMENT key OR the parent item key — both "
        "are accepted; attachment-to-parent resolution is automatic. "
        "Find the right key with zotero_get_item_children if unsure. "
        "Scope: PDFs only (EPUBs have no outline extraction here). "
        "Requires PyMuPDF (pip install zotero-mcp-server[pdf]). "
        "Read-only; works in local or web mode. "
        "Example: zotero_get_pdf_outline(item_key='RTKZQI8E')."
    )
)
@with_zotero_api_lock
def get_pdf_outline(
    item_key: str,
    *,
    ctx: Context
) -> str:
    try:
        zot = _client.get_zotero_client()
        ctx.info(f"Getting PDF outline for item {item_key}")

        # Find PDF attachment
        children = zot.children(item_key)
        pdf_child = None
        for child in children:
            if child.get("data", {}).get("contentType") == "application/pdf":
                pdf_child = child
                break

        if not pdf_child:
            return f"No PDF attachment found for item `{item_key}`."

        try:
            import fitz
        except ImportError:
            return "Error: PyMuPDF (fitz) is required for PDF outline extraction."

        attachment_key = pdf_child["key"]
        filename = pdf_child.get("data", {}).get("filename", "document.pdf")

        # Download PDF (works for both local/WebDAV/web storage)
        with tempfile.TemporaryDirectory() as tmpdir:
            zot.dump(attachment_key, filename=filename, path=tmpdir)
            pdf_path = os.path.join(tmpdir, filename)
            if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
                return f"Could not download PDF for attachment `{attachment_key}`."
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()
            doc.close()

        if not toc:
            return "This PDF does not contain a table of contents/outline."

        lines = [f"# PDF Outline for item `{item_key}`", ""]
        for level, title, page in toc:
            indent = "  " * (level - 1)
            lines.append(f"{indent}- {title} (p. {page})")

        return "\n".join(lines)

    except Exception as e:
        ctx.error(f"Error extracting PDF outline: {e}")
        return f"Error extracting PDF outline: {e}"


@mcp.tool(
    name="zotero_add_from_file",
    description=(
        "Add an item to the active Zotero library from a LOCAL .pdf or "
        ".epub file. Attempts to extract the DOI from the file content; "
        "if found, enriches metadata via CrossRef (title, creators, "
        "journal, year, abstract). If no DOI is found, falls back to "
        "best-effort title/author guesses from the filename or document "
        "text. "
        "Use this when the user has a file on disk but no DOI/URL handy. "
        "If you have a DOI use zotero_add_by_doi; for an online URL use "
        "zotero_add_by_url. "
        "file_path: ABSOLUTE path to a .pdf or .epub file (relative "
        "paths fail). Other extensions are rejected. "
        "title: optional override if metadata extraction misses. "
        "collections: optional list of collection keys, names, or "
        "'/'-separated paths to file under — resolved and validated "
        "before the item is created. "
        "tags: optional list of tag strings. "
        "if_exists: 'duplicate' (default) | 'file' (when the extracted "
        "DOI matches an existing item, reuse it: file into missing "
        "collections, attach the file to it unless an attachment with "
        "the same filename exists) | 'skip' (no item, no attachment). "
        "create_missing_collections: create unknown collection specs. "
        "Requires a writable library (fails in local-only mode). PDF "
        "uploads may hit the 300MB Zotero cloud free-tier quota — "
        "metadata still lands. Run zotero_update_search_database "
        "afterwards for semantic search. "
        "Example: zotero_add_from_file(file_path='/Users/me/paper.pdf', "
        "collections=['9SU943GB'])."
    )
)
@with_zotero_api_lock
def add_from_file(
    file_path: str,
    title: str | None = None,
    item_type: str = "document",
    collections: list[str] | str | None = None,
    tags: list[str] | str | None = None,
    if_exists: Literal["duplicate", "file", "skip"] = "duplicate",
    create_missing_collections: bool = False,
    *,
    ctx: Context
) -> str:
    try:
        read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        if if_exists not in _IF_EXISTS_VALUES:
            return f"Error: if_exists must be one of {_IF_EXISTS_VALUES}."
        # Path validation — check symlink BEFORE resolving
        if os.path.islink(file_path):
            return "Error: Symlinks are not allowed for security reasons."
        if not os.path.isabs(file_path):
            return "Error: file_path must be an absolute path."
        # Resolve ".." components after symlink check
        file_path = os.path.realpath(file_path)
        if not os.path.isfile(file_path):
            return f"Error: File not found: {file_path}"

        try:
            coll_keys = _resolve_collections_arg(
                read_zot, collections, ctx,
                create_missing=create_missing_collections, write_zot=write_zot,
            )
        except ValueError as e:
            return f"Error: {e}"

        ext = os.path.splitext(file_path)[1].lower()
        allowed_exts = {".pdf", ".epub", ".djvu", ".doc", ".docx", ".odt", ".rtf"}
        if ext not in allowed_exts:
            return f"Error: Unsupported file type '{ext}'. Allowed: {', '.join(sorted(allowed_exts))}"

        ctx.info(f"Adding file: {file_path}")

        # Try DOI extraction from PDF
        extracted_doi = None
        if ext == ".pdf":
            try:
                import fitz
                doc = fitz.open(file_path)

                # Check metadata
                meta = doc.metadata or {}
                for field in ("subject", "keywords", "title"):
                    candidate = meta.get(field, "")
                    if candidate:
                        found_doi = _helpers._normalize_doi(candidate)
                        if found_doi:
                            extracted_doi = found_doi
                            break

                # Scan first page text
                if not extracted_doi and doc.page_count > 0:
                    text = doc[0].get_text()[:3000]
                    m = re.search(r'10\.\d{4,9}/[^\s]+', text)
                    if m:
                        found_doi = _helpers._normalize_doi(m.group(0))
                        if found_doi:
                            extracted_doi = found_doi

                doc.close()
            except Exception as e:
                ctx.info(f"DOI extraction failed (non-fatal): {e}")

        # Create the metadata item. With if_exists='file' and a known DOI,
        # add_by_doi reuses the existing item — the attachment below then
        # lands on it instead of on a fresh duplicate.
        if extracted_doi:
            ctx.info(f"Found DOI: {extracted_doi}")
            result_msg = add_by_doi(doi=extracted_doi, collections=coll_keys,
                                    tags=tags, if_exists=if_exists, ctx=ctx)
            # Extract item key from result
            key_match = re.search(r'Item key: `([^`]+)`', result_msg)
            if key_match:
                parent_key = key_match.group(1)
            else:
                return f"DOI lookup succeeded but couldn't extract item key.\n\n{result_msg}"
        else:
            # Create a basic item
            template = write_zot.item_template(item_type)
            template["title"] = title or os.path.basename(file_path)

            tag_list = _helpers._normalize_str_list_input(tags, "tags")
            if tag_list:
                template["tags"] = [{"tag": t} for t in tag_list]
            if coll_keys:
                template["collections"] = coll_keys

            result = write_zot.create_items([template])
            if isinstance(result, dict) and result.get("success"):
                parent_key = next(iter(result["success"].values()))
                missing = _helpers.ensure_collection_membership(
                    write_zot, parent_key, coll_keys, ctx=ctx
                )
                if missing:
                    ctx.warning(f"Failed to file {parent_key} in {missing}")
            else:
                return f"Failed to create item: {result}"

        item_reused = bool(extracted_doi) and result_msg.startswith("Already in library")
        if item_reused and if_exists == "skip":
            return result_msg + "\n\nFile NOT attached (if_exists='skip')."

        # Attach the file. When reusing an existing item, skip the upload if
        # an attachment with the same filename is already there — re-running
        # the command must converge, not accumulate duplicate attachments.
        try:
            display_name = os.path.basename(file_path)
            if item_reused:
                try:
                    kids = write_zot.children(parent_key)
                except Exception:
                    kids = []
                if any(
                    (k.get("data", {}) or {}).get("filename") == display_name
                    for k in kids
                ):
                    return (
                        f"{result_msg}\n"
                        f"Attachment already present: {display_name} (not re-uploaded)\n\n"
                        "_Note: To include this item in semantic search, run "
                        "zotero_update_search_database._"
                    )

            attach_result = write_zot.attachment_both(
                [(display_name, file_path)],
                parentid=parent_key,
            )
            attach_info = (
                f"File attached: {display_name}"
                + _helpers._maybe_upload_to_webdav(attach_result, file_path, ctx)
            )
        except Exception as e:
            attach_info = f"Item created but file attachment failed: {e}"

        return (
            f"Item key: `{parent_key}`\n"
            f"{'DOI: ' + extracted_doi + chr(10) if extracted_doi else ''}"
            f"{attach_info}\n\n"
            "_Note: To include this item in semantic search, run "
            "zotero_update_search_database._"
        )

    except Exception as e:
        ctx.error(f"Error adding from file: {e}")
        return f"Error adding from file: {e}"


def _build_relation_uri(library_type: str, library_id: str, item_key: str) -> str:
    """Build a Zotero relation URI for the given item.

    Uses the canonical format based on library_type:
    - user library  → ``http://zotero.org/users/<id>/items/<key>``
    - group library → ``http://zotero.org/groups/<id>/items/<key>``

    Note: pyzotero internally pluralises the constructor argument
    (``'user'`` → ``'users'``, ``'group'`` → ``'groups'``), so we
    accept both singular and plural forms.
    """
    kind = "users" if library_type in ("user", "users") else "groups"
    return f"http://zotero.org/{kind}/{library_id}/items/{item_key}"


def _relation_exists(rel_list: list, library_id: str, item_key: str) -> bool:
    """Check whether a relation to *item_key* already exists (either URI variant)."""
    pattern = re.compile(
        rf"http://zotero\.org/(?:users|groups)/{re.escape(str(library_id))}/items/{re.escape(item_key)}$"
    )
    return any(isinstance(uri, str) and pattern.search(uri) for uri in rel_list)


def _find_matching_uri(rel_list: list, library_id: str, item_key: str) -> str | None:
    """Find and return the actual URI string for *item_key* regardless of prefix."""
    pattern = re.compile(
        rf"http://zotero\.org/(?:users|groups)/{re.escape(str(library_id))}/items/{re.escape(item_key)}$"
    )
    for uri in rel_list:
        if isinstance(uri, str) and pattern.search(uri):
            return uri
    return None


@mcp.tool(
    name="zotero_add_item_relation",
    description="Add a related item relationship to a Zotero item. Creates a bidirectional link between two items."
)
def add_item_relation(
    item_key: str,
    related_item_key: str,
    relation_type: str = "dc:relation",
    *,
    ctx: Context
) -> str:
    """
    Add a related item relationship to a Zotero item.

    Args:
        item_key: The key of the primary item
        related_item_key: The key of the item to relate to
        relation_type: The type of relationship (default: "dc:relation").
                       Common values: "dc:relation", "owl:sameAs"
        ctx: MCP context

    Returns:
        Confirmation message
    """
    try:
        read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        if item_key == related_item_key:
            return "Error: Cannot relate an item to itself."

        ctx.info(f"Adding relation from {item_key} to {related_item_key}")

        # Fetch the primary item
        try:
            item = write_zot.item(item_key)
        except Exception:
            return f"Error: Item '{item_key}' not found."

        # Verify the related item exists
        try:
            related_item = write_zot.item(related_item_key)
        except Exception:
            return f"Error: Related item '{related_item_key}' not found."

        data = item.get("data", {})
        related_data = related_item.get("data", {})

        # Get current relations or initialize empty dict
        relations = data.get("relations", {})
        if not isinstance(relations, dict):
            relations = {}

        # Build the relation URI using the canonical format for the library type
        library_type = write_zot.library_type
        library_id = write_zot.library_id
        related_uri = _build_relation_uri(library_type, library_id, related_item_key)

        # Add the relation to the primary item
        if relation_type not in relations:
            relations[relation_type] = []
        if not isinstance(relations[relation_type], list):
            relations[relation_type] = [relations[relation_type]]

        # Check if relation already exists (match both URI prefix variants)
        if _relation_exists(relations[relation_type], library_id, related_item_key):
            return f"Relation already exists: '{item_key}' is already related to '{related_item_key}'."

        relations[relation_type].append(related_uri)
        data["relations"] = relations

        # Update the primary item
        _helpers._strip_unwritable_fields(item)
        resp = write_zot.update_item(item)
        if not _helpers._handle_write_response(resp, ctx):
            return f"Failed to add relation to item '{item_key}'."

        # Also add reverse relation (bidirectional)
        try:
            # Re-fetch to get latest version
            item = write_zot.item(item_key)
            related_item = write_zot.item(related_item_key)
            related_data = related_item.get("data", {})
            reverse_relations = related_data.get("relations", {})
            if not isinstance(reverse_relations, dict):
                reverse_relations = {}

            item_uri = _build_relation_uri(library_type, library_id, item_key)

            if relation_type not in reverse_relations:
                reverse_relations[relation_type] = []
            if not isinstance(reverse_relations[relation_type], list):
                reverse_relations[relation_type] = [reverse_relations[relation_type]]

            if not _relation_exists(reverse_relations[relation_type], library_id, item_key):
                reverse_relations[relation_type].append(item_uri)
                related_data["relations"] = reverse_relations
                _helpers._strip_unwritable_fields(related_item)
                write_zot.update_item(related_item)
        except Exception as e:
            ctx.warn(f"Could not add reverse relation: {e}")

        item_title = data.get("title", "Untitled")
        related_title = related_data.get("title", "Untitled")

        return (
            f"Successfully added relation:\n\n"
            f"**From:** `{item_key}` — {item_title}\n"
            f"**To:** `{related_item_key}` — {related_title}\n"
            f"**Relation type:** `{relation_type}`"
        )

    except Exception as e:
        ctx.error(f"Error adding item relation: {e}")
        return f"Error adding item relation: {e}"


@mcp.tool(
    name="zotero_remove_item_relation",
    description="Remove a related item relationship from a Zotero item."
)
def remove_item_relation(
    item_key: str,
    related_item_key: str,
    relation_type: str = "dc:relation",
    remove_bidirectional: bool = True,
    *,
    ctx: Context
) -> str:
    """
    Remove a related item relationship from a Zotero item.

    Args:
        item_key: The key of the primary item
        related_item_key: The key of the related item to unlink
        relation_type: The type of relationship (default: "dc:relation")
        remove_bidirectional: Also remove the reverse relation (default: True)
        ctx: MCP context

    Returns:
        Confirmation message
    """
    try:
        read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        ctx.info(f"Removing relation from {item_key} to {related_item_key}")

        # Fetch the primary item
        try:
            item = write_zot.item(item_key)
        except Exception:
            return f"Error: Item '{item_key}' not found."

        data = item.get("data", {})
        relations = data.get("relations", {})

        if not isinstance(relations, dict):
            return f"Item '{item_key}' has no relations to remove."

        if relation_type not in relations:
            return f"Item '{item_key}' has no relations of type '{relation_type}'."

        # Match any URI variant (users/ or groups/) for this library
        library_id = write_zot.library_id

        rel_list = relations[relation_type]
        if not isinstance(rel_list, list):
            rel_list = [rel_list]

        # Find the matching URI regardless of users/ vs groups/ prefix
        matched_uri = _find_matching_uri(rel_list, library_id, related_item_key)
        if matched_uri is None:
            return f"Relation not found: '{item_key}' is not related to '{related_item_key}'."

        # Remove the relation
        rel_list.remove(matched_uri)
        if not rel_list:
            del relations[relation_type]
        else:
            relations[relation_type] = rel_list

        data["relations"] = relations

        # Update the item
        _helpers._strip_unwritable_fields(item)
        resp = write_zot.update_item(item)
        if not _helpers._handle_write_response(resp, ctx):
            return f"Failed to remove relation from item '{item_key}'."

        # Remove bidirectional relation if requested
        if remove_bidirectional:
            try:
                related_item = write_zot.item(related_item_key)
                related_data = related_item.get("data", {})
                reverse_relations = related_data.get("relations", {})

                if isinstance(reverse_relations, dict) and relation_type in reverse_relations:
                    reverse_list = reverse_relations[relation_type]
                    if not isinstance(reverse_list, list):
                        reverse_list = [reverse_list]

                    matched_reverse = _find_matching_uri(reverse_list, library_id, item_key)
                    if matched_reverse is not None:
                        reverse_list.remove(matched_reverse)
                        if not reverse_list:
                            del reverse_relations[relation_type]
                        else:
                            reverse_relations[relation_type] = reverse_list
                        related_data["relations"] = reverse_relations
                        _helpers._strip_unwritable_fields(related_item)
                        write_zot.update_item(related_item)
            except Exception as e:
                ctx.warn(f"Could not remove reverse relation: {e}")

        return (
            f"Successfully removed relation:\n\n"
            f"**From:** `{item_key}`\n"
            f"**To:** `{related_item_key}`\n"
            f"**Relation type:** `{relation_type}`"
        )

    except Exception as e:
        ctx.error(f"Error removing item relation: {e}")
        return f"Error removing item relation: {e}"


# ---------------------------------------------------------------------------
# Import-by-citation tools (BibTeX / CSL JSON)
# ---------------------------------------------------------------------------

_CITATION_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB — generous for citation files


def _read_citation_file(file_path: str, allowed_exts: set[str]) -> str:
    """Read a citation file as UTF-8 text with the same safety checks as add_from_file.

    Raises ValueError on any check failure. Returns the file contents.
    """
    if os.path.islink(file_path):
        raise ValueError("Symlinks are not allowed for security reasons.")
    if not os.path.isabs(file_path):
        raise ValueError("file_path must be an absolute path.")
    resolved = os.path.realpath(file_path)
    if not os.path.isfile(resolved):
        raise ValueError(f"File not found: {file_path}")

    ext = os.path.splitext(resolved)[1].lower()
    if ext not in allowed_exts:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Allowed: {', '.join(sorted(allowed_exts))}"
        )

    size = os.path.getsize(resolved)
    if size > _CITATION_FILE_MAX_BYTES:
        raise ValueError(
            f"File is too large ({size} bytes). "
            f"Maximum {_CITATION_FILE_MAX_BYTES} bytes."
        )

    try:
        with open(resolved, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError as e:
        raise ValueError(f"File is not valid UTF-8: {e}") from e


def _apply_caller_tags_and_collections(
    item_data: dict,
    caller_tags: list[str] | str | None,
    caller_collections: list[str] | str | None,
) -> None:
    """Merge caller tags with any source-tags already in ``item_data`` and set collections."""
    extra_tags = _helpers._normalize_str_list_input(caller_tags, "tags")
    source_tags = [t.get("tag", "") for t in item_data.get("tags", []) if t.get("tag")]
    merged = _citation_import.merge_tags(source_tags, extra_tags)
    if merged:
        item_data["tags"] = [{"tag": t} for t in merged]

    coll_keys = _helpers._normalize_str_list_input(caller_collections, "collections")
    if coll_keys:
        existing = list(item_data.get("collections") or [])
        # Preserve order while deduplicating
        seen = set(existing)
        for k in coll_keys:
            if k not in seen:
                existing.append(k)
                seen.add(k)
        item_data["collections"] = existing


def _create_and_attach(
    write_zot,
    item_data: dict,
    attach_mode: str,
    ctx: Context,
) -> dict:
    """Create one Zotero item and, if it has a DOI, try to attach an OA PDF.

    Returns a dict ``{"ok": bool, "key": str|None, "doi": str|None,
    "pdf_status": str|None, "error": str|None, "title": str,
    "collections_failed": list[str]}``.
    """
    title = item_data.get("title") or "(untitled)"
    try:
        result = write_zot.create_items([item_data])
    except Exception as e:
        return {"ok": False, "key": None, "doi": None, "pdf_status": None,
                "error": str(e), "title": title, "collections_failed": []}

    if not (isinstance(result, dict) and result.get("success")):
        return {"ok": False, "key": None, "doi": None, "pdf_status": None,
                "error": f"create_items failed: {result}", "title": title,
                "collections_failed": []}

    item_key = next(iter(result["success"].values()))

    # #235 backstop: atomic filing via item["collections"] is intermittent.
    collections_failed = _helpers.ensure_collection_membership(
        write_zot, item_key, item_data.get("collections") or [], ctx=ctx
    )

    doi_raw = item_data.get("DOI") or ""
    doi = _helpers._normalize_doi(doi_raw) if doi_raw else None

    pdf_status = None
    if doi:
        try:
            pdf_status = _helpers._try_attach_oa_pdf(
                write_zot, item_key, doi, ctx, attach_mode=attach_mode
            )
        except Exception as e:
            pdf_status = f"OA PDF attach failed: {e}"

    return {"ok": True, "key": item_key, "doi": doi, "pdf_status": pdf_status,
            "error": None, "title": title,
            "collections_failed": collections_failed}


def _maybe_reuse_existing(read_zot, write_zot, item_data, coll_keys, tags,
                          if_exists, ctx) -> dict | None:
    """Batch-import dedup: reuse an existing item matching the entry's DOI.

    Returns a result dict for _format_batch_result when if_exists is
    'file'/'skip' and a DOI match exists; otherwise None (proceed to
    create). Entries without a DOI always create — title matching is out
    of scope (#4).
    """
    if if_exists == "duplicate":
        return None
    doi_raw = item_data.get("DOI") or ""
    doi = _helpers._normalize_doi(doi_raw) if doi_raw else None
    if not doi:
        return None
    existing = _helpers.find_existing_items(read_zot, doi=doi, ctx=ctx)
    if not existing:
        return None

    item = existing[0]
    if if_exists == "skip":
        return {
            "ok": True, "key": item.get("key"), "doi": doi,
            "pdf_status": None, "error": None,
            "title": item.get("data", {}).get("title") or "(untitled)",
            "collections_failed": [],
            "existed": "skipped — already in library",
        }

    summary = _converge_existing_item(write_zot, item, coll_keys, tags, ctx)
    bits = []
    if summary["colls_added"]:
        bits.append(f"added to {summary['colls_added']}")
    if summary["colls_already"]:
        bits.append(f"already in {summary['colls_already']}")
    if summary["tags_added"]:
        bits.append(f"tags added {summary['tags_added']}")
    detail = "; ".join(bits) if bits else "already in requested state"
    return {
        "ok": True, "key": summary["key"], "doi": doi, "pdf_status": None,
        "error": None, "title": summary["title"],
        "collections_failed": summary["colls_failed"],
        "existed": f"reused existing — {detail}",
    }


def _format_batch_result(header: str, results: list[dict]) -> str:
    """Render a per-entry markdown summary for add_by_bibtex / add_by_csl_json."""
    ok_count = sum(1 for r in results if r["ok"])
    reused_count = sum(1 for r in results if r["ok"] and r.get("existed"))
    lines = [header, ""]
    if len(results) == 1:
        r = results[0]
        if r["ok"]:
            verb = "Already in library" if r.get("existed") else "Successfully added"
            lines.append(f"{verb}: **{r['title']}**")
            lines.append("")
            lines.append(f"Item key: `{r['key']}`")
            if r["doi"]:
                lines.append(f"DOI: {r['doi']}")
            if r.get("existed"):
                lines.append(f"Status: {r['existed']}")
            if r["pdf_status"]:
                lines.append(f"PDF: {r['pdf_status']}")
            if r.get("collections_failed"):
                lines.append(
                    f"WARNING: failed to file in {r['collections_failed']}"
                )
        else:
            lines.append(f"Failed to add **{r['title']}**: {r['error']}")
    else:
        summary_line = f"Added {ok_count - reused_count}/{len(results)} items."
        if reused_count:
            summary_line += f" {reused_count} already existed (reused, not duplicated)."
        lines.append(summary_line)
        lines.append("")
        for i, r in enumerate(results, 1):
            if r["ok"]:
                line = f"{i}. `{r['key']}` — {r['title']}"
                if r["doi"]:
                    line += f" (DOI: {r['doi']})"
                if r.get("existed"):
                    line += f" [{r['existed']}]"
                if r["pdf_status"]:
                    line += f" [{r['pdf_status']}]"
                if r.get("collections_failed"):
                    line += f" [failed to file in {r['collections_failed']}]"
                lines.append(line)
            else:
                lines.append(f"{i}. ❌ {r['title']}: {r['error']}")
    lines.append("")
    lines.append(
        "_Note: To include new items in semantic search, run "
        "zotero_update_search_database._"
    )
    return "\n".join(lines)


@mcp.tool(
    name="zotero_add_by_bibtex",
    description=(
        "Add one or more items to Zotero from BibTeX. "
        "Provide EITHER `bibtex` (inline string) OR `file_path` "
        "(absolute path to a .bib / .bibtex file) — not both. "
        "Supports multiple @entries per call. "
        "The citation key from each entry is preserved in the Extra field. "
        "If an entry has a DOI, an open-access PDF attachment is attempted. "
        "collections accepts keys, names, or '/'-paths (validated before "
        "create). if_exists: 'duplicate' (default) | 'file' (entries whose "
        "DOI already exists reuse that item — add missing collections/tags "
        "instead of duplicating) | 'skip' (leave existing matches "
        "untouched); entries without a DOI always create. "
        "create_missing_collections: create unknown collection specs."
    )
)
def add_by_bibtex(
    bibtex: str | None = None,
    file_path: str | None = None,
    collections: list[str] | str | None = None,
    tags: list[str] | str | None = None,
    attach_mode: str = "auto",
    if_exists: Literal["duplicate", "file", "skip"] = "duplicate",
    create_missing_collections: bool = False,
    *,
    ctx: Context
) -> str:
    try:
        _read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        if if_exists not in _IF_EXISTS_VALUES:
            return f"Error: if_exists must be one of {_IF_EXISTS_VALUES}."
        bibtex_provided = bool((bibtex or "").strip())
        if bibtex_provided and file_path:
            return "Error: Provide either `bibtex` or `file_path`, not both."
        if not bibtex_provided and not file_path:
            return "Error: Must provide `bibtex` (inline string) or `file_path`."

        if file_path:
            try:
                bibtex = _read_citation_file(
                    file_path, allowed_exts={".bib", ".bibtex"}
                )
            except ValueError as e:
                return f"Error: {e}"
            ctx.info(f"Loaded BibTeX from {file_path} ({len(bibtex)} bytes)")

        try:
            entries = _citation_import.parse_bibtex(bibtex)
        except Exception as e:
            return f"Error parsing BibTeX: {e}"

        if not entries:
            return "Error: No valid @entries found in the BibTeX input."

        try:
            coll_keys = _resolve_collections_arg(
                _read_zot, collections, ctx,
                create_missing=create_missing_collections, write_zot=write_zot,
            )
        except ValueError as e:
            return f"Error: {e}"

        ctx.info(f"Parsed {len(entries)} BibTeX entries")

        results = []
        for entry in entries:
            try:
                item_data = _citation_import.bibtex_entry_to_zotero(
                    entry, write_zot.item_template
                )
            except Exception as e:
                results.append({
                    "ok": False, "key": None, "doi": None, "pdf_status": None,
                    "error": f"conversion failed: {e}",
                    "title": entry.get("citekey") or "(unknown)",
                })
                continue

            reused = _maybe_reuse_existing(
                _read_zot, write_zot, item_data, coll_keys, tags, if_exists, ctx
            )
            if reused is not None:
                results.append(reused)
                continue

            _apply_caller_tags_and_collections(item_data, tags, coll_keys)
            results.append(_create_and_attach(write_zot, item_data, attach_mode, ctx))

        return _format_batch_result("# zotero_add_by_bibtex", results)

    except Exception as e:
        ctx.error(f"Error adding by BibTeX: {e}")
        return f"Error adding by BibTeX: {e}"


@mcp.tool(
    name="zotero_add_by_csl_json",
    description=(
        "Add one or more items to Zotero from CSL JSON. "
        "Provide EITHER `csl_json` (inline — a JSON string, object, or array) "
        "OR `file_path` (absolute path to a .json / .csljson file) — not both. "
        "The `id` field is preserved in the Extra field as the Citation Key. "
        "If an entry has a DOI, an open-access PDF attachment is attempted. "
        "collections accepts keys, names, or '/'-paths (validated before "
        "create). if_exists: 'duplicate' (default) | 'file' (entries whose "
        "DOI already exists reuse that item — add missing collections/tags) "
        "| 'skip'; entries without a DOI always create. "
        "create_missing_collections: create unknown collection specs."
    )
)
def add_by_csl_json(
    csl_json: str | list | dict | None = None,
    file_path: str | None = None,
    collections: list[str] | str | None = None,
    tags: list[str] | str | None = None,
    attach_mode: str = "auto",
    if_exists: Literal["duplicate", "file", "skip"] = "duplicate",
    create_missing_collections: bool = False,
    *,
    ctx: Context
) -> str:
    try:
        _read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        if if_exists not in _IF_EXISTS_VALUES:
            return f"Error: if_exists must be one of {_IF_EXISTS_VALUES}."
        csl_provided = csl_json not in (None, "", [], {})
        if csl_provided and file_path:
            return "Error: Provide either `csl_json` or `file_path`, not both."
        if not csl_provided and not file_path:
            return "Error: Must provide `csl_json` (inline) or `file_path`."

        if file_path:
            try:
                csl_json = _read_citation_file(
                    file_path, allowed_exts={".json", ".csljson"}
                )
            except ValueError as e:
                return f"Error: {e}"
            ctx.info(f"Loaded CSL JSON from {file_path} ({len(csl_json)} bytes)")

        try:
            entries = _citation_import.coerce_csl_json_input(csl_json)
        except ValueError as e:
            return f"Error: {e}"

        if not entries:
            return "Error: No valid CSL JSON objects provided."

        try:
            coll_keys = _resolve_collections_arg(
                _read_zot, collections, ctx,
                create_missing=create_missing_collections, write_zot=write_zot,
            )
        except ValueError as e:
            return f"Error: {e}"

        ctx.info(f"Processing {len(entries)} CSL JSON entries")

        results = []
        for entry in entries:
            try:
                item_data = _citation_import.csl_json_to_zotero(
                    entry, write_zot.item_template
                )
            except Exception as e:
                results.append({
                    "ok": False, "key": None, "doi": None, "pdf_status": None,
                    "error": f"conversion failed: {e}",
                    "title": str(entry.get("id") or entry.get("title") or "(unknown)"),
                })
                continue

            reused = _maybe_reuse_existing(
                _read_zot, write_zot, item_data, coll_keys, tags, if_exists, ctx
            )
            if reused is not None:
                results.append(reused)
                continue

            _apply_caller_tags_and_collections(item_data, tags, coll_keys)
            results.append(_create_and_attach(write_zot, item_data, attach_mode, ctx))

        return _format_batch_result("# zotero_add_by_csl_json", results)

    except Exception as e:
        ctx.error(f"Error adding by CSL JSON: {e}")
        return f"Error adding by CSL JSON: {e}"
