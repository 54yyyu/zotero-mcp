"""Retrieval tool functions — read-only access to Zotero items, collections, tags, libraries, and feeds."""

from typing import Literal
import json
import logging as _logging
import os
import re
import tempfile
import time as _time
from pathlib import Path

from fastmcp import Context

from zotero_mcp._app import mcp
from zotero_mcp import client as _client
from zotero_mcp import utils as _utils
from zotero_mcp.tools import _helpers


@mcp.tool(
    name="zotero_get_item_metadata",
    description="Get detailed metadata for a specific Zotero item by its key. If the metadata and abstract don't contain the specific information you need, use zotero_get_item_fulltext to read the full paper — but note that fulltext retrieval is resource-intensive and should not be used for searching; use zotero_search_items or zotero_semantic_search instead."
)
def get_item_metadata(
    item_key: str,
    include_abstract: bool = True,
    format: Literal["markdown", "bibtex"] = "markdown",
    *,
    ctx: Context
) -> str:
    """
    Get detailed metadata for a Zotero item.

    Args:
        item_key: Zotero item key/ID
        include_abstract: Whether to include the abstract in the output (markdown format only)
        format: Output format - 'markdown' for detailed metadata or 'bibtex' for BibTeX citation
        ctx: MCP context

    Returns:
        Formatted item metadata (markdown or BibTeX)
    """
    _ret_logger = _logging.getLogger("zotero_mcp.retrieval")
    try:
        ctx.info(f"Fetching metadata for item {item_key} in {format} format")
        zot = _client.get_zotero_client()

        t0 = _time.monotonic()
        item = zot.item(item_key)
        _ret_logger.debug(f"[METADATA] zot.item({item_key}): {_time.monotonic() - t0:.2f}s")
        if not item:
            return f"No item found with key: {item_key}"

        if format == "bibtex":
            return _client.generate_bibtex(item)
        else:
            return _client.format_item_metadata(item, include_abstract)

    except Exception as e:
        ctx.error(f"Error fetching item metadata: {str(e)}")
        return f"Error fetching item metadata: {str(e)}"


@mcp.tool(
    name="zotero_get_item_fulltext",
    description="Get the full text content of a Zotero item by its key. WARNING: Returns the entire paper text (often 10K+ tokens). Only use when you need to read the actual paper content, not just metadata. Do NOT use this for searching — use zotero_search_items or zotero_semantic_search instead. Avoid calling this on multiple papers in one conversation unless the user specifically asks to read them."
)
def get_item_fulltext(
    item_key: str,
    *,
    ctx: Context
) -> str:
    """
    Get the full text content of a Zotero item.

    Args:
        item_key: Zotero item key/ID
        ctx: MCP context

    Returns:
        Markdown-formatted item full text
    """
    try:
        ctx.info(f"Fetching full text for item {item_key}")
        zot = _client.get_zotero_client()

        # First get the item metadata
        item = zot.item(item_key)
        if not item:
            return f"No item found with key: {item_key}"

        # Get item metadata in markdown format
        metadata = _client.format_item_metadata(item, include_abstract=True)

        # In local mode, prefer direct local DB/storage extraction first.
        # This avoids pyzotero dump() failures on linked file:// attachments
        # when using remote clients over SSE/HTTP.
        local_extract_error_msg = None
        try:
            from zotero_mcp.local_db import LocalZoteroReader

            if _utils.is_local_mode():
                config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
                zotero_db_path = None
                pdf_max_pages = None
                fulltext_display_max = None

                if config_path.exists():
                    try:
                        with open(config_path, encoding="utf-8") as _f:
                            _cfg = json.load(_f)
                            semantic_cfg = _cfg.get("semantic_search", {})
                            zotero_db_path = semantic_cfg.get("zotero_db_path")
                            extraction_cfg = semantic_cfg.get("extraction", {})
                            pdf_max_pages = extraction_cfg.get("pdf_max_pages")
                            # Separate display limit for when Claude reads papers
                            # (reduces token usage vs. indexing which can be higher)
                            fulltext_display_max = extraction_cfg.get(
                                "fulltext_display_max_pages"
                            )
                    except Exception:
                        pass

                # Use display limit if configured, otherwise fall back to
                # pdf_max_pages, with a default cap of 10 pages.
                DEFAULT_FULLTEXT_DISPLAY_MAX = 10
                if fulltext_display_max is not None:
                    pdf_max_pages = fulltext_display_max
                elif pdf_max_pages is None:
                    pdf_max_pages = DEFAULT_FULLTEXT_DISPLAY_MAX

                with LocalZoteroReader(db_path=zotero_db_path, pdf_max_pages=pdf_max_pages) as reader:
                    local_item = reader.get_item_by_key(item_key)
                    if local_item:
                        extracted = reader.extract_fulltext_for_item(local_item.item_id)
                        if extracted and extracted[0]:
                            # Skip timeout sentinel — don't show "__EXTRACTION_TIMEOUT__" as content
                            if isinstance(extracted, tuple) and len(extracted) >= 2 and extracted[1] == "timeout":
                                ctx.info("PDF extraction timed out — skipping local fulltext")
                            else:
                                source = extracted[1] if len(extracted) > 1 else "file"
                                ctx.info(f"Retrieved full text from local storage ({source})")
                                return _helpers._prepend_size_warning(
                                    f"{metadata}\n\n---\n\n## Full Text\n\n{extracted[0]}",
                                    "Consider using zotero_semantic_search to find specific content instead of reading full papers."
                                )
        except Exception as local_extract_error:
            local_extract_error_msg = str(local_extract_error)
            ctx.info(f"Local extraction fallback not available: {str(local_extract_error)}")

        # Try to get attachment details
        attachment = _client.get_attachment_details(zot, item)
        if not attachment:
            return f"{metadata}\n\n---\n\nNo suitable attachment found for this item."

        ctx.info(f"Found attachment: {attachment.key} ({attachment.content_type})")

        # Try fetching full text from Zotero's full text index first
        try:
            full_text_data = zot.fulltext_item(attachment.key)
            if full_text_data and "content" in full_text_data and full_text_data["content"]:
                ctx.info("Successfully retrieved full text from Zotero's index")
                return _helpers._prepend_size_warning(
                    f"{metadata}\n\n---\n\n## Full Text\n\n{full_text_data['content']}",
                    "Consider using zotero_semantic_search to find specific content instead of reading full papers."
                )
        except Exception as fulltext_error:
            ctx.info(f"Couldn't retrieve indexed full text: {str(fulltext_error)}")

        # If we couldn't get indexed full text, try to download and convert the file
        try:
            ctx.info(f"Attempting to download and convert attachment {attachment.key}")

            # Download the file to a temporary location

            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, attachment.filename or f"{attachment.key}.pdf")
                zot.dump(attachment.key, filename=os.path.basename(file_path), path=tmpdir)

                if os.path.exists(file_path):
                    ctx.info(f"Downloaded file to {file_path}, converting to markdown")
                    converted_text = _client.convert_to_markdown(file_path)
                    return _helpers._prepend_size_warning(
                        f"{metadata}\n\n---\n\n## Full Text\n\n{converted_text}",
                        "Consider using zotero_semantic_search to find specific content instead of reading full papers."
                    )
                else:
                    return f"{metadata}\n\n---\n\nFile download failed."
        except Exception as download_error:
            ctx.error(f"Error downloading/converting file: {str(download_error)}")
            if local_extract_error_msg:
                return (
                    f"{metadata}\n\n---\n\nError accessing attachment: {str(download_error)}\n\n"
                    f"Local extraction fallback error: {local_extract_error_msg}"
                )
            return f"{metadata}\n\n---\n\nError accessing attachment: {str(download_error)}"

    except Exception as e:
        ctx.error(f"Error fetching item full text: {str(e)}")
        return f"Error fetching item full text: {str(e)}"


@mcp.tool(
    name="zotero_get_collections",
    description="List all collections in your Zotero library."
)
def get_collections(
    limit: int | str | None = None,
    *,
    ctx: Context
) -> str:
    """
    List all collections in your Zotero library.

    Args:
        limit: Maximum number of collections to return
        ctx: MCP context

    Returns:
        Markdown-formatted list of collections
    """
    try:
        ctx.info("Fetching collections")
        zot = _client.get_zotero_client()

        limit = _helpers._normalize_limit(limit, default=100, max_val=5000)

        collections = _helpers._paginate(zot.collections, max_items=limit)

        # Always return the header, even if empty
        output = ["# Zotero Collections", ""]

        if not collections:
            output.append("No collections found in your Zotero library.")
            return "\n".join(output)

        # Create a mapping of collection IDs to their data
        collection_map = {c["key"]: c for c in collections}

        # Create a mapping of parent to child collections
        # Only add entries for collections that actually exist
        hierarchy = {}
        for coll in collections:
            parent_key = coll["data"].get("parentCollection")
            # Handle various representations of "no parent"
            if parent_key in ["", None] or not parent_key:
                parent_key = None  # Normalize to None

            if parent_key not in hierarchy:
                hierarchy[parent_key] = []
            hierarchy[parent_key].append(coll["key"])

        # Function to recursively format collections
        def format_collection(key, level=0):
            if key not in collection_map:
                return []

            coll = collection_map[key]
            name = coll["data"].get("name", "Unnamed Collection")

            # Create indentation for hierarchy
            indent = "  " * level
            lines = [f"{indent}- **{name}** (Key: {key})"]

            # Add children if they exist
            child_keys = hierarchy.get(key, [])
            for child_key in sorted(child_keys):  # Sort for consistent output
                lines.extend(format_collection(child_key, level + 1))

            return lines

        # Start with top-level collections (those with None as parent)
        top_level_keys = hierarchy.get(None, [])

        if not top_level_keys:
            # If no clear hierarchy, just list all collections
            output.append("Collections (flat list):")
            for coll in sorted(collections, key=lambda x: x["data"].get("name", "")):
                name = coll["data"].get("name", "Unnamed Collection")
                key = coll["key"]
                output.append(f"- **{name}** (Key: {key})")
        else:
            # Display hierarchical structure
            for key in sorted(top_level_keys):
                output.extend(format_collection(key))

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error fetching collections: {str(e)}")
        error_msg = f"Error fetching collections: {str(e)}"
        return f"# Zotero Collections\n\n{error_msg}"


def _build_attachment_extra(info):
    """Build extra_fields dict from attachment_info for format_item_result."""
    if not info:
        return None
    parts = []
    if info.get("has_pdf"):
        parts.append("PDF")
    att_count = info.get("attachment_count", 0)
    if att_count:
        parts.append(f"{att_count} attachment{'s' if att_count != 1 else ''}")
    if info.get("has_notes"):
        parts.append("has notes")
    return {"Attachments": ", ".join(parts)} if parts else None


@mcp.tool(
    name="zotero_get_collection_items",
    description="Get all items in a specific Zotero collection. Supports detail='keys_only' (minimal), 'summary' (default, no abstracts), or 'full' (with abstracts). Includes PDF/notes indicators. TIP: To find papers on a specific topic, use zotero_semantic_search instead — it's faster and returns only relevant results."
)
def get_collection_items(
    collection_key: str,
    detail: Literal["keys_only", "summary", "full"] = "summary",
    limit: int | str | None = 50,
    *,
    ctx: Context
) -> str:
    """
    Get all items in a specific Zotero collection.

    Args:
        collection_key: The collection key/ID
        limit: Maximum number of items to return
        ctx: MCP context

    Returns:
        Markdown-formatted list of items in the collection
    """
    try:
        ctx.info(f"Fetching items for collection {collection_key}")
        zot = _client.get_zotero_client()

        # First get the collection details
        try:
            collection = zot.collection(collection_key)
            collection_name = collection["data"].get("name", "Unnamed Collection")
        except Exception:
            collection_name = f"Collection {collection_key}"

        limit = _helpers._normalize_limit(limit, default=50)

        # Fetch all items (includes children mixed in with parents)
        all_items = _helpers._paginate(zot.collection_items, collection_key)
        if not all_items:
            return f"No items found in collection: {collection_name} (Key: {collection_key})"

        # Build attachment/note summary from already-fetched children (zero extra API calls)
        attachment_info = {}
        for item in all_items:
            data = item.get("data", {})
            item_type = data.get("itemType", "")
            parent_key = data.get("parentItem", "")
            if not parent_key:
                continue
            if parent_key not in attachment_info:
                attachment_info[parent_key] = {
                    "has_pdf": False, "attachment_count": 0, "has_notes": False
                }
            if item_type == "attachment":
                attachment_info[parent_key]["attachment_count"] += 1
                if data.get("contentType", "") == "application/pdf":
                    attachment_info[parent_key]["has_pdf"] = True
            elif item_type == "note":
                attachment_info[parent_key]["has_notes"] = True

        # Filter to parent items only (exclude attachments, notes, annotations)
        child_types = {"attachment", "note", "annotation"}
        parent_items = [
            item for item in all_items
            if item.get("data", {}).get("itemType", "") not in child_types
        ]

        if not parent_items:
            return f"No items found in collection: {collection_name} (Key: {collection_key})"

        # Apply display limit after filtering
        if limit and len(parent_items) > limit:
            display_items = parent_items[:limit]
            truncated = True
        else:
            display_items = parent_items
            truncated = False

        # Format items as markdown based on detail level
        output = [f"# Items in Collection: {collection_name} ({len(parent_items)} items)", ""]

        for i, item in enumerate(display_items, 1):
            key = item.get("key", "")
            info = attachment_info.get(key, {})

            if detail == "keys_only":
                data = item.get("data", {})
                title = data.get("title", "Untitled")
                date = data.get("date", "")
                flags = []
                if info.get("has_pdf"):
                    flags.append("PDF")
                if info.get("has_notes"):
                    flags.append("Notes")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                output.append(f"- `{key}` | {title} ({date}){flag_str}")

            elif detail == "full":
                extra = _build_attachment_extra(info)
                output.extend(_utils.format_item_result(
                    item, index=i, abstract_len=None, include_tags=True,
                    extra_fields=extra
                ))

            else:  # "summary" (default)
                extra = _build_attachment_extra(info)
                output.extend(_utils.format_item_result(
                    item, index=i, abstract_len=0, include_tags=True,
                    extra_fields=extra
                ))

        if truncated:
            output.append(f"\n*Showing {limit} of {len(parent_items)} items. Increase the limit parameter to see more.*")

        result = "\n".join(output)
        if detail == "full":
            result = _helpers._prepend_size_warning(
                result,
                'Use detail="summary" for a lighter response.'
            )
        return result

    except Exception as e:
        ctx.error(f"Error fetching collection items: {str(e)}")
        return f"Error fetching collection items: {str(e)}"


@mcp.tool(
    name="zotero_get_item_children",
    description="Get all child items (attachments, notes) for a specific Zotero item."
)
def get_item_children(
    item_key: str,
    *,
    ctx: Context
) -> str:
    """
    Get all child items (attachments, notes) for a specific Zotero item.

    Args:
        item_key: Zotero item key/ID
        ctx: MCP context

    Returns:
        Markdown-formatted list of child items
    """
    try:
        ctx.info(f"Fetching children for item {item_key}")
        zot = _client.get_zotero_client()

        # First get the parent item details
        try:
            parent = zot.item(item_key)
            parent_title = parent["data"].get("title", "Untitled Item")
        except Exception:
            parent_title = f"Item {item_key}"

        # Then get the children
        children = zot.children(item_key)
        if not children:
            return f"No child items found for: {parent_title} (Key: {item_key})"

        # Format children as markdown
        output = [f"# Child Items for: {parent_title}", ""]

        # Group children by type
        attachments = []
        notes = []
        others = []

        for child in children:
            data = child.get("data", {})
            item_type = data.get("itemType", "unknown")

            if item_type == "attachment":
                attachments.append(child)
            elif item_type == "note":
                notes.append(child)
            else:
                others.append(child)

        # Format attachments
        if attachments:
            output.append("## Attachments")
            for i, att in enumerate(attachments, 1):
                data = att.get("data", {})
                title = data.get("title", "Untitled")
                key = att.get("key", "")
                content_type = data.get("contentType", "Unknown")
                filename = data.get("filename", "")

                output.append(f"{i}. **{title}**")
                output.append(f"   - Key: {key}")
                output.append(f"   - Type: {content_type}")
                if filename:
                    output.append(f"   - Filename: {filename}")
                output.append("")

        # Format notes
        if notes:
            output.append("## Notes")
            for i, note in enumerate(notes, 1):
                data = note.get("data", {})
                title = data.get("title", "Untitled Note")
                key = note.get("key", "")
                note_text = data.get("note", "")

                # Clean up HTML in notes
                note_text = note_text.replace("<p>", "").replace("</p>", "\n\n")
                note_text = note_text.replace("<br/>", "\n").replace("<br>", "\n")

                # Limit note length for display
                if len(note_text) > 500:
                    note_text = note_text[:500] + "...\n\n(Note truncated)"

                output.append(f"{i}. **{title}**")
                output.append(f"   - Key: {key}")
                output.append(f"   - Content:\n```\n{note_text}\n```")
                output.append("")

        # Format other item types
        if others:
            output.append("## Other Items")
            for i, other in enumerate(others, 1):
                data = other.get("data", {})
                title = data.get("title", "Untitled")
                key = other.get("key", "")
                item_type = data.get("itemType", "unknown")

                output.append(f"{i}. **{title}**")
                output.append(f"   - Key: {key}")
                output.append(f"   - Type: {item_type}")
                output.append("")

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error fetching item children: {str(e)}")
        return f"Error fetching item children: {str(e)}"


@mcp.tool(
    name="zotero_get_items_children",
    description="Get child items (attachments, notes) for MULTIPLE Zotero items in one call. Much more efficient than calling get_item_children repeatedly."
)
def get_items_children(
    item_keys: list[str] | str,
    *,
    ctx: Context
) -> str:
    """
    Get child items for multiple Zotero items in a single call.

    Args:
        item_keys: List of item keys (or JSON string, or comma-separated string)
        ctx: MCP context
    """
    try:
        zot = _client.get_zotero_client()
        keys = _helpers._normalize_str_list_input(item_keys, "item_keys")

        if not keys:
            return "Error: No item keys provided."

        # Batch-resolve parent titles (50 per API call)
        parent_titles = {}
        for batch_start in range(0, len(keys), 50):
            batch = keys[batch_start:batch_start + 50]
            try:
                items = zot.items(itemKey=",".join(batch))
                for item in items:
                    k = item.get("key", "")
                    parent_titles[k] = item.get("data", {}).get("title", "Untitled")
            except Exception as e:
                ctx.warning(f"Batch parent lookup failed: {e}")
                for k in batch:
                    parent_titles.setdefault(k, f"(key: {k})")

        output = [f"# Children for {len(keys)} items", ""]

        for key in keys:
            title = parent_titles.get(key, f"(key: {key})")
            output.append(f"## {title} (`{key}`)")

            try:
                children = zot.children(key)
            except Exception as e:
                output.append(f"  Error fetching children: {e}")
                output.append("")
                continue

            if not children:
                output.append("  No child items.")
                output.append("")
                continue

            for child in children:
                data = child.get("data", {})
                child_type = data.get("itemType", "unknown")
                child_key = child.get("key", "")

                if child_type == "attachment":
                    ct = data.get("contentType", "")
                    fn = data.get("filename", "")
                    link = data.get("linkMode", "")
                    output.append(f"  - [{child_key}] Attachment: {fn or '(no filename)'} ({ct}) [{link}]")

                elif child_type == "note":
                    note_text = _utils.clean_html(data.get("note", ""))[:150]
                    output.append(f"  - [{child_key}] Note: {note_text}...")

                elif child_type == "annotation":
                    ann_text = data.get("annotationText", "")[:100]
                    ann_type = data.get("annotationType", "")
                    output.append(f"  - [{child_key}] {ann_type}: {ann_text}...")

                else:
                    output.append(f"  - [{child_key}] {child_type}: {data.get('title', '')}")

            output.append("")

        return "\n".join(output)

    except ValueError as e:
        return f"Input error: {e}"
    except Exception as e:
        ctx.error(f"Error fetching items children: {str(e)}")
        return f"Error fetching items children: {str(e)}"


@mcp.tool(
    name="zotero_get_tags",
    description="Get all tags used in your Zotero library."
)
def get_tags(
    limit: int | str | None = None,
    *,
    ctx: Context
) -> str:
    """
    Get all tags used in your Zotero library.

    Args:
        limit: Maximum number of tags to return
        ctx: MCP context

    Returns:
        Markdown-formatted list of tags
    """
    try:
        ctx.info("Fetching tags")
        zot = _client.get_zotero_client()

        limit = _helpers._normalize_limit(limit, default=500, max_val=5000)

        # Use _paginate instead of zot.everything() to avoid RLock pickling
        tags = _helpers._paginate(zot.tags)
        if not tags:
            return "No tags found in your Zotero library."

        # Format tags as markdown
        total_count = len(tags)
        output = [f"# Zotero Tags ({total_count} total)", ""]

        # Sort tags alphabetically
        sorted_tags = sorted(tags)

        # Apply display limit
        truncated = False
        if limit and len(sorted_tags) > limit:
            sorted_tags = sorted_tags[:limit]
            truncated = True

        # Group tags alphabetically
        current_letter = None
        for tag in sorted_tags:
            first_letter = tag[0].upper() if tag else "#"

            if first_letter != current_letter:
                current_letter = first_letter
                output.append(f"## {current_letter}")

            output.append(f"- `{tag}`")

        if truncated:
            output.append(f"\n*Showing {limit} of {total_count} tags. Increase the limit parameter to see more.*")

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error fetching tags: {str(e)}")
        return f"Error fetching tags: {str(e)}"


@mcp.tool(
    name="zotero_list_libraries",
    description="List all accessible Zotero libraries (user library, group libraries, and RSS feeds). Use this to discover available libraries before switching with zotero_switch_library.",
)
def list_libraries(*, ctx: Context) -> str:
    """
    List all accessible Zotero libraries.

    In local mode, reads directly from the SQLite database.
    In web mode, queries groups via the Zotero API.

    Returns:
        Markdown-formatted list of libraries with item counts.
    """
    try:
        ctx.info("Listing accessible libraries")
        local = os.getenv("ZOTERO_LOCAL", "").lower() in ["true", "yes", "1"]
        override = _client.get_active_library()

        output = ["# Zotero Libraries", ""]

        # Show active library context
        if override:
            output.append(
                f"> **Active library:** ID={override['library_id']}, "
                f"type={override['library_type']}"
            )
            output.append("")

        if local:
            from zotero_mcp.local_db import LocalZoteroReader

            reader = LocalZoteroReader()
            try:
                libraries = reader.get_libraries()

                # User library
                user_libs = [l for l in libraries if l["type"] == "user"]
                if user_libs:
                    output.append("## User Library")
                    for lib in user_libs:
                        output.append(
                            f"- **My Library** — {lib['itemCount']} items "
                            f"(libraryID={lib['libraryID']})"
                        )
                    output.append("")

                # Group libraries
                group_libs = [l for l in libraries if l["type"] == "group"]
                if group_libs:
                    output.append("## Group Libraries")
                    for lib in group_libs:
                        desc = f" — {lib['groupDescription']}" if lib.get("groupDescription") else ""
                        output.append(
                            f"- **{lib['groupName']}** — {lib['itemCount']} items "
                            f"(groupID={lib['groupID']}){desc}"
                        )
                    output.append("")

                # Feeds
                feed_libs = [l for l in libraries if l["type"] == "feed"]
                if feed_libs:
                    output.append("## RSS Feeds")
                    for lib in feed_libs:
                        output.append(
                            f"- **{lib['feedName']}** — {lib['itemCount']} items "
                            f"(libraryID={lib['libraryID']})"
                        )
                    output.append("")
            finally:
                reader.close()
        else:
            # Web mode: query groups via pyzotero
            zot = _client.get_zotero_client()
            output.append("## User Library")
            output.append(
                f"- **My Library** (libraryID={os.getenv('ZOTERO_LIBRARY_ID', '?')})"
            )
            output.append("")

            try:
                groups = zot.groups()
                if groups:
                    output.append("## Group Libraries")
                    for group in groups:
                        gdata = group.get("data", {})
                        output.append(
                            f"- **{gdata.get('name', 'Unknown')}** "
                            f"(groupID={group.get('id', '?')})"
                        )
                    output.append("")
            except Exception:
                output.append("*Could not retrieve group libraries.*\n")

            output.append("*Note: RSS feeds are only accessible in local mode.*")

        output.append("")
        output.append(
            "Use `zotero_switch_library` to switch to a different library."
        )

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error listing libraries: {str(e)}")
        return f"Error listing libraries: {str(e)}"


@mcp.tool(
    name="zotero_switch_library",
    description="Switch the active Zotero library context. All subsequent tool calls will operate on the selected library. Use zotero_list_libraries first to see available options. Pass library_type='default' to reset to the original environment variable configuration.",
)
def switch_library(
    library_id: str,
    library_type: str = "group",
    *,
    ctx: Context,
) -> str:
    """
    Switch the active library for all subsequent MCP tool calls.

    Args:
        library_id: The library/group ID to switch to.
            For user library: "0" (local mode) or your user ID (web mode).
            For group libraries: the groupID (e.g. "6069773").
        library_type: "user", "group", or "default" to reset to env var defaults.
        ctx: MCP context

    Returns:
        Confirmation message with active library details.
    """
    try:
        # TODO(human): Implement validate_library_switch() below
        if library_type == "default":
            _client.clear_active_library()
            ctx.info("Reset to default library configuration")
            return (
                "Switched back to default library configuration "
                f"(ZOTERO_LIBRARY_ID={os.getenv('ZOTERO_LIBRARY_ID', '0')}, "
                f"ZOTERO_LIBRARY_TYPE={os.getenv('ZOTERO_LIBRARY_TYPE', 'user')})"
            )

        error = validate_library_switch(library_id, library_type)
        if error:
            return error

        _client.set_active_library(library_id, library_type)
        ctx.info(f"Switched to library {library_id} (type={library_type})")

        # Verify the switch works by making a test call
        try:
            zot = _client.get_zotero_client()
            zot.add_parameters(limit=1)
            zot.items()
            return (
                f"Successfully switched to library **{library_id}** "
                f"(type={library_type}). All tools now operate on this library."
            )
        except Exception as e:
            # Roll back on failure
            _client.clear_active_library()
            return (
                f"Error: Could not access library {library_id} "
                f"(type={library_type}): {e}. Reverted to default library."
            )

    except Exception as e:
        ctx.error(f"Error switching library: {str(e)}")
        return f"Error switching library: {str(e)}"


def validate_library_switch(library_id: str, library_type: str) -> str | None:
    """Validate a library switch request before applying it.

    Returns an error message string if the switch should be rejected,
    or None if the switch is valid and should proceed.
    """
    if library_type not in ("user", "group", "feed"):
        return f"Invalid library_type '{library_type}'. Must be 'user', 'group', or 'feed'."

    # In local mode, verify the library actually exists in the database
    local = os.getenv("ZOTERO_LOCAL", "").lower() in ["true", "yes", "1"]
    if local:
        try:
            from zotero_mcp.local_db import LocalZoteroReader

            reader = LocalZoteroReader()
            try:
                libraries = reader.get_libraries()
                if library_type == "group":
                    valid_ids = {str(l["groupID"]) for l in libraries if l["type"] == "group"}
                    if library_id not in valid_ids:
                        return (
                            f"Group '{library_id}' not found. "
                            f"Available groups: {', '.join(sorted(valid_ids))}"
                        )
                elif library_type == "feed":
                    valid_ids = {str(l["libraryID"]) for l in libraries if l["type"] == "feed"}
                    if library_id not in valid_ids:
                        return (
                            f"Feed with libraryID '{library_id}' not found. "
                            f"Available feeds: {', '.join(sorted(valid_ids))}"
                        )
            finally:
                reader.close()
        except Exception:
            pass  # If DB unavailable, skip validation — the test call will catch it

    return None


@mcp.tool(
    name="zotero_list_feeds",
    description="List all RSS feed subscriptions in your local Zotero installation. Shows feed names, URLs, item counts, and last check times. Local mode only.",
)
def list_feeds(*, ctx: Context) -> str:
    """
    List all RSS feed subscriptions from the local Zotero database.

    Returns:
        Markdown-formatted list of RSS feeds.
    """
    try:
        local = os.getenv("ZOTERO_LOCAL", "").lower() in ["true", "yes", "1"]
        if not local:
            return "RSS feeds are only accessible in local mode (ZOTERO_LOCAL=true)."

        ctx.info("Listing RSS feeds")
        from zotero_mcp.local_db import LocalZoteroReader

        reader = LocalZoteroReader()
        try:
            feeds = reader.get_feeds()
            if not feeds:
                return "No RSS feeds found in your Zotero installation."

            output = ["# RSS Feeds", ""]
            for feed in feeds:
                last_check = feed["lastCheck"] or "never"
                error = f" (error: {feed['lastCheckError']})" if feed.get("lastCheckError") else ""
                output.append(f"### {feed['name']}")
                output.append(f"- **URL:** {feed['url']}")
                output.append(f"- **Items:** {feed['itemCount']}")
                output.append(f"- **Last checked:** {last_check}{error}")
                output.append(f"- **Library ID:** {feed['libraryID']}")
                output.append("")

            output.append(
                "Use `zotero_get_feed_items` with a feed's library ID to view its items."
            )
            return "\n".join(output)
        finally:
            reader.close()

    except Exception as e:
        ctx.error(f"Error listing feeds: {str(e)}")
        return f"Error listing feeds: {str(e)}"


@mcp.tool(
    name="zotero_get_feed_items",
    description="Get items from a specific RSS feed by its library ID. Use zotero_list_feeds first to find feed library IDs. Local mode only.",
)
def get_feed_items(
    library_id: int,
    limit: int = 20,
    *,
    ctx: Context,
) -> str:
    """
    Retrieve items from a specific RSS feed.

    Args:
        library_id: The libraryID of the feed (from zotero_list_feeds).
        limit: Maximum number of items to return.
        ctx: MCP context

    Returns:
        Markdown-formatted list of feed items.
    """
    try:
        local = os.getenv("ZOTERO_LOCAL", "").lower() in ["true", "yes", "1"]
        if not local:
            return "RSS feed items are only accessible in local mode (ZOTERO_LOCAL=true)."

        ctx.info(f"Fetching items from feed (libraryID={library_id})")
        from zotero_mcp.local_db import LocalZoteroReader

        reader = LocalZoteroReader()
        try:
            # Verify this is actually a feed
            feeds = reader.get_feeds()
            feed_info = next((f for f in feeds if f["libraryID"] == library_id), None)
            if not feed_info:
                valid_ids = [str(f["libraryID"]) for f in feeds]
                return (
                    f"No feed found with libraryID={library_id}. "
                    f"Valid feed IDs: {', '.join(valid_ids)}"
                )

            items = reader.get_feed_items(library_id, limit=limit)
            if not items:
                return f"No items found in feed '{feed_info['name']}'."

            output = [f"# Feed: {feed_info['name']}", f"**URL:** {feed_info['url']}", ""]

            for item in items:
                read_status = "Read" if item.get("readTime") else "Unread"
                title = item.get("title") or "Untitled"
                output.append(f"### {title}")
                output.append(f"- **Status:** {read_status}")
                if item.get("creators"):
                    output.append(f"- **Authors:** {item['creators']}")
                if item.get("url"):
                    output.append(f"- **URL:** {item['url']}")
                output.append(f"- **Added:** {item.get('dateAdded', 'unknown')}")
                if item.get("abstract"):
                    abstract = _utils.clean_html(item["abstract"])
                    if len(abstract) > 200:
                        abstract = abstract[:200] + "..."
                    output.append(f"- **Abstract:** {abstract}")
                output.append("")

            return "\n".join(output)
        finally:
            reader.close()

    except Exception as e:
        ctx.error(f"Error fetching feed items: {str(e)}")
        return f"Error fetching feed items: {str(e)}"


@mcp.tool(
    name="zotero_get_recent",
    description="Get recently added items to your Zotero library, or to a specific collection."
)
def get_recent(
    limit: int | str = 10,
    collection_key: str | None = None,
    *,
    ctx: Context
) -> str:
    """
    Get recently added items to your Zotero library.

    Args:
        limit: Number of items to return
        collection_key: Optional collection key to scope results to a specific collection
        ctx: MCP context

    Returns:
        Markdown-formatted list of recent items
    """
    try:
        ctx.info(f"Fetching {limit} recent items")
        zot = _client.get_zotero_client()

        limit = _helpers._normalize_limit(limit, default=10)

        # Get recent items, optionally scoped to a collection
        if collection_key:
            try:
                _col = zot.collection(collection_key)
            except Exception:
                _col = None
            if not _col or _col.get("key") != collection_key:
                return f"Collection not found: '{collection_key}'. Use zotero_get_collections or zotero_search_collections to find valid collection keys."
            items = zot.collection_items(collection_key, sort="dateAdded", direction="desc", limit=limit)
        else:
            items = zot.items(limit=limit, sort="dateAdded", direction="desc")

        if not items:
            return "No items found in your Zotero library." if not collection_key else f"No items found in collection: {collection_key}"

        # Format items as markdown
        scope = f" in Collection {collection_key}" if collection_key else ""
        output = [f"# {limit} Most Recently Added Items{scope}", ""]

        for i, item in enumerate(items, 1):
            added = item.get("data", {}).get("dateAdded", "Unknown")
            output.extend(_utils.format_item_result(
                item, index=i, abstract_len=0, include_tags=False,
                extra_fields={"Added": added},
            ))

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error fetching recent items: {str(e)}")
        return f"Error fetching recent items: {str(e)}"


@mcp.tool(
    name="zotero_get_item_related",
    description="Get all related items for a specific Zotero item. Returns items that are linked via the relations field."
)
def get_item_related(
    item_key: str,
    *,
    ctx: Context
) -> str:
    """
    Get all related items for a specific Zotero item.

    Args:
        item_key: Zotero item key/ID
        ctx: MCP context

    Returns:
        Markdown-formatted list of related items
    """
    try:
        ctx.info(f"Fetching related items for {item_key}")
        zot = _client.get_zotero_client()

        # Fetch the item
        try:
            item = zot.item(item_key)
        except Exception as e:
            return f"Error: Item '{item_key}' not found."

        data = item.get("data", {})
        item_title = data.get("title", "Untitled")
        relations = data.get("relations", {})

        if not isinstance(relations, dict) or not relations:
            return f"No related items found for: **{item_title}** (Key: `{item_key}`)"

        # Extract related item keys from URIs
        # Zotero uses URIs like: http://zotero.org/users/{library_id}/items/{item_key}
        # or http://zotero.org/groups/{group_id}/items/{item_key}

        related_keys = []
        seen_keys = set()
        for rel_type, rel_values in relations.items():
            if not isinstance(rel_values, list):
                rel_values = [rel_values]
            for uri in rel_values:
                if not isinstance(uri, str):
                    continue
                # Extract item key from URI
                match = re.search(r'/items/([A-Z0-9]{8})$', uri)
                if match:
                    key = match.group(1)
                    # Deduplicate: same key may appear with both users/ and groups/ prefix
                    dedup_id = (rel_type, key)
                    if dedup_id not in seen_keys:
                        seen_keys.add(dedup_id)
                        related_keys.append((rel_type, key, uri))

        if not related_keys:
            return f"No related items found for: **{item_title}** (Key: `{item_key}`)"

        # Fetch details for related items
        output = [f"# Related Items for: {item_title}", f"**Item Key:** `{item_key}`", ""]

        # Group by relation type
        by_type = {}
        for rel_type, key, uri in related_keys:
            if rel_type not in by_type:
                by_type[rel_type] = []
            by_type[rel_type].append((key, uri))

        for rel_type, items in by_type.items():
            output.append(f"## Relation Type: `{rel_type}`")
            output.append("")

            for rel_key, uri in items:
                try:
                    rel_item = zot.item(rel_key)
                    rel_data = rel_item.get("data", {})
                    rel_title = rel_data.get("title", "Untitled")
                    rel_type_name = rel_data.get("itemType", "unknown")
                    rel_date = rel_data.get("date", "")
                    rel_creator = ""
                    if rel_data.get("creators"):
                        first_creator = rel_data["creators"][0]
                        if "lastName" in first_creator:
                            rel_creator = first_creator["lastName"]
                        elif "name" in first_creator:
                            rel_creator = first_creator["name"]

                    creator_info = f", {rel_creator}" if rel_creator else ""
                    date_info = f" ({rel_date})" if rel_date else ""

                    output.append(f"- `{rel_key}` — **{rel_title}**{creator_info}{date_info}")
                    output.append(f"  - Type: {rel_type_name}")
                    if doi := rel_data.get("DOI"):
                        output.append(f"  - DOI: {doi}")
                    output.append("")
                except Exception as e:
                    output.append(f"- `{rel_key}` — (Could not fetch details: {e})")
                    output.append("")

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error fetching related items: {str(e)}")
        return f"Error fetching related items: {str(e)}"
