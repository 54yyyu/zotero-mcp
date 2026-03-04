"""
Zotero MCP server implementation.

Note: ChatGPT requires specific tool names "search" and "fetch", and so they
are defined and used and piped through to the main server tools. See bottom of file for details.
"""

from typing import Dict, List, Literal, Optional, Union, Tuple
import os
import sys
import uuid
import asyncio
import json
import re
import xml.etree.ElementTree as ET
from urllib.parse import urljoin
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastmcp import Context, FastMCP

# Compatibility: some fastmcp Context versions don't expose `warn`.
# Provide a lightweight shim that maps warn -> warning -> info.
if not hasattr(Context, "warn"):
    def _ctx_warn(self, message: str):
        try:
            if hasattr(self, "warning"):
                return self.warning(message)
            if hasattr(self, "info"):
                return self.info(f"WARNING: {message}")
        except Exception:
            # Silently ignore logging failures to avoid impacting tool behavior
            return None

    setattr(Context, "warn", _ctx_warn)

from zotero_mcp.client import (
    clear_active_library,
    convert_to_markdown,
    format_item_metadata,
    generate_bibtex,
    get_active_library,
    get_attachment_details,
    get_zotero_client,
    set_active_library,
)
import requests

from zotero_mcp.utils import format_creators, clean_html, is_local_mode

@asynccontextmanager
async def server_lifespan(server: FastMCP):
    """Manage server startup and shutdown lifecycle."""
    sys.stderr.write("Starting Zotero MCP server...\n")
    background_task: asyncio.Task | None = None

    # Check for semantic search auto-update on startup
    try:
        from zotero_mcp.semantic_search import create_semantic_search

        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"

        if config_path.exists():
            search = create_semantic_search(str(config_path))

            if search.should_update_database():
                sys.stderr.write("Auto-updating semantic search database...\n")

                # Run update in background to avoid blocking server startup
                async def background_update():
                    try:
                        # Run sync indexing work in a worker thread.
                        stats = await asyncio.to_thread(
                            search.update_database, extract_fulltext=False
                        )
                        sys.stderr.write(f"Database update completed: {stats.get('processed_items', 0)} items processed\n")
                    except Exception as e:
                        sys.stderr.write(f"Background database update failed: {e}\n")

                # Start background task
                background_task = asyncio.create_task(background_update())

    except Exception as e:
        sys.stderr.write(f"Warning: Could not check semantic search auto-update: {e}\n")

    yield {}

    if background_task and not background_task.done():
        background_task.cancel()
        with suppress(asyncio.CancelledError):
            await background_task

    sys.stderr.write("Shutting down Zotero MCP server...\n")


# Create an MCP server (fastmcp 2.14+ no longer accepts `dependencies`)
mcp = FastMCP("Zotero", lifespan=server_lifespan)


def _normalize_str_list_input(
    value: Union[List[str], str, None],
    *,
    field_name: str
) -> List[str]:
    """Normalize list-like user input into a trimmed list of non-empty strings."""
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []

        parsed: object = None
        parsed_ok = False
        try:
            parsed = json.loads(raw)
            parsed_ok = True
        except json.JSONDecodeError:
            parsed_ok = False

        if parsed_ok:
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
            if isinstance(parsed, str):
                parsed_str = parsed.strip()
                return [parsed_str] if parsed_str else []
            raise ValueError(f"{field_name} must be a list of strings or a string")

        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) > 1:
            return parts
        return [raw]

    raise ValueError(f"{field_name} must be a list of strings or a string")


@mcp.tool(
    name="zotero_search_items",
    description="Search for items in your Zotero library, given a query string."
)
def search_items(
    query: str,
    qmode: Literal["titleCreatorYear", "everything"] = "titleCreatorYear",
    item_type: str = "-attachment",  # Exclude attachments by default
    limit: int | str | None = 10,
    tag: list[str] | None = None,
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
        tag: List of tags conditions to filter by
        ctx: MCP context

    Returns:
        Markdown-formatted search results
    """
    try:
        if not query.strip():
            return "Error: Search query cannot be empty"

        tag_condition_str = ""
        if tag:
            tag_condition_str = f" with tags: '{', '.join(tag)}'"
        else :
            tag = []

        ctx.info(f"Searching Zotero for '{query}'{tag_condition_str}")
        zot = get_zotero_client()

        if isinstance(limit, str):
            limit = int(limit)

        # Search using the query parameters
        zot.add_parameters(q=query, qmode=qmode, itemType=item_type, limit=limit, tag=tag)
        results = zot.items()

        if not results:
            return f"No items found matching query: '{query}'{tag_condition_str}"

        # Format results as markdown
        output = [f"# Search Results for '{query}'", f"{tag_condition_str}", ""]

        for i, item in enumerate(results, 1):
            data = item.get("data", {})
            title = data.get("title", "Untitled")
            item_type = data.get("itemType", "unknown")
            date = data.get("date", "No date")
            key = item.get("key", "")

            # Format creators
            creators = data.get("creators", [])
            creators_str = format_creators(creators)

            # Build the formatted entry
            output.append(f"## {i}. {title}")
            output.append(f"**Type:** {item_type}")
            output.append(f"**Item Key:** {key}")
            output.append(f"**Date:** {date}")
            output.append(f"**Authors:** {creators_str}")

            # Add abstract snippet if present
            if abstract := data.get("abstractNote"):
                # Limit abstract length for search results
                abstract_snippet = abstract[:200] + "..." if len(abstract) > 200 else abstract
                output.append(f"**Abstract:** {abstract_snippet}")

            # Add tags if present
            if tags := data.get("tags"):
                tag_list = [f"`{tag['tag']}`" for tag in tags]
                if tag_list:
                    output.append(f"**Tags:** {' '.join(tag_list)}")

            output.append("")  # Empty line between items

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error searching Zotero: {str(e)}")
        return f"Error searching Zotero: {str(e)}"

@mcp.tool(
    name="zotero_search_by_tag",
    description="Search for items in your Zotero library by tag. "
    "Conditions are ANDed, each term supports disjunction (`OR`) and exclusion (`-`)."
)
def search_by_tag(
    tag: list[str],
    item_type: str = "-attachment",
    limit: int | str | None = 10,
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
        ctx: MCP context

    Returns:
        Markdown-formatted search results
    """
    try:
        if not tag:
            return "Error: Tag cannot be empty"

        ctx.info(f"Searching Zotero for tag '{tag}'")
        zot = get_zotero_client()

        if isinstance(limit, str):
            limit = int(limit)

        # Search using the query parameters
        zot.add_parameters(q="", tag=tag, itemType=item_type, limit=limit)
        results = zot.items()

        if not results:
            return f"No items found with tag: '{tag}'"

        # Format results as markdown
        output = [f"# Search Results for Tag: '{tag}'", ""]

        for i, item in enumerate(results, 1):
            data = item.get("data", {})
            title = data.get("title", "Untitled")
            item_type = data.get("itemType", "unknown")
            date = data.get("date", "No date")
            key = item.get("key", "")

            # Format creators
            creators = data.get("creators", [])
            creators_str = format_creators(creators)

            # Build the formatted entry
            output.append(f"## {i}. {title}")
            output.append(f"**Type:** {item_type}")
            output.append(f"**Item Key:** {key}")
            output.append(f"**Date:** {date}")
            output.append(f"**Authors:** {creators_str}")

            # Add abstract snippet if present
            if abstract := data.get("abstractNote"):
                # Limit abstract length for search results
                abstract_snippet = abstract[:200] + "..." if len(abstract) > 200 else abstract
                output.append(f"**Abstract:** {abstract_snippet}")

            # Add tags if present
            if tags := data.get("tags"):
                tag_list = [f"`{tag['tag']}`" for tag in tags]
                if tag_list:
                    output.append(f"**Tags:** {' '.join(tag_list)}")

            output.append("")  # Empty line between items

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error searching Zotero: {str(e)}")
        return f"Error searching Zotero: {str(e)}"

@mcp.tool(
    name="zotero_get_item_metadata",
    description="Get detailed metadata for a specific Zotero item by its key."
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
    try:
        ctx.info(f"Fetching metadata for item {item_key} in {format} format")
        zot = get_zotero_client()

        item = zot.item(item_key)
        if not item:
            return f"No item found with key: {item_key}"

        if format == "bibtex":
            return generate_bibtex(item)
        else:
            return format_item_metadata(item, include_abstract)

    except Exception as e:
        ctx.error(f"Error fetching item metadata: {str(e)}")
        return f"Error fetching item metadata: {str(e)}"


@mcp.tool(
    name="zotero_get_item_fulltext",
    description="Get the full text content of a Zotero item by its key."
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
        zot = get_zotero_client()

        # First get the item metadata
        item = zot.item(item_key)
        if not item:
            return f"No item found with key: {item_key}"

        # Get item metadata in markdown format
        metadata = format_item_metadata(item, include_abstract=True)

        # Try to get attachment details
        attachment = get_attachment_details(zot, item)
        if not attachment:
            return f"{metadata}\n\n---\n\nNo suitable attachment found for this item."

        ctx.info(f"Found attachment: {attachment.key} ({attachment.content_type})")

        # Try fetching full text from Zotero's full text index first
        try:
            full_text_data = zot.fulltext_item(attachment.key)
            if full_text_data and "content" in full_text_data and full_text_data["content"]:
                ctx.info("Successfully retrieved full text from Zotero's index")
                return f"{metadata}\n\n---\n\n## Full Text\n\n{full_text_data['content']}"
        except Exception as fulltext_error:
            ctx.info(f"Couldn't retrieve indexed full text: {str(fulltext_error)}")

        # If we couldn't get indexed full text, try to download and convert the file
        try:
            ctx.info(f"Attempting to download and convert attachment {attachment.key}")

            # Download the file to a temporary location
            import tempfile
            import os

            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, attachment.filename or f"{attachment.key}.pdf")
                zot.dump(attachment.key, filename=os.path.basename(file_path), path=tmpdir)

                if os.path.exists(file_path):
                    ctx.info(f"Downloaded file to {file_path}, converting to markdown")
                    converted_text = convert_to_markdown(file_path)
                    return f"{metadata}\n\n---\n\n## Full Text\n\n{converted_text}"
                else:
                    return f"{metadata}\n\n---\n\nFile download failed."
        except Exception as download_error:
            ctx.error(f"Error downloading/converting file: {str(download_error)}")
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
        zot = get_zotero_client()

        if isinstance(limit, str):
            limit = int(limit)

        collections = zot.collections(limit=limit)

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


@mcp.tool(
    name="zotero_search_collections",
    description="Search for collections by name to find their keys for adding items."
)
def search_collections(
    query: str,
    *,
    ctx: Context
) -> str:
    """
    Search for collections by name.

    Args:
        query: Collection name or partial name to search for
        ctx: MCP context

    Returns:
        Markdown-formatted list of matching collections with their keys
    """
    try:
        ctx.info(f"Searching collections for '{query}'")
        zot = get_zotero_client()

        # Get all collections
        collections = zot.collections()

        if not collections:
            return "No collections found in your Zotero library."

        # Filter collections by name (case-insensitive)
        query_lower = query.lower()
        matching = [
            c for c in collections
            if query_lower in c["data"].get("name", "").lower()
        ]

        if not matching:
            return f"No collections found matching '{query}'"

        # Format results
        output = [f"# Collections matching '{query}'", ""]

        for i, coll in enumerate(matching, 1):
            name = coll["data"].get("name", "Unnamed Collection")
            key = coll["key"]
            parent_key = coll["data"].get("parentCollection")

            output.append(f"## {i}. {name}")
            output.append(f"**Key:** `{key}`")

            # Show parent collection if exists
            if parent_key:
                try:
                    parent = zot.collection(parent_key)
                    parent_name = parent["data"].get("name", "Unknown")
                    output.append(f"**Parent Collection:** {parent_name}")
                except Exception:
                    output.append(f"**Parent Collection Key:** {parent_key}")

            output.append("")

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error searching collections: {str(e)}")
        return f"Error searching collections: {str(e)}"


@mcp.tool(
    name="zotero_create_collection",
    description="Create a new collection (project/folder) in your Zotero library."
)
def create_collection(
    name: str,
    parent_collection: Optional[str] = None,
    *,
    ctx: Context
) -> str:
    """
    Create a new collection in your Zotero library.

    Args:
        name: Name of the collection to create
        parent_collection: Optional parent collection key to create a subcollection
        ctx: MCP context

    Returns:
        Confirmation message with the new collection key
    """
    try:
        ctx.info(f"Creating collection '{name}'")
        zot = get_zotero_client()

        # Build collection data
        collection_data = {"name": name}

        if parent_collection:
            collection_data["parentCollection"] = parent_collection

        # Create the collection
        result = zot.create_collections([collection_data])

        # Check if creation was successful
        if "success" in result and result["success"]:
            successful = result["success"]
            if len(successful) > 0:
                collection_key = next(iter(successful.values()))
                parent_info = f" as subcollection of {parent_collection}" if parent_collection else ""
                return f"Successfully created collection: \"{name}\"{parent_info}\n\nCollection key: `{collection_key}`\n\nYou can now use this key to add items to this collection."
            else:
                return f"Collection creation response was successful but no key was returned: {result}"
        else:
            return f"Failed to create collection: {result.get('failed', 'Unknown error')}"

    except Exception as e:
        ctx.error(f"Error creating collection: {str(e)}")
        return f"Error creating collection: {str(e)}"


@mcp.tool(
    name="zotero_add_items_to_collection",
    description="Add existing items to a collection by their keys."
)
def add_items_to_collection(
    collection_key: str,
    item_keys: List[str],
    *,
    ctx: Context
) -> str:
    """
    Add existing items to a collection.

    Args:
        collection_key: The collection key to add items to
        item_keys: List of item keys to add to the collection
        ctx: MCP context

    Returns:
        Confirmation message
    """
    try:
        ctx.info(f"Adding {len(item_keys)} items to collection {collection_key}")
        zot = get_zotero_client()

        # Verify collection exists
        try:
            collection = zot.collection(collection_key)
            collection_name = collection["data"].get("name", "Unknown Collection")
        except Exception:
            return f"Error: Collection with key '{collection_key}' not found"

        # Add items to collection
        result = zot.addto_collection(collection_key, item_keys)

        if result:
            return f"Successfully added {len(item_keys)} item(s) to collection: \"{collection_name}\""
        else:
            return f"Failed to add items to collection"

    except Exception as e:
        ctx.error(f"Error adding items to collection: {str(e)}")
        return f"Error adding items to collection: {str(e)}"


@mcp.tool(
    name="zotero_get_collection_items",
    description="Get all items in a specific Zotero collection."
)
def get_collection_items(
    collection_key: str,
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
        zot = get_zotero_client()

        # First get the collection details
        try:
            collection = zot.collection(collection_key)
            collection_name = collection["data"].get("name", "Unnamed Collection")
        except Exception:
            collection_name = f"Collection {collection_key}"

        if isinstance(limit, str):
            limit = int(limit)

        # Then get the items
        items = zot.collection_items(collection_key, limit=limit)
        if not items:
            return f"No items found in collection: {collection_name} (Key: {collection_key})"

        # Format items as markdown
        output = [f"# Items in Collection: {collection_name}", ""]

        for i, item in enumerate(items, 1):
            data = item.get("data", {})
            title = data.get("title", "Untitled")
            item_type = data.get("itemType", "unknown")
            date = data.get("date", "No date")
            key = item.get("key", "")

            # Format creators
            creators = data.get("creators", [])
            creators_str = format_creators(creators)

            # Build the formatted entry
            output.append(f"## {i}. {title}")
            output.append(f"**Type:** {item_type}")
            output.append(f"**Item Key:** {key}")
            output.append(f"**Date:** {date}")
            output.append(f"**Authors:** {creators_str}")

            output.append("")  # Empty line between items

        return "\n".join(output)

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
        zot = get_zotero_client()

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
        zot = get_zotero_client()

        if isinstance(limit, str):
            limit = int(limit)

        tags = zot.tags(limit=limit)
        if not tags:
            return "No tags found in your Zotero library."

        # Format tags as markdown
        output = ["# Zotero Tags", ""]

        # Sort tags alphabetically
        sorted_tags = sorted(tags)

        # Group tags alphabetically
        current_letter = None
        for tag in sorted_tags:
            first_letter = tag[0].upper() if tag else "#"

            if first_letter != current_letter:
                current_letter = first_letter
                output.append(f"## {current_letter}")

            output.append(f"- `{tag}`")

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
        override = get_active_library()

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
            zot = get_zotero_client()
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
            clear_active_library()
            ctx.info("Reset to default library configuration")
            return (
                "Switched back to default library configuration "
                f"(ZOTERO_LIBRARY_ID={os.getenv('ZOTERO_LIBRARY_ID', '0')}, "
                f"ZOTERO_LIBRARY_TYPE={os.getenv('ZOTERO_LIBRARY_TYPE', 'user')})"
            )

        error = validate_library_switch(library_id, library_type)
        if error:
            return error

        set_active_library(library_id, library_type)
        ctx.info(f"Switched to library {library_id} (type={library_type})")

        # Verify the switch works by making a test call
        try:
            zot = get_zotero_client()
            zot.add_parameters(limit=1)
            zot.items()
            return (
                f"Successfully switched to library **{library_id}** "
                f"(type={library_type}). All tools now operate on this library."
            )
        except Exception as e:
            # Roll back on failure
            clear_active_library()
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
                    abstract = clean_html(item["abstract"])
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
    description="Get recently added items to your Zotero library."
)
def get_recent(
    limit: int | str = 10,
    *,
    ctx: Context
) -> str:
    """
    Get recently added items to your Zotero library.

    Args:
        limit: Number of items to return
        ctx: MCP context

    Returns:
        Markdown-formatted list of recent items
    """
    try:
        ctx.info(f"Fetching {limit} recent items")
        zot = get_zotero_client()

        if isinstance(limit, str):
            limit = int(limit)

        # Ensure limit is a reasonable number
        if limit <= 0:
            limit = 10
        elif limit > 100:
            limit = 100

        # Get recent items
        items = zot.items(limit=limit, sort="dateAdded", direction="desc")
        if not items:
            return "No items found in your Zotero library."

        # Format items as markdown
        output = [f"# {limit} Most Recently Added Items", ""]

        for i, item in enumerate(items, 1):
            data = item.get("data", {})
            title = data.get("title", "Untitled")
            item_type = data.get("itemType", "unknown")
            date = data.get("date", "No date")
            key = item.get("key", "")
            date_added = data.get("dateAdded", "Unknown")

            # Format creators
            creators = data.get("creators", [])
            creators_str = format_creators(creators)

            # Build the formatted entry
            output.append(f"## {i}. {title}")
            output.append(f"**Type:** {item_type}")
            output.append(f"**Item Key:** {key}")
            output.append(f"**Date:** {date}")
            output.append(f"**Added:** {date_added}")
            output.append(f"**Authors:** {creators_str}")

            output.append("")  # Empty line between items

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error fetching recent items: {str(e)}")
        return f"Error fetching recent items: {str(e)}"


@mcp.tool(
    name="zotero_batch_update_tags",
    description="Batch update tags across multiple items matching a search query."
)
def batch_update_tags(
    query: str,
    add_tags: list[str] | str | None = None,
    remove_tags: list[str] | str | None = None,
    limit: int | str = 50,
    *,
    ctx: Context
) -> str:
    """
    Batch update tags across multiple items matching a search query.

    Args:
        query: Search query to find items to update
        add_tags: List of tags to add to matched items (can be list or JSON string)
        remove_tags: List of tags to remove from matched items (can be list or JSON string)
        limit: Maximum number of items to process
        ctx: MCP context

    Returns:
        Summary of the batch update
    """
    try:
        if not query:
            return "Error: Search query cannot be empty"

        if not add_tags and not remove_tags:
            return "Error: You must specify either tags to add or tags to remove"

        def _normalize_tag_list(
            raw_value: list[str] | str | None, field_name: str
        ) -> list[str]:
            if raw_value is None:
                return []

            parsed_value = raw_value
            if isinstance(parsed_value, str):
                try:
                    parsed_value = json.loads(parsed_value)
                    ctx.info(f"Parsed {field_name} from JSON string: {parsed_value}")
                except json.JSONDecodeError:
                    raise ValueError(
                        f"{field_name} appears to be malformed JSON: {raw_value}"
                    )

            if not isinstance(parsed_value, list):
                raise ValueError(
                    f"{field_name} must be a JSON array or a list of strings"
                )

            normalized = []
            for tag_value in parsed_value:
                if not isinstance(tag_value, str):
                    raise ValueError(f"{field_name} entries must all be strings")
                stripped = tag_value.strip()
                if stripped:
                    normalized.append(stripped)
            return normalized

        try:
            add_tags = _normalize_tag_list(add_tags, "add_tags")
            remove_tags = _normalize_tag_list(remove_tags, "remove_tags")
        except ValueError as validation_error:
            return f"Error: {validation_error}"

        if not add_tags and not remove_tags:
            return "Error: After parsing, no valid tags were provided to add or remove"

        ctx.info(f"Batch updating tags for items matching '{query}'")
        zot = get_zotero_client()

        if isinstance(limit, str):
            limit = int(limit)

        # Search for items matching the query
        zot.add_parameters(q=query, limit=limit)
        items = zot.items()

        if not items:
            return f"No items found matching query: '{query}'"

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

            # Process tags to add
            if add_tags:
                for tag in add_tags:
                    if tag and tag not in current_tag_values:
                        current_tags.append({"tag": tag})
                        added_tag_counts[tag] += 1
                        needs_update = True

            # Update the item if needed
            # Since we are logging errors we might as well log the update.
            if needs_update:
                try:
                    item["data"]["tags"] = current_tags
                    ctx.info(f"Updating item {item.get('key', 'unknown')} with tags: {current_tags}")
                    result = zot.update_item(item)
                    ctx.info(f"Update result: {result}")
                    updated_count += 1
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


@mcp.tool(
    name="zotero_advanced_search",
    description="Perform an advanced search with multiple criteria."
)
def advanced_search(
    conditions: list[dict[str, str]],
    join_mode: Literal["all", "any"] = "all",
    sort_by: str | None = None,
    sort_direction: Literal["asc", "desc"] = "asc",
    limit: int | str = 50,
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
        ctx: MCP context

    Returns:
        Markdown-formatted search results
    """
    try:
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

        if isinstance(limit, str):
            limit = int(limit)
        if limit <= 0:
            return "Error: limit must be greater than 0"
        if limit > 500:
            limit = 500

        ctx.info(f"Performing advanced search with {len(conditions)} conditions")
        zot = get_zotero_client()

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

            if field_lower == "year":
                date_value = str(data.get("date", "")).strip()
                return [date_value[:4]] if len(date_value) >= 4 else []

            field_aliases = {
                "itemtype": "itemType",
                "dateadded": "dateAdded",
                "datemodified": "dateModified",
                "doi": "DOI",
            }
            source_field = field_aliases.get(field_lower, field)
            raw_value = data.get(source_field, "")
            if raw_value is None:
                return []
            return [str(raw_value).strip()]

        def _as_float(text: str) -> float | None:
            try:
                return float(text)
            except ValueError:
                return None

        def _compare(candidate: str, expected: str, operation: str) -> bool:
            left = candidate.lower()
            right = expected.lower()

            if operation == "is":
                return left == right
            if operation == "isNot":
                return left != right
            if operation == "contains":
                return right in left
            if operation == "doesNotContain":
                return right not in left
            if operation == "beginsWith":
                return left.startswith(right)
            if operation == "endsWith":
                return left.endswith(right)

            left_num = _as_float(left)
            right_num = _as_float(right)
            if (
                operation in {"isGreaterThan", "isLessThan", "isBefore", "isAfter"}
                and left_num is not None
                and right_num is not None
            ):
                if operation in {"isGreaterThan", "isAfter"}:
                    return left_num > right_num
                return left_num < right_num

            if operation in {"isGreaterThan", "isAfter"}:
                return left > right
            return left < right

        def _matches_condition(data: dict[str, object], condition: dict[str, str]) -> bool:
            values = _extract_values(data, condition["field"])
            if not values:
                return False

            operation = condition["operation"]
            target = condition["value"]
            comparisons = [_compare(value, target, operation) for value in values]

            if operation in {"isNot", "doesNotContain"}:
                return all(comparisons)
            return any(comparisons)

        # Execute advanced search by iterating items and filtering client-side.
        results = []
        batch_size = 100
        start = 0
        while True:
            batch = zot.items(start=start, limit=batch_size)
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

            if len(batch) < batch_size:
                break
            start += batch_size

        if sort_by:
            sort_field = sort_by.strip()
            reverse = sort_direction == "desc"

            def _sort_key(item: dict[str, object]) -> str:
                data = item.get("data", {}) if isinstance(item, dict) else {}
                if sort_field in {"creator", "author"}:
                    return format_creators(data.get("creators", []))
                return str(data.get(sort_field, "")).lower()

            results.sort(key=_sort_key, reverse=reverse)

        if not results:
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
        output.append("")
        output.append("## Results")

        for i, item in enumerate(results, 1):
            data = item.get("data", {})
            title = data.get("title", "Untitled")
            item_type = data.get("itemType", "unknown")
            date = data.get("date", "No date")
            key = item.get("key", "")

            creators = data.get("creators", [])
            creators_str = format_creators(creators)

            output.append(f"### {i}. {title}")
            output.append(f"**Type:** {item_type}")
            output.append(f"**Item Key:** {key}")
            output.append(f"**Date:** {date}")
            output.append(f"**Authors:** {creators_str}")

            if abstract := data.get("abstractNote"):
                abstract_snippet = abstract[:150] + "..." if len(abstract) > 150 else abstract
                output.append(f"**Abstract:** {abstract_snippet}")

            if tags := data.get("tags"):
                tag_list = [f"`{tag['tag']}`" for tag in tags]
                if tag_list:
                    output.append(f"**Tags:** {' '.join(tag_list)}")

            output.append("")

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error in advanced search: {str(e)}")
        return f"Error in advanced search: {str(e)}"


@mcp.tool(
    name="zotero_get_annotations",
    description="Get all annotations for a specific item or across your entire Zotero library."
)
def get_annotations(
    item_key: str | None = None,
    use_pdf_extraction: bool = False,
    limit: int | str | None = None,
    *,
    ctx: Context
) -> str:
    """
    Get annotations from your Zotero library.

    Args:
        item_key: Optional Zotero item key/ID to filter annotations by parent item
        use_pdf_extraction: Whether to attempt direct PDF extraction as a fallback
        limit: Maximum number of annotations to return
        ctx: MCP context

    Returns:
        Markdown-formatted list of annotations
    """
    return _get_annotations(
        item_key=item_key,
        use_pdf_extraction=use_pdf_extraction,
        limit=limit,
        ctx=ctx
    )

def _get_annotations(
    item_key: str | None = None,
    use_pdf_extraction: bool = False,
    limit: int | str | None = None,
    *,
    ctx: Context
) -> str:
    try:
        # Initialize Zotero client
        zot = get_zotero_client()

        # Prepare annotations list
        annotations = []
        parent_title = "Untitled Item"

        # If an item key is provided, use specialized retrieval
        if item_key:
            # First, verify the item exists and get its details
            try:
                parent = zot.item(item_key)
                parent_title = parent["data"].get("title", "Untitled Item")
                ctx.info(f"Fetching annotations for item: {parent_title}")
            except Exception:
                return f"Error: No item found with key: {item_key}"

            # Initialize annotation sources
            better_bibtex_annotations = []
            zotero_api_annotations = []
            pdf_annotations = []

            # Try Better BibTeX method (local Zotero only)
            if os.environ.get("ZOTERO_LOCAL", "").lower() in ["true", "yes", "1"]:
                try:
                    # Import Better BibTeX dependencies
                    from zotero_mcp.better_bibtex_client import (
                        ZoteroBetterBibTexAPI,
                        process_annotation,
                        get_color_category
                    )

                    # Initialize Better BibTeX client
                    bibtex = ZoteroBetterBibTexAPI()

                    # Check if Zotero with Better BibTeX is running
                    if bibtex.is_zotero_running():
                        # Extract citation key
                        citation_key = None

                        # Try to find citation key in Extra field
                        try:
                            extra_field = parent["data"].get("extra", "")
                            for line in extra_field.split("\n"):
                                if line.lower().startswith("citation key:"):
                                    citation_key = line.replace("Citation Key:", "").strip()
                                    break
                                elif line.lower().startswith("citationkey:"):
                                    citation_key = line.replace("citationkey:", "").strip()
                                    break
                        except Exception as e:
                            ctx.warn(f"Error extracting citation key from Extra field: {e}")

                        # Fallback to searching by title if no citation key found
                        if not citation_key:
                            title = parent["data"].get("title", "")
                            try:
                                if title:
                                    # Use the search_citekeys method
                                    search_results = bibtex.search_citekeys(title)

                                    # Find the matching item
                                    for result in search_results:
                                        ctx.info(f"Checking result: {result}")

                                        # Try to match with item key if possible
                                        if result.get('citekey'):
                                            citation_key = result['citekey']
                                            break
                            except Exception as e:
                                ctx.warn(f"Error searching for citation key: {e}")

                        # Process annotations if citation key found
                        if citation_key:
                            try:
                                # Determine library
                                library = "*"  # Default all libraries
                                search_results = bibtex._make_request("item.search", [citation_key])
                                if search_results:
                                    matched_item = next((item for item in search_results if item.get('citekey') == citation_key), None)
                                    if matched_item:
                                        library = matched_item.get('library', "*")

                                # Get attachments
                                attachments = bibtex.get_attachments(citation_key, library)

                                # Process annotations from attachments
                                for attachment in attachments:
                                    annotations = bibtex.get_annotations_from_attachment(attachment)

                                    for anno in annotations:
                                        processed = process_annotation(anno, attachment)
                                        if processed:
                                            # Create Zotero-like annotation object
                                            bibtex_anno = {
                                                "key": processed.get("id", ""),
                                                "data": {
                                                    "itemType": "annotation",
                                                    "annotationType": processed.get("type", "highlight"),
                                                    "annotationText": processed.get("annotatedText", ""),
                                                    "annotationComment": processed.get("comment", ""),
                                                    "annotationColor": processed.get("color", ""),
                                                    "parentItem": item_key,
                                                    "tags": [],
                                                    "_pdf_page": processed.get("page", 0),
                                                    "_pageLabel": processed.get("pageLabel", ""),
                                                    "_attachment_title": attachment.get("title", ""),
                                                    "_color_category": get_color_category(processed.get("color", "")),
                                                    "_from_better_bibtex": True
                                                }
                                            }
                                            better_bibtex_annotations.append(bibtex_anno)

                                ctx.info(f"Retrieved {len(better_bibtex_annotations)} annotations via Better BibTeX")
                            except Exception as e:
                                ctx.warn(f"Error processing Better BibTeX annotations: {e}")
                except Exception as bibtex_error:
                    ctx.warn(f"Error initializing Better BibTeX: {bibtex_error}")

            # Fallback to Zotero API annotations
            if not better_bibtex_annotations:
                try:
                    # Get child annotations via Zotero API
                    children = zot.children(item_key)
                    zotero_api_annotations = [
                        item for item in children
                        if item.get("data", {}).get("itemType") == "annotation"
                    ]
                    ctx.info(f"Retrieved {len(zotero_api_annotations)} annotations via Zotero API")
                except Exception as api_error:
                    ctx.warn(f"Error retrieving Zotero API annotations: {api_error}")

            # PDF Extraction fallback
            if use_pdf_extraction and not (better_bibtex_annotations or zotero_api_annotations):
                try:
                    from zotero_mcp.pdfannots_helper import extract_annotations_from_pdf, ensure_pdfannots_installed
                    import tempfile
                    import uuid

                    # Ensure PDF annotation tool is installed
                    if ensure_pdfannots_installed():
                        # Get PDF attachments
                        children = zot.children(item_key)
                        pdf_attachments = [
                            item for item in children
                            if item.get("data", {}).get("contentType") == "application/pdf"
                        ]

                        # Extract annotations from PDFs
                        for attachment in pdf_attachments:
                            with tempfile.TemporaryDirectory() as tmpdir:
                                att_key = attachment.get("key", "")
                                file_path = os.path.join(tmpdir, f"{att_key}.pdf")
                                zot.dump(
                                    att_key,
                                    filename=os.path.basename(file_path),
                                    path=tmpdir,
                                )

                                if os.path.exists(file_path):
                                    extracted = extract_annotations_from_pdf(file_path, tmpdir)

                                    for ext in extracted:
                                        # Skip empty annotations
                                        if not ext.get("annotatedText") and not ext.get("comment"):
                                            continue

                                        # Create Zotero-like annotation object
                                        pdf_anno = {
                                            "key": f"pdf_{att_key}_{ext.get('id', uuid.uuid4().hex[:8])}",
                                            "data": {
                                                "itemType": "annotation",
                                                "annotationType": ext.get("type", "highlight"),
                                                "annotationText": ext.get("annotatedText", ""),
                                                "annotationComment": ext.get("comment", ""),
                                                "annotationColor": ext.get("color", ""),
                                                "parentItem": item_key,
                                                "tags": [],
                                                "_pdf_page": ext.get("page", 0),
                                                "_from_pdf_extraction": True,
                                                "_attachment_title": attachment.get("data", {}).get("title", "PDF")
                                            }
                                        }

                                        # Handle image annotations
                                        if ext.get("type") == "image" and ext.get("imageRelativePath"):
                                            pdf_anno["data"]["_image_path"] = os.path.join(tmpdir, ext.get("imageRelativePath"))

                                        pdf_annotations.append(pdf_anno)

                        ctx.info(f"Retrieved {len(pdf_annotations)} annotations via PDF extraction")
                except Exception as pdf_error:
                    ctx.warn(f"Error during PDF annotation extraction: {pdf_error}")

            # Combine annotations from all sources
            annotations = better_bibtex_annotations + zotero_api_annotations + pdf_annotations

        else:
            # Retrieve all annotations in the library
            if isinstance(limit, str):
                limit = int(limit)
            zot.add_parameters(itemType="annotation", limit=limit or 50)
            annotations = zot.everything(zot.items())

        # Handle no annotations found
        if not annotations:
            return f"No annotations found{f' for item: {parent_title}' if item_key else ''}."

        # Generate markdown output
        output = [f"# Annotations{f' for: {parent_title}' if item_key else ''}", ""]

        for i, anno in enumerate(annotations, 1):
            data = anno.get("data", {})

            # Annotation details
            anno_type = data.get("annotationType", "Unknown type")
            anno_text = data.get("annotationText", "")
            anno_comment = data.get("annotationComment", "")
            anno_color = data.get("annotationColor", "")
            anno_key = anno.get("key", "")

            # Parent item context for library-wide retrieval
            parent_info = ""
            if not item_key and (parent_key := data.get("parentItem")):
                try:
                    parent = zot.item(parent_key)
                    parent_title = parent["data"].get("title", "Untitled")
                    parent_info = f" (from \"{parent_title}\")"
                except Exception:
                    parent_info = f" (parent key: {parent_key})"

            # Annotation source details
            source_info = ""
            if data.get("_from_better_bibtex", False):
                source_info = " (extracted via Better BibTeX)"
            elif data.get("_from_pdf_extraction", False):
                source_info = " (extracted directly from PDF)"

            # Attachment context
            attachment_info = ""
            if "_attachment_title" in data and data["_attachment_title"]:
                attachment_info = f" in {data['_attachment_title']}"

            # Build markdown annotation entry
            output.append(f"## Annotation {i}{parent_info}{attachment_info}{source_info}")
            output.append(f"**Type:** {anno_type}")
            output.append(f"**Key:** {anno_key}")

            # Color information
            if anno_color:
                output.append(f"**Color:** {anno_color}")
                if "_color_category" in data and data["_color_category"]:
                    output.append(f"**Color Category:** {data['_color_category']}")

            # Page information
            if "_pdf_page" in data:
                label = data.get("_pageLabel", str(data["_pdf_page"]))
                output.append(f"**Page:** {data['_pdf_page']} (Label: {label})")

            # Annotation content
            if anno_text:
                output.append(f"**Text:** {anno_text}")

            if anno_comment:
                output.append(f"**Comment:** {anno_comment}")

            # Image annotation
            if "_image_path" in data and os.path.exists(data["_image_path"]):
                output.append("**Image:** This annotation includes an image (not displayed in this interface)")

            # Tags
            if tags := data.get("tags"):
                tag_list = [f"`{tag['tag']}`" for tag in tags]
                if tag_list:
                    output.append(f"**Tags:** {' '.join(tag_list)}")

            output.append("")  # Empty line between annotations

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error fetching annotations: {str(e)}")
        return f"Error fetching annotations: {str(e)}"


@mcp.tool(
    name="zotero_get_notes",
    description="Retrieve notes from your Zotero library, with options to filter by parent item."
)
def get_notes(
    item_key: str | None = None,
    limit: int | str | None = 20,
    truncate: bool = True,
    *,
    ctx: Context
) -> str:
    """
    Retrieve notes from your Zotero library.

    Args:
        item_key: Optional Zotero item key/ID to filter notes by parent item
        limit: Maximum number of notes to return
        truncate: Whether to truncate long notes for display
        ctx: MCP context

    Returns:
        Markdown-formatted list of notes
    """
    try:
        ctx.info(f"Fetching notes{f' for item {item_key}' if item_key else ''}")
        zot = get_zotero_client()

        # Prepare search parameters
        params = {"itemType": "note"}

        if isinstance(limit, str):
            limit = int(limit)

        # Get notes
        notes = []
        if item_key:
            notes = zot.children(item_key, **params) if not limit else zot.children(item_key, limit=limit, **params)
        else: 
            notes = zot.items(**params) if not limit else zot.items(limit=limit, **params)

        if not notes:
            return f"No notes found{f' for item {item_key}' if item_key else ''}."

        # Generate markdown output
        output = [f"# Notes{f' for Item: {item_key}' if item_key else ''}", ""]

        for i, note in enumerate(notes, 1):
            data = note.get("data", {})
            note_key = note.get("key", "")

            # Parent item context
            parent_info = ""
            if parent_key := data.get("parentItem"):
                try:
                    parent = zot.item(parent_key)
                    parent_title = parent["data"].get("title", "Untitled")
                    parent_info = f" (from \"{parent_title}\")"
                except Exception:
                    parent_info = f" (parent key: {parent_key})"

            # Prepare note text
            note_text = data.get("note", "")

            # Clean up HTML formatting
            note_text = clean_html(note_text)

            # Limit note length for display
            if truncate and len(note_text) > 500:
                note_text = note_text[:500] + "..."

            # Build markdown entry
            output.append(f"## Note {i}{parent_info}")
            output.append(f"**Key:** {note_key}")

            # Tags
            if tags := data.get("tags"):
                tag_list = [f"`{tag['tag']}`" for tag in tags]
                if tag_list:
                    output.append(f"**Tags:** {' '.join(tag_list)}")

            output.append(f"**Content:**\n{note_text}")
            output.append("")  # Empty line between notes

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error fetching notes: {str(e)}")
        return f"Error fetching notes: {str(e)}"


@mcp.tool(
    name="zotero_search_notes",
    description="Search for notes across your Zotero library."
)
def search_notes(
    query: str,
    limit: int | str | None = 20,
    *,
    ctx: Context
) -> str:
    """
    Search for notes in your Zotero library.

    Args:
        query: Search query string
        limit: Maximum number of results to return
        ctx: MCP context

    Returns:
        Markdown-formatted search results
    """
    try:
        if not query.strip():
            return "Error: Search query cannot be empty"

        ctx.info(f"Searching Zotero notes for '{query}'")
        zot = get_zotero_client()

        # Search for notes and annotations

        if isinstance(limit, str):
            limit = int(limit)

        # First search notes
        zot.add_parameters(q=query, itemType="note", limit=limit or 20)
        notes = zot.items()

        # Then search annotations (reusing the get_annotations function)
        annotation_results = _get_annotations(
            item_key=None,  # Search all annotations
            use_pdf_extraction=True,
            limit=limit or 20,
            ctx=ctx
        )

        # Parse annotation markdown blocks from get_annotations output.
        annotation_lines = annotation_results.split("\n")
        current_annotation = None
        annotations = []

        for line in annotation_lines:
            if line.startswith("## "):
                if current_annotation:
                    annotations.append(current_annotation)
                current_annotation = {"lines": [line], "type": "annotation"}
            elif current_annotation is not None:
                current_annotation["lines"].append(line)

        if current_annotation:
            annotations.append(current_annotation)

        # Filter and highlight notes
        query_lower = query.lower()
        note_results = []

        for note in notes:
            data = note.get("data", {})
            note_text = data.get("note", "").lower()

            if query_lower in note_text:
                # Prepare full note details
                note_result = {
                    "type": "note",
                    "key": note.get("key", ""),
                    "data": data
                }
                note_results.append(note_result)

        # Keep only annotation blocks that contain the query text.
        annotation_results_filtered = []
        for annotation in annotations:
            block_text = "\n".join(annotation.get("lines", []))
            if query_lower in block_text.lower():
                annotation_results_filtered.append(annotation)

        # Combine and sort results
        all_results = note_results + annotation_results_filtered
        if not all_results:
            return f"No results found for '{query}'"

        # Format results
        output = [f"# Search Results for '{query}'", ""]

        for i, result in enumerate(all_results, 1):
            if result["type"] == "note":
                # Note formatting
                data = result["data"]
                key = result["key"]

                # Parent item context
                parent_info = ""
                if parent_key := data.get("parentItem"):
                    try:
                        parent = zot.item(parent_key)
                        parent_title = parent["data"].get("title", "Untitled")
                        parent_info = f" (from \"{parent_title}\")"
                    except Exception:
                        parent_info = f" (parent key: {parent_key})"

                # Note text with query highlight
                note_text = data.get("note", "")
                note_text = note_text.replace("<p>", "").replace("</p>", "\n\n")
                note_text = note_text.replace("<br/>", "\n").replace("<br>", "\n")

                # Highlight query in note text
                try:
                    # Find first occurrence of query and extract context
                    text_lower = note_text.lower()
                    pos = text_lower.find(query_lower)
                    if pos >= 0:
                        # Extract context around the query
                        start = max(0, pos - 100)
                        end = min(len(note_text), pos + 200)
                        context = note_text[start:end]

                        # Highlight the query in the context
                        highlighted = context.replace(
                            context[context.lower().find(query_lower):context.lower().find(query_lower)+len(query)],
                            f"**{context[context.lower().find(query_lower):context.lower().find(query_lower)+len(query)]}**"
                        )

                        note_text = highlighted + "..."
                except Exception:
                    # Fallback to first 500 characters if highlighting fails
                    note_text = note_text[:500] + "..."

                output.append(f"## Note {i}{parent_info}")
                output.append(f"**Key:** {key}")

                # Tags
                if tags := data.get("tags"):
                    tag_list = [f"`{tag['tag']}`" for tag in tags]
                    if tag_list:
                        output.append(f"**Tags:** {' '.join(tag_list)}")

                output.append(f"**Content:**\n{note_text}")
                output.append("")

            elif result["type"] == "annotation":
                # Add the entire annotation block
                output.extend(result["lines"])
                output.append("")

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error searching notes: {str(e)}")
        return f"Error searching notes: {str(e)}"


@mcp.tool(
    name="zotero_add_by_identifier",
    description="Add new Zotero item(s) by DOI or arXiv identifier."
)
def add_by_identifier(
    identifiers: Union[List[str], str],
    collections: Optional[Union[List[str], str]] = None,
    collection_names: Optional[Union[List[str], str]] = None,
    tags: Optional[Union[List[str], str]] = None,
    attach_pdfs: bool = True,
    attach_mode: Literal["auto", "import_file", "linked_url"] = "auto",
    *,
    ctx: Context
) -> str:
    """
    Create Zotero items using DOI or arXiv identifiers.

    Args:
        identifiers: Single identifier or list/JSON string of identifiers
        collections: Collection keys to add created items to
        collection_names: Collection names to resolve and add created items to
        tags: Optional tags to apply to the created items
        attach_pdfs: If true, attempts to attach a PDF when discoverable
        attach_mode: PDF attachment mode:
            - auto: try imported file attachment first, then linked URL fallback
            - import_file: try imported file only
            - linked_url: use linked URL attachment only
        ctx: MCP context

    Returns:
        Markdown summary with created item keys or errors per identifier.
    """
    def _as_list(value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            s = value.strip()
            # Try JSON list first
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except Exception:
                pass
            # Fallback to comma-separated
            parts = [p.strip() for p in s.split(',') if p.strip()]
            if parts:
                return parts
            # Single string
            return [s]
        return [str(value).strip()]

    def _normalize_doi_candidate(raw: str) -> Optional[str]:
        s = (raw or "").strip()
        if not s:
            return None

        # Strip common prefix
        if s.lower().startswith("doi:"):
            s = s[4:].strip()

        # DOI URL format
        if s.lower().startswith("http://") or s.lower().startswith("https://"):
            m = re.search(r"doi\.org/(10\.\d{4,9}/[^\s?#]+)", s, flags=re.IGNORECASE)
            if not m:
                return None
            s = m.group(1)

        # Trim common trailing punctuation from copied text
        s = s.rstrip(".,);]")

        # Basic DOI shape validation
        if re.match(r"^10\.\d{4,9}/\S+$", s):
            return s
        return None

    def _normalize_arxiv_candidate(raw: str) -> Optional[str]:
        s = (raw or "").strip()
        if not s:
            return None

        if s.lower().startswith("arxiv:"):
            s = s[6:].strip()

        # URL forms:
        # - https://arxiv.org/abs/2401.00001
        # - https://arxiv.org/pdf/2401.00001.pdf
        if s.lower().startswith("http://") or s.lower().startswith("https://"):
            m = re.search(
                r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z\-]+/\d{7}(?:v\d+)?)(?:\.pdf)?",
                s,
                flags=re.IGNORECASE,
            )
            if not m:
                return None
            s = m.group(1)

        if re.match(r"^[0-9]{4}\.[0-9]{4,5}(?:v\d+)?$", s):
            return s
        if re.match(r"^[a-z\-]+/\d{7}(?:v\d+)?$", s, flags=re.IGNORECASE):
            return s
        return None

    def _arxiv_id_from_doi_namespace(doi: str) -> Optional[str]:
        """
        Detect arXiv DOI namespace forms like:
        - 10.48550/arXiv.2603.02553
        - 10.48550/arXiv.hep-th/9901001
        """
        m = re.match(r"^10\.48550/arxiv\.(.+)$", doi or "", flags=re.IGNORECASE)
        if not m:
            return None
        return _normalize_arxiv_candidate(m.group(1))

    def _first_or(seq, default=""):
        if isinstance(seq, list) and seq:
            return seq[0]
        return default

    def _date_from_crossref(issued) -> str:
        try:
            parts = (issued or {}).get('date-parts', [])
            if parts and parts[0]:
                p = parts[0]
                if len(p) >= 3:
                    return f"{p[0]:04d}-{p[1]:02d}-{p[2]:02d}"
                if len(p) == 2:
                    return f"{p[0]:04d}-{p[1]:02d}"
                if len(p) == 1:
                    return str(p[0])
        except Exception:
            pass
        return ""

    def _strip_html(text: str) -> str:
        try:
            return re.sub(r"<[^>]+>", "", text or "").strip()
        except Exception:
            return text or ""

    def _authors_from_names(names: List[Dict[str, str]]) -> List[Dict[str, str]]:
        creators = []
        for a in names or []:
            first = a.get('given') or a.get('first') or ""
            last = a.get('family') or a.get('last') or ""
            literal = a.get('literal') or a.get('name')
            if literal and (not first and not last):
                # Try to split literal into first/last
                parts = literal.split()
                if len(parts) >= 2:
                    first = " ".join(parts[:-1])
                    last = parts[-1]
                else:
                    last = literal
            creators.append({
                "creatorType": "author",
                "firstName": first,
                "lastName": last,
            })
        return creators

    def _fetch_crossref(doi: str) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
        url = f"https://api.crossref.org/works/{requests.utils.quote(doi)}"
        headers = {"User-Agent": "zotero-mcp (identifier import)"}
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        msg = (r.json() or {}).get('message', {})
        typ = (msg.get('type') or '').lower()
        type_map = {
            'journal-article': 'journalArticle',
            'proceedings-article': 'conferencePaper',
            'book': 'book',
            'book-chapter': 'bookSection',
            'chapter': 'bookSection',
            'report': 'report',
            'thesis': 'thesis',
            'posted-content': 'preprint',
        }
        item_type = type_map.get(typ, 'journalArticle')
        creators = _authors_from_names(msg.get('author', []))
        title = _first_or(msg.get('title'), '')
        container = _first_or(msg.get('container-title'), '')
        abstract = _strip_html(msg.get('abstract', '') or '')
        date_str = _date_from_crossref(msg.get('issued'))
        pages = msg.get('page') or ''
        volume = msg.get('volume') or ''
        issue = msg.get('issue') or ''
        publisher = msg.get('publisher') or ''
        url_final = msg.get('URL') or (f"https://doi.org/{doi}")
        data: Dict[str, object] = {
            "itemType": item_type,
            "title": title or f"DOI {doi}",
            "creators": creators,
            "DOI": doi,
            "url": url_final,
        }
        if container:
            data["publicationTitle"] = container
        if date_str:
            data["date"] = date_str
        if pages:
            data["pages"] = pages
        if volume:
            data["volume"] = volume
        if issue:
            data["issue"] = issue
        if publisher:
            data["publisher"] = publisher
        if abstract:
            data["abstractNote"] = abstract
        # Try Crossref-provided PDF link if present
        pdf_url: Optional[str] = None
        try:
            for link in msg.get('link', []) or []:
                ct = (link.get('content-type') or link.get('content_type') or '').lower()
                if ct == 'application/pdf':
                    candidate = link.get('URL') or link.get('url')
                    if candidate:
                        pdf_url = candidate
                        break
        except Exception:
            pass
        return data, pdf_url

    def _fetch_arxiv(arxiv_id: str) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
        url = (
            "https://export.arxiv.org/api/query"
            f"?search_query=id:{requests.utils.quote(arxiv_id)}&max_results=1"
        )
        r = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "zotero-mcp (identifier import)"},
        )
        r.raise_for_status()

        root = ET.fromstring(r.text)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None, None

        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
        abs_url = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        doi = (entry.findtext("arxiv:doi", default="", namespaces=ns) or "").strip()

        authors = []
        for author in entry.findall("atom:author", ns):
            name = (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
            if name:
                authors.append({"literal": name})

        creators = _authors_from_names(authors)
        item_data: Dict[str, object] = {
            "itemType": "preprint",
            "title": title or f"arXiv:{arxiv_id}",
            "creators": creators,
            "url": abs_url or f"https://arxiv.org/abs/{arxiv_id}",
        }
        if published:
            item_data["date"] = published[:10]
        if summary:
            item_data["abstractNote"] = summary
        if doi:
            item_data["DOI"] = doi

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return item_data, pdf_url

    def _discover_pdf_from_landing_page(landing_url: str) -> Optional[str]:
        """Best-effort PDF discovery from DOI landing pages."""
        try:
            resp = requests.get(
                landing_url,
                timeout=20,
                headers={"User-Agent": "zotero-mcp (identifier import)"},
                allow_redirects=True,
            )
            if not resp.ok:
                return None

            html = resp.text or ""

            # Common scholarly metadata tag
            m_meta = re.search(
                r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
                html,
                flags=re.IGNORECASE,
            )
            if m_meta:
                return urljoin(resp.url, m_meta.group(1).strip())

            # Generic href fallback
            m_href = re.search(
                r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
                html,
                flags=re.IGNORECASE,
            )
            if m_href:
                return urljoin(resp.url, m_href.group(1).strip())
        except Exception:
            return None
        return None

    def _download_pdf_to_temp(pdf_url: str) -> Tuple[Optional[str], bool, Optional[str]]:
        """Download candidate PDF URL to a temporary file."""
        import tempfile

        try:
            resp = requests.get(
                pdf_url,
                timeout=30,
                allow_redirects=True,
                headers={"User-Agent": "zotero-mcp (identifier import)"},
                stream=True,
            )
            if not resp.ok:
                return None, False, f"download failed (HTTP {resp.status_code})"

            ctype = (resp.headers.get("Content-Type") or "").lower()
            mime_ok = "pdf" in ctype or pdf_url.lower().endswith(".pdf")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        tmpf.write(chunk)
                return tmpf.name, mime_ok, None
        except Exception as e:
            return None, False, f"download error: {e}"

    def _attach_pdf_linked_url(zot_client, parent_item_key: str, pdf_url: str) -> Tuple[bool, str]:
        """Attach PDF as linked URL."""
        try:
            probe = requests.head(pdf_url, timeout=10, allow_redirects=True)
            ctype = (probe.headers.get("Content-Type") or "").lower()
            if not (probe.ok and ("pdf" in ctype or pdf_url.lower().endswith(".pdf"))):
                return False, "linked_url rejected (not a PDF URL)"

            attachment_item = {
                "itemType": "attachment",
                "parentItem": parent_item_key,
                "linkMode": "linked_url",
                "title": "PDF",
                "url": pdf_url,
                "contentType": "application/pdf",
            }
            att_res = zot_client.create_items([attachment_item])
            if att_res.get("success"):
                return True, "linked_url"
            return False, f"linked_url create failed: {att_res}"
        except Exception as e:
            return False, f"linked_url error: {e}"

    def _attach_pdf_imported(zot_client, parent_item_key: str, pdf_path: str) -> Tuple[bool, str]:
        """Attempt imported file attachment through available pyzotero upload helpers."""
        fn = getattr(zot_client, "attachment_simple", None)
        if not callable(fn):
            return False, "import_file unsupported by active zotero client"

        call_attempts = [
            lambda: fn([pdf_path], parentid=parent_item_key),
            lambda: fn([pdf_path], parent_item=parent_item_key),
            lambda: fn([pdf_path], parent_item_key),
            lambda: fn(pdf_path, parentid=parent_item_key),
            lambda: fn(pdf_path, parent_item=parent_item_key),
            lambda: fn(pdf_path, parent_item_key),
        ]
        last_err: Optional[Exception] = None
        for attempt in call_attempts:
            try:
                result = attempt()
                # pyzotero upload helpers may return dict/None/other truthy values on success
                if isinstance(result, dict):
                    if result.get("success") or result.get("successful") or result.get("key"):
                        return True, "imported"
                else:
                    return True, "imported"
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue
        return False, f"import_file failed: {last_err or 'unknown error'}"

    try:
        # Prepare inputs
        id_list = _as_list(identifiers)
        if not id_list:
            return "Error: identifiers cannot be empty"
        if attach_mode not in {"auto", "import_file", "linked_url"}:
            return "Error: attach_mode must be one of: auto, import_file, linked_url"
        ctx.info(f"PDF attachment configuration: attach_pdfs={attach_pdfs}, attach_mode={attach_mode}")

        tag_list = _as_list(tags)

        # Resolve collection names to keys if provided
        zot = get_zotero_client()
        resolved_collections: List[str] = []
        if collections:
            resolved_collections.extend(_as_list(collections))
        if collection_names:
            names = _as_list(collection_names)
            if names:
                ctx.info(f"Resolving collection names: {names}")
                try:
                    all_collections = zot.collections()
                    collection_map = {c["data"].get("name", "").lower(): c["key"] for c in all_collections}
                    for name in names:
                        key = collection_map.get(name.lower())
                        if key:
                            resolved_collections.append(key)
                            ctx.info(f"Resolved '{name}' to key: {key}")
                        else:
                            # Partial match
                            matches = [k for n, k in collection_map.items() if name.lower() in n]
                            if matches:
                                resolved_collections.append(matches[0])
                                ctx.info(f"Resolved '{name}' to key: {matches[0]} (partial)")
                            else:
                                ctx.warn(f"Collection '{name}' not found.")
                except Exception as ce:
                    ctx.warn(f"Could not resolve collection names: {ce}")

        results_md: List[str] = ["# Identifier Import Results", ""]
        created_count = 0

        for ident in id_list:
            doi = _normalize_doi_candidate(ident)
            arxiv_id = _normalize_arxiv_candidate(ident)
            if doi and not arxiv_id:
                arxiv_id = _arxiv_id_from_doi_namespace(doi)
            ctx.info(f"Processing identifier '{ident}'")

            try:
                if not doi and not arxiv_id:
                    results_md.append(
                        f"- {ident}: Unsupported identifier format. "
                        "Supported formats: DOI (`10.x/...`, `https://doi.org/...`) "
                        "or arXiv (`arXiv:2401.00001`, `https://arxiv.org/abs/...`)."
                    )
                    continue

                pdf_hint: Optional[str] = None
                if doi and not arxiv_id:
                    item_data, pdf_hint = _fetch_crossref(doi)
                else:
                    item_data, pdf_hint = _fetch_arxiv(arxiv_id or "")
                    if doi and item_data and not item_data.get("DOI"):
                        item_data["DOI"] = doi

                if not item_data:
                    results_md.append(f"- {ident}: No metadata found")
                    continue

                if doi and attach_pdfs and not pdf_hint:
                    landing_url = str(item_data.get("url") or f"https://doi.org/{doi}")
                    pdf_hint = _discover_pdf_from_landing_page(landing_url)

                # Attach tags if given
                if tag_list:
                    item_data["tags"] = [{"tag": t} for t in tag_list]

                # Force empty collections in item_data; we'll add after creation
                item_data["collections"] = []

                # Create the item
                result = zot.create_items([item_data])
                if "success" in result and result["success"]:
                    item_key = next(iter(result["success"].values()))

                    # Attach PDF according to attachment mode
                    pdf_status = "not requested"
                    if attach_pdfs:
                        if not pdf_hint:
                            pdf_status = "failed (no discoverable PDF URL)"
                        elif attach_mode == "linked_url":
                            ok, detail = _attach_pdf_linked_url(zot, item_key, pdf_hint)
                            pdf_status = "linked_url" if ok else f"failed ({detail})"
                        else:
                            file_path = None
                            try:
                                file_path, mime_ok, dl_err = _download_pdf_to_temp(pdf_hint)
                                if not file_path:
                                    pdf_status = f"failed ({dl_err})"
                                elif not mime_ok:
                                    pdf_status = "failed (downloaded resource is not PDF)"
                                else:
                                    ok, detail = _attach_pdf_imported(zot, item_key, file_path)
                                    if ok:
                                        pdf_status = "imported"
                                    elif attach_mode == "auto":
                                        ctx.warn(f"import_file path failed; falling back to linked_url: {detail}")
                                        ok2, detail2 = _attach_pdf_linked_url(zot, item_key, pdf_hint)
                                        pdf_status = "linked_url" if ok2 else f"failed ({detail}; fallback {detail2})"
                                    else:
                                        pdf_status = f"failed ({detail})"
                            finally:
                                if file_path and os.path.exists(file_path):
                                    with suppress(Exception):
                                        os.remove(file_path)

                    # Add to collections
                    added_to = []
                    errors = []
                    for coll_key in resolved_collections:
                        try:
                            coll = zot.collection(coll_key)
                            coll_name = coll["data"].get("name", coll_key)
                            add_result = zot.addto_collection(coll_key, [item_key])
                            if add_result:
                                added_to.append(coll_name)
                            else:
                                errors.append(f"Failed to add to '{coll_name}'")
                        except Exception as ce:
                            errors.append(f"Error with collection {coll_key}: {str(ce)}")

                    created_count += 1
                    extra = f" (collections: {', '.join(added_to)})" if added_to else ""
                    if errors:
                        extra += f"; warnings: {'; '.join(errors)}"
                    results_md.append(f"- {ident}: created item key `{item_key}`{extra}; pdf: {pdf_status}")
                else:
                    error_details = result.get('failed', {})
                    if error_details:
                        results_md.append(f"- {ident}: Failed to create item: {str(error_details)}")
                    else:
                        results_md.append(f"- {ident}: Failed to create item")
            except requests.RequestException as net_err:
                ctx.warn(f"Network error for {ident}: {net_err}")
                results_md.append(f"- {ident}: Network error fetching metadata: {net_err}")
            except Exception as e:
                ctx.error(f"Error importing {ident}: {e}")
                results_md.append(f"- {ident}: Error: {e}")

        if created_count == 0:
            results_md.insert(1, "No items were created.")
        else:
            results_md.insert(1, f"Created {created_count} item(s).")

        return "\n".join(results_md)

    except Exception as e:
        ctx.error(f"Error adding by identifier: {str(e)}")
        return f"Error adding by identifier: {str(e)}"


@mcp.tool(
    name="zotero_create_item",
    description="""Create a new item in your Zotero library (article, book, webpage, etc.).

📝 FLEXIBLE INPUT FORMATS - All structured parameters accept multiple formats:

PARAMETER TYPES (accepts both Python types AND JSON strings):
┌─────────────────┬─────────────────────────────────────────────────────────┐
│ Parameter       │ Accepted Formats (all work!)                            │
├─────────────────┼─────────────────────────────────────────────────────────┤
│ tags            │ ✓ ["optimization", "ML"]  (Python list)                │
│                 │ ✓ '["optimization", "ML"]'  (JSON string)               │
│                 │ ✓ "optimization, ML"  (comma-separated)                 │
│                 │ ✓ "single-tag"  (single string)                         │
├─────────────────┼─────────────────────────────────────────────────────────┤
│ creators        │ ✓ [{"creatorType": "author", ...}]  (Python list)      │
│                 │ ✓ '[{"creatorType": "author", ...}]'  (JSON string)    │
├─────────────────┼─────────────────────────────────────────────────────────┤
│ collection_names│ ✓ ["PhD Research"]  (Python list)                      │
│                 │ ✓ '["PhD Research"]'  (JSON string)                     │
│                 │ ✓ "PhD Research"  (single string)                       │
├─────────────────┼─────────────────────────────────────────────────────────┤
│ extra_fields    │ ✓ {"key": "value"}  (Python dict)                      │
│                 │ ✓ '{"key": "value"}'  (JSON string)                     │
└─────────────────┴─────────────────────────────────────────────────────────┘

RECOMMENDED EXAMPLE (Python types - best practice):
zotero_create_item(
    item_type="journalArticle",
    title="Deep Learning in Healthcare",
    creators=[
        {"creatorType": "author", "firstName": "Jane", "lastName": "Smith"},
        {"creatorType": "author", "firstName": "John", "lastName": "Doe"}
    ],
    date="2024",
    publication_title="Journal of AI",
    tags=["deep learning", "healthcare", "AI"],
    collection_names=["PhD Research"]
)

NOTE: Both Python native types and JSON strings are accepted and will work correctly.
      The function will automatically convert JSON strings to the appropriate types."""
)
def create_item(
    item_type: str,
    title: str,
    creators: Optional[Union[List[Dict[str, str]], str]] = None,
    date: Optional[str] = None,
    publication_title: Optional[str] = None,
    volume: Optional[str] = None,
    issue: Optional[str] = None,
    pages: Optional[str] = None,
    publisher: Optional[str] = None,
    place: Optional[str] = None,
    doi: Optional[str] = None,
    url: Optional[str] = None,
    attach_pdf_url: Optional[str] = None,
    abstract: Optional[str] = None,
    tags: Optional[Union[List[str], str]] = None,
    collections: Optional[Union[List[str], str]] = None,
    collection_names: Optional[Union[List[str], str]] = None,
    extra_fields: Optional[Union[Dict[str, str], str]] = None,
    *,
    ctx: Context
) -> str:
    """
    Create a new item in your Zotero library.

    Args:
        item_type: Type of item (e.g., "journalArticle", "book", "webpage", "conferencePaper")
        title: Title of the item
        creators: List of creator dictionaries. Each dict must have:
                 - creatorType: "author", "editor", "contributor", etc.
                 - firstName: Author's first name
                 - lastName: Author's last name
                 OR for single-field names (organizations):
                 - creatorType: "author"
                 - name: Full name/organization name
                 Example: [{"creatorType": "author", "firstName": "John", "lastName": "Doe"}]
        date: Publication date (flexible format like "2024", "2024-01", "2024-01-15")
        publication_title: Journal/publication name (for articles)
        volume: Volume number (as string)
        issue: Issue number (as string)
        pages: Page range (e.g., "123-145")
        publisher: Publisher name
        place: Publication place
        doi: Digital Object Identifier
        url: URL of the item
        attach_pdf_url: Optional direct PDF URL to attach (linked URL attachment)
        abstract: Abstract or summary text
        tags: Python list of strings (NOT JSON string). Example: ["optimization", "machine learning"]
        collections: Python list of collection keys. Example: ["ABC123XY"]
        collection_names: Python list of collection names (preferred). Example: ["PhD Research"]
        extra_fields: Python dict of additional fields. Example: {"ISBN": "978-0-123456-78-9"}
        ctx: MCP context

    Returns:
        Confirmation message with the new item key
    """
    try:
        ctx.info(f"Creating new {item_type} item: {title}")
        zot = get_zotero_client()

        # Input validation and auto-correction for common mistakes
        # Fix tags if passed as JSON string instead of list
        if tags is not None:
            if isinstance(tags, str):
                tags_stripped = tags.strip()
                ctx.info(f"Received tags as string (first 100 chars): {repr(tags_stripped[:100])}")

                # Try JSON parsing first (handles ["tag1", "tag2"])
                try:
                    tags = json.loads(tags_stripped)
                    ctx.info(f"✓ Auto-corrected tags from JSON string to Python list ({len(tags)} tags)")
                except json.JSONDecodeError as json_err:
                    ctx.info(f"JSON parse failed: {str(json_err)}, trying alternative formats")

                    # Try unescaping if string contains backslash-escaped quotes
                    if '\\' in tags_stripped:
                        try:
                            # Remove extra escaping that might have been added
                            unescaped = tags_stripped.replace('\\"', '"').replace("\\'", "'")
                            tags = json.loads(unescaped)
                            ctx.info(f"✓ Auto-corrected tags from escaped JSON string to Python list ({len(tags)} tags)")
                        except json.JSONDecodeError:
                            # Fall through to comma-separated parsing
                            pass

                    # If still a string, try comma-separated parsing
                    if isinstance(tags, str):
                        if ',' in tags_stripped:
                            tags = [tag.strip() for tag in tags_stripped.split(',') if tag.strip()]
                            ctx.info(f"✓ Auto-corrected tags from comma-separated string to Python list ({len(tags)} tags)")
                        else:
                            # Single tag without comma
                            tags = [tags_stripped] if tags_stripped else []
                            ctx.info(f"✓ Auto-corrected tags from single string to Python list (1 tag)")

            if not isinstance(tags, list):
                return f"Error: tags must be a list of strings. Example: ['optimization', 'machine learning']"
            # Ensure all tags are strings
            if not all(isinstance(tag, str) for tag in tags):
                return f"Error: All tags must be strings. Received: {tags}"

        # Fix creators if passed as JSON string instead of list
        if creators is not None:
            if isinstance(creators, str):
                creators_stripped = creators.strip()
                try:
                    creators = json.loads(creators_stripped)
                    ctx.info(f"✓ Auto-corrected creators from JSON string to Python list: {len(creators)} creator(s)")
                except json.JSONDecodeError as e:
                    return f"Error: creators parameter appears to be a string but couldn't be parsed as JSON.\n" \
                           f"Parse error: {str(e)}\n" \
                           f"Expected: Python list like [{{'creatorType': 'author', 'firstName': 'Jane', 'lastName': 'Doe'}}]\n" \
                           f"Received: {repr(creators[:100])}..."
            if not isinstance(creators, list):
                return f"Error: creators must be a list of dictionaries. Example: [{{'creatorType': 'author', 'firstName': 'Jane', 'lastName': 'Doe'}}]"

            # Validate creator structure
            if creators:  # Only validate if not empty
                for i, creator in enumerate(creators):
                    if not isinstance(creator, dict):
                        return f"Error: Each creator must be a dictionary. Creator {i+1} is not a dictionary."

                    if "creatorType" not in creator:
                        return f"Error: Creator {i+1} is missing 'creatorType' field. Must be 'author', 'editor', 'contributor', etc."

                    # Check if it has either (firstName + lastName) OR name
                    has_split_name = "firstName" in creator and "lastName" in creator
                    has_single_name = "name" in creator

                    if not has_split_name and not has_single_name:
                        return f"Error: Creator {i+1} must have EITHER ('firstName' AND 'lastName') OR 'name'. Example: {{'creatorType': 'author', 'firstName': 'Jane', 'lastName': 'Doe'}}"

                    # Warn about empty names
                    if has_split_name:
                        if not creator.get("firstName") and not creator.get("lastName"):
                            ctx.warn(f"Creator {i+1} has empty firstName and lastName")
                    elif has_single_name:
                        if not creator.get("name"):
                            ctx.warn(f"Creator {i+1} has empty name field")

                ctx.info(f"Validated {len(creators)} creator(s) successfully")

        # Fix collections if passed as JSON string
        if collections is not None:
            if isinstance(collections, str):
                try:
                    collections = json.loads(collections.strip())
                    ctx.info(f"✓ Auto-corrected collections from JSON string to Python list")
                except json.JSONDecodeError as e:
                    return f"Error: collections must be a list of strings. Parse error: {str(e)}"
            if not isinstance(collections, list):
                return f"Error: collections must be a Python list of strings"

        # Fix collection_names if passed as JSON string / plain string / comma-separated string
        if collection_names is not None:
            try:
                collection_names = _normalize_str_list_input(
                    collection_names, field_name="collection_names"
                )
                ctx.info(
                    "✓ Auto-corrected collection_names to Python list "
                    f"({len(collection_names)} value(s))"
                )
            except ValueError as e:
                return f"Error: {str(e)}"

        # Fix extra_fields if passed as JSON string
        if extra_fields is not None:
            if isinstance(extra_fields, str):
                extra_fields_stripped = extra_fields.strip()
                try:
                    extra_fields = json.loads(extra_fields_stripped)
                    ctx.info(f"✓ Auto-corrected extra_fields from JSON string to Python dict: {extra_fields}")
                except json.JSONDecodeError as e:
                    return f"Error: extra_fields parameter appears to be a string but couldn't be parsed as JSON.\n" \
                           f"Parse error: {str(e)}\n" \
                           f"Expected: Python dict like {{'key': 'value'}}\n" \
                           f"Received: {repr(extra_fields[:100])}..."
            if not isinstance(extra_fields, dict):
                return f"Error: extra_fields must be a dictionary. Example: {{'ISBN': '123-456'}}"

        # Validate item type specific fields
        if item_type == "conferencePaper" and publication_title:
            ctx.warn(f"Note: 'publication_title' is not a standard field for conferencePaper. Consider using extra_fields with 'proceedingsTitle' or 'conferenceName' instead.")
            # Move it to extra_fields automatically
            if extra_fields is None:
                extra_fields = {}
            extra_fields["proceedingsTitle"] = publication_title
            publication_title = None
            ctx.info("Automatically moved publication_title to extra_fields['proceedingsTitle']")

        # Resolve collection names to keys if provided
        resolved_collections = []
        if collections:
            resolved_collections.extend(collections)

        if collection_names:
            ctx.info(f"Resolving collection names: {collection_names}")
            all_collections = zot.collections()
            collection_map = {c["data"].get("name", "").lower(): c["key"] for c in all_collections}

            for name in collection_names:
                name_lower = name.lower()
                if name_lower in collection_map:
                    resolved_collections.append(collection_map[name_lower])
                    ctx.info(f"Resolved '{name}' to key: {collection_map[name_lower]}")
                else:
                    # Try partial match
                    matches = [key for coll_name, key in collection_map.items() if name_lower in coll_name]
                    if matches:
                        resolved_collections.append(matches[0])
                        ctx.info(f"Resolved '{name}' to key: {matches[0]} (partial match)")
                    else:
                        ctx.warn(f"Collection '{name}' not found. Use zotero_search_collections to find it or zotero_create_collection to create it.")

        # Build item data structure
        item_data = {
            "itemType": item_type,
            "title": title
        }

        # Add creators if provided
        if creators:
            item_data["creators"] = creators
        else:
            item_data["creators"] = []

        # Add optional fields
        if date:
            item_data["date"] = date
        if publication_title:
            item_data["publicationTitle"] = publication_title
        if volume:
            item_data["volume"] = volume
        if issue:
            item_data["issue"] = issue
        if pages:
            item_data["pages"] = pages
        if publisher:
            item_data["publisher"] = publisher
        if place:
            item_data["place"] = place
        if doi:
            item_data["DOI"] = doi
        if url:
            item_data["url"] = url
        if abstract:
            item_data["abstractNote"] = abstract

        # Add tags
        if tags:
            item_data["tags"] = [{"tag": tag} for tag in tags]
        else:
            item_data["tags"] = []

        # Don't add collections to item_data - we'll add them after creation
        # The Zotero API works better when collections are added via addto_collection()
        item_data["collections"] = []

        # Add extra fields
        if extra_fields:
            for key, value in extra_fields.items():
                if key not in item_data:  # Don't override existing fields
                    item_data[key] = value

        # Create the item
        result = zot.create_items([item_data])

        # Check if creation was successful
        if "success" in result and result["success"]:
            successful = result["success"]
            if len(successful) > 0:
                item_key = next(iter(successful.values()))

                # Optionally attach a PDF link if provided
                attachment_info = ""
                if attach_pdf_url:
                    try:
                        probe = requests.head(attach_pdf_url, timeout=10, allow_redirects=True)
                        ctype = (probe.headers.get('Content-Type') or '').lower()
                        if probe.ok and ('pdf' in ctype or attach_pdf_url.lower().endswith('.pdf')):
                            attachment_item = {
                                "itemType": "attachment",
                                "parentItem": item_key,
                                "linkMode": "linked_url",
                                "title": "PDF",
                                "url": attach_pdf_url,
                                "contentType": "application/pdf",
                            }
                            att_res = zot.create_items([attachment_item])
                            if att_res.get('success'):
                                attachment_info = "\n\nAttached PDF link."
                                ctx.info("Attached PDF as linked URL")
                            else:
                                ctx.warn(f"Failed to attach PDF link: {att_res}")
                        else:
                            ctx.warn(f"PDF URL not accessible or not a PDF: {attach_pdf_url}")
                    except Exception as attach_err:
                        ctx.warn(f"Error attaching PDF: {attach_err}")

                # Now add the item to collections if specified
                collection_info = ""
                if resolved_collections:
                    collection_errors = []
                    successfully_added = []

                    for coll_key in resolved_collections:
                        try:
                            # Verify collection exists and add item to it
                            coll = zot.collection(coll_key)
                            coll_name = coll["data"].get("name", coll_key)
                            add_result = zot.addto_collection(coll_key, [item_key])

                            if add_result:
                                successfully_added.append(coll_name)
                                ctx.info(f"Added item to collection: {coll_name}")
                            else:
                                collection_errors.append(f"Failed to add to '{coll_name}'")
                                ctx.warn(f"Failed to add item to collection: {coll_name}")
                        except Exception as coll_error:
                            collection_errors.append(f"Error with collection {coll_key}: {str(coll_error)}")
                            ctx.error(f"Error adding to collection {coll_key}: {str(coll_error)}")

                    if successfully_added:
                        collection_info = f"\n\nAdded to collection(s): {', '.join(successfully_added)}"
                    if collection_errors:
                        collection_info += f"\n\n**Collection warnings:**\n" + "\n".join(f"- {err}" for err in collection_errors)

                return f"Successfully created {item_type}: \"{title}\"\n\nItem key: `{item_key}`{attachment_info}{collection_info}"
            else:
                return f"Item creation response was successful but no key was returned: {result}"
        else:
            error_details = result.get('failed', {})
            if error_details:
                # Extract error message from failed response
                error_msg = str(error_details)
                return f"Failed to create item: {error_msg}\n\nPlease check that all required fields for '{item_type}' are provided."
            return f"Failed to create item: {result}"

    except Exception as e:
        ctx.error(f"Error creating item: {str(e)}")
        return f"Error creating item: {str(e)}"


@mcp.tool(
    name="zotero_update_item",
    description="""Update an existing item in your Zotero library.

📝 FLEXIBLE INPUT FORMATS - All structured parameters accept multiple formats:

PARAMETER TYPES (accepts both Python types AND JSON strings):
┌─────────────────┬─────────────────────────────────────────────────────────┐
│ Parameter       │ Accepted Formats (all work!)                            │
├─────────────────┼─────────────────────────────────────────────────────────┤
│ tags            │ Replaces ALL tags                                       │
│ add_tags        │ Adds to existing tags                                   │
│ remove_tags     │ Removes from existing tags                              │
│                 │ ✓ ["optimization", "AI"]  (Python list)                 │
│                 │ ✓ '["optimization", "AI"]'  (JSON string)               │
│                 │ ✓ "optimization, AI"  (comma-separated)                 │
├─────────────────┼─────────────────────────────────────────────────────────┤
│ creators        │ Replaces ALL creators                                   │
│                 │ ✓ [{"creatorType": "author", ...}]  (Python list)      │
│                 │ ✓ '[{"creatorType": "author", ...}]'  (JSON string)    │
└─────────────────┴─────────────────────────────────────────────────────────┘

RECOMMENDED EXAMPLE (Python types - best practice):
zotero_update_item(
    item_key="ABC12345",
    abstract="Updated abstract text",
    add_tags=["reviewed", "important"],
    creators=[
        {"creatorType": "author", "firstName": "Jane", "lastName": "Smith"}
    ],
    collection_names=["PhD Research"]
)

NOTE: Both Python native types and JSON strings are accepted and will work correctly.
      The function will automatically convert JSON strings to the appropriate types."""
)
def update_item(
    item_key: str,
    title: Optional[str] = None,
    creators: Optional[Union[List[Dict[str, str]], str]] = None,
    date: Optional[str] = None,
    publication_title: Optional[str] = None,
    volume: Optional[str] = None,
    issue: Optional[str] = None,
    pages: Optional[str] = None,
    publisher: Optional[str] = None,
    place: Optional[str] = None,
    doi: Optional[str] = None,
    url: Optional[str] = None,
    abstract: Optional[str] = None,
    tags: Optional[Union[List[str], str]] = None,
    add_tags: Optional[Union[List[str], str]] = None,
    remove_tags: Optional[Union[List[str], str]] = None,
    collections: Optional[Union[List[str], str]] = None,
    collection_names: Optional[Union[List[str], str]] = None,
    extra_fields: Optional[Union[Dict[str, str], str]] = None,
    *,
    ctx: Context
) -> str:
    """
    Update an existing item in your Zotero library.

    Args:
        item_key: Zotero item key/ID to update
        title: New title for the item
        creators: New list of creator dictionaries (replaces existing creators)
        date: New publication date
        publication_title: New journal/publication name
        volume: New volume number
        issue: New issue number
        pages: New page range
        publisher: New publisher name
        place: New publication place
        doi: New Digital Object Identifier
        url: New URL
        abstract: New abstract or summary text
        tags: New list of tags (replaces existing tags)
        add_tags: Tags to add to existing tags (doesn't replace)
        remove_tags: Tags to remove from existing tags
        collections: New list of collection keys (replaces existing collections)
        collection_names: Collection names to add (will be automatically resolved to keys)
        extra_fields: Additional fields to update as key-value pairs
        ctx: MCP context

    Returns:
        Confirmation message with update details
    """
    try:
        ctx.info(f"Updating item {item_key}")
        zot = get_zotero_client()

        # Input validation and auto-correction for common mistakes
        def _normalize_update_tags_input(
            value: Optional[Union[List[str], str]],
            field_name: str,
        ) -> Tuple[Optional[List[str]], Optional[str]]:
            if value is None:
                return None, None

            parsed_value: object = value
            parsed_from_json = False
            if isinstance(value, str):
                raw = value.strip()
                try:
                    parsed_value = json.loads(raw)
                    parsed_from_json = True
                    ctx.info(f"✓ Auto-corrected {field_name} from JSON string to Python list")
                except json.JSONDecodeError:
                    if "," in raw:
                        parsed_value = [tag.strip() for tag in raw.split(",") if tag.strip()]
                        ctx.info(
                            f"✓ Auto-corrected {field_name} from comma-separated string to Python list"
                        )
                    else:
                        parsed_value = [raw] if raw else []
                        ctx.info(f"✓ Auto-corrected {field_name} from single string to Python list")

            if parsed_from_json and not isinstance(parsed_value, list):
                return (
                    None,
                    f"Error: {field_name} must be a list of strings "
                    "(JSON arrays only when passing JSON).",
                )

            if not isinstance(parsed_value, list):
                return None, f"Error: {field_name} must be a list of strings."

            normalized: List[str] = []
            for i, tag_value in enumerate(parsed_value):
                if not isinstance(tag_value, str):
                    return (
                        None,
                        f"Error: {field_name} entries must all be strings. "
                        f"Entry {i + 1} is {type(tag_value).__name__}.",
                    )
                stripped = tag_value.strip()
                if stripped:
                    normalized.append(stripped)

            return normalized, None

        tags, tags_error = _normalize_update_tags_input(tags, "tags")
        if tags_error:
            return tags_error

        add_tags, add_tags_error = _normalize_update_tags_input(add_tags, "add_tags")
        if add_tags_error:
            return add_tags_error

        remove_tags, remove_tags_error = _normalize_update_tags_input(
            remove_tags, "remove_tags"
        )
        if remove_tags_error:
            return remove_tags_error

        # Fix creators if passed as JSON string
        if creators is not None:
            if isinstance(creators, str):
                try:
                    creators = json.loads(creators)
                    ctx.info(f"Auto-corrected creators from JSON string to list")
                except json.JSONDecodeError:
                    return f"Error: creators parameter must be a list of dictionaries, not a JSON string."

            # Validate creator structure
            if creators:  # Only validate if not empty
                for i, creator in enumerate(creators):
                    if not isinstance(creator, dict):
                        return f"Error: Each creator must be a dictionary. Creator {i+1} is not a dictionary."

                    if "creatorType" not in creator:
                        return f"Error: Creator {i+1} is missing 'creatorType' field. Must be 'author', 'editor', 'contributor', etc."

                    # Check if it has either (firstName + lastName) OR name
                    has_split_name = "firstName" in creator and "lastName" in creator
                    has_single_name = "name" in creator

                    if not has_split_name and not has_single_name:
                        return f"Error: Creator {i+1} must have EITHER ('firstName' AND 'lastName') OR 'name'. Example: {{'creatorType': 'author', 'firstName': 'Jane', 'lastName': 'Doe'}}"

                    # Warn about empty names
                    if has_split_name:
                        if not creator.get("firstName") and not creator.get("lastName"):
                            ctx.warn(f"Creator {i+1} has empty firstName and lastName")
                    elif has_single_name:
                        if not creator.get("name"):
                            ctx.warn(f"Creator {i+1} has empty name field")

                ctx.info(f"Validated {len(creators)} creator(s) successfully")

        # Fix collections if passed as JSON string / plain string / comma-separated string
        if collections is not None:
            try:
                collections = _normalize_str_list_input(collections, field_name="collections")
                ctx.info(
                    "Auto-corrected collections to list "
                    f"({len(collections)} value(s))"
                )
            except ValueError as e:
                return f"Error: {str(e)}"

        # Fix collection_names if passed as JSON string / plain string / comma-separated string
        if collection_names is not None:
            try:
                collection_names = _normalize_str_list_input(
                    collection_names, field_name="collection_names"
                )
                ctx.info(
                    "Auto-corrected collection_names to list "
                    f"({len(collection_names)} value(s))"
                )
            except ValueError as e:
                return f"Error: {str(e)}"

        # Fix extra_fields if passed as JSON string
        if extra_fields is not None:
            if isinstance(extra_fields, str):
                try:
                    extra_fields = json.loads(extra_fields)
                    ctx.info(f"Auto-corrected extra_fields from JSON string to dict")
                except json.JSONDecodeError:
                    return f"Error: extra_fields parameter must be a dictionary, not a JSON string."
            if not isinstance(extra_fields, dict):
                return f"Error: extra_fields must be a dictionary. Example: {{'ISBN': '123-456'}}"

        # First, fetch the existing item
        try:
            item = zot.item(item_key)
            if not item:
                return f"Error: No item found with key: {item_key}"
        except Exception as e:
            return f"Error: Could not fetch item {item_key}: {str(e)}"

        # Get the item data
        item_data = item.get("data", {})
        original_title = item_data.get("title", "Untitled")

        # Track what we're updating
        updates = []

        # Update basic fields
        if title is not None:
            item_data["title"] = title
            updates.append(f"title: '{title}'")

        if date is not None:
            item_data["date"] = date
            updates.append(f"date: '{date}'")

        if publication_title is not None:
            item_data["publicationTitle"] = publication_title
            updates.append(f"publication title: '{publication_title}'")

        if volume is not None:
            item_data["volume"] = volume
            updates.append(f"volume: '{volume}'")

        if issue is not None:
            item_data["issue"] = issue
            updates.append(f"issue: '{issue}'")

        if pages is not None:
            item_data["pages"] = pages
            updates.append(f"pages: '{pages}'")

        if publisher is not None:
            item_data["publisher"] = publisher
            updates.append(f"publisher: '{publisher}'")

        if place is not None:
            item_data["place"] = place
            updates.append(f"place: '{place}'")

        if doi is not None:
            item_data["DOI"] = doi
            updates.append(f"DOI: '{doi}'")

        if url is not None:
            item_data["url"] = url
            updates.append(f"URL: '{url}'")

        if abstract is not None:
            item_data["abstractNote"] = abstract
            updates.append(f"abstract updated")

        # Update creators (replaces existing)
        if creators is not None:
            item_data["creators"] = creators
            updates.append(f"creators: {len(creators)} creator(s)")

        # Handle tags
        current_tags = item_data.get("tags", [])
        current_tag_values = {t["tag"] for t in current_tags}

        if tags is not None:
            # Replace all tags
            item_data["tags"] = [{"tag": tag} for tag in tags]
            updates.append(f"tags replaced with {len(tags)} tag(s)")
        else:
            # Add/remove specific tags
            if remove_tags:
                new_tags = [t for t in current_tags if t["tag"] not in remove_tags]
                item_data["tags"] = new_tags
                updates.append(f"removed {len(remove_tags)} tag(s)")
                current_tag_values = {t["tag"] for t in new_tags}

            if add_tags:
                for tag in add_tags:
                    if tag and tag not in current_tag_values:
                        item_data["tags"].append({"tag": tag})
                        current_tag_values.add(tag)
                updates.append(f"added {len(add_tags)} tag(s)")

        # Track collection operations (don't modify item_data["collections"] directly)
        collections_to_set: List[str] = []
        collections_to_add: List[str] = []

        if collections is not None:
            # User wants to replace collections - we'll handle this after update
            deduped = []
            seen = set()
            for key in collections:
                if key and key not in seen:
                    deduped.append(key)
                    seen.add(key)
            collections_to_set = deduped
        elif collection_names:
            # Resolve collection names to add
            ctx.info(f"Resolving collection names: {collection_names}")
            all_collections = zot.collections()
            collection_map = {c["data"].get("name", "").lower(): c["key"] for c in all_collections}

            for name in collection_names:
                name_lower = name.lower()
                if name_lower in collection_map:
                    resolved_key = collection_map[name_lower]
                    collections_to_add.append(resolved_key)
                    ctx.info(f"Resolved '{name}' to key: {resolved_key}")
                else:
                    # Try partial match
                    matches = [key for coll_name, key in collection_map.items() if name_lower in coll_name]
                    if matches:
                        collections_to_add.append(matches[0])
                        ctx.info(f"Resolved '{name}' to key: {matches[0]} (partial match)")
                    else:
                        ctx.warn(f"Collection '{name}' not found.")

            if collections_to_add:
                updates.append(f"will add to {len(collections_to_add)} collection(s)")

        # Update extra fields
        if extra_fields:
            for key, value in extra_fields.items():
                item_data[key] = value
                updates.append(f"{key}: '{value}'")

        collection_change_requested = collections is not None or bool(collections_to_add)

        # Check if there are any updates
        if not updates and not collection_change_requested:
            return f"No updates specified for item: \"{original_title}\" (Key: {item_key})"

        # Update the item in Zotero (if there are field updates)
        if updates:
            try:
                result = zot.update_item(item)
                ctx.info(f"Update result: {result}")
            except Exception as update_error:
                ctx.error(f"Failed to update item: {str(update_error)}")
                return f"Failed to update item: {str(update_error)}"

        # Apply collection changes
        collection_info = ""
        if collections is not None:
            collection_errors = []
            added_collections = []
            removed_collections = []

            current_collection_keys = []
            for coll_key in item_data.get("collections", []) or []:
                coll_key_str = str(coll_key).strip()
                if coll_key_str and coll_key_str not in current_collection_keys:
                    current_collection_keys.append(coll_key_str)

            target_set = set(collections_to_set)
            current_set = set(current_collection_keys)
            to_add = [key for key in collections_to_set if key not in current_set]
            to_remove = [key for key in current_collection_keys if key not in target_set]

            if to_remove:
                if not hasattr(zot, "deletefrom_collection"):
                    collection_errors.append(
                        "Collection removal is not supported by the current Zotero client. "
                        f"Could not remove: {', '.join(to_remove)}"
                    )
                    ctx.warn("deletefrom_collection not available; cannot remove existing collections")
                else:
                    for coll_key in to_remove:
                        try:
                            coll_name = coll_key
                            try:
                                coll = zot.collection(coll_key)
                                coll_name = coll["data"].get("name", coll_key)
                            except Exception:
                                pass

                            remove_result = zot.deletefrom_collection(coll_key, [item_key])
                            if remove_result:
                                removed_collections.append(coll_name)
                                ctx.info(f"Removed item from collection: {coll_name}")
                            else:
                                collection_errors.append(f"Failed to remove from '{coll_name}'")
                                ctx.warn(f"Failed to remove item from collection: {coll_name}")
                        except Exception as coll_error:
                            collection_errors.append(f"Error removing from {coll_key}: {str(coll_error)}")
                            ctx.warn(f"Error removing from collection {coll_key}: {str(coll_error)}")

            for coll_key in to_add:
                try:
                    coll = zot.collection(coll_key)
                    coll_name = coll["data"].get("name", coll_key)
                    add_result = zot.addto_collection(coll_key, [item_key])

                    if add_result:
                        added_collections.append(coll_name)
                        ctx.info(f"Added item to collection: {coll_name}")
                    else:
                        collection_errors.append(f"Failed to add to '{coll_name}'")
                        ctx.warn(f"Failed to add item to collection: {coll_name}")
                except Exception as coll_error:
                    collection_errors.append(f"Error with collection {coll_key}: {str(coll_error)}")
                    ctx.warn(f"Error adding to collection {coll_key}: {str(coll_error)}")

            if added_collections:
                collection_info += (
                    f"\n\n**Added to collection(s):** {', '.join(added_collections)}"
                )
            if removed_collections:
                collection_info += (
                    f"\n\n**Removed from collection(s):** {', '.join(removed_collections)}"
                )
            if collection_errors:
                collection_info += (
                    f"\n\n**Collection warnings:**\n"
                    + "\n".join(f"- {err}" for err in collection_errors)
                )
        elif collections_to_add:
            collection_errors = []
            successfully_added = []

            for coll_key in collections_to_add:
                try:
                    # Verify collection exists and add item to it
                    coll = zot.collection(coll_key)
                    coll_name = coll["data"].get("name", coll_key)
                    add_result = zot.addto_collection(coll_key, [item_key])

                    if add_result:
                        successfully_added.append(coll_name)
                        ctx.info(f"Added item to collection: {coll_name}")
                    else:
                        collection_errors.append(f"Failed to add to '{coll_name}'")
                        ctx.warn(f"Failed to add item to collection: {coll_name}")
                except Exception as coll_error:
                    collection_errors.append(f"Error with collection {coll_key}: {str(coll_error)}")
                    ctx.warn(f"Error adding to collection {coll_key}: {str(coll_error)}")

            if successfully_added:
                collection_info = f"\n\n**Added to collection(s):** {', '.join(successfully_added)}"
            if collection_errors:
                collection_info += f"\n\n**Collection warnings:**\n" + "\n".join(f"- {err}" for err in collection_errors)

        # Format the response
        response = [
            f"Successfully updated item: \"{original_title}\"",
            f"Item key: `{item_key}`",
        ]
        if updates:
            response.append("")
            response.append("**Updated fields:**")
            for update in updates:
                response.append(f"- {update}")

        if collection_info:
            response.append(collection_info)

        return "\n".join(response)

    except Exception as e:
        ctx.error(f"Error updating item: {str(e)}")
        return f"Error updating item: {str(e)}"


@mcp.tool(
    name="zotero_create_note",
    description="Create a new note for a Zotero item."
)
def create_note(
    item_key: str,
    note_title: str,
    note_text: str,
    tags: list[str] | None = None,
    *,
    ctx: Context
) -> str:
    """
    Create a new note for a Zotero item.

    Args:
        item_key: Zotero item key/ID to attach the note to
        note_title: Title for the note
        note_text: Content of the note (can include simple HTML formatting)
        tags: List of tags to apply to the note
        ctx: MCP context

    Returns:
        Confirmation message with the new note key
    """
    try:
        ctx.info(f"Creating note for item {item_key}")
        zot = get_zotero_client()

        # First verify the parent item exists
        try:
            parent = zot.item(item_key)
            parent_title = parent["data"].get("title", "Untitled Item")
        except Exception:
            return f"Error: No item found with key: {item_key}"

        # Format the note content with proper HTML
        # If the note_text already has HTML, use it directly
        if "<p>" in note_text or "<div>" in note_text:
            html_content = note_text
        else:
            # Convert plain text to HTML paragraphs - avoiding f-strings with replacements
            paragraphs = note_text.split("\n\n")
            html_parts = []
            for p in paragraphs:
                # Replace newlines with <br/> tags
                p_with_br = p.replace("\n", "<br/>")
                html_parts.append("<p>" + p_with_br + "</p>")
            html_content = "".join(html_parts)

        # Use note_title as a visible heading so the argument is not ignored.
        clean_title = (note_title or "").strip()
        if clean_title:
            safe_title = (
                clean_title.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            html_content = f"<h1>{safe_title}</h1>{html_content}"

        # Prepare the note data
        note_data = {
            "itemType": "note",
            "parentItem": item_key,
            "note": html_content,
            "tags": [{"tag": tag} for tag in (tags or [])]
        }

        # Use connector/saveItems for local mode since the local API
        # does not support POST to /api/users/0/items
        if is_local_mode():
            port = os.getenv("ZOTERO_LOCAL_PORT", "23119")
            connector_url = f"http://127.0.0.1:{port}/connector/saveItems"
            payload = {
                "items": [
                    {
                        "itemType": "note",
                        "note": html_content,
                        "tags": [tag for tag in (tags or [])],
                        "parentItem": item_key,
                    }
                ],
                "uri": "about:blank",
            }
            resp = requests.post(
                connector_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            if resp.status_code == 201:
                return f"Successfully created note for \"{parent_title}\" (parent key: {item_key})"
            else:
                return f"Failed to create note via local connector (HTTP {resp.status_code}): {resp.text}"
        else:
            # Remote API: use pyzotero's create_items
            result = zot.create_items([note_data])

            # Check if creation was successful
            if "success" in result and result["success"]:
                successful = result["success"]
                if len(successful) > 0:
                    note_key = next(iter(successful.keys()))
                    return f"Successfully created note for \"{parent_title}\"\n\nNote key: {note_key}"
                else:
                    return f"Note creation response was successful but no key was returned: {result}"
            else:
                return f"Failed to create note: {result.get('failed', 'Unknown error')}"

    except Exception as e:
        ctx.error(f"Error creating note: {str(e)}")
        return f"Error creating note: {str(e)}"


@mcp.tool(
    name="zotero_create_annotation",
    description="Create a highlight annotation on a PDF or EPUB attachment with optional comment."
)
def create_annotation(
    attachment_key: str,
    page: int,
    text: str,
    comment: str | None = None,
    color: str = "#ffd400",
    *,
    ctx: Context
) -> str:
    """
    Create a highlight annotation on a PDF or EPUB attachment.

    This tool handles multiple storage configurations:
    - Zotero Cloud Storage: Downloads file via Web API
    - WebDAV Storage: Downloads file via local Zotero (requires Zotero desktop running)
    - Annotations are always created via the Web API (required for write operations)

    Args:
        attachment_key: Attachment key (e.g., "NHZFE5A7")
        page: For PDF: 1-indexed page number. For EPUB: 1-indexed chapter number.
        text: Exact text to highlight (used to find coordinates/CFI)
        comment: Optional comment on the annotation
        color: Highlight color in hex format (default: "#ffd400" yellow)
        ctx: MCP context

    Returns:
        Confirmation message with the new annotation key
    """
    import tempfile

    from zotero_mcp.client import (
        get_local_zotero_client,
        get_web_zotero_client,
    )
    from zotero_mcp.pdf_utils import (
        find_text_position,
        get_page_label,
        build_annotation_position,
        verify_pdf_attachment,
    )

    try:
        ctx.info(f"Creating annotation on attachment {attachment_key}, page {page}")

        # Get clients for different operations
        local_client = get_local_zotero_client()
        web_client = get_web_zotero_client()

        # REQUIREMENT: Web API is required for creating annotations
        # Zotero's local API (port 23119) is read-only
        if not web_client:
            return (
                "Error: Web API credentials required for creating annotations.\n\n"
                "Please configure the following environment variables:\n"
                "- ZOTERO_API_KEY: Your Zotero API key (from zotero.org/settings/keys)\n"
                "- ZOTERO_LIBRARY_ID: Your library ID\n"
                "- ZOTERO_LIBRARY_TYPE: 'user' or 'group'\n\n"
                "Note: Zotero's local API is read-only and cannot create annotations."
            )

        # Use web client for metadata (it has the credentials)
        metadata_client = web_client

        # Verify the attachment exists and is a PDF
        try:
            attachment = metadata_client.item(attachment_key)
            attachment_data = attachment.get("data", {})

            if attachment_data.get("itemType") != "attachment":
                return f"Error: Item {attachment_key} is not an attachment"

            content_type = attachment_data.get("contentType", "")
            supported_types = {
                "application/pdf": "pdf",
                "application/epub+zip": "epub",
            }
            if content_type not in supported_types:
                return f"Error: Attachment {attachment_key} is not a PDF or EPUB (type: {content_type})"

            file_type = supported_types[content_type]
            filename = attachment_data.get("filename", f"{attachment_key}.{file_type}")

        except Exception as e:
            return f"Error: No attachment found with key: {attachment_key} ({e})"

        # Download the PDF to a temporary location
        # Strategy: Try multiple sources in order of likelihood to succeed
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, filename)
            ctx.info(f"Downloading PDF to {file_path}")

            download_errors = []
            downloaded = False

            # Source 1: Try local Zotero first (works for WebDAV and local storage)
            if local_client and not downloaded:
                try:
                    ctx.info("Trying local Zotero (WebDAV/local storage)...")
                    local_client.dump(attachment_key, filename=filename, path=tmpdir)
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        downloaded = True
                        ctx.info("PDF downloaded via local Zotero")
                except Exception as e:
                    download_errors.append(f"Local Zotero: {e}")

            # Source 2: Try Web API (works for Zotero Cloud Storage)
            if not downloaded:
                try:
                    ctx.info("Trying Zotero Web API (cloud storage)...")
                    web_client.dump(attachment_key, filename=filename, path=tmpdir)
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        downloaded = True
                        ctx.info("PDF downloaded via Web API")
                except Exception as e:
                    download_errors.append(f"Web API: {e}")

            if not downloaded:
                error_details = "\n".join(f"  - {err}" for err in download_errors)
                return (
                    f"Error: Could not download PDF attachment.\n\n"
                    f"Attempted sources:\n{error_details}\n\n"
                    "Possible solutions:\n"
                    "- **Zotero Cloud Storage**: Ensure file syncing is enabled in Zotero preferences\n"
                    "- **WebDAV Storage**: Ensure Zotero desktop is running with "
                    "'Allow other applications to communicate with Zotero' enabled\n"
                    "- **Linked files**: Linked attachments (not imported) cannot be accessed remotely"
                )

            # Verify the file is valid
            if file_type == "pdf":
                if not verify_pdf_attachment(file_path):
                    return f"Error: Downloaded file is not a valid PDF"
            else:  # epub
                from zotero_mcp.epub_utils import verify_epub_attachment
                if not verify_epub_attachment(file_path):
                    return f"Error: Downloaded file is not a valid EPUB"

            # Search for the text and get position data
            search_preview = text[:50] + "..." if len(text) > 50 else text
            location_type = "page" if file_type == "pdf" else "chapter"
            ctx.info(f"Searching for text in {location_type} {page}: '{search_preview}'")

            if file_type == "pdf":
                position_data = find_text_position(file_path, page, text)
            else:  # epub
                from zotero_mcp.epub_utils import find_text_in_epub
                position_data = find_text_in_epub(file_path, page, text)

            if "error" in position_data:
                # Build debug info message
                debug_lines = [
                    f"Error: {position_data['error']}",
                    f"",
                    f"Text searched: \"{text[:100]}{'...' if len(text) > 100 else ''}\"",
                ]

                best_score = position_data.get("best_score", 0)
                best_match = position_data.get("best_match")

                # Add "Did you mean" suggestion if we found a reasonable match
                if best_score >= 0.5 and best_match:
                    debug_lines.append("")
                    debug_lines.append("=" * 50)
                    debug_lines.append(f"DID YOU MEAN (score: {best_score:.0%}):")
                    debug_lines.append("")
                    # Show a useful preview - first 150 chars of the match
                    suggestion = best_match[:150].strip()
                    if len(best_match) > 150:
                        suggestion += "..."
                    debug_lines.append(f'  "{suggestion}"')
                    debug_lines.append("")
                    if position_data.get("page_found"):
                        debug_lines.append(f"  (Found on page {position_data['page_found']})")
                    debug_lines.append("=" * 50)
                    debug_lines.append("")
                    debug_lines.append("TIP: Copy the exact text from the PDF instead of paraphrasing.")
                elif best_score > 0:
                    debug_lines.append(f"")
                    debug_lines.append(f"Debug info:")
                    debug_lines.append(f"  Best match score: {best_score:.2f} (too low for suggestion)")
                    if best_match:
                        preview = best_match[:80]
                        debug_lines.append(f"  Best match text: \"{preview}...\"")
                    # Handle both PDF (page_found) and EPUB (chapter_found)
                    found_location = position_data.get("page_found") or position_data.get("chapter_found")
                    if found_location:
                        debug_lines.append(f"  Found in {location_type}: {found_location}")

                # Handle both PDF (pages_searched) and EPUB (chapters_searched)
                searched = position_data.get("pages_searched") or position_data.get("chapters_searched")
                if searched:
                    debug_lines.append(f"  {location_type.title()}s searched: {searched}")

                if best_score < 0.5:
                    debug_lines.extend([
                        "",
                        "Tips:",
                        f"- Copy the exact text from the {file_type.upper()} (don't paraphrase)",
                        "- Try a shorter, unique phrase from the beginning",
                        f"- Check that the {location_type} number is correct",
                    ])

                return "\n".join(debug_lines)

            # Build annotation data based on file type
            if file_type == "pdf":
                # Get page label (might differ from page number in some PDFs)
                page_label = get_page_label(file_path, page)

                # Build annotation position JSON for PDF
                annotation_position = build_annotation_position(
                    position_data["pageIndex"],
                    position_data["rects"]
                )
                sort_index = position_data["sort_index"]
            else:  # epub
                # For EPUB: leave pageLabel EMPTY for proper navigation
                # Zotero's manual EPUB annotations have empty pageLabel and it works
                page_label = ""  # Empty, not chapter number!
                annotation_position = position_data["annotation_position"]
                # EPUB sort index format: "spine_index|character_offset"
                # Use actual character position from CFI generation
                chapter = position_data.get("chapter_found", page)
                char_position = position_data.get("char_position", chapter * 1000)
                sort_index = f"{chapter:05d}|{char_position:08d}"

            # Prepare the annotation data
            annotation_data = {
                "itemType": "annotation",
                "parentItem": attachment_key,
                "annotationType": "highlight",
                "annotationText": text,
                "annotationComment": comment or "",
                "annotationColor": color,
                "annotationSortIndex": sort_index,
                "annotationPosition": annotation_position,
            }
            # Only add pageLabel if not empty (EPUB should not have it)
            if page_label:
                annotation_data["annotationPageLabel"] = page_label

            ctx.info(f"Creating annotation via Web API...")

            # Create the annotation using web client
            result = web_client.create_items([annotation_data])

            # Check if creation was successful
            if "success" in result and result["success"]:
                successful = result["success"]
                if len(successful) > 0:
                    annotation_key = list(successful.values())[0]
                    location_label = "Page" if file_type == "pdf" else "Chapter"
                    response = [
                        f"Successfully created highlight annotation",
                        f"",
                        f"**Annotation Key:** {annotation_key}",
                        f"**{location_label}:** {page_label}",
                    ]
                    # For EPUB, show if text was found in different chapter than requested
                    if file_type == "epub":
                        chapter_found = position_data.get("chapter_found", page)
                        if chapter_found != page:
                            response.append(f"**Note:** Text was found in chapter {chapter_found} (you specified {page})")
                        chapter_href = position_data.get("chapter_href", "")
                        if chapter_href:
                            response.append(f"**Section:** {chapter_href}")
                    response.append(f"**Text:** \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
                    if comment:
                        response.append(f"**Comment:** {comment}")
                    response.append(f"**Color:** {color}")
                    return "\n".join(response)
                else:
                    return f"Annotation creation response was successful but no key was returned: {result}"
            else:
                failed_info = result.get("failed", {})
                return f"Failed to create annotation: {failed_info}"

    except Exception as e:
        ctx.error(f"Error creating annotation: {str(e)}")
        return f"Error creating annotation: {str(e)}"


@mcp.tool(
    name="zotero_semantic_search",
    description="Prioritized search tool. Perform semantic search over your Zotero library using AI-powered embeddings."
)
def semantic_search(
    query: str,
    limit: int = 10,
    filters: dict[str, str] | str | None = None,
    *,
    ctx: Context
) -> str:
    """
    Perform semantic search over your Zotero library.

    Args:
        query: Search query text - can be concepts, topics, or natural language descriptions
        limit: Maximum number of results to return (default: 10)
        filters: Optional metadata filters as dict or JSON string. Example: {"item_type": "note"}
        ctx: MCP context

    Returns:
        Markdown-formatted search results with similarity scores
    """
    try:
        if not query.strip():
            return "Error: Search query cannot be empty"

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
        from zotero_mcp.semantic_search import create_semantic_search
        from pathlib import Path

        # Determine config path
        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"

        # Create semantic search instance
        search = create_semantic_search(str(config_path))

        # Perform search
        results = search.search(query=query, limit=limit, filters=filters)

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
            _ = result.get("metadata", {})
            zotero_item = result.get("zotero_item", {})

            if zotero_item:
                data = zotero_item.get("data", {})
                title = data.get("title", "Untitled")
                item_type = data.get("itemType", "unknown")
                key = result.get("item_key", "")

                # Format creators
                creators = data.get("creators", [])
                creators_str = format_creators(creators)

                output.append(f"## {i}. {title}")
                output.append(f"**Similarity Score:** {similarity_score:.3f}")
                output.append(f"**Type:** {item_type}")
                output.append(f"**Item Key:** {key}")
                output.append(f"**Authors:** {creators_str}")

                # Add date if available
                if date := data.get("date"):
                    output.append(f"**Date:** {date}")

                # Add abstract snippet if present
                if abstract := data.get("abstractNote"):
                    abstract_snippet = abstract[:200] + "..." if len(abstract) > 200 else abstract
                    output.append(f"**Abstract:** {abstract_snippet}")

                # Add tags if present
                if tags := data.get("tags"):
                    tag_list = [f"`{tag['tag']}`" for tag in tags]
                    if tag_list:
                        output.append(f"**Tags:** {' '.join(tag_list)}")

                # Show matched text snippet
                matched_text = result.get("matched_text", "")
                if matched_text:
                    snippet = matched_text[:300] + "..." if len(matched_text) > 300 else matched_text
                    output.append(f"**Matched Content:** {snippet}")

                output.append("")  # Empty line between items
            else:
                # Fallback if full Zotero item not available
                output.append(f"## {i}. Item {result.get('item_key', 'Unknown')}")
                output.append(f"**Similarity Score:** {similarity_score:.3f}")
                if error := result.get("error"):
                    output.append(f"**Error:** {error}")
                output.append("")

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error in semantic search: {str(e)}")
        return f"Error in semantic search: {str(e)}"


@mcp.tool(
    name="zotero_update_search_database",
    description="Update the semantic search database with latest Zotero items."
)
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
        from zotero_mcp.semantic_search import create_semantic_search
        from pathlib import Path

        # Determine config path
        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"

        # Create semantic search instance
        search = create_semantic_search(str(config_path))

        # Perform update with no fulltext extraction (for speed)
        stats = search.update_database(
            force_full_rebuild=force_rebuild,
            limit=limit,
            extract_fulltext=False
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
    description="Get status information about the semantic search database."
)
def get_search_database_status(*, ctx: Context) -> str:
    """
    Get semantic search database status.

    Args:
        ctx: MCP context

    Returns:
        Database status information
    """
    try:
        ctx.info("Getting semantic search database status...")

        # Import semantic search module
        from zotero_mcp.semantic_search import create_semantic_search
        from pathlib import Path

        # Determine config path
        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"

        # Create semantic search instance
        search = create_semantic_search(str(config_path))

        # Get status
        status = search.get_database_status()

        # Format results
        output = ["# Semantic Search Database Status", ""]

        collection_info = status.get("collection_info", {})
        output.append("## Collection Information")
        output.append(f"**Name:** {collection_info.get('name', 'Unknown')}")
        output.append(f"**Document Count:** {collection_info.get('count', 0)}")
        output.append(f"**Embedding Model:** {collection_info.get('embedding_model', 'Unknown')}")
        output.append(f"**Database Path:** {collection_info.get('persist_directory', 'Unknown')}")

        if collection_info.get('error'):
            output.append(f"**Error:** {collection_info['error']}")

        output.append("")

        update_config = status.get("update_config", {})
        output.append("## Update Configuration")
        output.append(f"**Auto Update:** {update_config.get('auto_update', False)}")
        output.append(f"**Frequency:** {update_config.get('update_frequency', 'manual')}")
        output.append(f"**Last Update:** {update_config.get('last_update', 'Never')}")
        output.append(f"**Should Update Now:** {status.get('should_update', False)}")

        if update_config.get('update_days'):
            output.append(f"**Update Interval:** Every {update_config['update_days']} days")

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error getting database status: {str(e)}")
        return f"Error getting database status: {str(e)}"


# --- Minimal wrappers for ChatGPT connectors ---
# These are required for ChatGPT custom MCP servers via web "connectors"
# specific tools required are "search" and "fetch"
# See: https://platform.openai.com/docs/mcp

def _extract_item_key_from_input(value: str) -> str | None:
    """Extract a Zotero item key from a Zotero URL, web URL, or bare key.
    Returns None if no plausible key is found.
    """
    if not value:
        return None
    text = value.strip()

    # Common patterns:
    # - zotero://select/items/<KEY>
    # - zotero://select/library/items/<KEY>
    # - https://www.zotero.org/.../items/<KEY>
    # - bare <KEY>
    patterns = [
        r"zotero://select/(?:library/)?items/([A-Za-z0-9]{8})",
        r"/items/([A-Za-z0-9]{8})(?:[^A-Za-z0-9]|$)",
        r"\b([A-Za-z0-9]{8})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None

@mcp.tool(
    name="search",
    description="ChatGPT-compatible search wrapper. Performs semantic search and returns JSON results."
)
def chatgpt_connector_search(
    query: str,
    *,
    ctx: Context
) -> str:
    """
    Returns a JSON-encoded string with shape {"results": [{"id","title","url"}, ...]}.
    The MCP runtime wraps this string as a single text content item.
    """
    try:
        default_limit = 10

        from zotero_mcp.semantic_search import create_semantic_search

        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
        search = create_semantic_search(str(config_path))

        result_list: list[dict[str, str]] = []
        results = search.search(query=query, limit=default_limit, filters=None) or {}
        for r in results.get("results", []):
            item_key = r.get("item_key") or ""
            title = ""
            if r.get("zotero_item"):
                data = (r.get("zotero_item") or {}).get("data", {})
                title = data.get("title", "")
            if not title:
                title = f"Zotero Item {item_key}" if item_key else "Zotero Item"
            url = f"zotero://select/items/{item_key}" if item_key else ""
            result_list.append({
                "id": item_key or uuid.uuid4().hex[:8],
                "title": title,
                "url": url,
            })

        return json.dumps({"results": result_list}, separators=(",", ":"))
    except Exception as e:
        ctx.error(f"Error in search wrapper: {str(e)}")
        return json.dumps({"results": []}, separators=(",", ":"))


@mcp.tool(
    name="fetch",
    description="ChatGPT-compatible fetch wrapper. Retrieves fulltext/metadata for a Zotero item by ID."
)
def connector_fetch(
    id: str,
    *,
    ctx: Context
) -> str:
    """
    Returns a JSON-encoded string with shape {"id","title","text","url","metadata":{...}}.
    The MCP runtime wraps this string as a single text content item.
    """
    try:
        item_key = (id or "").strip()
        if not item_key:
            return json.dumps({
                "id": id,
                "title": "",
                "text": "",
                "url": "",
                "metadata": {"error": "missing item key"}
            }, separators=(",", ":"))

        # Fetch item metadata for title and context
        zot = get_zotero_client()
        try:
            item = zot.item(item_key)
            data = item.get("data", {}) if item else {}
        except Exception:
            item = None
            data = {}

        title = data.get("title", f"Zotero Item {item_key}")
        zotero_url = f"zotero://select/items/{item_key}"
        # Prefer web URL for connectors; fall back to zotero:// if unknown
        lib_type = (os.getenv("ZOTERO_LIBRARY_TYPE", "user") or "user").lower()
        lib_id = os.getenv("ZOTERO_LIBRARY_ID", "")
        if lib_type not in ["user", "group"]:
            lib_type = "user"
        web_url = f"https://www.zotero.org/{'users' if lib_type=='user' else 'groups'}/{lib_id}/items/{item_key}" if lib_id else ""
        url = web_url or zotero_url

        # Use existing tool to get best-effort fulltext/markdown
        text_md = get_item_fulltext(item_key=item_key, ctx=ctx)
        # Extract the actual full text section if present, else keep as-is
        text_clean = text_md
        try:
            marker = "## Full Text"
            pos = text_md.find(marker)
            if pos >= 0:
                text_clean = text_md[pos + len(marker):].lstrip("\n #")
        except Exception:
            pass
        if (not text_clean or len(text_clean.strip()) < 40) and data:
            abstract = data.get("abstractNote", "")
            creators = data.get("creators", [])
            byline = format_creators(creators)
            text_clean = (f"{title}\n\n" + (f"Authors: {byline}\n" if byline else "") +
                          (f"Abstract:\n{abstract}" if abstract else "")) or text_md

        metadata = {
            "itemType": data.get("itemType", ""),
            "date": data.get("date", ""),
            "key": item_key,
            "doi": data.get("DOI", ""),
            "authors": format_creators(data.get("creators", [])),
            "tags": [t.get("tag", "") for t in (data.get("tags", []) or [])],
            "zotero_url": zotero_url,
            "web_url": web_url,
            "source": "zotero-mcp"
        }

        return json.dumps({
            "id": item_key,
            "title": title,
            "text": text_clean,
            "url": url,
            "metadata": metadata
        }, separators=(",", ":"))
    except Exception as e:
        ctx.error(f"Error in fetch wrapper: {str(e)}")
        return json.dumps({
            "id": id,
            "title": "",
            "text": "",
            "url": "",
            "metadata": {"error": str(e)}
        }, separators=(",", ":"))
