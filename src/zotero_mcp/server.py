"""
Zotero MCP server implementation.

Note: ChatGPT requires specific tool names "search" and "fetch", and so they
are defined and used and piped through to the main server tools. See bottom of file for details.
"""

from typing import Dict, List, Literal, Optional, Union
import os
import sys
import uuid
import asyncio
import json
import re
import tempfile
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastmcp import Context, FastMCP

from zotero_mcp.client import (
    clear_active_library,
    convert_to_markdown,
    format_item_metadata,
    generate_bibtex,
    get_active_library,
    get_attachment_details,
    get_web_zotero_client,
    get_zotero_client,
    set_active_library,
)
import requests
import httpx
import xml.etree.ElementTree as ET

from zotero_mcp.utils import format_creators, clean_html, is_local_mode


# ---------------------------------------------------------------------------
# Shared helpers for write operations
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


def _get_write_client(ctx):
    """Return (read_client, write_client) for hybrid-mode operations.

    In web-only mode: both are the web client.
    In local mode with web credentials: read from local, write to web.
    In local-only mode: raises ValueError with clear message.
    """
    read_zot = get_zotero_client()
    if not is_local_mode():
        return read_zot, read_zot
    web_zot = get_web_zotero_client()
    if web_zot is not None:
        override = get_active_library()
        if override:
            web_zot.library_id = override.get("library_id", web_zot.library_id)
            web_zot.library_type = override.get("library_type", web_zot.library_type)
        return read_zot, web_zot
    raise ValueError(
        "Cannot perform write operations in local-only mode. "
        "Add ZOTERO_API_KEY and ZOTERO_LIBRARY_ID to enable hybrid mode."
    )


def _handle_write_response(response, ctx=None):
    """Check if a pyzotero write operation succeeded.

    Handles httpx.Response, dict (from create_items/create_collections),
    or bool. Logs error details on failure if ctx is provided.
    """
    if hasattr(response, "status_code"):
        ok = response.status_code in (200, 204)
        if not ok and ctx is not None:
            ctx.error(f"Write failed ({response.status_code}): {response.text[:500]}")
        return ok
    if isinstance(response, dict):
        return bool(response.get("success"))
    return bool(response)


def _normalize_str_list_input(value, field_name="value"):
    """Normalize list-like user input into a list of non-empty strings.

    Handles: None, list, JSON string, comma-separated string, single string.
    """
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


def _resolve_collection_names(zot, names, ctx=None):
    """Resolve collection names to keys (case-insensitive).

    Returns a list of collection keys. If a name matches multiple collections,
    all are returned with a warning. Raises ValueError if a name has zero matches.
    """
    if not names:
        return []
    all_collections = zot.collections()
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
            ctx.warn(
                f"Multiple collections match '{name}': {matches}. "
                "Using all. Pass collection keys directly to disambiguate."
            )
        results.extend(matches)
    return results


def _strip_xml_tags(text):
    """Strip XML/HTML tags from text (e.g., JATS tags in CrossRef abstracts)."""
    if not text:
        return ""
    cleaned = re.sub(r'<[^>]+>', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _try_attach_oa_pdf(write_zot, item_key, doi, ctx):
    """Attempt to find and attach an open-access PDF for a DOI.

    Checks Unpaywall for OA availability, downloads to temp file,
    and attaches via pyzotero. Returns a status string for the user.
    Fails gracefully — item creation is never affected by PDF failure.
    """
    try:
        # Query Unpaywall
        unpaywall_resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": "zotero-mcp@users.noreply.github.com"},
            timeout=10,
        )
        if unpaywall_resp.status_code != 200:
            return "no open-access PDF found"

        oa_data = unpaywall_resp.json()
        best = oa_data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf") or best.get("url")

        if not pdf_url:
            return "no open-access PDF found"

        ctx.info(f"Found OA PDF: {pdf_url}")

        # Download PDF to temp file
        pdf_resp = requests.get(pdf_url, timeout=30, stream=True)
        pdf_resp.raise_for_status()

        # Verify it's actually a PDF (check content-type or magic bytes)
        content_type = pdf_resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and "octet-stream" not in content_type:
            return "OA link found but was not a PDF"

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f"{doi.replace('/', '_')}.pdf"
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "wb") as f:
                for chunk in pdf_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Attach to item
            write_zot.attachment_both(
                [(filename, filepath)],
                parentid=item_key,
            )

        return "open-access PDF attached"

    except Exception as e:
        ctx.info(f"PDF attachment failed (non-fatal): {e}")
        return f"no PDF attached ({e})"


def _normalize_doi(raw):
    """Normalize a DOI string from various input formats.

    Handles: doi:10.xxx, https://doi.org/10.xxx, http://dx.doi.org/10.xxx,
    bare 10.xxx/yyy, and strips trailing punctuation.
    Returns the bare DOI or None if not a valid DOI.
    """
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


def _normalize_arxiv_id(raw):
    """Normalize an arXiv ID from various input formats.

    Handles: arXiv:2401.00001, https://arxiv.org/abs/2401.00001,
    https://arxiv.org/pdf/2401.00001.pdf, old format hep-ph/9901234.
    Returns the bare arXiv ID or None.
    """
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
    description="Batch update tags across multiple items matching a search query or tag filter."
)
def batch_update_tags(
    query: str = "",
    add_tags: list[str] | str | None = None,
    remove_tags: list[str] | str | None = None,
    tag: str | None = None,
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
        import httpx

        if not query and not tag:
            return "Error: Must provide a search query and/or tag filter"

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

        # In local mode, the local API is read-only and cannot update items.
        # If web API credentials are available, use the web client for writes.
        # The local client is still used for reading/searching (faster).
        write_zot = zot
        if is_local_mode():
            web_zot = get_web_zotero_client()
            if web_zot is not None:
                write_zot = web_zot
                ctx.info("Local mode: using web API for tag writes")
            else:
                return (
                    "Error: Cannot update tags in local-only mode.\n\n"
                    "Zotero's local API is read-only. To enable tag updates, add these "
                    "environment variables to your Claude Desktop config:\n"
                    "- ZOTERO_API_KEY: Your Zotero API key (from zotero.org/settings/keys)\n"
                    "- ZOTERO_LIBRARY_ID: Your library ID (from the same page)\n"
                    "- ZOTERO_LIBRARY_TYPE: 'user' or 'group'\n\n"
                    "You can keep ZOTERO_LOCAL=true alongside these for fast reads."
                )

        if isinstance(limit, str):
            limit = int(limit)

        # Search for items matching the query and/or tag filter
        params = {"limit": limit}
        if query:
            params["q"] = query
        if tag:
            params["tag"] = tag
        zot.add_parameters(**params)
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

                    ctx.info(f"Update result: {result}")
                    # pyzotero's update_item returns an httpx.Response.
                    # Success = 204 No Content. The response is truthy for 2xx codes.
                    if isinstance(result, httpx.Response) and result.is_success:
                        updated_count += 1
                    elif isinstance(result, dict) and result.get("success"):
                        # Fallback check in case pyzotero behavior changes
                        updated_count += 1
                    elif result is True:
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
        zot.add_parameters(q=query, qmode="everything", itemType="note", limit=limit or 20)
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
        query_terms = query_lower.split()
        note_results = []

        for note in notes:
            data = note.get("data", {})
            note_text = data.get("note", "").lower()

            if all(term in note_text for term in query_terms):
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
    name="zotero_create_note",
    description="Create a new note for a Zotero item."
)
def create_note(
    item_key: str,
    note_title: str,
    note_text: str,
    tags: list[str] | str | None = None,
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
        # Normalize tags (LLMs often pass JSON strings instead of lists)
        tags = _normalize_str_list_input(tags, "tags") if tags is not None else []
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

        # In local mode, the local API does not support POST to create items,
        # and the connector/saveItems endpoint ignores parentItem (creating
        # standalone notes instead of child notes). If an API key is available,
        # use the web API which properly supports parentItem.
        if is_local_mode():
            web_zot = get_web_zotero_client()
            if web_zot is not None:
                result = web_zot.create_items([note_data])
                if "success" in result and result["success"]:
                    successful = result["success"]
                    if len(successful) > 0:
                        note_key = next(iter(successful.values()))
                        return f"Successfully created note for \"{parent_title}\"\n\nNote key: {note_key}"
                    else:
                        return f"Note creation response was successful but no key was returned: {result}"
                else:
                    return f"Failed to create note: {result.get('failed', 'Unknown error')}"
            else:
                # Fallback: connector endpoint (note will NOT be attached as child)
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
                    return (
                        f"Note created for \"{parent_title}\" but it is a standalone note, not attached "
                        f"to the paper.\n\n"
                        f"To create properly attached child notes, add these environment variables "
                        f"to your Claude Desktop config alongside ZOTERO_LOCAL=true:\n"
                        f"- ZOTERO_API_KEY: Your Zotero API key (from zotero.org/settings/keys)\n"
                        f"- ZOTERO_LIBRARY_ID: Your library ID (from the same page)\n"
                        f"- ZOTERO_LIBRARY_TYPE: 'user' or 'group'"
                    )
                else:
                    return f"Failed to create note via local connector (HTTP {resp.status_code}): {resp.text}"
        else:
            # Remote API: use pyzotero's create_items
            result = zot.create_items([note_data])

            # Check if creation was successful
            if "success" in result and result["success"]:
                successful = result["success"]
                if len(successful) > 0:
                    note_key = next(iter(successful.values()))
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


# ---------------------------------------------------------------------------
# Feature 1: Create Collection
# ---------------------------------------------------------------------------

@mcp.tool(
    name="zotero_create_collection",
    description="Create a new collection (project/folder) in your Zotero library."
)
def create_collection(
    name: str,
    parent_collection: str | None = None,
    *,
    ctx: Context
) -> str:
    try:
        read_zot, write_zot = _get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        ctx.info(f"Creating collection '{name}'")

        # Resolve parent_collection name if it doesn't look like a key
        parent_key = parent_collection
        if parent_collection and not re.match(r'^[A-Z0-9]{8}$', parent_collection):
            try:
                keys = _resolve_collection_names(read_zot, [parent_collection], ctx=ctx)
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


# ---------------------------------------------------------------------------
# Feature 2: Search Collections
# ---------------------------------------------------------------------------

@mcp.tool(
    name="zotero_search_collections",
    description="Search for collections by name to find their keys."
)
def search_collections(
    query: str,
    *,
    ctx: Context
) -> str:
    try:
        zot = get_zotero_client()
        ctx.info(f"Searching collections for '{query}'")

        collections = zot.collections()
        if not collections:
            return "No collections found in your Zotero library."

        query_lower = query.lower()
        matching = [
            c for c in collections
            if query_lower in c.get("data", {}).get("name", "").lower()
        ]

        if not matching:
            return f"No collections found matching '{query}'"

        lines = [f"# Collections matching '{query}'", ""]
        for i, coll in enumerate(matching, 1):
            name = coll["data"].get("name", "Unnamed")
            key = coll["key"]
            parent_key = coll["data"].get("parentCollection")
            lines.append(f"## {i}. {name}")
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


# ---------------------------------------------------------------------------
# Feature 3: Manage Collection Membership
# ---------------------------------------------------------------------------

@mcp.tool(
    name="zotero_manage_collections",
    description="Add or remove items from collections."
)
def manage_collections(
    item_keys: list[str] | str,
    add_to: list[str] | str | None = None,
    remove_from: list[str] | str | None = None,
    *,
    ctx: Context
) -> str:
    try:
        read_zot, write_zot = _get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        keys = _normalize_str_list_input(item_keys, "item_keys")
        add_colls = _normalize_str_list_input(add_to, "add_to")
        remove_colls = _normalize_str_list_input(remove_from, "remove_from")

        if not keys:
            return "Error: No item keys provided."
        if not add_colls and not remove_colls:
            return "Error: Must specify add_to and/or remove_from."

        results = []

        for coll_key in add_colls:
            for item_key in keys:
                item_dict = write_zot.item(item_key)
                resp = write_zot.addto_collection(coll_key, item_dict)
                if _handle_write_response(resp, ctx):
                    results.append(f"Added {item_key} to {coll_key}")
                else:
                    results.append(f"Failed to add {item_key} to {coll_key}")

        for coll_key in remove_colls:
            for item_key in keys:
                item_dict = write_zot.item(item_key)
                resp = write_zot.deletefrom_collection(coll_key, item_dict)
                if _handle_write_response(resp, ctx):
                    results.append(f"Removed {item_key} from {coll_key}")
                else:
                    results.append(f"Failed to remove {item_key} from {coll_key}")

        return "\n".join(results)

    except ValueError as e:
        return f"Input error: {e}"
    except Exception as e:
        ctx.error(f"Error managing collections: {e}")
        return f"Error managing collections: {e}"


# ---------------------------------------------------------------------------
# Feature 4: Add by DOI
# ---------------------------------------------------------------------------

@mcp.tool(
    name="zotero_add_by_doi",
    description="Add a paper to your Zotero library by DOI. Fetches metadata from CrossRef."
)
def add_by_doi(
    doi: str,
    collections: list[str] | str | None = None,
    tags: list[str] | str | None = None,
    *,
    ctx: Context
) -> str:
    try:
        read_zot, write_zot = _get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        normalized = _normalize_doi(doi)
        if not normalized:
            return f"Error: '{doi}' does not appear to be a valid DOI."

        ctx.info(f"Fetching metadata for DOI: {normalized}")

        resp = requests.get(
            f"https://api.crossref.org/works/{normalized}",
            headers={
                "User-Agent": "zotero-mcp/1.0 (https://github.com/ehawkin/zotero-mcp)",
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

        abstract = _strip_xml_tags(cr.get("abstract", ""))
        if abstract:
            field_map["abstractNote"] = abstract

        for field, value in field_map.items():
            if field in item_data and value:
                item_data[field] = value

        # Tags
        tag_list = _normalize_str_list_input(tags, "tags")
        if tag_list:
            item_data["tags"] = [{"tag": t} for t in tag_list]

        # Collections
        coll_keys = _normalize_str_list_input(collections, "collections")
        if coll_keys:
            item_data["collections"] = coll_keys

        # Create item
        result = write_zot.create_items([item_data])

        if isinstance(result, dict) and result.get("success"):
            item_key = next(iter(result["success"].values()))
            title = item_data.get("title", normalized)

            # Attempt open-access PDF attachment
            pdf_status = _try_attach_oa_pdf(write_zot, item_key, normalized, ctx)

            return (
                f"Successfully added: **{title}**\n\n"
                f"Item key: `{item_key}`\n"
                f"Type: {zot_type}\n"
                f"DOI: {normalized}\n"
                f"PDF: {pdf_status}"
            )
        return f"Failed to create item: {result}"

    except requests.Timeout:
        return "Error: CrossRef API request timed out. Please try again."
    except requests.RequestException as e:
        return f"Error fetching from CrossRef: {e}"
    except Exception as e:
        ctx.error(f"Error adding by DOI: {e}")
        return f"Error adding by DOI: {e}"


# ---------------------------------------------------------------------------
# Feature 5: Add by URL
# ---------------------------------------------------------------------------

@mcp.tool(
    name="zotero_add_by_url",
    description="Add a paper by URL. Supports DOI URLs, arXiv URLs, and general web pages."
)
def add_by_url(
    url: str,
    collections: list[str] | str | None = None,
    tags: list[str] | str | None = None,
    *,
    ctx: Context
) -> str:
    try:
        read_zot, write_zot = _get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        url = (url or "").strip()
        if not url:
            return "Error: No URL provided."

        # DOI URL routing
        doi = _normalize_doi(url)
        if doi:
            return add_by_doi(doi=url, collections=collections, tags=tags, ctx=ctx)

        # arXiv URL routing
        arxiv_id = _normalize_arxiv_id(url)
        if arxiv_id:
            return _add_by_arxiv(arxiv_id, collections, tags, write_zot, ctx)

        # Generic webpage
        ctx.info(f"Creating webpage item for: {url}")
        template = write_zot.item_template("webpage")
        template["url"] = url
        template["title"] = url
        template["accessDate"] = ""

        tag_list = _normalize_str_list_input(tags, "tags")
        if tag_list:
            template["tags"] = [{"tag": t} for t in tag_list]
        coll_keys = _normalize_str_list_input(collections, "collections")
        if coll_keys:
            template["collections"] = coll_keys

        result = write_zot.create_items([template])
        if isinstance(result, dict) and result.get("success"):
            item_key = next(iter(result["success"].values()))
            return f"Created webpage item for: {url}\n\nItem key: `{item_key}`"
        return f"Failed to create item: {result}"

    except Exception as e:
        ctx.error(f"Error adding by URL: {e}")
        return f"Error adding by URL: {e}"


def _add_by_arxiv(arxiv_id, collections, tags, write_zot, ctx):
    """Add an arXiv paper by ID. Internal helper for add_by_url."""
    ctx.info(f"Fetching arXiv metadata for: {arxiv_id}")

    resp = requests.get(
        f"https://export.arxiv.org/api/query?id_list={arxiv_id}",
        timeout=15,
    )
    resp.raise_for_status()

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

    tag_list = _normalize_str_list_input(tags, "tags")
    if tag_list:
        template["tags"] = [{"tag": t} for t in tag_list]
    coll_keys = _normalize_str_list_input(collections, "collections")
    if coll_keys:
        template["collections"] = coll_keys

    result = write_zot.create_items([template])
    if isinstance(result, dict) and result.get("success"):
        item_key = next(iter(result["success"].values()))

        # arXiv always has a free PDF — try to attach it
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_status = "no PDF attached"
        try:
            pdf_resp = requests.get(pdf_url, timeout=30, stream=True)
            pdf_resp.raise_for_status()
            with tempfile.TemporaryDirectory() as tmpdir:
                filename = f"arxiv_{arxiv_id.replace('/', '_')}.pdf"
                filepath = os.path.join(tmpdir, filename)
                with open(filepath, "wb") as f:
                    for chunk in pdf_resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                write_zot.attachment_both(
                    [(filename, filepath)],
                    parentid=item_key,
                )
            pdf_status = "PDF attached"
        except Exception as e:
            ctx.info(f"arXiv PDF attachment failed (non-fatal): {e}")
            pdf_status = f"no PDF attached ({e})"

        return (
            f"Successfully added arXiv paper: **{title}**\n\n"
            f"Item key: `{item_key}`\n"
            f"arXiv ID: {arxiv_id}\n"
            f"PDF: {pdf_status}"
        )
    return f"Failed to create arXiv item: {result}"


# ---------------------------------------------------------------------------
# Feature 6: Update Item Metadata
# ---------------------------------------------------------------------------

@mcp.tool(
    name="zotero_update_item",
    description="Update metadata for an existing item in your Zotero library."
)
def update_item(
    item_key: str,
    title: str | None = None,
    creators: list[dict] | str | None = None,
    date: str | None = None,
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
    *,
    ctx: Context
) -> str:
    try:
        read_zot, write_zot = _get_write_client(ctx)
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
        item = write_zot.item(item_key)
        data = item.get("data", {})
        changes = []

        # Apply field updates
        field_updates = {}
        if title is not None:
            field_updates["title"] = title
        if date is not None:
            field_updates["date"] = date
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

        for field, value in field_updates.items():
            if field in data:
                old = data[field]
                if old != value:
                    changes.append(f"- **{field}**: '{old}' -> '{value}'")
                data[field] = value

        # Creators
        if creators is not None:
            if isinstance(creators, str):
                creators = json.loads(creators)
            data["creators"] = creators
            changes.append("- **creators**: updated")

        # Tags
        if tags is not None:
            tag_list = _normalize_str_list_input(tags, "tags")
            data["tags"] = [{"tag": t} for t in tag_list]
            changes.append(f"- **tags**: replaced with {tag_list}")
        elif add_tags is not None or remove_tags is not None:
            existing = {t["tag"] for t in data.get("tags", [])}
            if add_tags is not None:
                to_add = _normalize_str_list_input(add_tags, "add_tags")
                existing.update(to_add)
                changes.append(f"- **tags**: added {to_add}")
            if remove_tags is not None:
                to_remove = set(_normalize_str_list_input(remove_tags, "remove_tags"))
                existing -= to_remove
                changes.append(f"- **tags**: removed {list(to_remove)}")
            data["tags"] = [{"tag": t} for t in sorted(existing)]

        # Collections — both params ADD to existing collections (never replace)
        if collections is not None:
            coll_keys = _normalize_str_list_input(collections, "collections")
            existing_colls = set(data.get("collections", []))
            existing_colls.update(coll_keys)
            data["collections"] = list(existing_colls)
            changes.append(f"- **collections**: added {coll_keys}")
        if collection_names is not None:
            names = _normalize_str_list_input(collection_names, "collection_names")
            resolved = _resolve_collection_names(read_zot, names, ctx=ctx)
            existing_colls = set(data.get("collections", []))
            existing_colls.update(resolved)
            data["collections"] = list(existing_colls)
            changes.append(f"- **collections**: added {resolved}")

        if not changes:
            return "No changes to apply."

        resp = write_zot.update_item(item)
        if _handle_write_response(resp, ctx):
            return f"Successfully updated item `{item_key}`:\n\n" + "\n".join(changes)
        return f"Failed to update item: write operation returned failure"

    except ValueError as e:
        return f"Input error: {e}"
    except Exception as e:
        ctx.error(f"Error updating item: {e}")
        return f"Error updating item: {e}"


# ---------------------------------------------------------------------------
# Feature 7: Find Duplicates
# ---------------------------------------------------------------------------

@mcp.tool(
    name="zotero_find_duplicates",
    description="Find duplicate items in your library by title and/or DOI."
)
def find_duplicates(
    method: Literal["title", "doi", "both"] = "both",
    collection_key: str | None = None,
    limit: int = 50,
    *,
    ctx: Context
) -> str:
    try:
        zot = get_zotero_client()
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


# ---------------------------------------------------------------------------
# Feature 8: Merge Duplicates
# ---------------------------------------------------------------------------

@mcp.tool(
    name="zotero_merge_duplicates",
    description=(
        "Merge duplicate items. Consolidates tags, collections, notes, annotations, "
        "and all child items into the keeper. Duplicates are moved to Trash (recoverable). "
        "Dry-run by default — call with confirm=True to execute."
    )
)
def merge_duplicates(
    keeper_key: str,
    duplicate_keys: list[str] | str,
    confirm: bool = False,
    *,
    ctx: Context
) -> str:
    try:
        read_zot, write_zot = _get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        dup_keys = _normalize_str_list_input(duplicate_keys, "duplicate_keys")

        # Safety: remove keeper from duplicates
        if keeper_key in dup_keys:
            dup_keys.remove(keeper_key)
            ctx.warn(f"Keeper key '{keeper_key}' was in duplicate list — removed.")

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
                f"**Child items to re-parent:** {total_children_to_move}",
                f"  (notes, PDFs, annotations, highlights, etc.)",
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
            resp = write_zot.update_item(keeper)
            if not _handle_write_response(resp, ctx):
                return "Error: Failed to merge tags into keeper."
            keeper = write_zot.item(keeper_key)  # re-fetch for version

        # Step 4: Consolidate collections
        for coll_key in new_collections:
            resp = write_zot.addto_collection(coll_key, keeper)
            if not _handle_write_response(resp, ctx):
                ctx.warn(f"Failed to add keeper to collection {coll_key}")
            keeper = write_zot.item(keeper_key)  # re-fetch for version

        # Step 5: Re-parent children
        moved = []
        failed = []
        for dup in duplicates:
            for child in dup["children"]:
                child_key = child.get("key", "?")
                try:
                    fresh_child = write_zot.item(child_key)
                    fresh_child.get("data", {})["parentItem"] = keeper_key
                    resp = write_zot.update_item(fresh_child)
                    if _handle_write_response(resp, ctx):
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
                    ctx.warn(f"Failed to trash {dup_key}: HTTP {resp.status_code}")
            except Exception as e:
                ctx.warn(f"Failed to trash {dup_key}: {e}")

        return (
            f"Merge complete.\n\n"
            f"- Tags merged: {len(new_tags)} new\n"
            f"- Collections added: {len(new_collections)} new\n"
            f"- Children re-parented: {len(moved)}\n"
            f"- Duplicates trashed: {', '.join(f'`{k}`' for k in trashed)}\n\n"
            "Trashed items can be restored from Zotero's Trash."
        )

    except ValueError as e:
        return f"Input error: {e}"
    except Exception as e:
        ctx.error(f"Error merging duplicates: {e}")
        return f"Error merging duplicates: {e}"


# ---------------------------------------------------------------------------
# Feature 9: PDF Outline Extraction
# ---------------------------------------------------------------------------

@mcp.tool(
    name="zotero_get_pdf_outline",
    description="Extract the table of contents / outline from a PDF attachment."
)
def get_pdf_outline(
    item_key: str,
    *,
    ctx: Context
) -> str:
    try:
        import fitz
        import tempfile

        zot = get_zotero_client()
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

    except ImportError:
        return "Error: PyMuPDF (fitz) is required for PDF outline extraction."
    except Exception as e:
        ctx.error(f"Error extracting PDF outline: {e}")
        return f"Error extracting PDF outline: {e}"


# ---------------------------------------------------------------------------
# Feature 10: Add from File
# ---------------------------------------------------------------------------

@mcp.tool(
    name="zotero_add_from_file",
    description=(
        "Add an item to Zotero from a local PDF file. "
        "Attempts DOI extraction for rich metadata. "
        "File path must be absolute and point to a .pdf or .epub file."
    )
)
def add_from_file(
    file_path: str,
    title: str | None = None,
    item_type: str = "document",
    collections: list[str] | str | None = None,
    tags: list[str] | str | None = None,
    *,
    ctx: Context
) -> str:
    try:
        read_zot, write_zot = _get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        # Path validation
        if not os.path.isabs(file_path):
            return "Error: file_path must be an absolute path."
        if os.path.islink(file_path):
            return "Error: Symlinks are not allowed for security reasons."
        if not os.path.isfile(file_path):
            return f"Error: File not found: {file_path}"

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
                        found_doi = _normalize_doi(candidate)
                        if found_doi:
                            extracted_doi = found_doi
                            break

                # Scan first page text
                if not extracted_doi and doc.page_count > 0:
                    text = doc[0].get_text()[:3000]
                    m = re.search(r'10\.\d{4,9}/[^\s]+', text)
                    if m:
                        found_doi = _normalize_doi(m.group(0))
                        if found_doi:
                            extracted_doi = found_doi

                doc.close()
            except Exception as e:
                ctx.info(f"DOI extraction failed (non-fatal): {e}")

        # Create the metadata item
        if extracted_doi:
            ctx.info(f"Found DOI: {extracted_doi}")
            result_msg = add_by_doi(doi=extracted_doi, collections=collections, tags=tags, ctx=ctx)
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

            tag_list = _normalize_str_list_input(tags, "tags")
            if tag_list:
                template["tags"] = [{"tag": t} for t in tag_list]
            coll_keys = _normalize_str_list_input(collections, "collections")
            if coll_keys:
                template["collections"] = coll_keys

            result = write_zot.create_items([template])
            if isinstance(result, dict) and result.get("success"):
                parent_key = next(iter(result["success"].values()))
            else:
                return f"Failed to create item: {result}"

        # Attach the file
        try:
            display_name = os.path.basename(file_path)
            attach_result = write_zot.attachment_both(
                [(display_name, file_path)],
                parentid=parent_key,
            )
            attach_info = f"File attached: {display_name}"
        except Exception as e:
            attach_info = f"Item created but file attachment failed: {e}"

        return (
            f"Item key: `{parent_key}`\n"
            f"{'DOI: ' + extracted_doi + chr(10) if extracted_doi else ''}"
            f"{attach_info}"
        )

    except Exception as e:
        ctx.error(f"Error adding from file: {e}")
        return f"Error adding from file: {e}"
