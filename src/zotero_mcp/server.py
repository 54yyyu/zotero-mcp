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
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import Context, FastMCP

from zotero_mcp.client import (
    convert_to_markdown,
    format_item_metadata,
    generate_bibtex,
    get_attachment_details,
    get_zotero_client,
)
from zotero_mcp.utils import format_creators

@asynccontextmanager
async def server_lifespan(server: FastMCP):
    """Manage server startup and shutdown lifecycle."""
    sys.stderr.write("Starting Zotero MCP server...\n")
    
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
                        stats = search.update_database(extract_fulltext=False)
                        sys.stderr.write(f"Database update completed: {stats.get('processed_items', 0)} items processed\n")
                    except Exception as e:
                        sys.stderr.write(f"Background database update failed: {e}\n")
                
                # Start background task
                asyncio.create_task(background_update())
    
    except Exception as e:
        sys.stderr.write(f"Warning: Could not check semantic search auto-update: {e}\n")
    
    yield {}
    
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
    limit: Optional[Union[int, str]] = 10,
    tag: Optional[List[str]] = None,
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
    description="Search for items in your Zotero library by tag. " \
    "Conditions are ANDed, each term supports disjunction`||` and exclusion`-`."
)
def search_by_tag(
    tag: List[str],
    item_type: str = "-attachment",
    limit: Optional[Union[int, str]] = 10,
    *,
    ctx: Context
) -> str:
    """
    Search for items in your Zotero library by tag。
    Conditions are ANDed, each term supports disjunction`||` and exclusion`-`.
    
    Args:
        tag: List of tag conditions. Items are returned only if they satisfy 
            ALL conditions in the list. Each tag condition can be expressed 
            in two ways:
                As alternatives: tag1 || tag2 (matches items with either tag1 OR tag2)
                As exclusions: -tag (matches items that do NOT have this tag)
            For example, a tag field with ["research || important", "-draft"] would 
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
    limit: Optional[Union[int, str]] = None,
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
    limit: Optional[Union[int, str]] = 50,
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
    limit: Optional[Union[int, str]] = None,
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
    name="zotero_get_recent",
    description="Get recently added items to your Zotero library."
)
def get_recent(
    limit: Union[int, str] = 10,
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
    add_tags: Optional[Union[List[str], str]] = None,
    remove_tags: Optional[Union[List[str], str]] = None,
    limit: Union[int, str] = 50,
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
        
        # Debug logging... commented out for now but could be useful in future.
        # ctx.info(f"add_tags type: {type(add_tags)}, value: {add_tags}")
        # ctx.info(f"remove_tags type: {type(remove_tags)}, value: {remove_tags}")
        
        # Handle case where add_tags might be a JSON string instead of list
        if add_tags and isinstance(add_tags, str):
            try:
                import json
                add_tags = json.loads(add_tags)
                ctx.info(f"Parsed add_tags from JSON string: {add_tags}")
            except json.JSONDecodeError:
                return f"Error: add_tags appears to be malformed JSON string: {add_tags}"
        
        # Handle case where remove_tags might be a JSON string instead of list  
        if remove_tags and isinstance(remove_tags, str):
            try:
                import json
                remove_tags = json.loads(remove_tags)
                ctx.info(f"Parsed remove_tags from JSON string: {remove_tags}")
            except json.JSONDecodeError:
                return f"Error: remove_tags appears to be malformed JSON string: {remove_tags}"
        
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
    conditions: List[Dict[str, str]],
    join_mode: Literal["all", "any"] = "all",
    sort_by: Optional[str] = None,
    sort_direction: Literal["asc", "desc"] = "asc",
    limit: Union[int, str] = 50,
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
        if not conditions:
            return "Error: No search conditions provided"
        
        ctx.info(f"Performing advanced search with {len(conditions)} conditions")
        zot = get_zotero_client()
        
        # Prepare search parameters
        params = {}
        
        # Add sorting parameters if specified
        if sort_by:
            params["sort"] = sort_by
            params["direction"] = sort_direction
        
        if isinstance(limit, str):
            limit = int(limit)
        
        # Add limit parameter
        params["limit"] = limit
        
        # Build search conditions
        search_conditions = []
        for i, condition in enumerate(conditions):
            if "field" not in condition or "operation" not in condition or "value" not in condition:
                return f"Error: Condition {i+1} is missing required fields (field, operation, value)"
            
            # Map common field names to Zotero API fields if needed
            field = condition["field"]
            operation = condition["operation"]
            value = condition["value"]
            
            # Handle special fields
            if field == "author" or field == "creator":
                field = "creator"
            elif field == "year":
                field = "date"
                # Convert year to partial date format for matching
                value = str(value)
            
            search_conditions.append({
                "condition": field,
                "operator": operation,
                "value": value
            })
        
        # Add join mode condition
        search_conditions.append({
            "condition": "joinMode",
            "operator": join_mode,
            "value": ""
        })
        
        # Create a saved search
        search_name = f"temp_search_{uuid.uuid4().hex[:8]}"
        saved_search = zot.saved_search(
            search_name,
            search_conditions
        )
        
        # Extract the search key from the result
        if not saved_search.get("success"):
            return f"Error creating saved search: {saved_search.get('failed', 'Unknown error')}"
        
        search_key = next(iter(saved_search.get("success", {}).values()), None)
        
        # Execute the saved search
        try:
            results = zot.collection_items(search_key)
        finally:
            # Clean up the temporary saved search
            try:
                zot.delete_saved_search([search_key])
            except Exception as cleanup_error:
                ctx.warn(f"Error cleaning up saved search: {str(cleanup_error)}")
        
        # Format the results
        if not results:
            return "No items found matching the search criteria."
        
        output = ["# Advanced Search Results", ""]
        output.append(f"Found {len(results)} items matching the search criteria:")
        output.append("")
        
        # Add search criteria summary
        output.append("## Search Criteria")
        output.append(f"Join mode: {join_mode.upper()}")
        
        for i, condition in enumerate(conditions, 1):
            output.append(f"{i}. {condition['field']} {condition['operation']} \"{condition['value']}\"")
        
        output.append("")
        
        # Format results
        output.append("## Results")
        
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
            output.append(f"### {i}. {title}")
            output.append(f"**Type:** {item_type}")
            output.append(f"**Item Key:** {key}")
            output.append(f"**Date:** {date}")
            output.append(f"**Authors:** {creators_str}")
            
            # Add abstract snippet if present
            if abstract := data.get("abstractNote"):
                # Limit abstract length for search results
                abstract_snippet = abstract[:150] + "..." if len(abstract) > 150 else abstract
                output.append(f"**Abstract:** {abstract_snippet}")
            
            # Add tags if present
            if tags := data.get("tags"):
                tag_list = [f"`{tag['tag']}`" for tag in tags]
                if tag_list:
                    output.append(f"**Tags:** {' '.join(tag_list)}")
            
            output.append("")  # Empty line between items
        
        return "\n".join(output)
    
    except Exception as e:
        ctx.error(f"Error in advanced search: {str(e)}")
        return f"Error in advanced search: {str(e)}"


@mcp.tool(
    name="zotero_get_annotations",
    description="Get all annotations for a specific item or across your entire Zotero library."
)
def get_annotations(
    item_key: Optional[str] = None,
    use_pdf_extraction: bool = False,
    limit: Optional[Union[int, str]] = None,
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
                                zot.dump(att_key, file_path)
                                
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
    item_key: Optional[str] = None,
    limit: Optional[Union[int, str]] = 20,
    *,
    ctx: Context
) -> str:
    """
    Retrieve notes from your Zotero library.
    
    Args:
        item_key: Optional Zotero item key/ID to filter notes by parent item
        limit: Maximum number of notes to return
        ctx: MCP context
    
    Returns:
        Markdown-formatted list of notes
    """
    try:
        ctx.info(f"Fetching notes{f' for item {item_key}' if item_key else ''}")
        zot = get_zotero_client()
        
        # Prepare search parameters
        params = {"itemType": "note"}
        if item_key:
            params["parentItem"] = item_key
        
        if isinstance(limit, str):
            limit = int(limit)
        
        # Get notes
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
            note_text = note_text.replace("<p>", "").replace("</p>", "\n\n")
            note_text = note_text.replace("<br/>", "\n").replace("<br>", "\n")
            
            # Limit note length for display
            if len(note_text) > 500:
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
    limit: Optional[Union[int, str]] = 20,
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
        annotation_results = get_annotations(
            item_key=None,  # Search all annotations
            use_pdf_extraction=True,
            limit=limit or 20,
            ctx=ctx
        )
        
        # Parse the annotation results to extract annotation items
        # This is a bit hacky and depends on the exact formatting of get_annotations
        # You might want to modify get_annotations to return a more structured result
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
        
        # Format results
        output = [f"# Search Results for '{query}'", ""]
        
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
        
        # Combine and sort results
        all_results = note_results + annotations
        
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
        
        return "\n".join(output) if output else f"No results found for '{query}'"
    
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
    tags: Optional[List[str]] = None,
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
        
        # Prepare the note data
        note_data = {
            "itemType": "note",
            "parentItem": item_key,
            "note": html_content,
            "tags": [{"tag": tag} for tag in (tags or [])]
        }
        
        # Create the note
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
    name="zotero_semantic_search",
    description="Prioritized search tool. Perform semantic search over your Zotero library using AI-powered embeddings."
)
def semantic_search(
    query: str,
    limit: int = 10,
    filters: Optional[Union[Dict[str, str], str]] = None,
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
    limit: Optional[int] = None,
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

def _extract_item_key_from_input(value: str) -> Optional[str]:
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

        result_list: List[Dict[str, str]] = []
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
