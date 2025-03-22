"""
Zotero MCP server implementation.
"""

from typing import Any, Dict, List, Literal, Optional, Union
import uuid
import uuid

from mcp.server.fastmcp import Context, FastMCP

from zotero_mcp.client import (
    AttachmentDetails,
    convert_to_markdown,
    format_item_metadata,
    get_attachment_details,
    get_zotero_client,
)
from zotero_mcp.utils import format_creators

# Create an MCP server with appropriate dependencies
mcp = FastMCP(
    "Zotero",
    dependencies=["pyzotero", "mcp[cli]", "python-dotenv", "markitdown"],
)


@mcp.tool(
    name="zotero_search_items",
    description="Search for items in your Zotero library, given a query string."
)
def search_items(
    query: str,
    qmode: Literal["titleCreatorYear", "everything"] = "titleCreatorYear",
    item_type: str = "-attachment",  # Exclude attachments by default
    limit: Optional[int] = 10,
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
        ctx: MCP context
    
    Returns:
        Markdown-formatted search results
    """
    try:
        if not query.strip():
            return "Error: Search query cannot be empty"
        
        ctx.info(f"Searching Zotero for '{query}'")
        zot = get_zotero_client()
        
        # Search using the query parameters
        zot.add_parameters(q=query, qmode=qmode, itemType=item_type, limit=limit)
        results = zot.items()
        
        if not results:
            return f"No items found matching query: '{query}'"
        
        # Format results as markdown
        output = [f"# Search Results for '{query}'", ""]
        
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
    *,
    ctx: Context
) -> str:
    """
    Get detailed metadata for a Zotero item.
    
    Args:
        item_key: Zotero item key/ID
        include_abstract: Whether to include the abstract in the output
        ctx: MCP context
    
    Returns:
        Markdown-formatted item metadata
    """
    try:
        ctx.info(f"Fetching metadata for item {item_key}")
        zot = get_zotero_client()
        
        item = zot.item(item_key)
        if not item:
            return f"No item found with key: {item_key}"
        
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
    limit: Optional[int] = None,
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
        
        collections = zot.collections(limit=limit)
        if not collections:
            return "No collections found in your Zotero library."
        
        # Format collections as markdown
        output = ["# Zotero Collections", ""]
        
        # First create a mapping of collection IDs to their data
        collection_map = {c["key"]: c for c in collections}
        
        # Then create a mapping of parent to child collections
        hierarchy = {}
        for coll in collections:
            parent_key = coll["data"].get("parentCollection")
            if parent_key not in hierarchy:
                hierarchy[parent_key] = []
            hierarchy[parent_key].append(coll["key"])
        
        # Function to recursively format collections
        def format_collection(key, level=0):
            if key not in collection_map:
                return []
            
            coll = collection_map[key]
            name = coll["data"].get("name", "Unnamed Collection")
            key = coll["key"]
            
            # Create indentation for hierarchy
            indent = "  " * level
            lines = [f"{indent}- **{name}** (Key: {key})"]
            
            # Add children
            child_keys = hierarchy.get(key, [])
            for child_key in child_keys:
                lines.extend(format_collection(child_key, level + 1))
            
            return lines
        
        # Start with top-level collections (those with no parent or empty parent)
        top_level = hierarchy.get("", []) + hierarchy.get(None, [])
        for key in top_level:
            output.extend(format_collection(key))
        
        return "\n".join(output)
    
    except Exception as e:
        ctx.error(f"Error fetching collections: {str(e)}")
        return f"Error fetching collections: {str(e)}"


@mcp.tool(
    name="zotero_get_collection_items",
    description="Get all items in a specific Zotero collection."
)
def get_collection_items(
    collection_key: str,
    limit: Optional[int] = 50,
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
    limit: Optional[int] = None,
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
    limit: int = 10,
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
    add_tags: Optional[List[str]] = None,
    remove_tags: Optional[List[str]] = None,
    limit: int = 50,
    *,
    ctx: Context
) -> str:
    """
    Batch update tags across multiple items matching a search query.
    
    Args:
        query: Search query to find items to update
        add_tags: List of tags to add to matched items
        remove_tags: List of tags to remove from matched items
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
        
        ctx.info(f"Batch updating tags for items matching '{query}'")
        zot = get_zotero_client()
        
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
            if needs_update:
                item["data"]["tags"] = current_tags
                zot.update_item(item)
                updated_count += 1
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
    limit: int = 50,
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
    name="zotero_update_item",
    description="Update fields of existing Zotero items, including titles, authors, abstracts, and other metadata."
)
def update_item(
    item_key: str,
    fields: Dict[str, Any],
    *,
    ctx: Context
) -> str:
    """
    Update fields of an existing Zotero item.
    
    Args:
        item_key: Zotero item key/ID
        fields: Dictionary of fields to update (e.g., {"title": "New Title", "abstractNote": "New abstract"})
        ctx: MCP context
    
    Returns:
        Markdown-formatted summary of the update
    """
    try:
        ctx.info(f"Updating item {item_key}")
        zot = get_zotero_client()
        
        # Get the current item data
        item = zot.item(item_key)
        if not item:
            return f"Error: No item found with key {item_key}"
        
        # Store original values for reporting changes
        original_data = {k: v for k, v in item["data"].items() if k in fields}
        
        # Update the fields
        for field, value in fields.items():
            # Special handling for creators
            if field == "creators":
                # Ensure creators have the correct format
                if isinstance(value, list):
                    # Validate each creator has the required fields
                    for creator in value:
                        if not isinstance(creator, dict):
                            return f"Error: Each creator must be a dictionary with 'creatorType' and name fields"
                        if "creatorType" not in creator:
                            return f"Error: Each creator must have a 'creatorType' field"
                        # Ensure name is provided in the correct format
                        if not any(key in creator for key in ["name", "firstName", "lastName"]):
                            return f"Error: Each creator must have either 'name' or 'firstName'/'lastName' fields"
                    
                    item["data"]["creators"] = value
                else:
                    return f"Error: 'creators' field must be a list of creator dictionaries"
            else:
                # Update regular fields
                item["data"][field] = value
        
        # Submit the update
        updated = zot.update_item(item)
        
        # Check for successful update
        if not updated:
            return f"Error: Failed to update item {item_key}"
        
        # Format response showing before and after
        response = [f"# Item Updated: {item['data'].get('title', 'Untitled')}", ""]
        response.append(f"Item key: {item_key}")
        response.append("")
        response.append("## Changes Made")
        
        for field, new_value in fields.items():
            old_value = original_data.get(field, "N/A")
            
            # Special handling for display of different field types
            if field == "creators":
                response.append(f"### {field}")
                response.append("**Before:**")
                for creator in old_value:
                    if "name" in creator:
                        response.append(f"- {creator.get('creatorType', 'author')}: {creator['name']}")
                    else:
                        response.append(f"- {creator.get('creatorType', 'author')}: {creator.get('lastName', '')}, {creator.get('firstName', '')}")
                
                response.append("**After:**")
                for creator in new_value:
                    if "name" in creator:
                        response.append(f"- {creator.get('creatorType', 'author')}: {creator['name']}")
                    else:
                        response.append(f"- {creator.get('creatorType', 'author')}: {creator.get('lastName', '')}, {creator.get('firstName', '')}")
            elif field == "tags":
                response.append(f"### {field}")
                old_tags = [tag["tag"] for tag in old_value] if isinstance(old_value, list) else []
                new_tags = [tag["tag"] for tag in new_value] if isinstance(new_value, list) else []
                
                response.append(f"**Before:** {', '.join(f'`{tag}`' for tag in old_tags) or 'None'}")
                response.append(f"**After:** {', '.join(f'`{tag}`' for tag in new_tags) or 'None'}")
            else:
                response.append(f"### {field}")
                # Truncate long values for display
                if isinstance(old_value, str) and len(old_value) > 100:
                    old_display = old_value[:100] + "..."
                else:
                    old_display = old_value
                
                if isinstance(new_value, str) and len(new_value) > 100:
                    new_display = new_value[:100] + "..."
                else:
                    new_display = new_value
                
                response.append(f"**Before:** {old_display}")
                response.append(f"**After:** {new_display}")
            
            response.append("")
        
        # Return a link to view the updated item
        response.append("## View Item")
        response.append(f"To see all item details, use `zotero_get_item_metadata` with item key: `{item_key}`")
        
        return "\n".join(response)
    
    except Exception as e:
        ctx.error(f"Error updating item: {str(e)}")
        return f"Error updating item: {str(e)}"


@mcp.tool(
    name="zotero_add_tags",
    description="Add new tags to existing Zotero items, making them easier to organize and find."
)
def add_tags(
    item_key: str,
    tags: List[str],
    *,
    ctx: Context
) -> str:
    """
    Add tags to an existing Zotero item.
    
    Args:
        item_key: Zotero item key/ID
        tags: List of tags to add
        ctx: MCP context
    
    Returns:
        Markdown-formatted summary of the update
    """
    try:
        if not tags:
            return "Error: No tags provided to add"
        
        ctx.info(f"Adding {len(tags)} tags to item {item_key}")
        zot = get_zotero_client()
        
        # Get the current item data
        item = zot.item(item_key)
        if not item:
            return f"Error: No item found with key {item_key}"
        
        # Get the current tags
        current_tags = item["data"].get("tags", [])
        current_tag_values = {tag["tag"] for tag in current_tags}
        
        # Prepare new tags to add (avoid duplicates)
        new_tags_added = []
        for tag in tags:
            if tag and tag not in current_tag_values:
                current_tags.append({"tag": tag})
                new_tags_added.append(tag)
        
        # If no new tags were added, return early
        if not new_tags_added:
            return f"No new tags added. All provided tags already exist on item: {item['data'].get('title', 'Untitled')}"
        
        # Update the item with new tags
        item["data"]["tags"] = current_tags
        updated = zot.update_item(item)
        
        # Check for successful update
        if not updated:
            return f"Error: Failed to update tags for item {item_key}"
        
        # Format response
        response = [f"# Tags Added to: {item['data'].get('title', 'Untitled')}", ""]
        response.append(f"Item key: {item_key}")
        response.append("")
        response.append("## Tags Added")
        for tag in new_tags_added:
            response.append(f"- `{tag}`")
        
        response.append("")
        response.append("## All Tags")
        all_tags = [tag["tag"] for tag in current_tags]
        all_tags.sort()  # Sort tags alphabetically
        for tag in all_tags:
            response.append(f"- `{tag}`")
        
        return "\n".join(response)
    
    except Exception as e:
        ctx.error(f"Error adding tags: {str(e)}")
        return f"Error adding tags: {str(e)}"


@mcp.tool(
    name="zotero_create_collection",
    description="Create new collections (folders) to organize your Zotero library, including support for nested collections."
)
def create_collection(
    name: str,
    parent_collection_key: Optional[str] = None,
    *,
    ctx: Context
) -> str:
    """
    Create a new collection in your Zotero library.
    
    Args:
        name: Name of the new collection
        parent_collection_key: Key of the parent collection (for nested collections)
        ctx: MCP context
    
    Returns:
        Markdown-formatted summary of the created collection
    """
    try:
        if not name.strip():
            return "Error: Collection name cannot be empty"
        
        ctx.info(f"Creating new collection: {name}")
        zot = get_zotero_client()
        
        # Prepare collection data
        collection_data = {"name": name}
        
        # If parent collection is specified, validate it exists
        if parent_collection_key:
            try:
                parent = zot.collection(parent_collection_key)
                parent_name = parent["data"].get("name", "Unknown Collection")
                collection_data["parentCollection"] = parent_collection_key
                ctx.info(f"Adding as sub-collection of: {parent_name}")
            except Exception as parent_error:
                return f"Error: Parent collection not found with key {parent_collection_key}: {str(parent_error)}"
        
        # Create the collection
        result = zot.create_collection(collection_data)
        
        # Check for successful creation
        if not result or not result.get("success"):
            return f"Error: Failed to create collection '{name}'"
        
        # Extract the new collection key
        collection_key = next(iter(result["success"].values()), None)
        if not collection_key:
            return f"Error: Could not retrieve key for created collection '{name}'"
        
        # Format response
        response = ["# Collection Created", ""]
        response.append(f"**Name:** {name}")
        response.append(f"**Collection Key:** {collection_key}")
        
        if parent_collection_key:
            response.append(f"**Parent Collection:** {parent_name} (Key: {parent_collection_key})")
        
        response.append("")
        response.append("## Usage")
        response.append("To add items to this collection, update the item and include this collection key in its 'collections' field.")
        response.append("")
        response.append("To view items in this collection, use `zotero_get_collection_items` with this collection key.")
        
        return "\n".join(response)
    
    except Exception as e:
        ctx.error(f"Error creating collection: {str(e)}")
        return f"Error creating collection: {str(e)}"


@mcp.tool(
    name="zotero_add_note",
    description="Add notes to Zotero items, which is useful for summarizing content, adding comments, or attaching additional information."
)
def add_note(
    item_key: str,
    note_text: str,
    note_title: Optional[str] = None,
    tags: Optional[List[str]] = None,
    *,
    ctx: Context
) -> str:
    """
    Add a note to a Zotero item.
    
    Args:
        item_key: Zotero item key/ID to attach the note to
        note_text: Content of the note (supports HTML formatting)
        note_title: Title of the note (optional)
        tags: List of tags to add to the note (optional)
        ctx: MCP context
    
    Returns:
        Markdown-formatted summary of the created note
    """
    try:
        if not note_text.strip():
            return "Error: Note text cannot be empty"
        
        ctx.info(f"Adding note to item {item_key}")
        zot = get_zotero_client()
        
        # First check if the parent item exists
        parent_item = zot.item(item_key)
        if not parent_item:
            return f"Error: No item found with key {item_key}"
        
        parent_title = parent_item["data"].get("title", "Untitled Item")
        
        # Prepare note data
        note_data = {
            "itemType": "note",
            "parentItem": item_key,
            "note": note_text
        }
        
        # If a title was provided, use it for the note title field
        if note_title:
            note_data["title"] = note_title
        
        # If tags were provided, format them correctly
        if tags:
            note_data["tags"] = [{"tag": tag} for tag in tags if tag]
        
        # Create the note
        result = zot.create_items([note_data])
        
        # Check for successful creation
        if not result or not result.get("success"):
            return f"Error: Failed to create note for item '{parent_title}'"
        
        # Extract the new note key
        note_keys = list(result["success"].keys())
        if not note_keys:
            return f"Error: Could not retrieve key for created note"
        
        note_key = note_keys[0]
        
        # Format response
        response = ["# Note Added", ""]
        response.append(f"**Parent Item:** {parent_title}")
        response.append(f"**Parent Key:** {item_key}")
        response.append(f"**Note Key:** {note_key}")
        
        if note_title:
            response.append(f"**Note Title:** {note_title}")
        
        if tags:
            response.append(f"**Tags:** {', '.join(f'`{tag}`' for tag in tags)}")
        
        # Add a preview of the note content
        response.append("")
        response.append("## Note Preview")
        
        # Convert HTML to plaintext for preview
        preview_text = note_text
        preview_text = preview_text.replace("<p>", "").replace("</p>", "\n\n")
        preview_text = preview_text.replace("<br/>", "\n").replace("<br>", "\n")
        
        # Limit preview length
        if len(preview_text) > 500:
            preview_text = preview_text[:500] + "...\n\n(Note truncated)"
        
        response.append(f"```\n{preview_text}\n```")
        
        return "\n".join(response)
    
    except Exception as e:
        ctx.error(f"Error adding note: {str(e)}")
        return f"Error adding note: {str(e)}"
