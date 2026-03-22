"""Write / mutation tool functions for the Zotero MCP server."""

from typing import Literal
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET

import requests
from fastmcp import Context

from zotero_mcp._app import mcp
from zotero_mcp import client as _client
from zotero_mcp import utils as _utils
from zotero_mcp.tools import _helpers

# Accessed as _helpers.X so that monkeypatch/mock on the module attribute works.
CROSSREF_TYPE_MAP = _helpers.CROSSREF_TYPE_MAP


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
    name="zotero_search_collections",
    description="Search for collections by name to find their keys."
)
def search_collections(
    query: str,
    *,
    ctx: Context
) -> str:
    try:
        zot = _client.get_zotero_client()
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
        read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        keys = _helpers._normalize_str_list_input(item_keys, "item_keys")
        add_colls = _helpers._normalize_str_list_input(add_to, "add_to")
        remove_colls = _helpers._normalize_str_list_input(remove_from, "remove_from")

        if not keys:
            return "Error: No item keys provided."
        if not add_colls and not remove_colls:
            return "Error: Must specify add_to and/or remove_from."

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
    description="Add a paper to your Zotero library by DOI. Fetches metadata from CrossRef."
)
def add_by_doi(
    doi: str,
    collections: list[str] | str | None = None,
    tags: list[str] | str | None = None,
    attach_mode: str = "auto",
    *,
    ctx: Context
) -> str:
    try:
        read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        normalized = _helpers._normalize_doi(doi)
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

        # Collections
        coll_keys = _helpers._normalize_str_list_input(collections, "collections")
        if coll_keys:
            item_data["collections"] = coll_keys

        # Create item
        result = write_zot.create_items([item_data])

        if isinstance(result, dict) and result.get("success"):
            item_key = next(iter(result["success"].values()))
            title = item_data.get("title", normalized)

            # Attempt open-access PDF attachment (pass CrossRef metadata for arXiv fallback)
            pdf_status = _helpers._try_attach_oa_pdf(write_zot, item_key, normalized, ctx,
                                            crossref_metadata=cr,
                                            attach_mode=attach_mode)

            return (
                f"Successfully added: **{title}**\n\n"
                f"Item key: `{item_key}`\n"
                f"Type: {zot_type}\n"
                f"DOI: {normalized}\n"
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
    description="Add a paper by URL. Supports DOI URLs, arXiv URLs, and general web pages."
)
def add_by_url(
    url: str,
    collections: list[str] | str | None = None,
    tags: list[str] | str | None = None,
    attach_mode: str = "auto",
    *,
    ctx: Context
) -> str:
    try:
        read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        url = (url or "").strip()
        if not url:
            return "Error: No URL provided."

        # DOI URL routing
        doi = _helpers._normalize_doi(url)
        if doi:
            return add_by_doi(doi=url, collections=collections, tags=tags,
                              attach_mode=attach_mode, ctx=ctx)

        # arXiv URL routing
        arxiv_id = _helpers._normalize_arxiv_id(url)
        if arxiv_id:
            return _add_by_arxiv(arxiv_id, collections, tags, write_zot, ctx)

        # Generic webpage
        ctx.info(f"Creating webpage item for: {url}")
        template = write_zot.item_template("webpage")
        template["url"] = url
        template["title"] = url
        template["accessDate"] = ""

        tag_list = _helpers._normalize_str_list_input(tags, "tags")
        if tag_list:
            template["tags"] = [{"tag": t} for t in tag_list]
        coll_keys = _helpers._normalize_str_list_input(collections, "collections")
        if coll_keys:
            template["collections"] = coll_keys

        result = write_zot.create_items([template])
        if isinstance(result, dict) and result.get("success"):
            item_key = next(iter(result["success"].values()))
            return (
                f"Created webpage item for: {url}\n\nItem key: `{item_key}`\n\n"
                "_Note: To include this item in semantic search, run "
                "zotero_update_search_database._"
            )
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

    tag_list = _helpers._normalize_str_list_input(tags, "tags")
    if tag_list:
        template["tags"] = [{"tag": t} for t in tag_list]
    coll_keys = _helpers._normalize_str_list_input(collections, "collections")
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
            f"PDF: {pdf_status}\n\n"
            "_Note: To include this item in semantic search, run "
            "zotero_update_search_database._"
        )
    return f"Failed to create arXiv item: {result}"


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

        # Collections — both params ADD to existing collections (never replace)
        if collections is not None:
            coll_keys = _helpers._normalize_str_list_input(collections, "collections")
            existing_colls = set(data.get("collections", []))
            existing_colls.update(coll_keys)
            data["collections"] = list(existing_colls)
            changes.append(f"- **collections**: added {coll_keys}")
        if collection_names is not None:
            names = _helpers._normalize_str_list_input(collection_names, "collection_names")
            resolved = _helpers._resolve_collection_names(read_zot, names, ctx=ctx)
            existing_colls = set(data.get("collections", []))
            existing_colls.update(resolved)
            data["collections"] = list(existing_colls)
            changes.append(f"- **collections**: added {resolved}")

        if not changes:
            return "No changes to apply."

        resp = write_zot.update_item(item)
        if _helpers._handle_write_response(resp, ctx):
            return f"Successfully updated item `{item_key}`:\n\n" + "\n".join(changes)
        return f"Failed to update item: write operation returned failure"

    except ValueError as e:
        return f"Input error: {e}"
    except Exception as e:
        ctx.error(f"Error updating item: {e}")
        return f"Error updating item: {e}"


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
        zot = _client.get_zotero_client()
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
        read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        dup_keys = _helpers._normalize_str_list_input(duplicate_keys, "duplicate_keys")

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
            if not _helpers._handle_write_response(resp, ctx):
                return "Error: Failed to merge tags into keeper."
            keeper = write_zot.item(keeper_key)  # re-fetch for version

        # Step 4: Consolidate collections
        for coll_key in new_collections:
            resp = write_zot.addto_collection(coll_key, keeper)
            if not _helpers._handle_write_response(resp, ctx):
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
        read_zot, write_zot = _helpers._get_write_client(ctx)
    except ValueError as e:
        return str(e)

    try:
        # Path validation — check symlink BEFORE resolving
        if os.path.islink(file_path):
            return "Error: Symlinks are not allowed for security reasons."
        if not os.path.isabs(file_path):
            return "Error: file_path must be an absolute path."
        # Resolve ".." components after symlink check
        file_path = os.path.realpath(file_path)
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

            tag_list = _helpers._normalize_str_list_input(tags, "tags")
            if tag_list:
                template["tags"] = [{"tag": t} for t in tag_list]
            coll_keys = _helpers._normalize_str_list_input(collections, "collections")
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
            f"{attach_info}\n\n"
            "_Note: To include this item in semantic search, run "
            "zotero_update_search_database._"
        )

    except Exception as e:
        ctx.error(f"Error adding from file: {e}")
        return f"Error adding from file: {e}"
