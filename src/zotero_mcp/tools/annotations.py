"""Annotation and note tool functions for the Zotero MCP server."""

import json
import os
import shutil
import tempfile
import uuid
from typing import Literal

import requests

from zotero_mcp import client as _client
from zotero_mcp import library as _library
from zotero_mcp import utils as _utils
from zotero_mcp._app import mcp
from zotero_mcp._context import Context
from zotero_mcp.better_bibtex_client import get_color_category
from zotero_mcp.client import with_zotero_api_lock
from zotero_mcp.tools import _helpers

_WEB_API_ENV_VARS = (
    "- ZOTERO_API_KEY: Your Zotero API key (from zotero.org/settings/keys)\n"
    "- ZOTERO_LIBRARY_ID: Your library ID\n"
    "- ZOTERO_LIBRARY_TYPE: 'user' or 'group'"
)


def _page_index(data: dict) -> int | None:
    """Return the 0-based page index of an annotation, whatever its source.

    ``_pdf_page`` (Better BibTeX, direct PDF extraction) is a 1-based page
    number, while the Zotero API's ``annotationPosition.pageIndex`` is already
    0-based. Both are normalized to 0-based here so the field means the same
    thing no matter which source produced the annotation.
    """
    pdf_page = data.get("_pdf_page")
    if pdf_page is not None:
        try:
            return max(int(pdf_page) - 1, 0)
        except (TypeError, ValueError):
            return None

    position = data.get("annotationPosition")
    if isinstance(position, str):
        try:
            position = json.loads(position)
        except json.JSONDecodeError:
            return None
    if isinstance(position, dict):
        index = position.get("pageIndex")
        return index if isinstance(index, int) else None
    return None


def _annotation_to_record(
    annotation: dict,
    parent_context: dict[str, str | None] | None = None,
) -> dict:
    """Normalize an annotation from any supported source into a JSON record."""
    data = annotation.get("data", {})
    context = parent_context or {}

    # page is always the human-facing label string, page_index always the
    # 0-based number — downstream tooling can rely on both types.
    page_index = _page_index(data)
    page = data.get("_pageLabel") or data.get("annotationPageLabel") or None
    if page is None and page_index is not None:
        page = str(page_index + 1)

    color = data.get("annotationColor") or ""

    if data.get("_from_better_bibtex"):
        source = "better_bibtex"
    elif data.get("_from_pdf_extraction"):
        source = "pdf_extraction"
    else:
        source = "zotero"

    tags = [
        tag["tag"]
        for tag in _helpers._normalize_item_tags(data.get("tags"))
    ]

    return {
        "annotation_key": annotation.get("key") or None,
        "item_key": context.get("item_key"),
        "attachment_key": context.get("attachment_key"),
        "parent_title": context.get("parent_title"),
        "attachment_title": (
            context.get("attachment_title")
            or data.get("_attachment_title")
            or None
        ),
        "type": data.get("annotationType") or None,
        "page": page,
        "page_index": page_index,
        "text": data.get("annotationText") or "",
        "comment": data.get("annotationComment") or "",
        "color": color or None,
        # Only the Better BibTeX path precomputes this; derive it for the
        # other sources so the field isn't null for most records.
        "color_category": data.get("_color_category") or get_color_category(color) or None,
        "tags": tags,
        "created": data.get("dateAdded") or None,
        "modified": data.get("dateModified") or None,
        "source": source,
    }


def _download_attachment_for_processing(
    attachment_key: str,
    filename: str,
    tmpdir: str,
    *,
    local_client,
    web_client,
    ctx: Context,
) -> tuple[str | None, str | None]:
    """Download an attachment for PDF/EPUB processing and return (path, error)."""
    download = _client.download_attachment_file(
        attachment_key,
        tmpdir,
        filename,
        local_client=local_client,
        web_client=web_client,
    )
    if download.path and download.path.exists():
        ctx.info(f"Attachment downloaded via {download.source}")
        return str(download.path), None

    error_details = "\n".join(f"  - {err}" for err in download.errors) or "  - No download source succeeded"
    return (
        None,
        "Error: Could not download attachment.\n\n"
        f"Attempted sources:\n{error_details}\n\n"
        "Possible solutions:\n"
        "- **Zotero Cloud Storage**: Ensure file syncing is enabled in Zotero preferences\n"
        "- **WebDAV Storage**: Configure ZOTERO_WEBDAV_URL, ZOTERO_WEBDAV_USERNAME, and "
        "ZOTERO_WEBDAV_PASSWORD in the MCP env file, or run local Zotero with remote access enabled\n"
        "- **Linked files**: Linked attachments are readable only from the machine holding "
        "the file — set ZOTERO_LOCAL=true so they can be resolved from local Zotero storage"
    )


def _get_note_write_client(op_description: str):
    """Return (client, None) or (None, error_msg) for note-write operations.

    Zotero's local API is read-only, so in local mode this falls back to the
    web client and propagates any active library override.
    """
    if _utils.is_local_mode():
        zot = _client.get_web_zotero_client()
        if zot is None:
            return None, (
                f"Error: Web API credentials required for {op_description}.\n\n"
                "Please configure the following environment variables:\n"
                + _WEB_API_ENV_VARS
            )
        _helpers.apply_library_override(zot, _client.get_active_library())
    else:
        zot = _client.get_zotero_client()
    return zot, None


@mcp.tool(
    name="zotero_get_annotations",
    description=(
        "Get annotations (highlights and attached notes on PDF/EPUB "
        "attachments) for a specific item or across the active Zotero "
        "library. item_key: pass the parent item key OR an attachment key — "
        "both work; attachment-to-parent resolution is automatic. ALWAYS "
        "pass item_key when you know which item you want; calling without "
        "it returns every annotation in the library (potentially thousands). "
        "use_pdf_extraction=True falls back to direct PDF parsing when the "
        "Zotero API has no stored annotation record — useful for "
        "annotations made outside Zotero desktop. limit: cap on annotations "
        "returned; None (default) returns all. format='markdown' (default) "
        "returns a readable list; format='json' returns normalized records "
        "with stable keys for downstream scripts and other MCP tools. "
        "Uses Better BibTeX when Zotero desktop is running locally, "
        "otherwise the Zotero web API. "
        "Example: zotero_get_annotations(item_key='ABC12345') → every "
        "highlight/note on that paper."
    )
)
@with_zotero_api_lock
def get_annotations(
    item_key: str | None = None,
    use_pdf_extraction: bool = False,
    limit: int | str | None = None,
    format: Literal["markdown", "json"] = "markdown",
    *,
    ctx: Context
) -> str:
    """
    Get annotations from your Zotero library.

    Args:
        item_key: Optional Zotero item key/ID to filter annotations by parent item
        use_pdf_extraction: Whether to attempt direct PDF extraction as a fallback
        limit: Maximum number of annotations to return
        format: ``markdown`` for human-readable output or ``json`` for
            normalized structured records.
        ctx: MCP context

    Returns:
        Markdown-formatted annotations or a JSON array of normalized records.
    """
    try:
        backend = _library.get_library_backend()

        # The pyzotero client is only needed by the paths below that have no
        # local equivalent (library-wide listing, PDF download). Built on
        # demand so a read the backend can serve never constructs one — with
        # Zotero closed, constructing it eagerly failed the whole tool.
        _zot_cache = []

        def _api_client():
            if not _zot_cache:
                _zot_cache.append(_client.get_zotero_client())
            return _zot_cache[0]

        # Prepare annotations list
        annotations = []
        parent_title = "Untitled Item"

        # If an item key is provided, use specialized retrieval
        if item_key:
            # First, verify the item exists and get its details
            parent = backend.get_item(item_key)
            if parent is None:
                return f"Error: No item found with key: {item_key}"
            parent_title = parent["data"].get("title", "Untitled Item")
            ctx.info(f"Fetching annotations for item: {parent_title}")

            # Determine whether item_key is an attachment or a parent item.
            # In the Zotero API, annotations are children of attachments, not of
            # the parent item (two-hop: parent → attachment → annotation).
            # We need to know this for both the API annotations path and the PDF fallback.
            _item_data = parent.get("data", {})
            _is_attachment = _item_data.get("itemType") == "attachment"

            # parent_item_key is used by the PDF fallback to find PDF attachments.
            # If the caller passed an attachment key, resolve up to the parent item.
            parent_item_key = (
                _item_data.get("parentItem", item_key)
                if _is_attachment
                else item_key
            )

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
                            ctx.warning(f"Error extracting citation key from Extra field: {e}")

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
                                ctx.warning(f"Error searching for citation key: {e}")

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
                                                    "parentItem": (
                                                        attachment.get("itemKey")
                                                        or attachment.get("key")
                                                        or item_key
                                                    ),
                                                    # Better BibTeX hands tags back as
                                                    # bare strings; normalize to Zotero's
                                                    # {"tag": ...} shape so they render (#377).
                                                    "tags": _helpers._normalize_item_tags(
                                                        processed.get("tags") or anno.get("tags")
                                                    ),
                                                    # Carry timestamps through so
                                                    # JSON records aren't null-dated
                                                    # on the local BBT path.
                                                    "dateAdded": anno.get("dateAdded", ""),
                                                    "dateModified": anno.get("dateModified", ""),
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
                                ctx.warning(f"Error processing Better BibTeX annotations: {e}")
                except Exception as bibtex_error:
                    ctx.warning(f"Error initializing Better BibTeX: {bibtex_error}")

            # Fallback to Zotero API annotations.
            #
            # In Zotero's data model annotations are children of the PDF
            # attachment, not the parent paper. So if item_key points to a
            # paper we need to descend through its attachment children.
            # If item_key is itself an attachment, its annotations are
            # returned directly. We do both and dedupe by key for safety.
            if not better_bibtex_annotations:
                try:
                    if _is_attachment:
                        # item_key is already a PDF attachment -- annotations
                        # are its direct children.
                        zotero_api_annotations = backend.get_children(
                            [item_key], item_type="annotation"
                        ).get(item_key, [])
                    else:
                        # item_key is a parent item. Annotations live under
                        # attachments (parent -> attachment -> annotation).
                        # Only PDF, EPUB, and snapshot attachments carry annotations.
                        _annotatable = {"application/pdf", "application/epub+zip", "text/html"}
                        all_children = backend.get_children([item_key]).get(item_key, [])
                        att_keys = [
                            c["key"] for c in all_children
                            if c.get("data", {}).get("itemType") == "attachment"
                            and c.get("data", {}).get("contentType") in _annotatable
                        ]
                        # One call for every attachment, not one per
                        # attachment — the second hop used to be N+1.
                        seen = set()
                        for children in backend.get_children(
                            att_keys, item_type="annotation"
                        ).values():
                            for a in children:
                                k = a.get("key")
                                if k and k not in seen:
                                    seen.add(k)
                                    zotero_api_annotations.append(a)
                    ctx.info(f"Retrieved {len(zotero_api_annotations)} annotations via Zotero API")
                except Exception as api_error:
                    ctx.warning(f"Error retrieving Zotero API annotations: {api_error}")

            # PDF Extraction fallback
            if use_pdf_extraction and not (better_bibtex_annotations or zotero_api_annotations):
                try:
                    from zotero_mcp.pdfannots_helper import extract_annotations_from_pdf, ensure_pdfannots_installed

                    # Ensure PDF annotation tool is installed
                    if ensure_pdfannots_installed():
                        # Get PDF attachments via the resolved parent key
                        children = backend.get_children([parent_item_key]).get(parent_item_key, [])
                        pdf_attachments = [
                            item for item in children
                            if item.get("data", {}).get("contentType") == "application/pdf"
                        ]

                        # Extract annotations from PDFs
                        for attachment in pdf_attachments:
                            with tempfile.TemporaryDirectory() as tmpdir:
                                att_key = attachment.get("key", "")
                                file_path = os.path.join(tmpdir, f"{att_key}.pdf")
                                local_pdf = _library.attachment_path_for(att_key)
                                if local_pdf is not None:
                                    shutil.copyfile(local_pdf, file_path)
                                else:
                                    _api_client().dump(
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
                                                "parentItem": att_key,
                                                # pdfannots2json only emits tags for some
                                                # annotation kinds; carry them when present (#377).
                                                "tags": _helpers._normalize_item_tags(ext.get("tags")),
                                                "dateModified": ext.get("date", ""),
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
                    ctx.warning(f"Error during PDF annotation extraction: {pdf_error}")

            # Combine annotations from all sources
            annotations = better_bibtex_annotations + zotero_api_annotations + pdf_annotations

        else:
            # Retrieve all annotations in the library. No backend operation
            # covers an unfiltered library-wide annotation listing, so this
            # stays on the API client.
            limit = _helpers._normalize_limit(limit, default=100)
            # Use _paginate helper instead of inline manual pagination
            annotations = _helpers._paginate(
                _api_client().items, max_items=limit, itemType="annotation"
            )

        # Handle no annotations found
        if not annotations:
            if format == "json":
                return "[]"
            return f"No annotations found{f' for item: {parent_title}' if item_key else ''}."

        # Structured output and library-wide Markdown need attachment -> paper
        # context. Item-scoped Markdown keeps its existing no-extra-lookup path.
        parent_keys = set()
        if format == "json" or not item_key:
            parent_keys = {
                parent_key
                for anno in annotations
                if (parent_key := anno.get("data", {}).get("parentItem"))
            }
        parent_contexts = (
            _batch_resolve_annotation_contexts(backend, parent_keys, ctx)
            if parent_keys
            else {}
        )
        parent_titles = {
            key: context["parent_title"] or f"(parent key: {key})"
            for key, context in parent_contexts.items()
        }

        if format == "json":
            def _fallback_context(parent_key: str | None) -> dict[str, str | None]:
                """Context for an attachment we couldn't re-resolve.

                Falls back to what the caller already told us rather than
                passing the attachment key off as the paper key.
                """
                return {
                    "item_key": parent_item_key if item_key else None,
                    "attachment_key": parent_key,
                    "parent_title": parent_title if item_key else None,
                    "attachment_title": None,
                }

            records = []
            for anno in annotations:
                parent_key = anno.get("data", {}).get("parentItem")
                context = (
                    parent_contexts.get(parent_key) or _fallback_context(parent_key)
                )
                records.append(_annotation_to_record(anno, context))
            return json.dumps(records, ensure_ascii=False, indent=2)

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
                resolved_title = parent_titles.get(parent_key, f"(parent key: {parent_key})")
                parent_info = f" (from \"{resolved_title}\")"

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
            elif data.get("annotationPageLabel"):
                page_label = data["annotationPageLabel"]
                page_index = None
                position_raw = data.get("annotationPosition", "")
                if position_raw:
                    try:
                        position = json.loads(position_raw) if isinstance(position_raw, str) else position_raw
                        if "pageIndex" in position:
                            page_index = position["pageIndex"]
                    except (json.JSONDecodeError, TypeError):
                        pass
                if page_index is not None:
                    output.append(f"**Page:** {page_label} (index: {page_index})")
                else:
                    output.append(f"**Page:** {page_label}")

            # Annotation content
            if anno_text:
                output.append(f"**Text:** {anno_text}")

            if anno_comment:
                output.append(f"**Comment:** {anno_comment}")

            # Image annotation
            if "_image_path" in data and os.path.exists(data["_image_path"]):
                output.append("**Image:** This annotation includes an image (not displayed in this interface)")

            # Tags (normalized so a non-web-API source can't KeyError here)
            if tags := _helpers._normalize_item_tags(data.get("tags")):
                tag_list = [f"`{t['tag']}`" for t in tags]
                output.append(f"**Tags:** {' '.join(tag_list)}")

            output.append("")  # Empty line between annotations

        result = "\n".join(output)
        # Warn about large responses for library-wide queries
        if not item_key:
            result = _helpers._prepend_size_warning(
                result,
                "Pass item_key to get annotations for a specific item instead of library-wide."
            )
        return result

    except Exception as e:
        ctx.error(f"Error fetching annotations: {str(e)}")
        return f"Error fetching annotations: {str(e)}"


# Alias kept for backward compatibility (imported by server.py).
_get_annotations = get_annotations


@with_zotero_api_lock
def get_notes(
    item_key: str | None = None,
    limit: int | str | None = 20,
    truncate: bool = True,
    raw_html: bool = False,
    *,
    ctx: Context
) -> str:
    """
    Retrieve notes from your Zotero library.

    Args:
        item_key: Optional Zotero item key/ID to filter notes by parent item
        limit: Maximum number of notes to return
        truncate: Whether to truncate long notes for display
        raw_html: If True, return the note's raw HTML instead of stripped text.
            Useful for fetching exact content to pass back for an update.
        ctx: MCP context

    Returns:
        Markdown-formatted list of notes
    """
    try:
        ctx.info(f"Fetching notes{f' for item {item_key}' if item_key else ''}")
        backend = _library.get_library_backend()

        limit = _helpers._normalize_limit(limit, default=20)

        notes = []
        if item_key:
            notes = backend.get_children([item_key], item_type="note").get(item_key, [])[:limit]
        else:
            zot = _client.get_zotero_client()
            notes = _helpers._paginate(zot.items, max_items=limit, itemType="note")

        if not notes and item_key:
            # `item_key` may name a note itself rather than its parent. A note
            # has no children, so the query above finds nothing and the tool
            # reported "No notes found" for a note that plainly exists —
            # leaving no way to read one whose key you already hold (#447).
            candidate = backend.get_item(item_key)
            if candidate and candidate.get("data", {}).get("itemType") == "note":
                notes = [candidate]

        if not notes:
            return f"No notes found{f' for item {item_key}' if item_key else ''}."

        # Batch-resolve parent titles (Fix 2)
        note_parent_titles = {}
        note_parent_keys = set()
        for note in notes:
            pk = note.get("data", {}).get("parentItem")
            if pk:
                note_parent_keys.add(pk)
        if note_parent_keys:
            note_parent_titles = _batch_resolve_parent_titles(backend, note_parent_keys, ctx)

        # Generate markdown output
        output = [f"# Notes{f' for Item: {item_key}' if item_key else ''}", ""]

        for i, note in enumerate(notes, 1):
            data = note.get("data", {})
            note_key = note.get("key", "")

            # Parent item context
            parent_info = ""
            if parent_key := data.get("parentItem"):
                resolved_title = note_parent_titles.get(parent_key, f"(parent key: {parent_key})")
                parent_info = f" (from \"{resolved_title}\")"

            # Prepare note text
            note_text = data.get("note", "")

            if not raw_html:
                note_text = _utils.clean_html(note_text)

            # Limit note length for display
            if truncate and len(note_text) > 500:
                note_text = note_text[:500] + "..."

            # Build markdown entry
            output.append(f"## Note {i}{parent_info}")
            output.append(f"**Key:** {note_key}")

            # Tags
            if tags := data.get("tags"):
                tag_list = [f"`{t['tag']}`" for t in tags]
                if tag_list:
                    output.append(f"**Tags:** {' '.join(tag_list)}")

            output.append(f"**Content:**\n{note_text}")
            output.append("")  # Empty line between notes

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error fetching notes: {str(e)}")
        return f"Error fetching notes: {str(e)}"


# ---------------------------------------------------------------------------
# Helpers for search_notes
# ---------------------------------------------------------------------------

@with_zotero_api_lock
def _batch_resolve_parent_titles(
    backend, parent_keys: set[str], ctx: Context
) -> dict[str, str]:
    """Parent item titles for a set of keys, resolved in one batch.

    The backend does its own batching (the API one still chunks at the
    Zotero itemKey limit, the SQLite one answers in a single query), so
    there is no chunk loop or per-key fallback left here.
    """
    keys_list = list(parent_keys)
    try:
        items = backend.get_items(keys_list)
    except Exception as e:
        ctx.warning(f"Batch parent lookup failed: {e}")
        items = {}
    return {
        key: items[key].get("data", {}).get("title", "Untitled")
        if key in items else f"(parent key: {key})"
        for key in keys_list
    }


@with_zotero_api_lock
def _batch_resolve_annotation_contexts(
    backend, parent_keys: set[str], ctx: Context
) -> dict[str, dict[str, str | None]]:
    """Resolve annotation parent keys to paper and attachment metadata.

    Annotations are children of PDF attachments, which are children of papers.
    This does a two-hop lookup: annotation → attachment → paper. Parents that
    aren't attachments degrade to a one-hop context; parents that couldn't be
    fetched at all are omitted so the caller can supply its own fallback.

    Two backend calls total, one per hop. The chunk loops and per-key
    fallbacks this replaced belonged to the API's 50-key itemKey limit,
    which the backend now handles on its own side.
    """
    try:
        attachment_data = backend.get_items(list(parent_keys))
    except Exception as e:
        ctx.info(f"Batch attachment lookup failed: {e}")
        attachment_data = {}

    grandparent_keys = {
        gp_key
        for item in attachment_data.values()
        if item.get("data", {}).get("itemType") == "attachment"
        and (gp_key := item.get("data", {}).get("parentItem"))
    }

    grandparent_titles: dict[str, str] = {}
    if grandparent_keys:
        try:
            grandparent_titles = {
                key: item.get("data", {}).get("title", "Untitled")
                for key, item in backend.get_items(list(grandparent_keys)).items()
            }
        except Exception as e:
            ctx.info(f"Batch grandparent lookup failed: {e}")

    # Step 3: Map immediate parent keys to stable paper/attachment context.
    # Keys we couldn't fetch are left out so callers can apply their own
    # fallback instead of receiving a placeholder that looks resolved.
    result: dict[str, dict[str, str | None]] = {}
    for parent_key, parent_item in attachment_data.items():
        data = parent_item.get("data", {})
        is_attachment = data.get("itemType") == "attachment"
        gp_key = data.get("parentItem") if is_attachment else None
        if gp_key:
            result[parent_key] = {
                "item_key": gp_key,
                "attachment_key": parent_key,
                "parent_title": (
                    grandparent_titles.get(gp_key)
                    or data.get("title")
                    or f"(key: {gp_key})"
                ),
                "attachment_title": data.get("title") or None,
            }
        else:
            result[parent_key] = {
                "item_key": parent_key,
                "attachment_key": parent_key if is_attachment else None,
                "parent_title": data.get("title") or f"(key: {parent_key})",
                "attachment_title": data.get("title") if is_attachment else None,
            }

    return result


@with_zotero_api_lock
def _batch_resolve_grandparent_titles(
    backend, parent_keys: set[str], ctx: Context
) -> dict[str, str]:
    """Backward-compatible title-only view of annotation parent contexts."""
    contexts = _batch_resolve_annotation_contexts(backend, parent_keys, ctx)
    return {
        key: context["parent_title"] or f"(parent key: {key})"
        for key, context in contexts.items()
    }


@with_zotero_api_lock
def _format_search_results(
    query: str,
    note_results: list[dict],
    annotation_results: list[dict],
    raw_html: bool = False,
) -> str:
    """Format note and annotation search results as consistent markdown."""
    all_results = note_results + annotation_results
    if not all_results:
        return f"No results found for '{query}'"

    output = [f"# Search Results for '{query}'", ""]
    for i, result in enumerate(all_results, 1):
        parent_title = result.get("parent_title")
        parent_info = f' (from "{parent_title}")' if parent_title else ""
        key = result.get("key", "")

        if result.get("type") == "note":
            note_html = result.get("text", "")
            if raw_html:
                # Contextual windowing requires plain-text positions, so in
                # raw mode just emit the full HTML (optionally head-truncated).
                note_text = note_html if len(note_html) <= 2000 else note_html[:2000] + "..."
            else:
                note_text = _utils.clean_html(note_html)
                # Show context around match
                pos = note_text.lower().find(query.lower())
                if pos >= 0:
                    start = max(0, pos - 100)
                    end = min(len(note_text), pos + len(query) + 200)
                    note_text = note_text[start:end] + "..."
                else:
                    note_text = note_text[:500] + "..."

            output.append(f"## Note {i}{parent_info}")
            output.append(f"**Key:** {key}")
            if tags := result.get("tags"):
                tag_list = [f"`{t}`" for t in tags]
                output.append(f"**Tags:** {' '.join(tag_list)}")
            output.append(f"**Content:**\n{note_text}")
            output.append("")

        elif result.get("type") == "annotation":
            anno_type = result.get("annotation_type", "highlight")
            anno_text = result.get("text", "")
            anno_comment = result.get("comment", "")
            page_label = result.get("page_label")
            output.append(f"## Annotation {i}{parent_info}")
            output.append(f"**Type:** {anno_type}")
            output.append(f"**Key:** {key}")
            if page_label:
                output.append(f"**Page:** {page_label}")
            if anno_text:
                output.append(f"**Text:** {anno_text}")
            if anno_comment:
                output.append(f"**Comment:** {anno_comment}")
            output.append("")

    return "\n".join(output)


@with_zotero_api_lock
def search_notes(
    query: str,
    limit: int | str | None = 20,
    raw_html: bool = False,
    *,
    ctx: Context
) -> str:
    """
    Search for notes and annotations in your Zotero library.

    Args:
        query: Search query string
        limit: Maximum number of results to return
        raw_html: If True, return matching notes as raw HTML instead of
            stripped text. Query matching still uses stripped text.
        ctx: MCP context

    Returns:
        Markdown-formatted search results
    """
    if not query or not query.strip():
        return "Error: Search query cannot be empty"

    ctx.info(f"Searching Zotero notes for '{query}'")

    limit = _helpers._normalize_limit(limit, default=20)

    note_results: list[dict] = []
    annotation_results: list[dict] = []

    # ---------- Local mode: fast SQLite queries ----------
    if _utils.is_local_mode():
        try:
            from zotero_mcp.local_db import get_local_zotero_reader
            reader = get_local_zotero_reader()
            if reader:
                try:
                    note_results = reader.search_notes_local(query, limit)
                    ctx.info(f"Local note search: {len(note_results)} results")
                except Exception as e:
                    ctx.warning(f"Local note search failed: {e}")

                try:
                    annotation_results = reader.search_annotations_local(query, limit)
                    ctx.info(f"Local annotation search: {len(annotation_results)} results")
                except Exception as e:
                    ctx.warning(f"Local annotation search failed: {e}")
                finally:
                    reader.close()

                return _format_search_results(query, note_results, annotation_results, raw_html=raw_html)
        except Exception as e:
            ctx.warning(f"Local search unavailable, falling back to API: {e}")

    # ---------- API mode: separate try/except blocks ----------
    zot = _client.get_zotero_client()
    # The batch resolvers below speak the backend port, not raw pyzotero, so
    # wrap this client rather than reaching past the port with it.
    api_backend = _library.ApiBackend(zot)

    # Notes — always try (this works since upstream PR #136)
    try:
        zot.add_parameters(q=query, qmode="everything", itemType="note", limit=limit)
        notes = zot.items()

        # Batch-resolve parent titles
        parent_keys = {n.get("data", {}).get("parentItem") for n in notes
                       if n.get("data", {}).get("parentItem")}
        parent_titles = _batch_resolve_parent_titles(api_backend, parent_keys, ctx) if parent_keys else {}

        query_lower = query.lower()
        for note in notes:
            data = note.get("data", {})
            note_html = data.get("note", "")
            clean_text = _utils.clean_html(note_html)
            if query_lower not in clean_text.lower():
                continue

            parent_key = data.get("parentItem")
            tags = [t["tag"] for t in data.get("tags", [])]
            note_results.append({
                "type": "note",
                "key": note.get("key", ""),
                "text": note_html,
                "tags": tags,
                "parent_key": parent_key,
                "parent_title": parent_titles.get(parent_key) if parent_key else None,
            })
        ctx.info(f"API note search: {len(note_results)} results")
    except Exception as e:
        ctx.warning(f"Note search failed: {e}")

    # Annotations — separate block so note results survive if this crashes
    try:
        # Use _paginate helper for consistent pagination
        annotations = _helpers._paginate(zot.items, max_items=limit, itemType="annotation")

        # Batch-resolve parent titles for annotations
        anno_parent_keys = set()
        for anno in annotations:
            pk = anno.get("data", {}).get("parentItem")
            if pk:
                anno_parent_keys.add(pk)
        anno_parent_titles = _batch_resolve_grandparent_titles(api_backend, anno_parent_keys, ctx) if anno_parent_keys else {}

        query_lower = query.lower()
        for anno in annotations:
            data = anno.get("data", {})
            anno_text = data.get("annotationText", "")
            anno_comment = data.get("annotationComment", "")
            if query_lower not in (anno_text + " " + anno_comment).lower():
                continue

            parent_key = data.get("parentItem")
            annotation_results.append({
                "type": "annotation",
                "key": anno.get("key", ""),
                "text": anno_text,
                "comment": anno_comment,
                "annotation_type": data.get("annotationType", "highlight"),
                "page_label": data.get("annotationPageLabel"),
                "parent_key": parent_key,
                "parent_title": anno_parent_titles.get(parent_key) if parent_key else None,
            })
        ctx.info(f"API annotation search: {len(annotation_results)} results")
    except Exception as e:
        ctx.warning(f"Annotation search failed: {e}")

    return _format_search_results(query, note_results, annotation_results, raw_html=raw_html)


@mcp.tool(
    name="zotero_get_notes",
    description=(
        "Read notes from the active Zotero library. "
        "Omit query to LIST notes: with item_key, that item's child notes; "
        "without it, notes library-wide (capped by limit). "
        "Pass query to SEARCH note and annotation text instead — "
        "case-insensitive substring over the stripped-text body, "
        "library-wide, so query and item_key cannot be combined. "
        "limit: max results (default 20). "
        "truncate=True (default) shortens long bodies for display; pass "
        "False for complete content (list mode only). "
        "raw_html=True returns a note's original HTML instead of stripped "
        "text — use it when you intend to edit and round-trip via "
        "zotero_manage_note(action='update'). "
        "Scope: active library only (zotero_switch_library to change). "
        "Example: zotero_get_notes(item_key='ABC12345', raw_html=True); "
        "zotero_get_notes(query='mindfulness')."
    )
)
def get_notes_tool(
    item_key: str | None = None,
    query: str | None = None,
    limit: int | str | None = 20,
    truncate: bool = True,
    raw_html: bool = False,
    *,
    ctx: Context
) -> str:
    """Dispatch the note-read surface to listing or full-text search."""
    if query is not None and query.strip():
        if item_key:
            return (
                "Error: query and item_key cannot be combined — note search "
                "is library-wide. Call with item_key alone to list one "
                "item's notes, or with query alone to search the library."
            )
        return search_notes(query=query, limit=limit, raw_html=raw_html, ctx=ctx)

    if query is not None:
        return "Error: Search query cannot be empty"

    return get_notes(
        item_key=item_key,
        limit=limit,
        truncate=truncate,
        raw_html=raw_html,
        ctx=ctx,
    )


@mcp.tool(
    name="zotero_manage_note",
    description=(
        "Create, update, or trash a Zotero note. "
        "item_key: the PARENT item's key for action='create', the NOTE's "
        "own key for 'update' and 'delete' (zotero_get_notes finds it). "
        "create: needs note_text — plain text, or simple HTML (p, strong, "
        "em, ul/li, a, code), which is preserved; note_title becomes a "
        "heading; tags optional. "
        "update: needs note_text. append=False (default) REPLACES the "
        "whole body, append=True concatenates. To keep formatting, fetch "
        "with zotero_get_notes(raw_html=True), edit that HTML, and pass it "
        "back whole. "
        "delete: moves the note to the Trash — recoverable; emptying the "
        "Trash is manual in Zotero. Notes only, not items/collections/"
        "attachments. "
        "Requires a writable library (web API key or hybrid mode). "
        "Example: (action='create', item_key='ABC12345', "
        "note_title='Reading notes', note_text='<p>Key claim ...</p>')."
    )
)
def manage_note(
    action: Literal["create", "update", "delete"],
    item_key: str,
    note_text: str | None = None,
    note_title: str | None = None,
    tags: list[str] | str | None = None,
    append: bool = False,
    *,
    ctx: Context
) -> str:
    """Dispatch the note-write surface to create/update/delete."""
    if action == "create":
        if not note_text:
            return (
                "Error: action='create' requires note_text (the note body). "
                "item_key must be the parent item the note is attached to."
            )
        return create_note(
            item_key=item_key,
            note_title=note_title or "",
            note_text=note_text,
            tags=tags,
            ctx=ctx,
        )

    if action == "update":
        if note_text is None:
            return (
                "Error: action='update' requires note_text (the new HTML "
                "body). item_key must be the note's own key."
            )
        return update_note(
            item_key=item_key, note_text=note_text, append=append, ctx=ctx
        )

    if action == "delete":
        if note_text is not None:
            return (
                "Error: action='delete' takes only item_key (the note's own "
                "key); note_text is not applicable. Use action='update' to "
                "change a note's content."
            )
        return delete_note(item_key=item_key, ctx=ctx)

    return (
        f"Error: unknown action '{action}'. "
        "Expected one of: create, update, delete."
    )


@with_zotero_api_lock
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
        tags = _helpers._normalize_str_list_input(tags, "tags") if tags is not None else []
        zot = _client.get_zotero_client()

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
        if _utils.is_local_mode():
            web_zot = _client.get_web_zotero_client()
            if web_zot is not None:
                # Propagate library override if user switched libraries
                _helpers.apply_library_override(web_zot, _client.get_active_library())
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
                        "To create properly attached child notes, add these environment variables "
                        "to your Claude Desktop config alongside ZOTERO_LOCAL=true:\n"
                        + _WEB_API_ENV_VARS
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


def update_note(
    item_key: str,
    note_text: str,
    append: bool = False,
    *,
    ctx: Context
) -> str:
    """
    Update an existing Zotero note.

    Args:
        item_key: Zotero item key/ID of the note to update
        note_text: New HTML content of the note
        append: If True, concatenate note_text to existing note content;
            if False (default), replace existing content.
        ctx: MCP context

    Returns:
        Confirmation message
    """
    try:
        ctx.info(f"Updating note {item_key} (append={append})")

        zot, err = _get_note_write_client("updating notes")
        if err:
            return err

        try:
            item = zot.item(item_key)
        except Exception:
            return f"Error: No item found with key: {item_key}"

        data = item.get("data", {})
        if data.get("itemType") != "note":
            return f"Error: Item {item_key} is not a note (itemType={data.get('itemType')})"

        if append:
            data["note"] = (data.get("note", "") or "") + note_text
        else:
            data["note"] = note_text

        resp = zot.update_item(item)
        if _helpers._handle_write_response(resp, ctx):
            return f"Successfully updated note {item_key}"
        return f"Failed to update note {item_key}"

    except Exception as e:
        ctx.error(f"Error updating note: {str(e)}")
        return f"Error updating note: {str(e)}"


def delete_note(
    item_key: str,
    *,
    ctx: Context
) -> str:
    """
    Move a Zotero note to the Trash.

    Args:
        item_key: Zotero item key/ID of the note to trash
        ctx: MCP context

    Returns:
        Confirmation message
    """
    try:
        ctx.info(f"Trashing note {item_key}")

        zot, err = _get_note_write_client("deleting notes")
        if err:
            return err

        try:
            item = zot.item(item_key)
        except Exception:
            return f"Error: No item found with key: {item_key}"

        data = item.get("data", {})
        if data.get("itemType") != "note":
            return f"Error: Item {item_key} is not a note (itemType={data.get('itemType')})"

        # pyzotero's delete_item() permanently destroys items, and update_item()
        # strips the "deleted" field. We send a direct PATCH with {"deleted": 1}
        # to move the note to Zotero's Trash (recoverable by the user).
        from pyzotero.zotero import build_url
        url = build_url(
            zot.endpoint,
            f"/{zot.library_type}/{zot.library_id}/items/{item_key}",
        )
        resp = zot.client.patch(
            url=url,
            headers={"If-Unmodified-Since-Version": str(item["version"])},
            content=json.dumps({"deleted": 1}),
        )
        if resp.status_code in (200, 204):
            return f"Successfully trashed note {item_key} (recoverable from Zotero's Trash)"
        return f"Failed to trash note {item_key} (HTTP {resp.status_code}): {resp.text[:200]}"

    except Exception as e:
        ctx.error(f"Error trashing note: {str(e)}")
        return f"Error trashing note: {str(e)}"


@mcp.tool(
    name="zotero_create_annotation",
    description=(
        "Create an annotation on a PDF attachment (EPUB: highlights only). "
        "Exactly one of two modes per call: text= HIGHLIGHTS selectable "
        "text; rect= draws an AREA box over a figure, table, or other "
        "non-text region (PDF only). Passing both or neither is an error. "
        "attachment_key: the PDF/EPUB attachment key, NOT the parent item "
        "key (zotero_get_item_children finds it). "
        "page: 1-indexed page (EPUB: 1-indexed chapter). "
        "text: exact text to highlight, matched against the text layer — "
        "scanned/image-only PDFs will not match. "
        "rect: [x, y, width, height] normalized to [0, 1], with (0, 0) at "
        "the page's top-left; width/height are page-relative and the box "
        "must fit the page. Call zotero_get_page_layout first and reuse a "
        "detected region's bbox instead of guessing coordinates. "
        "comment, color (hex, default '#ffd400'), tags: optional. "
        "Requires PyMuPDF (the [pdf] extra) and a writable library (web "
        "API key or hybrid mode). "
        "Examples: (attachment_key='NHZFE5A7', page=4, text='working "
        "memory'); (attachment_key='NHZFE5A7', page=7, rect=[0.15, 0.22, "
        "0.6, 0.35], comment='Figure 3')."
    )
)
def create_annotation(
    attachment_key: str,
    page: int,
    text: str | None = None,
    rect: list[float] | str | None = None,
    comment: str | None = None,
    color: str = "#ffd400",
    tags: list[str] | str | None = None,
    *,
    ctx: Context
) -> str:
    """Dispatch to the highlight or area annotation implementation.

    Area mode is selected by the presence of ``rect``. Geometry is mutually
    exclusive with ``text``: Zotero stores highlights and image annotations
    as different annotationTypes with different position semantics, so a
    call carrying both is ambiguous rather than additive.
    """
    has_text = text is not None and text != ""

    if rect is not None and has_text:
        return (
            "Error: pass either text (highlight mode) or rect (area mode), "
            "not both. Highlights anchor to the text layer; area "
            "annotations anchor to a page rectangle."
        )

    if rect is not None:
        # MCP clients that stringify untyped params (same as the sibling
        # `tags` parameter, see _helpers._normalize_str_list_input) may send
        # rect as JSON text rather than a list.
        normalized_rect = _helpers._normalize_float_list_input(rect, 4, "rect")
        if normalized_rect is None:
            return (
                "Error: rect must be exactly four numbers, "
                "[x, y, width, height], normalized to 0-1. "
                f"Got: {rect!r}"
            )
        x, y, width, height = normalized_rect
        return create_area_annotation(
            attachment_key=attachment_key,
            page=page,
            x=x,
            y=y,
            width=width,
            height=height,
            comment=comment,
            color=color,
            tags=tags,
            ctx=ctx,
        )

    if not has_text:
        return (
            "Error: nothing to annotate. Pass text= with the exact text to "
            "highlight, or rect=[x, y, width, height] (normalized 0-1) to "
            "draw an area box — zotero_get_page_layout returns usable "
            "coordinates."
        )

    return _create_highlight_annotation(
        attachment_key=attachment_key,
        page=page,
        text=text,
        comment=comment,
        color=color,
        tags=tags,
        ctx=ctx,
    )


@with_zotero_api_lock
def _create_highlight_annotation(
    attachment_key: str,
    page: int,
    text: str,
    comment: str | None = None,
    color: str = "#ffd400",
    tags: list[str] | str | None = None,
    *,
    ctx: Context
) -> str:
    """
    Create a highlight annotation on a PDF or EPUB attachment.

    This tool handles multiple storage configurations:
    - Zotero Cloud Storage: Downloads file via Web API
    - WebDAV Storage: Downloads file via local Zotero or direct WebDAV access
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

    from zotero_mcp.pdf_utils import (
        find_text_position,
        get_page_label,
        build_annotation_position,
        verify_pdf_attachment,
    )

    try:
        ctx.info(f"Creating annotation on attachment {attachment_key}, page {page}")

        # Get clients for different operations
        local_client = _client.get_local_zotero_client()
        web_client = _client.get_web_zotero_client()

        # Propagate library override if user switched libraries
        if web_client:
            _helpers.apply_library_override(web_client, _client.get_active_library())

        # REQUIREMENT: Web API is required for creating annotations
        # Zotero's local API (port 23119) is read-only
        if not web_client:
            return (
                "Error: Web API credentials required for creating annotations.\n\n"
                "Please configure the following environment variables:\n"
                + _WEB_API_ENV_VARS
                + "\n\nNote: Zotero's local API is read-only and cannot create annotations."
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
            file_path, error_message = _download_attachment_for_processing(
                attachment_key,
                filename,
                tmpdir,
                local_client=local_client,
                web_client=web_client,
                ctx=ctx,
            )
            if error_message:
                return error_message

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

            tag_list = (
                _helpers._normalize_str_list_input(tags, "tags")
                if tags is not None else []
            )

            annotation_data = {
                "itemType": "annotation",
                "parentItem": attachment_key,
                "annotationType": "highlight",
                "annotationText": text,
                "annotationComment": comment or "",
                "annotationColor": color,
                "annotationSortIndex": sort_index,
                "annotationPosition": annotation_position,
                "tags": [{"tag": t} for t in tag_list],
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


def create_area_annotation(
    attachment_key: str,
    page: int,
    x: float,
    y: float,
    width: float,
    height: float,
    comment: str | None = None,
    color: str = "#ffd400",
    tags: list[str] | str | None = None,
    *,
    ctx: Context
) -> str:
    """
    Create an area/image annotation on a PDF attachment.

    Args:
        attachment_key: PDF attachment key (e.g., "NHZFE5A7")
        page: 1-indexed PDF page number
        x: Normalized left coordinate (0..1)
        y: Normalized top coordinate (0..1)
        width: Normalized width (0..1)
        height: Normalized height (0..1)
        comment: Optional comment on the annotation
        color: Annotation color in hex format
        ctx: MCP context

    Returns:
        Confirmation message with the new annotation key
    """
    from math import isfinite

    from zotero_mcp.pdf_utils import (
        build_annotation_position,
        build_area_position_data,
        get_page_label,
        verify_pdf_attachment,
    )

    try:
        ctx.info(f"Creating area annotation on attachment {attachment_key}, page {page}")

        values = {"x": x, "y": y, "width": width, "height": height}
        for name, value in values.items():
            if not isinstance(value, (int, float)) or not isfinite(value):
                return f"Error: {name} must be a finite number"

        if x < 0 or x > 1:
            return "Error: x must be between 0 and 1"
        if y < 0 or y > 1:
            return "Error: y must be between 0 and 1"
        if width <= 0 or width > 1:
            return "Error: width must be greater than 0 and at most 1"
        if height <= 0 or height > 1:
            return "Error: height must be greater than 0 and at most 1"
        if x + width > 1:
            return "Error: Rectangle must fit within the page width (x + width must be <= 1)"
        if y + height > 1:
            return "Error: Rectangle must fit within the page height (y + height must be <= 1)"

        local_client = _client.get_local_zotero_client()
        web_client = _client.get_web_zotero_client()

        if web_client:
            _helpers.apply_library_override(web_client, _client.get_active_library())

        if not web_client:
            return (
                "Error: Web API credentials required for creating annotations.\n\n"
                "Please configure the following environment variables:\n"
                "- ZOTERO_API_KEY: Your Zotero API key (from zotero.org/settings/keys)\n"
                "- ZOTERO_LIBRARY_ID: Your library ID\n"
                "- ZOTERO_LIBRARY_TYPE: 'user' or 'group'\n\n"
                "Note: Zotero's local API is read-only and cannot create annotations."
            )

        try:
            attachment = web_client.item(attachment_key)
            attachment_data = attachment.get("data", {})

            if attachment_data.get("itemType") != "attachment":
                return f"Error: Item {attachment_key} is not an attachment"

            content_type = attachment_data.get("contentType", "")
            if content_type != "application/pdf":
                return f"Error: Attachment {attachment_key} is not a PDF attachment (type: {content_type})"

            filename = attachment_data.get("filename", f"{attachment_key}.pdf")
        except Exception as e:
            return f"Error: No attachment found with key: {attachment_key} ({e})"

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path, error_message = _download_attachment_for_processing(
                attachment_key,
                filename,
                tmpdir,
                local_client=local_client,
                web_client=web_client,
                ctx=ctx,
            )
            if error_message:
                return error_message

            if not verify_pdf_attachment(file_path):
                return "Error: Downloaded file is not a valid PDF"

            position_data = build_area_position_data(file_path, page, x, y, width, height)
            if "error" in position_data:
                return f"Error: {position_data['error']}"

            page_label = get_page_label(file_path, page)
            annotation_position = build_annotation_position(
                position_data["pageIndex"],
                position_data["rects"],
            )

            tag_list = (
                _helpers._normalize_str_list_input(tags, "tags")
                if tags is not None else []
            )

            annotation_data = {
                "itemType": "annotation",
                "parentItem": attachment_key,
                "annotationType": "image",
                "annotationComment": comment or "",
                "annotationColor": color,
                "annotationSortIndex": position_data["sort_index"],
                "annotationPosition": annotation_position,
                "annotationPageLabel": page_label,
                "tags": [{"tag": t} for t in tag_list],
            }

            ctx.info("Creating area annotation via Web API...")
            result = web_client.create_items([annotation_data])

            if "success" in result and result["success"]:
                successful = result["success"]
                if successful:
                    annotation_key = next(iter(successful.values()))
                    response = [
                        "Successfully created area annotation",
                        "",
                        f"**Annotation Key:** {annotation_key}",
                        f"**Page:** {page_label}",
                        f"**Rect (normalized):** x={x:.4f}, y={y:.4f}, width={width:.4f}, height={height:.4f}",
                        f"**Color:** {color}",
                    ]
                    if comment:
                        response.append(f"**Comment:** {comment}")
                    return "\n".join(response)
                return f"Annotation creation response was successful but no key was returned: {result}"

            failed_info = result.get("failed", {})
            return f"Failed to create annotation: {failed_info}"

    except Exception as e:
        ctx.error(f"Error creating area annotation: {str(e)}")
        return f"Error creating area annotation: {str(e)}"


def _format_page_layout(
    layout: dict,
    attachment_key: str,
    page: int,
    filename: str,
) -> str:
    """Render detect_page_regions output as markdown for the LLM."""
    regions = layout["regions"]
    warnings = layout.get("warnings", [])
    page_label = layout.get("pageLabel", str(page))

    lines = [
        f"# Page layout: {filename}, page {page} (label: {page_label})",
        "",
    ]

    for warning in warnings:
        lines.append(f"**Warning:** {warning}")
    if warnings:
        lines.append("")

    if not regions:
        lines.append(
            f"No figure/table regions detected on page {page}. "
            "The page may be text-only or a scanned image. "
            "You can still create an area annotation by passing explicit coordinates."
        )
        return "\n".join(lines)

    plural = "s" if len(regions) != 1 else ""
    lines.extend([
        f"**{len(regions)} region{plural} detected.**",
        "",
        "| # | source | x | y | width | height | caption | confidence |",
        "|---|--------|---|---|-------|--------|---------|------------|",
    ])

    for region in regions:
        x, y, width, height = region["bbox"]
        caption = region.get("caption_text") or "(none)"
        if len(caption) > 80:
            caption = caption[:77] + "..."
        caption = caption.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {region['region_id']} | {region['source']} | {x:.4f} | {y:.4f} "
            f"| {width:.4f} | {height:.4f} | {caption} | {region['confidence']} |"
        )

    first_x, first_y, first_w, first_h = regions[0]["bbox"]
    lines.extend([
        "",
        "To annotate a region, pass its bbox to zotero_create_annotation "
        "as rect — e.g. for region 1:",
        "```",
        f"zotero_create_annotation(attachment_key='{attachment_key}', "
        f"page={page}, rect=[{first_x:.4f}, {first_y:.4f}, "
        f"{first_w:.4f}, {first_h:.4f}], comment='...')",
        "```",
    ])

    return "\n".join(lines)


@mcp.tool(
    name="zotero_get_page_layout",
    description=(
        "Detect candidate figure/table regions on a PDF page and return "
        "their normalized bounding boxes, so area annotations can be placed "
        "on detected content instead of guessed positions. ALWAYS call this "
        "before zotero_create_annotation's area mode unless exact "
        "coordinates are already known. Returns each region's bounding box "
        "(x, y, width, height in [0, 1]), source "
        "(image/drawing/table/merged), associated caption (e.g. 'Figure 3: "
        "...'), confidence level, and a ready-to-paste "
        "zotero_create_annotation call. "
        "Note: detection is geometric — boxes cover the graphical core of a "
        "figure/table; text labels inside figures or unruled table headers "
        "may fall outside the box. Confidence reflects caption matching, "
        "not box completeness. "
        "attachment_key: PDF attachment key — NOT the parent item key (use "
        "zotero_get_item_children to find attachments). "
        "page: 1-indexed page number (page 1 is the first page). "
        "Scope: PDFs only — EPUB attachments are NOT supported. Read-only: "
        "works in both local and web API modes. "
        "Example: zotero_get_page_layout(attachment_key='NHZFE5A7', page=7)."
    )
)
def get_page_layout(
    attachment_key: str,
    page: int,
    *,
    ctx: Context
) -> str:
    """
    Detect figure/table regions on a PDF page for area annotation grounding.

    Args:
        attachment_key: PDF attachment key (e.g., "NHZFE5A7")
        page: 1-indexed PDF page number
        ctx: MCP context

    Returns:
        Markdown table of detected regions with normalized bounding boxes
    """
    from zotero_mcp import pdf_layout

    try:
        ctx.info(f"Detecting page layout on attachment {attachment_key}, page {page}")

        if not isinstance(page, int) or page < 1:
            return "Error: page must be a positive 1-indexed page number"

        local_client = _client.get_local_zotero_client()
        web_client = _client.get_web_zotero_client()

        if web_client:
            _helpers.apply_library_override(web_client, _client.get_active_library())

        meta_client = web_client or local_client
        if meta_client is None:
            return (
                "Error: No Zotero client configured.\n\n"
                "Configure either local Zotero (with 'Allow other applications "
                "on this computer to communicate with Zotero' enabled) or Web "
                "API credentials:\n" + _WEB_API_ENV_VARS
            )

        try:
            attachment = meta_client.item(attachment_key)
            attachment_data = attachment.get("data", {})

            if attachment_data.get("itemType") != "attachment":
                return f"Error: Item {attachment_key} is not an attachment"

            content_type = attachment_data.get("contentType", "")
            if content_type != "application/pdf":
                return f"Error: Attachment {attachment_key} is not a PDF attachment (type: {content_type})"

            filename = attachment_data.get("filename", f"{attachment_key}.pdf")
        except Exception as e:
            return f"Error: No attachment found with key: {attachment_key} ({e})"

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path, error_message = _download_attachment_for_processing(
                attachment_key,
                filename,
                tmpdir,
                local_client=local_client,
                web_client=web_client,
                ctx=ctx,
            )
            if error_message:
                return error_message

            ctx.info(f"Detecting regions on page {page}...")
            layout = pdf_layout.detect_page_regions(file_path, page)

        if "error" in layout:
            return f"Error: {layout['error']}"

        return _format_page_layout(layout, attachment_key, page, filename)

    except Exception as e:
        ctx.error(f"Error detecting page layout: {str(e)}")
        return f"Error detecting page layout: {str(e)}"


@mcp.tool(
    name="zotero_update_annotation",
    description=(
        "Update an existing Zotero annotation. Editable fields: text (highlight text), "
        "comment, color (hex like '#ffd400'), and tags. "
        "Tags can be replaced wholesale via `tags`, or edited incrementally via "
        "`add_tags`/`remove_tags` (mutually exclusive with `tags`). "
        "Position/page/sortIndex are anchored to the PDF/EPUB geometry and are not editable."
    )
)
def update_annotation(
    annotation_key: str,
    text: str | None = None,
    comment: str | None = None,
    color: str | None = None,
    tags: list[str] | str | None = None,
    add_tags: list[str] | str | None = None,
    remove_tags: list[str] | str | None = None,
    *,
    ctx: Context
) -> str:
    try:
        if tags is not None and (add_tags is not None or remove_tags is not None):
            return (
                "Error: Cannot use 'tags' (replace all) together with "
                "'add_tags'/'remove_tags' (incremental). Use one approach or the other."
            )

        ctx.info(f"Updating annotation {annotation_key}")

        zot, err = _get_note_write_client("updating annotations")
        if err:
            return err

        try:
            item = zot.item(annotation_key)
        except Exception:
            return f"Error: No item found with key: {annotation_key}"

        data = item.get("data", {})
        if data.get("itemType") != "annotation":
            return (
                f"Error: Item {annotation_key} is not an annotation "
                f"(itemType={data.get('itemType')})"
            )

        changes = []
        if text is not None:
            data["annotationText"] = text
            changes.append("- **text**: updated")
        if comment is not None:
            data["annotationComment"] = comment
            changes.append("- **comment**: updated")
        if color is not None:
            data["annotationColor"] = color
            changes.append(f"- **color**: {color}")

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
                to_remove = set(
                    _helpers._normalize_str_list_input(remove_tags, "remove_tags")
                )
                existing -= to_remove
                changes.append(f"- **tags**: removed {list(to_remove)}")
            data["tags"] = [{"tag": t} for t in sorted(existing)]

        if not changes:
            return "No changes to apply."

        resp = zot.update_item(item)
        if _helpers._handle_write_response(resp, ctx):
            return (
                f"Successfully updated annotation `{annotation_key}`:\n\n"
                + "\n".join(changes)
            )
        return f"Failed to update annotation {annotation_key}"

    except ValueError as e:
        return f"Input error: {e}"
    except Exception as e:
        ctx.error(f"Error updating annotation: {e}")
        return f"Error updating annotation: {e}"


@mcp.tool(
    name="zotero_delete_annotation",
    description=(
        "Move a Zotero annotation to the Trash. Trashed annotations are recoverable "
        "from Zotero's Trash — empty the Trash in the Zotero UI for permanent deletion."
    )
)
def delete_annotation(
    annotation_key: str,
    *,
    ctx: Context
) -> str:
    try:
        ctx.info(f"Trashing annotation {annotation_key}")

        zot, err = _get_note_write_client("deleting annotations")
        if err:
            return err

        try:
            item = zot.item(annotation_key)
        except Exception:
            return f"Error: No item found with key: {annotation_key}"

        data = item.get("data", {})
        if data.get("itemType") != "annotation":
            return (
                f"Error: Item {annotation_key} is not an annotation "
                f"(itemType={data.get('itemType')})"
            )

        from pyzotero.zotero import build_url
        url = build_url(
            zot.endpoint,
            f"/{zot.library_type}/{zot.library_id}/items/{annotation_key}",
        )
        resp = zot.client.patch(
            url=url,
            headers={"If-Unmodified-Since-Version": str(item["version"])},
            content=json.dumps({"deleted": 1}),
        )
        if resp.status_code in (200, 204):
            return (
                f"Successfully trashed annotation {annotation_key} "
                "(recoverable from Zotero's Trash)"
            )
        return (
            f"Failed to trash annotation {annotation_key} "
            f"(HTTP {resp.status_code}): {resp.text[:200]}"
        )

    except Exception as e:
        ctx.error(f"Error trashing annotation: {str(e)}")
        return f"Error trashing annotation: {str(e)}"
