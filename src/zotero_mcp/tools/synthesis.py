"""Synthesis and export tool functions for the Zotero MCP server.

These tools gather and structure existing library content (annotations,
notes, citations) so an LLM agent can synthesize literature summaries or
drop formatted references into a manuscript. They do NOT call an LLM
themselves; they only collect and format.
"""

import json
from collections import Counter
from typing import Literal

from zotero_mcp import client as _client
from zotero_mcp import utils as _utils
from zotero_mcp._app import mcp
from zotero_mcp._context import Context
from zotero_mcp.client import with_zotero_api_lock
from zotero_mcp.tools import _helpers
from zotero_mcp.tools.annotations import _annotation_to_record


def _resolve_paper_context(
    zot,
    parent_key: str,
    cache: dict[str, dict[str, str | None]],
) -> dict[str, str | None]:
    """Resolve an annotation/note parent key to its paper context.

    Annotations are children of PDF/EPUB attachments, which are children of
    the paper (two hops: annotation.parentItem -> attachment ->
    attachment.parentItem -> paper). Notes are usually direct children of the
    paper (one hop). This helper tolerates either shape and any missing hop,
    falling back to the attachment title or the bare key. Results are cached.
    """
    if parent_key in cache:
        return cache[parent_key]

    context: dict[str, str | None] = {
        "item_key": parent_key or None,
        "attachment_key": None,
        "parent_title": parent_key or "(unknown source)",
        "attachment_title": None,
    }
    try:
        parent = zot.item(parent_key)
        data = parent.get("data", {}) if parent else {}
        if data.get("itemType") == "attachment" and data.get("parentItem"):
            gp_key = data["parentItem"]
            attachment_title = data.get("title") or None
            try:
                grandparent = zot.item(gp_key)
                gp_data = grandparent.get("data", {}) if grandparent else {}
                title = gp_data.get("title") or attachment_title or parent_key
            except Exception:
                title = attachment_title or parent_key
            context = {
                "item_key": gp_key,
                "attachment_key": parent_key,
                "parent_title": title,
                "attachment_title": attachment_title,
            }
        else:
            context["parent_title"] = data.get("title") or parent_key
    except Exception:
        pass

    cache[parent_key] = context
    return context


@mcp.tool(
    name="zotero_synthesize_annotations",
    description=(
        "Collect every highlight, annotation comment, and child note across "
        "a scope and organize them into a structured, per-paper digest that "
        "YOU (the agent) can then synthesize into a literature summary. This "
        "tool does NOT call an LLM — it only gathers and groups the raw "
        "material, so the synthesis step is yours. "
        "collection_key: optional 8-character collection key; when given, "
        "only annotations/notes whose resolved paper is a member of that "
        "collection are included. When omitted, the whole active library is "
        "scanned (capped by limit). "
        "tag: optional tag or list of tags to filter items by (accepts a "
        "string, a JSON list, or a list). "
        "limit: cap on annotations/notes scanned (default 200) to keep the "
        "call tractable. "
        "format='markdown' (default) groups the digest by paper; "
        "format='json' returns the same highlights and notes as structured "
        "records for downstream processing. Markdown output has each paper "
        "heading followed by "
        "its highlights (with attached comments) and any note excerpts — "
        "plus a top summary line counting papers, highlights, and notes. "
        "Use this before writing a thematic review so you can spot themes "
        "and contradictions across sources. "
        "Example: zotero_synthesize_annotations(collection_key='MT53KB66')."
    ),
)
@with_zotero_api_lock
def synthesize_annotations(
    collection_key: str | None = None,
    tag: list[str] | str | None = None,
    limit: int | str | None = 200,
    format: Literal["markdown", "json"] = "markdown",
    *,
    ctx: Context,
) -> str:
    """Gather annotations and notes into a per-paper digest for synthesis.

    Args:
        collection_key: Optional collection to restrict the digest to.
        tag: Optional tag filter (string, JSON list, or list).
        limit: Maximum annotations/notes to scan.
        format: ``markdown`` for a readable digest or ``json`` for structured
            per-paper annotation and note records.
        ctx: MCP context.

    Returns:
        Markdown digest or a JSON object containing summary counts and papers.
    """
    try:
        ctx.info("Gathering annotations and notes for synthesis")
        zot = _client.get_zotero_client()

        limit = _helpers._normalize_limit(limit, default=200, max_val=5000)
        tags = _helpers._normalize_tag_filter(tag)

        # Determine the set of allowed paper keys if a collection is scoped.
        allowed_keys: set[str] | None = None
        if collection_key:
            try:
                coll_items = _helpers._paginate(
                    zot.collection_items,
                    collection_key,
                    itemType="-attachment",
                )
                allowed_keys = {it.get("key") for it in coll_items if it.get("key")}
            except Exception as e:
                ctx.warning(f"Could not load collection items: {e}")
                allowed_keys = set()

        anno_params = {"itemType": "annotation"}
        note_params = {"itemType": "note"}
        if tags:
            anno_params["tag"] = tags
            note_params["tag"] = tags

        try:
            annotations = _helpers._paginate(zot.items, max_items=limit, **anno_params)
        except Exception as e:
            ctx.warning(f"Annotation fetch failed: {e}")
            annotations = []
        try:
            notes = _helpers._paginate(zot.items, max_items=limit, **note_params)
        except Exception as e:
            ctx.warning(f"Note fetch failed: {e}")
            notes = []

        if not annotations and not notes:
            if format == "json":
                return json.dumps({
                    "summary": {"papers": 0, "highlights": 0, "notes": 0},
                    "papers": [],
                }, indent=2)
            scope = f" in collection {collection_key}" if collection_key else ""
            return f"No annotations or notes found{scope}."

        # Group by resolved paper key so same-title papers stay distinct.
        context_cache: dict[str, dict[str, str | None]] = {}
        papers: dict[str, dict] = {}

        def _bucket(context: dict[str, str | None]) -> dict:
            item_key = context.get("item_key")
            title = context.get("parent_title") or "(unknown source)"
            bucket_key = item_key or title
            return papers.setdefault(bucket_key, {
                "item_key": item_key,
                "title": title,
                "highlights": [],
                "notes": [],
            })

        def _in_scope(parent_key: str) -> bool:
            if allowed_keys is None:
                return True
            # Member if the immediate parent, or its grandparent paper, is in scope.
            if parent_key in allowed_keys:
                return True
            try:
                parent = zot.item(parent_key)
                data = parent.get("data", {}) if parent else {}
                gp = data.get("parentItem")
                if gp and gp in allowed_keys:
                    return True
            except Exception:
                pass
            return False

        highlight_count = 0
        for anno in annotations:
            data = anno.get("data", {})
            parent_key = data.get("parentItem")
            text = (data.get("annotationText") or "").strip()
            comment = (data.get("annotationComment") or "").strip()
            if not text and not comment:
                continue
            if parent_key and not _in_scope(parent_key):
                continue
            context = (
                _resolve_paper_context(zot, parent_key, context_cache)
                if parent_key
                else {
                    "item_key": None,
                    "attachment_key": None,
                    "parent_title": "(unknown source)",
                    "attachment_title": None,
                }
            )
            record = _annotation_to_record(anno, context)
            # Preserve this tool's existing whitespace-trimming behaviour.
            record["text"] = text
            record["comment"] = comment
            _bucket(context)["highlights"].append(record)
            highlight_count += 1

        note_count = 0
        for note in notes:
            data = note.get("data", {})
            parent_key = data.get("parentItem")
            raw = data.get("note") or ""
            text = _utils.clean_html(raw).strip()
            if not text:
                continue
            if parent_key and not _in_scope(parent_key):
                continue
            context = (
                _resolve_paper_context(zot, parent_key, context_cache)
                if parent_key
                else {
                    "item_key": None,
                    "attachment_key": None,
                    "parent_title": "(standalone note)",
                    "attachment_title": None,
                }
            )
            if len(text) > 400:
                text = text[:400] + "..."
            _bucket(context)["notes"].append({
                "note_key": note.get("key") or None,
                "item_key": context.get("item_key"),
                "parent_title": context.get("parent_title"),
                "text": text,
                "tags": [
                    item["tag"]
                    for item in _helpers._normalize_item_tags(data.get("tags"))
                ],
                "created": data.get("dateAdded") or None,
                "modified": data.get("dateModified") or None,
            })
            note_count += 1

        if not papers:
            if format == "json":
                return json.dumps({
                    "summary": {"papers": 0, "highlights": 0, "notes": 0},
                    "papers": [],
                }, indent=2)
            scope = f" in collection {collection_key}" if collection_key else ""
            return f"No annotations or notes found{scope}."

        sorted_papers = sorted(papers.values(), key=lambda paper: paper["title"])
        if format == "json":
            return json.dumps({
                "summary": {
                    "papers": len(papers),
                    "highlights": highlight_count,
                    "notes": note_count,
                },
                "papers": sorted_papers,
            }, ensure_ascii=False, indent=2)

        output = [
            "# Annotation & Note Digest",
            "",
            (f"**{len(papers)} papers, {highlight_count} highlights, {note_count} notes**"),
            "",
        ]

        # Buckets are keyed by item key, so two distinct papers can share a
        # title — qualify the heading with the key when that happens.
        title_counts = Counter(paper["title"] for paper in sorted_papers)
        for bucket in sorted_papers:
            heading = bucket["title"]
            if title_counts[heading] > 1 and bucket["item_key"]:
                heading = f"{heading} ({bucket['item_key']})"
            output.append(f"## {heading}")
            if bucket["highlights"]:
                output.append("**Highlights:**")
                for highlight in bucket["highlights"]:
                    text = highlight["text"]
                    comment = highlight["comment"]
                    line = f"- {text}" if text else "- (comment only)"
                    if comment:
                        line += f" — *{comment}*"
                    output.append(line)
            if bucket["notes"]:
                output.append("**Notes:**")
                for note in bucket["notes"]:
                    output.append(f"- {note['text']}")
            output.append("")

        output.append(
            "*You can now synthesize themes, agreements, and contradictions across these papers from the digest above.*"
        )

        result = "\n".join(output)
        return _helpers._prepend_size_warning(
            result,
            "Scope to a collection_key or narrow with tag to reduce size.",
        )

    except Exception as e:
        ctx.error(f"Error synthesizing annotations: {str(e)}")
        return f"Error synthesizing annotations: {str(e)}"


def _render_entries(rendered) -> list[str]:
    """Normalize pyzotero content output into a list of plain-text entries."""
    if rendered is None:
        return []
    if isinstance(rendered, str):
        return [rendered]
    entries: list[str] = []
    for item in rendered:
        entries.append(item if isinstance(item, str) else str(item))
    return entries


@mcp.tool(
    name="zotero_export_bibliography",
    description=(
        "Render a formatted bibliography or in-text citations for a set of "
        "Zotero items using Zotero's own CSL citation engine, so you can drop "
        "references straight into a manuscript. "
        "item_keys: optional list of 8-character item keys (also accepts a "
        "JSON list string); takes precedence over collection_key. "
        "collection_key: optional collection to export instead; if neither is "
        "given, the active library is exported (capped). "
        "style: CSL style short name (default 'apa'); e.g. 'modern-language-"
        "association', 'chicago-note-bibliography', 'ieee'. Ignored for "
        "bibtex. "
        "export_format: 'bib' (formatted reference-list entries, default), "
        "'citation' (in-text citation strings), or 'bibtex' (raw BibTeX for "
        ".bib files). "
        "Output: markdown naming the style/format, then the rendered entries "
        "(a fenced block for bibtex, a numbered list otherwise). "
        "Rendering uses Zotero's own CSL engine and works in local mode with "
        "no API credentials, as well as over the web API. "
        "Capped at 100 items per call; scope with item_keys or collection_key "
        "for anything larger. "
        "Example: zotero_export_bibliography(item_keys=['RTKZQI8E'], "
        "style='apa', export_format='bib')."
    ),
)
@with_zotero_api_lock
def export_bibliography(
    item_keys: list[str] | str | None = None,
    collection_key: str | None = None,
    style: str = "apa",
    export_format: Literal["bib", "citation", "bibtex"] = "bib",
    *,
    ctx: Context,
) -> str:
    """Render a formatted bibliography/citations via Zotero's CSL engine.

    Args:
        item_keys: Optional list of item keys (or JSON/comma string).
        collection_key: Optional collection to export.
        style: CSL style short name (default "apa").
        export_format: "bib", "citation", or "bibtex".
        ctx: MCP context.

    Returns:
        Markdown-formatted bibliography or citations.
    """
    try:
        if not isinstance(style, str) or not style.strip():
            style = "apa"
        style = style.strip()

        keys: list[str] = []
        if item_keys is not None:
            keys = _helpers._normalize_str_list_input(item_keys, "item_keys")

        ctx.info(f"Exporting bibliography (format={export_format}, style={style})")
        # Rendering asks for JSON, never Atom. `content=` implies format=atom,
        # which the local API rejects with 501; `include=`/`format=bibtex` are
        # served by both the local and web APIs, so one path covers every mode
        # and local-only users no longer need web credentials (#371).
        zot = _helpers._get_bibliography_client(ctx)

        try:
            if export_format == "bibtex":
                # A whole-file export, not per-item entries: the API returns the
                # concatenated .bib as raw bytes rather than a list.
                if keys:
                    raw = zot.items(itemKey=",".join(keys), format="bibtex", limit=100)
                elif collection_key:
                    raw = zot.collection_items(collection_key, format="bibtex", limit=100)
                else:
                    # Top-level items only. Attachments and notes have no
                    # bibliography entry, so including them would pad the
                    # export with blanks and crowd out real references.
                    raw = zot.top(format="bibtex", limit=100)
                rendered = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            else:
                include = "bib" if export_format == "bib" else "citation"
                fetch_kwargs = {"include": include, "style": style}
                if keys:
                    rows = zot.items(itemKey=",".join(keys), limit=100, **fetch_kwargs)
                    # The local API answers an itemKey filter on this endpoint
                    # with the requested items *plus* others, so the response
                    # cannot be trusted as the selection. Filter to what was
                    # asked for and return it in the caller's order — a no-op
                    # against the web API, which already filters correctly.
                    by_key = {
                        row.get("key"): row for row in rows if isinstance(row, dict)
                    }
                    rows = [by_key[k] for k in keys if k in by_key]
                elif collection_key:
                    rows = _helpers._paginate(
                        zot.collection_items, collection_key, max_items=100, **fetch_kwargs
                    )
                else:
                    rows = zot.top(limit=100, **fetch_kwargs)
                # Items with nothing to render (attachments, notes) come back
                # with the field empty; drop them rather than emitting blanks.
                rendered = [
                    row.get(include)
                    for row in rows
                    if isinstance(row, dict) and row.get(include)
                ]
        except Exception as api_error:
            ctx.error(f"Bibliography rendering failed: {api_error}")
            return (
                f"Error rendering bibliography: {api_error}\n\n"
                "Rendering uses Zotero's CSL engine via the local or web API. "
                "In local mode, check that Zotero is running with the local "
                "API enabled; otherwise verify ZOTERO_API_KEY and "
                "ZOTERO_LIBRARY_ID."
            )

        entries = _render_entries(rendered)
        if not entries:
            scope = (
                f" for collection {collection_key}" if collection_key else (" for the requested items" if keys else "")
            )
            return f"No bibliography entries produced{scope}."

        format_label = {
            "bib": "Bibliography",
            "citation": "Citations",
            "bibtex": "BibTeX",
        }[export_format]

        if export_format == "bibtex":
            body = "\n\n".join(e.strip() for e in entries if e.strip())
            return f"# {format_label}\n\n```bibtex\n{body}\n```"

        header = f"# {format_label} ({style})"
        lines = [header, ""]
        for i, entry in enumerate(entries, 1):
            clean = _utils.clean_html(entry).strip()
            if not clean:
                continue
            lines.append(f"{i}. {clean}")
        return "\n".join(lines)

    except Exception as e:
        ctx.error(f"Error exporting bibliography: {str(e)}")
        return f"Error exporting bibliography: {str(e)}"
