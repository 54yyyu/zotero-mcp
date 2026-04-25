"""
Standalone CLI for Zotero — direct tool access without an MCP server.

Usage:
    zotero-cli search "Einstein 1905"
    zotero-cli get metadata ITEM_KEY
    zotero-cli get collections
    zotero-cli add doi 10.1234/example
    zotero-cli add url https://arxiv.org/abs/2301.00001
    zotero-cli edit ITEM_KEY --title "New Title"
    zotero-cli notes list
    zotero-cli annotations list --item-key ITEM_KEY
    zotero-cli db status
"""

import argparse
import json
import sys

# Reuse environment setup from the original CLI module
from zotero_mcp.cli import (
    obfuscate_config_for_display,
    setup_zotero_environment,
)

# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

class CLIContext:
    """Drop-in replacement for fastmcp.Context that writes to stderr."""

    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    def info(self, message: str) -> None:
        if self._verbose:
            print(f"[INFO] {message}", file=sys.stderr)

    def warning(self, message: str) -> None:
        print(f"[WARN] {message}", file=sys.stderr)

    def error(self, message: str) -> None:
        print(f"[ERROR] {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Lazy tool imports
# ---------------------------------------------------------------------------

def _import_tools():
    """Import tool modules lazily to avoid heavy startup cost."""
    from zotero_mcp import client as _client
    from zotero_mcp.tools import annotations, retrieval, search, write
    return search, retrieval, annotations, write, _client


def _ctx(args) -> CLIContext:
    return CLIContext(verbose=getattr(args, "verbose", False))


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_config(args):
    import os
    setup_zotero_environment()
    config = {
        k: v for k, v in os.environ.items()
        if k.startswith("ZOTERO_") or k in ("OPENAI_API_KEY", "GOOGLE_API_KEY")
    }
    if not getattr(args, "show_secrets", False):
        config = obfuscate_config_for_display(config)
    print("=== Zotero Configuration ===")
    for k, v in sorted(config.items()):
        print(f"  {k}={v}")


def cmd_search(args):
    setup_zotero_environment()
    search_mod, retrieval, annotations, write_mod, _client = _import_tools()
    ctx = _ctx(args)

    if args.mode == "tag":
        result = search_mod.search_by_tag(
            tag=args.query.split(","), limit=args.limit,
            collection_key=getattr(args, "collection", None), ctx=ctx,
        )
    elif args.mode == "citekey":
        result = search_mod.search_by_citation_key(citekey=args.query, ctx=ctx)
    elif args.mode == "advanced":
        try:
            conditions = json.loads(args.conditions)
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON in --conditions: {e}", file=sys.stderr)
            sys.exit(1)
        result = search_mod.advanced_search(
            conditions=conditions, join_mode=args.join_mode,
            sort_by=args.sort_by, sort_direction=args.sort_direction,
            limit=args.limit, ctx=ctx,
        )
    elif args.mode == "semantic":
        filters = None
        if getattr(args, "filters", None):
            try:
                filters = json.loads(args.filters)
            except json.JSONDecodeError as e:
                print(f"Error: invalid JSON in --filters: {e}", file=sys.stderr)
                sys.exit(1)
        result = search_mod.semantic_search(
            query=args.query, limit=args.limit, filters=filters, ctx=ctx,
        )
    elif args.mode == "notes":
        result = annotations.search_notes(query=args.query, limit=args.limit, ctx=ctx)
    else:
        result = search_mod.search_items(
            query=args.query, qmode=args.qmode, limit=args.limit,
            collection_key=getattr(args, "collection", None), ctx=ctx,
        )
    print(result)


def cmd_get(args):
    setup_zotero_environment()
    search_mod, retrieval, annotations, write_mod, _client = _import_tools()
    ctx = _ctx(args)
    sub = args.subcommand

    if sub == "metadata":
        print(retrieval.get_item_metadata(
            item_key=args.item_key, include_abstract=not args.no_abstract,
            format=args.output_format, ctx=ctx,
        ))
    elif sub == "fulltext":
        print(retrieval.get_item_fulltext(item_key=args.item_key, ctx=ctx))
    elif sub == "bibtex":
        print(retrieval.get_item_metadata(item_key=args.item_key, format="bibtex", ctx=ctx))
    elif sub == "collections":
        print(retrieval.get_collections(limit=args.limit, ctx=ctx))
    elif sub == "collection-items":
        print(retrieval.get_collection_items(
            collection_key=args.collection_key, detail=args.detail, limit=args.limit, ctx=ctx,
        ))
    elif sub == "children":
        if getattr(args, "item_keys", None):
            print(retrieval.get_items_children(item_keys=args.item_keys, ctx=ctx))
        else:
            print(retrieval.get_item_children(item_key=args.item_key, ctx=ctx))
    elif sub == "tags":
        print(retrieval.get_tags(limit=args.limit, ctx=ctx))
    elif sub == "recent":
        print(retrieval.get_recent(
            limit=args.limit, collection_key=getattr(args, "collection", None), ctx=ctx,
        ))
    elif sub == "libraries":
        print(retrieval.list_libraries(ctx=ctx))
    elif sub == "feeds":
        print(retrieval.list_feeds(ctx=ctx))
    elif sub == "feed-items":
        print(retrieval.get_feed_items(library_id=args.library_id, limit=args.limit, ctx=ctx))
    else:
        print(f"Unknown 'get' subcommand: {sub}", file=sys.stderr)
        sys.exit(1)


def cmd_annotations(args):
    setup_zotero_environment()
    search_mod, retrieval, annotations, write_mod, _client = _import_tools()
    ctx = _ctx(args)

    if args.subcommand == "list":
        print(annotations.get_annotations(
            item_key=getattr(args, "item_key", None),
            use_pdf_extraction=args.pdf_extraction,
            limit=args.limit, ctx=ctx,
        ))
    elif args.subcommand == "create":
        print(annotations.create_annotation(
            attachment_key=args.attachment_key, page=args.page,
            text=args.text, comment=getattr(args, "comment", None),
            color=args.color, ctx=ctx,
        ))
    else:
        print(f"Unknown 'annotations' subcommand: {args.subcommand}", file=sys.stderr)
        sys.exit(1)


def cmd_notes(args):
    setup_zotero_environment()
    search_mod, retrieval, annotations, write_mod, _client = _import_tools()
    ctx = _ctx(args)

    if args.subcommand == "list":
        print(annotations.get_notes(
            item_key=getattr(args, "item_key", None), limit=args.limit,
            truncate=not args.full, raw_html=args.raw_html, ctx=ctx,
        ))
    elif args.subcommand == "create":
        note_text = sys.stdin.read() if args.text == "-" else (args.text or "")
        if not note_text:
            print("Error: provide note text via --text TEXT or --text - (reads stdin)",
                  file=sys.stderr)
            sys.exit(1)
        tags = args.tags.split(",") if args.tags else []
        print(annotations.create_note(
            item_key=args.item_key, note_title=args.title or "CLI Note",
            note_text=note_text, tags=tags, ctx=ctx,
        ))
    elif args.subcommand == "update":
        note_text = sys.stdin.read() if args.text == "-" else args.text
        print(annotations.update_note(item_key=args.item_key, note_text=note_text, ctx=ctx))
    elif args.subcommand == "delete":
        print(annotations.delete_note(item_key=args.item_key, ctx=ctx))
    else:
        print(f"Unknown 'notes' subcommand: {args.subcommand}", file=sys.stderr)
        sys.exit(1)


def cmd_add(args):
    setup_zotero_environment()
    search_mod, retrieval, annotations, write_mod, _client = _import_tools()
    ctx = _ctx(args)
    tags = args.tags.split(",") if args.tags else None
    collections = args.collections.split(",") if args.collections else None

    if args.subcommand == "doi":
        print(write_mod.add_by_doi(
            doi=args.doi, collections=collections, tags=tags,
            attach_mode=args.attach_mode, ctx=ctx,
        ))
    elif args.subcommand == "url":
        print(write_mod.add_by_url(
            url=args.url, collections=collections, tags=tags,
            attach_mode=args.attach_mode, ctx=ctx,
        ))
    elif args.subcommand == "file":
        print(write_mod.add_from_file(
            file_path=args.filepath, parent_key=getattr(args, "parent_key", None),
            collections=collections, tags=tags, ctx=ctx,
        ))
    else:
        print(f"Unknown 'add' subcommand: {args.subcommand}", file=sys.stderr)
        sys.exit(1)


def cmd_collections(args):
    setup_zotero_environment()
    search_mod, retrieval, annotations, write_mod, _client = _import_tools()
    ctx = _ctx(args)

    if args.subcommand == "create":
        print(write_mod.create_collection(
            name=args.name, parent_collection=getattr(args, "parent", None), ctx=ctx,
        ))
    elif args.subcommand == "search":
        print(write_mod.search_collections(query=args.query, ctx=ctx))
    elif args.subcommand == "manage":
        print(write_mod.manage_collections(
            item_keys=args.item_keys.split(","),
            add_to=args.add_to.split(",") if args.add_to else None,
            remove_from=args.remove_from.split(",") if args.remove_from else None,
            ctx=ctx,
        ))
    else:
        print(f"Unknown 'collections' subcommand: {args.subcommand}", file=sys.stderr)
        sys.exit(1)


def cmd_tags(args):
    setup_zotero_environment()
    search_mod, retrieval, annotations, write_mod, _client = _import_tools()
    ctx = _ctx(args)
    print(write_mod.batch_update_tags(
        query=args.query or "",
        add_tags=args.add.split(",") if args.add else None,
        remove_tags=args.remove.split(",") if args.remove else None,
        tag=args.tag.split(",") if args.tag else None,
        limit=args.limit,
        ctx=ctx,
    ))


def cmd_edit(args):
    """Edit metadata fields of an existing Zotero item."""
    setup_zotero_environment()
    search_mod, retrieval, annotations, write_mod, _client = _import_tools()
    ctx = _ctx(args)

    creators = None
    if args.creators:
        try:
            creators = json.loads(args.creators)
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON in --creators: {e}", file=sys.stderr)
            sys.exit(1)

    print(write_mod.update_item(
        item_key=args.item_key,
        title=args.title,
        creators=creators,
        date=args.date,
        publication_title=args.publication_title,
        abstract=args.abstract,
        tags=args.tags.split(",") if args.tags else None,
        add_tags=args.add_tags.split(",") if args.add_tags else None,
        remove_tags=args.remove_tags.split(",") if args.remove_tags else None,
        collections=args.collections.split(",") if args.collections else None,
        collection_names=args.collection_names.split(",") if args.collection_names else None,
        doi=args.doi,
        url=args.url,
        extra=args.extra,
        volume=args.volume,
        issue=args.issue,
        pages=args.pages,
        publisher=args.publisher,
        issn=args.issn,
        language=args.language,
        short_title=args.short_title,
        edition=args.edition,
        isbn=args.isbn,
        book_title=args.book_title,
        ctx=ctx,
    ))


def cmd_duplicates(args):
    setup_zotero_environment()
    search_mod, retrieval, annotations, write_mod, _client = _import_tools()
    ctx = _ctx(args)

    if args.subcommand == "find":
        print(write_mod.find_duplicates(
            method=args.method, collection_key=getattr(args, "collection", None),
            limit=args.limit, ctx=ctx,
        ))
    elif args.subcommand == "merge":
        print(write_mod.merge_duplicates(
            keeper_key=args.keeper_key,
            duplicate_keys=args.duplicate_keys.split(","),
            confirm=not args.dry_run, ctx=ctx,
        ))
    else:
        print(f"Unknown 'duplicates' subcommand: {args.subcommand}", file=sys.stderr)
        sys.exit(1)


def cmd_db(args):
    """Manage the semantic search database."""
    from pathlib import Path
    setup_zotero_environment()
    from zotero_mcp.cli import _save_zotero_db_path_to_config
    from zotero_mcp.semantic_search import create_semantic_search

    config_path_arg = getattr(args, "config_path", None)
    config_path = (
        Path(config_path_arg) if config_path_arg
        else Path.home() / ".config" / "zotero-mcp" / "config.json"
    )

    if args.subcommand == "update":
        db_path = getattr(args, "db_path", None)
        if db_path:
            _save_zotero_db_path_to_config(config_path, db_path)
        search = create_semantic_search(str(config_path), db_path=db_path)
        fulltext = getattr(args, "fulltext", False)
        if fulltext:
            from zotero_mcp.utils import is_local_mode
            if not is_local_mode():
                print("Error: --fulltext requires local mode (ZOTERO_LOCAL=true).", file=sys.stderr)
                sys.exit(1)
        stats = search.update_database(
            force_full_rebuild=args.force_rebuild,
            limit=args.limit,
            extract_fulltext=fulltext,
        )
        print("Database update completed:")
        print(f"- Total items: {stats.get('total_items', 0)}")
        print(f"- Processed: {stats.get('processed_items', 0)}")
        print(f"- Added: {stats.get('added_items', 0)}")
        print(f"- Updated: {stats.get('updated_items', 0)}")
        print(f"- Skipped: {stats.get('skipped_items', 0)}")
        print(f"- Errors: {stats.get('errors', 0)}")
        print(f"- Duration: {stats.get('duration', 'Unknown')}")
        if stats.get("error"):
            print(f"Error: {stats['error']}", file=sys.stderr)
            sys.exit(1)

    elif args.subcommand == "status":
        search = create_semantic_search(str(config_path))
        status = search.get_database_status()
        ci = status.get("collection_info", {})
        uc = status.get("update_config", {})
        print("=== Semantic Search Database Status ===")
        print(f"Collection: {ci.get('name', 'Unknown')}")
        print(f"Document count: {ci.get('count', 0)}")
        print(f"Embedding model: {ci.get('embedding_model', 'Unknown')}")
        print(f"Database path: {ci.get('persist_directory', 'Unknown')}")
        print("\nUpdate configuration:")
        print(f"- Auto update: {uc.get('auto_update', False)}")
        print(f"- Frequency: {uc.get('update_frequency', 'manual')}")
        print(f"- Last update: {uc.get('last_update', 'Never')}")
        print(f"- Should update: {status.get('should_update', False)}")
        if ci.get("error"):
            print(f"\nError: {ci['error']}")

    elif args.subcommand == "inspect":
        from collections import Counter
        search = create_semantic_search(str(config_path))
        client = search.chroma_client
        col = client.collection

        if args.stats:
            meta = col.get(include=["metadatas"])
            metas = meta.get("metadatas", [])
            info = client.get_collection_info()
            print("=== Semantic DB Stats ===")
            print(f"Collection: {info.get('name')} @ {info.get('persist_directory')}")
            print(f"Count: {info.get('count')}")
            ct = Counter((m or {}).get("item_type", "") for m in metas)
            print("Item types:")
            for t, c in ct.most_common(20):
                print(f"  {t or '(missing)'}: {c}")
            coverage: dict = {}
            for m in metas:
                m = m or {}
                t = m.get("item_type", "") or "(missing)"
                cov = coverage.setdefault(t, {"total": 0, "with_fulltext": 0, "pdf": 0, "html": 0})
                cov["total"] += 1
                if m.get("has_fulltext"):
                    cov["with_fulltext"] += 1
                    src = (m.get("fulltext_source") or "").lower()
                    if src == "pdf":
                        cov["pdf"] += 1
                    elif src == "html":
                        cov["html"] += 1
            print("Fulltext coverage (by type):")
            for t, cov in coverage.items():
                print(f"  {t}: {cov['with_fulltext']}/{cov['total']} (pdf:{cov['pdf']}, html:{cov['html']})")
            titles = [(m or {}).get("title", "") for m in metas]
            ct_titles = Counter(t for t in titles if t)
            common = ct_titles.most_common(10)
            if common:
                print("Common titles:")
                for t, c in common:
                    print(f"  {t[:80]}{'...' if len(t) > 80 else ''}: {c}")
        else:
            include = ["metadatas"]
            if args.show_documents:
                include.append("documents")
            data = col.get(limit=args.limit, include=include)
            print("=== Semantic DB Inspection ===")
            print(f"Total documents: {client.get_collection_info().get('count', 0)}")
            print(f"Showing up to: {args.limit}")
            shown = 0
            for i, meta in enumerate(data.get("metadatas", [])):
                meta = meta or {}
                title = meta.get("title", "")
                creators = meta.get("creators", "")
                if args.filter_text:
                    needle = args.filter_text.lower()
                    if needle not in (title or "").lower() and needle not in (creators or "").lower():
                        continue
                print(f"- {title} | {creators}")
                if args.show_documents:
                    doc = (data.get("documents", [""])[i] or "").strip()
                    snippet = doc[:200].replace("\n", " ") + ("..." if len(doc) > 200 else "")
                    if snippet:
                        print(f"  doc: {snippet}")
                shown += 1
                if shown >= args.limit:
                    break
            if shown == 0:
                print("No records matched your filter.")
    else:
        print(f"Unknown 'db' subcommand: {args.subcommand}", file=sys.stderr)
        sys.exit(1)


def cmd_library(args):
    setup_zotero_environment()
    search_mod, retrieval, annotations, write_mod, _client = _import_tools()
    ctx = _ctx(args)

    if args.action == "switch":
        print(retrieval.switch_library(
            library_id=args.library_id, library_type=args.library_type, ctx=ctx,
        ))
    elif args.action == "list":
        print(retrieval.list_libraries(ctx=ctx))
    elif args.action == "reset":
        _client.clear_active_library()
        print("Switched back to default library configuration.")
    else:
        print(f"Unknown library action: {args.action}", file=sys.stderr)
        sys.exit(1)


def cmd_outline(args):
    setup_zotero_environment()
    search_mod, retrieval, annotations, write_mod, _client = _import_tools()
    print(write_mod.get_pdf_outline(item_key=args.item_key, ctx=_ctx(args)))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zotero CLI — standalone library access without an MCP server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  zotero-cli search "Smith 2020"\n'
            '  zotero-cli search --mode tag "important,research"\n'
            '  zotero-cli search --mode semantic "machine learning"\n'
            "  zotero-cli get metadata ITEM_KEY\n"
            "  zotero-cli get collections\n"
            "  zotero-cli get recent --limit 20\n"
            "  zotero-cli annotations list --item-key ITEM_KEY\n"
            "  zotero-cli notes create --item-key ITEM_KEY --text -\n"
            "  zotero-cli add doi 10.1234/example\n"
            "  zotero-cli edit ITEM_KEY --title \"New Title\" --add-tags reviewed\n"
            "  zotero-cli db status\n"
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output")

    sub = parser.add_subparsers(dest="command", help="Command to run")

    # config
    cfg_p = sub.add_parser("config", help="Show current Zotero configuration")
    cfg_p.add_argument("--show-secrets", action="store_true", help="Show full API keys")

    # search
    s_p = sub.add_parser("search", help="Search your Zotero library", aliases=["s"])
    s_p.add_argument("query", nargs="?", default="", help="Search query")
    s_p.add_argument("--mode", choices=["items", "tag", "citekey", "advanced", "semantic", "notes"],
                     default="items", help="Search mode (default: items)")
    s_p.add_argument("--qmode", choices=["titleCreatorYear", "everything"],
                     default="titleCreatorYear")
    s_p.add_argument("--collection", help="Scope to a collection key")
    s_p.add_argument("--limit", type=int, default=10)
    s_p.add_argument("--conditions", help='JSON conditions for advanced mode')
    s_p.add_argument("--join-mode", choices=["all", "any"], default="all")
    s_p.add_argument("--sort-by")
    s_p.add_argument("--sort-direction", choices=["asc", "desc"], default="asc")
    s_p.add_argument("--filters", help='JSON filters for semantic mode')

    # get
    g_p = sub.add_parser("get", help="Get items, collections, tags, etc.", aliases=["g"])
    g_sub = g_p.add_subparsers(dest="subcommand")
    gm = g_sub.add_parser("metadata", help="Get item metadata")
    gm.add_argument("item_key")
    gm.add_argument("--no-abstract", action="store_true")
    gm.add_argument("--output-format", choices=["markdown", "bibtex"], default="markdown")
    gf = g_sub.add_parser("fulltext", help="Get full text of an item")
    gf.add_argument("item_key")
    gb = g_sub.add_parser("bibtex", help="Get BibTeX for an item")
    gb.add_argument("item_key")
    gc = g_sub.add_parser("collections", help="List all collections")
    gc.add_argument("--limit", type=int, default=500)
    gci = g_sub.add_parser("collection-items", help="Get items in a collection")
    gci.add_argument("collection_key")
    gci.add_argument("--detail", choices=["keys_only", "summary", "full"], default="summary")
    gci.add_argument("--limit", type=int, default=50)
    gch = g_sub.add_parser("children", help="Get child items (attachments, notes)")
    gch.add_argument("item_key", nargs="?")
    gch.add_argument("--item-keys", help="Comma-separated keys for batch mode")
    gt = g_sub.add_parser("tags", help="List all tags")
    gt.add_argument("--limit", type=int, default=500)
    gr = g_sub.add_parser("recent", help="Get recently added items")
    gr.add_argument("--limit", type=int, default=10)
    gr.add_argument("--collection")
    g_sub.add_parser("libraries", help="List accessible libraries")
    g_sub.add_parser("feeds", help="List RSS feeds")
    gfi = g_sub.add_parser("feed-items", help="Get items from an RSS feed")
    gfi.add_argument("library_id", type=int)
    gfi.add_argument("--limit", type=int, default=20)

    # annotations
    a_p = sub.add_parser("annotations", help="Manage annotations", aliases=["ann"])
    a_sub = a_p.add_subparsers(dest="subcommand")
    al = a_sub.add_parser("list", help="Get annotations")
    al.add_argument("--item-key")
    al.add_argument("--pdf-extraction", action="store_true")
    al.add_argument("--limit", type=int, default=100)
    ac = a_sub.add_parser("create", help="Create an annotation on a PDF/EPUB")
    ac.add_argument("--attachment-key", required=True)
    ac.add_argument("--page", required=True, type=int)
    ac.add_argument("--text", required=True)
    ac.add_argument("--comment")
    ac.add_argument("--color", default="#ffd400")

    # notes
    n_p = sub.add_parser("notes", help="Manage notes", aliases=["n"])
    n_sub = n_p.add_subparsers(dest="subcommand")
    nl = n_sub.add_parser("list", help="List notes")
    nl.add_argument("--item-key")
    nl.add_argument("--limit", type=int, default=20)
    nl.add_argument("--full", action="store_true")
    nl.add_argument("--raw-html", action="store_true")
    nc = n_sub.add_parser("create", help="Create a note")
    nc.add_argument("--item-key", required=True)
    nc.add_argument("--title")
    nc.add_argument("--text", help="Note text (use - to read from stdin)")
    nc.add_argument("--tags")
    nu = n_sub.add_parser("update", help="Update a note")
    nu.add_argument("--item-key", required=True)
    nu.add_argument("--text", help="New text (use - for stdin)")
    nd = n_sub.add_parser("delete", help="Delete a note")
    nd.add_argument("--item-key", required=True)

    # add
    add_p = sub.add_parser("add", help="Add items to your library")
    add_sub = add_p.add_subparsers(dest="subcommand")
    adoi = add_sub.add_parser("doi", help="Add item by DOI")
    adoi.add_argument("doi")
    adoi.add_argument("--collections")
    adoi.add_argument("--tags")
    adoi.add_argument("--attach-mode", choices=["auto", "linked_url", "import_file"], default="auto")
    aurl = add_sub.add_parser("url", help="Add item by URL")
    aurl.add_argument("url")
    aurl.add_argument("--collections")
    aurl.add_argument("--tags")
    aurl.add_argument("--attach-mode", choices=["auto", "linked_url", "import_file"], default="auto")
    afil = add_sub.add_parser("file", help="Add item from local file")
    afil.add_argument("--filepath", required=True)
    afil.add_argument("--parent-key")
    afil.add_argument("--collections")
    afil.add_argument("--tags")

    # collections
    col_p = sub.add_parser("collections", help="Manage collections", aliases=["coll"])
    col_sub = col_p.add_subparsers(dest="subcommand")
    ccs = col_sub.add_parser("create", help="Create a collection")
    ccs.add_argument("name")
    ccs.add_argument("--parent")
    css = col_sub.add_parser("search", help="Search collections by name")
    css.add_argument("query")
    cmg = col_sub.add_parser("manage", help="Add/remove items from collections")
    cmg.add_argument("--item-keys", required=True)
    cmg.add_argument("--add-to")
    cmg.add_argument("--remove-from")

    # tags
    t_p = sub.add_parser("tags", help="Batch update tags on matched items")
    t_p.add_argument("--query")
    t_p.add_argument("--tag")
    t_p.add_argument("--add")
    t_p.add_argument("--remove")
    t_p.add_argument("--limit", type=int, default=50)

    # edit (item metadata)
    e_p = sub.add_parser("edit", help="Edit metadata fields of an existing item")
    e_p.add_argument("item_key")
    e_p.add_argument("--title")
    e_p.add_argument("--creators", help="JSON array of creators")
    e_p.add_argument("--date")
    e_p.add_argument("--publication-title")
    e_p.add_argument("--abstract")
    e_p.add_argument("--tags", help="Replace all tags (comma-separated)")
    e_p.add_argument("--add-tags")
    e_p.add_argument("--remove-tags")
    e_p.add_argument("--collections", help="Add to collections (comma-separated keys)")
    e_p.add_argument("--collection-names", help="Add to collections (comma-separated names)")
    e_p.add_argument("--doi")
    e_p.add_argument("--url")
    e_p.add_argument("--extra")
    e_p.add_argument("--volume")
    e_p.add_argument("--issue")
    e_p.add_argument("--pages")
    e_p.add_argument("--publisher")
    e_p.add_argument("--issn")
    e_p.add_argument("--language")
    e_p.add_argument("--short-title")
    e_p.add_argument("--edition")
    e_p.add_argument("--isbn")
    e_p.add_argument("--book-title")

    # duplicates
    d_p = sub.add_parser("duplicates", help="Find or merge duplicate items")
    d_sub = d_p.add_subparsers(dest="subcommand")
    df = d_sub.add_parser("find")
    df.add_argument("--method", choices=["title", "doi", "both"], default="both")
    df.add_argument("--collection")
    df.add_argument("--limit", type=int, default=50)
    dm = d_sub.add_parser("merge")
    dm.add_argument("--keeper-key", required=True)
    dm.add_argument("--duplicate-keys", required=True)
    dm.add_argument("--dry-run", action="store_true")

    # db
    db_p = sub.add_parser("db", help="Manage the semantic search database")
    db_sub = db_p.add_subparsers(dest="subcommand")
    dbu = db_sub.add_parser("update")
    dbu.add_argument("--force-rebuild", action="store_true")
    dbu.add_argument("--limit", type=int)
    dbu.add_argument("--fulltext", action="store_true")
    dbu.add_argument("--config-path")
    dbu.add_argument("--db-path")
    dbs = db_sub.add_parser("status")
    dbs.add_argument("--config-path")
    dbi = db_sub.add_parser("inspect")
    dbi.add_argument("--limit", type=int, default=20)
    dbi.add_argument("--filter-text")
    dbi.add_argument("--show-documents", action="store_true")
    dbi.add_argument("--stats", action="store_true")
    dbi.add_argument("--config-path")

    # library
    lib_p = sub.add_parser("library", help="Switch or list libraries")
    lib_p.add_argument("action", choices=["switch", "list", "reset"], nargs="?", default="list")
    lib_p.add_argument("--library-id")
    lib_p.add_argument("--library-type", choices=["user", "group"], default="group")

    # outline
    out_p = sub.add_parser("outline", help="Get PDF outline/table of contents")
    out_p.add_argument("item_key")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_CMD_MAP = {
    "config": cmd_config,
    "search": cmd_search, "s": cmd_search,
    "get": cmd_get, "g": cmd_get,
    "annotations": cmd_annotations, "ann": cmd_annotations,
    "notes": cmd_notes, "n": cmd_notes,
    "add": cmd_add,
    "collections": cmd_collections, "coll": cmd_collections,
    "tags": cmd_tags,
    "edit": cmd_edit,
    "duplicates": cmd_duplicates,
    "db": cmd_db,
    "library": cmd_library,
    "outline": cmd_outline,
}


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    handler = _CMD_MAP.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import os
        if os.environ.get("ZOTERO_CLI_DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
