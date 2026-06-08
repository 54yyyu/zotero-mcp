"""Data layer for the Zotero TUI.

Two kinds of access sit here:

* **Structured accessors** (``collections``, ``recent``, ``search``, …) call
  ``get_zotero_client()`` directly and return raw pyzotero item dicts so the
  TUI can render selectable tables and trees. These bypass only the final
  markdown formatter in the tool layer — the actual querying/normalization is
  shared.
* **Markdown / action wrappers** (``metadata``, ``fulltext``, ``add_by_doi``,
  …) are thin pass-throughs over the existing tool functions in
  ``zotero_mcp.tools`` (the same ones ``zotero-cli`` drives), invoked with a
  lightweight :class:`CLIContext`. This keeps the fallback cascade, citation
  resolution and write logic in one place.

Every method is synchronous and may block on the Zotero API; the TUI calls
them from worker threads so the UI stays responsive.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Context shim (mirrors cli_standalone.CLIContext, but buffers messages so the
# TUI can surface them instead of writing to stderr).
# ---------------------------------------------------------------------------


class TUIContext:
    """Drop-in replacement for fastmcp.Context used by tool functions.

    Tool functions call ``ctx.info / .warning / .error``. We buffer the most
    recent messages so the app can show them in a status bar / log, and also
    mirror warnings and errors to stderr for debugging.
    """

    def __init__(self, sink=None, verbose: bool = False) -> None:
        self._sink = sink  # optional callable(level: str, message: str)
        self._verbose = verbose

    def _emit(self, level: str, message: str) -> None:
        if self._sink is not None:
            try:
                self._sink(level, message)
            except Exception:
                pass
        if level != "info" or self._verbose:
            print(f"[{level.upper()}] {message}", file=sys.stderr)

    def info(self, message: str) -> None:
        self._emit("info", message)

    def warning(self, message: str) -> None:
        self._emit("warning", message)

    def error(self, message: str) -> None:
        self._emit("error", message)


# ---------------------------------------------------------------------------
# Lightweight value objects
# ---------------------------------------------------------------------------


@dataclass
class Collection:
    key: str
    name: str
    parent: str | None  # parent collection key, or None for top-level


@dataclass
class ItemRow:
    """Flattened view of a Zotero item for table display."""

    key: str
    item_type: str
    title: str
    creators: str
    year: str
    tags: int
    extra: dict[str, Any] = field(default_factory=dict)  # e.g. similarity score
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data facade
# ---------------------------------------------------------------------------


class ZoteroData:
    """Facade the TUI uses for all Zotero access."""

    def __init__(self, *, verbose: bool = False) -> None:
        from zotero_mcp.cli import setup_zotero_environment

        # Configure ZOTERO_* env vars from config / Claude Desktop (idempotent).
        setup_zotero_environment()

        # Imported after env setup; cheap (no heavy ML deps until semantic search).
        from zotero_mcp import client as _client
        from zotero_mcp import utils as _utils
        from zotero_mcp.tools import annotations, retrieval, search, write

        self._client = _client
        self._utils = _utils
        self._search = search
        self._retrieval = retrieval
        self._annotations = annotations
        self._write = write
        self.verbose = verbose
        self._log_sink = None

    # -- context / logging ------------------------------------------------

    def set_log_sink(self, sink) -> None:
        """Register a callable(level, message) to receive tool-context logs."""
        self._log_sink = sink

    def _ctx(self) -> TUIContext:
        return TUIContext(sink=self._log_sink, verbose=self.verbose)

    # -- connection -------------------------------------------------------

    def check_connection(self) -> tuple[bool, str]:
        """Return (ok, message). Cheap probe used at startup."""
        try:
            zot = self._client.get_zotero_client()
            zot.add_parameters(limit=1)
            zot.items()
            mode = "local" if self._utils.is_local_mode() else "web API"
            return True, f"Connected ({mode})"
        except Exception as e:  # noqa: BLE001
            return False, str(e)

    def current_library(self) -> str:
        try:
            override = getattr(self._client, "_active_library_override", {}) or {}
            if override.get("library_id"):
                return f"{override.get('library_type', 'user')}:{override['library_id']}"
            import os

            return f"{os.getenv('ZOTERO_LIBRARY_TYPE', 'user')}:{os.getenv('ZOTERO_LIBRARY_ID', '?')}"
        except Exception:  # noqa: BLE001
            return "?"

    # -- row helpers ------------------------------------------------------

    def _to_row(self, item: dict, extra: dict | None = None) -> ItemRow:
        data = item.get("data", {}) or {}
        meta = item.get("meta", {}) or {}
        parsed = meta.get("parsedDate") or data.get("date") or ""
        year = str(parsed)[:4] if parsed else ""
        creators = self._utils.format_creators(data.get("creators", []) or [])
        title = data.get("title") or data.get("note") or data.get("caption") or "(untitled)"
        # Notes carry HTML in `note`; show a short plaintext-ish preview.
        if data.get("itemType") == "note" and not data.get("title"):
            title = _strip_html(data.get("note", ""))[:80] or "(note)"
        return ItemRow(
            key=item.get("key", ""),
            item_type=data.get("itemType", ""),
            title=title,
            creators=creators,
            year=year,
            tags=len(data.get("tags", []) or []),
            extra=extra or {},
            raw=item,
        )

    # -- structured reads (tables / tree) ---------------------------------

    def collections(self) -> list[Collection]:
        zot = self._client.get_zotero_client()
        raw = zot.everything(zot.collections())
        out: list[Collection] = []
        for c in raw:
            d = c.get("data", {}) or {}
            parent = d.get("parentCollection")
            out.append(
                Collection(
                    key=c.get("key", ""),
                    name=d.get("name", "(unnamed)"),
                    parent=parent if parent else None,
                )
            )
        return out

    def recent(self, limit: int = 50, collection_key: str | None = None) -> list[ItemRow]:
        zot = self._client.get_zotero_client()
        if collection_key:
            zot.add_parameters(
                sort="dateAdded", direction="desc", limit=limit, itemType="-attachment"
            )
            items = zot.collection_items_top(collection_key)
        else:
            zot.add_parameters(
                sort="dateAdded", direction="desc", limit=limit, itemType="-attachment"
            )
            items = zot.items()
        return [self._to_row(it) for it in items]

    def collection_items(self, collection_key: str, limit: int = 200) -> list[ItemRow]:
        zot = self._client.get_zotero_client()
        zot.add_parameters(limit=limit, itemType="-attachment", sort="dateAdded", direction="desc")
        items = zot.collection_items_top(collection_key)
        return [self._to_row(it) for it in items]

    def search(self, query: str, qmode: str = "titleCreatorYear", limit: int = 50) -> list[ItemRow]:
        """Keyword search returning structured rows (reuses variant logic)."""
        zot = self._client.get_zotero_client()
        items = self._search._search_with_variants(
            zot, query, qmode, limit, item_type="-attachment"
        )
        return [self._to_row(it) for it in items]

    def search_by_tag(self, tags: list[str], limit: int = 50) -> list[ItemRow]:
        zot = self._client.get_zotero_client()
        zot.add_parameters(q="", tag=tags, itemType="-attachment", limit=limit)
        items = zot.items()
        return [self._to_row(it) for it in items]

    def semantic_search(self, query: str, limit: int = 25) -> list[ItemRow]:
        """Semantic search returning rows with a similarity score in `extra`.

        Raises RuntimeError with a friendly message if the [semantic] extras
        or the database aren't available.
        """
        from pathlib import Path

        try:
            from zotero_mcp.semantic_search import create_semantic_search
        except ImportError as e:  # noqa: BLE001
            raise RuntimeError(
                "Semantic search needs the [semantic] extra: "
                "pip install zotero-mcp-server[semantic]"
            ) from e

        config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
        sem = create_semantic_search(str(config_path))
        results = sem.search(query=query, limit=limit)
        if results.get("error"):
            raise RuntimeError(results["error"])
        rows: list[ItemRow] = []
        for r in results.get("results", []):
            item = r.get("zotero_item") or {}
            if "key" not in item:
                item["key"] = r.get("item_key", "")
            score = r.get("similarity_score")
            extra = {"score": f"{score:.3f}"} if isinstance(score, (int, float)) else {}
            rows.append(self._to_row(item, extra=extra))
        return rows

    # -- markdown detail panes (via tool functions) -----------------------

    def metadata(self, key: str, fmt: str = "markdown") -> str:
        return self._retrieval.get_item_metadata(item_key=key, format=fmt, ctx=self._ctx())

    def fulltext(self, key: str) -> str:
        return self._retrieval.get_item_fulltext(item_key=key, ctx=self._ctx())

    def children(self, key: str) -> str:
        return self._retrieval.get_item_children(item_key=key, ctx=self._ctx())

    def annotations(self, key: str, use_pdf: bool = True, limit: int = 100) -> str:
        return self._annotations.get_annotations(
            item_key=key, use_pdf_extraction=use_pdf, limit=limit, ctx=self._ctx()
        )

    def notes(self, key: str, limit: int = 20) -> str:
        return self._annotations.get_notes(
            item_key=key, limit=limit, truncate=False, ctx=self._ctx()
        )

    def bibtex(self, key: str) -> str:
        return self._retrieval.get_item_metadata(item_key=key, format="bibtex", ctx=self._ctx())

    def outline(self, key: str) -> str:
        return self._write.get_pdf_outline(item_key=key, ctx=self._ctx())

    # -- searches that stay as markdown (less navigation-critical) ---------

    def advanced_search(self, conditions: list[dict], join_mode: str = "all", limit: int = 50) -> str:
        return self._search.advanced_search(
            conditions=conditions, join_mode=join_mode, limit=limit, ctx=self._ctx()
        )

    def search_by_citation_key(self, citekey: str) -> str:
        return self._search.search_by_citation_key(citekey=citekey, ctx=self._ctx())

    def search_notes(self, query: str, limit: int = 20) -> str:
        return self._annotations.search_notes(query=query, limit=limit, ctx=self._ctx())

    # -- writes (return markdown result strings) --------------------------

    def add_by_doi(self, doi: str, collections=None, tags=None, attach_mode="auto") -> str:
        return self._write.add_by_doi(
            doi=doi, collections=collections, tags=tags, attach_mode=attach_mode, ctx=self._ctx()
        )

    def add_by_url(self, url: str, collections=None, tags=None, attach_mode="auto") -> str:
        return self._write.add_by_url(
            url=url, collections=collections, tags=tags, attach_mode=attach_mode, ctx=self._ctx()
        )

    def add_from_file(self, file_path: str, parent_key=None, collections=None, tags=None) -> str:
        return self._write.add_from_file(
            file_path=file_path, parent_key=parent_key, collections=collections,
            tags=tags, ctx=self._ctx(),
        )

    def update_item(self, item_key: str, **fields) -> str:
        return self._write.update_item(item_key=item_key, ctx=self._ctx(), **fields)

    def create_note(self, item_key: str, note_text: str, note_title="TUI Note", tags=None) -> str:
        return self._annotations.create_note(
            item_key=item_key, note_title=note_title, note_text=note_text,
            tags=tags or [], ctx=self._ctx(),
        )

    def update_note(self, item_key: str, note_text: str) -> str:
        return self._annotations.update_note(item_key=item_key, note_text=note_text, ctx=self._ctx())

    def delete_note(self, item_key: str) -> str:
        return self._annotations.delete_note(item_key=item_key, ctx=self._ctx())

    def create_annotation(self, attachment_key: str, page: int, text: str, comment=None,
                          color="#ffd400") -> str:
        return self._annotations.create_annotation(
            attachment_key=attachment_key, page=page, text=text, comment=comment,
            color=color, ctx=self._ctx(),
        )

    def batch_update_tags(self, query="", add_tags=None, remove_tags=None, tag=None, limit=50) -> str:
        return self._write.batch_update_tags(
            query=query, add_tags=add_tags, remove_tags=remove_tags, tag=tag,
            limit=limit, ctx=self._ctx(),
        )

    def create_collection(self, name: str, parent_collection=None) -> str:
        return self._write.create_collection(
            name=name, parent_collection=parent_collection, ctx=self._ctx()
        )

    def search_collections(self, query: str) -> str:
        return self._write.search_collections(query=query, ctx=self._ctx())

    def manage_collections(self, item_keys: list[str], add_to=None, remove_from=None) -> str:
        return self._write.manage_collections(
            item_keys=item_keys, add_to=add_to, remove_from=remove_from, ctx=self._ctx()
        )

    def find_duplicates(self, method="both", collection_key=None, limit=50) -> str:
        return self._write.find_duplicates(
            method=method, collection_key=collection_key, limit=limit, ctx=self._ctx()
        )

    def merge_duplicates(self, keeper_key: str, duplicate_keys: list[str], confirm=False) -> str:
        return self._write.merge_duplicates(
            keeper_key=keeper_key, duplicate_keys=duplicate_keys, confirm=confirm, ctx=self._ctx()
        )

    # -- semantic DB management ------------------------------------------

    def db_status(self) -> str:
        return self._search.get_search_database_status(ctx=self._ctx())

    def db_update(self, force_rebuild=False, limit=None) -> str:
        return self._search.update_search_database(
            force_rebuild=force_rebuild, limit=limit, ctx=self._ctx()
        )

    # -- libraries --------------------------------------------------------

    def list_libraries(self) -> str:
        return self._retrieval.list_libraries(ctx=self._ctx())

    def switch_library(self, library_id: str, library_type: str = "group") -> str:
        return self._retrieval.switch_library(
            library_id=library_id, library_type=library_type, ctx=self._ctx()
        )

    def reset_library(self) -> str:
        self._client.clear_active_library()
        return "Switched back to the default library configuration."


def _strip_html(text: str) -> str:
    """Very small HTML→text helper for note previews."""
    import re

    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    return re.sub(r"\s+", " ", text).strip()
