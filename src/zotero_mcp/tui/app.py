"""Zotero TUI — an interactive terminal front-end over the zotero-cli tools.

Layout:

    ┌──────────────┬─────────────────────────────────────────┐
    │ Collections  │  search bar  [mode ▼] [ query........... ]│
    │  tree        │  ┌───────────────────────────────────────┐│
    │              │  │ items table (Type/Year/Authors/Title) ││
    │              │  └───────────────────────────────────────┘│
    │              │  ┌─ detail tabs ─────────────────────────┐│
    │              │  │ Metadata | Full Text | Notes | …       ││
    │              │  └───────────────────────────────────────┘│
    └──────────────┴─────────────────────────────────────────┘
                                                       status bar

All Zotero calls run in worker threads so the UI never blocks.
"""

from __future__ import annotations

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Markdown,
    Select,
    Static,
    TabbedContent,
    TabPane,
    Tree,
)
from textual.widgets.tree import TreeNode

from zotero_mcp.tui.actions import ActionsMixin
from zotero_mcp.tui.data import Collection, ItemRow, ZoteroData

SEARCH_MODES = [
    ("Keyword", "items"),
    ("Tag", "tag"),
    ("Semantic", "semantic"),
]

# Pseudo-nodes shown at the top of the collections tree.
NODE_RECENT = "__recent__"
NODE_ALL = "__all__"


class ZoteroTUI(ActionsMixin, App):
    """Interactive terminal UI for a Zotero library."""

    CSS_PATH = "app.tcss"
    TITLE = "Zotero"

    BINDINGS = [
        ("/", "focus_search", "Search"),
        ("ctrl+r", "refresh", "Refresh"),
        ("a", "add_menu", "Add"),
        ("e", "edit_item", "Edit"),
        ("n", "new_note", "Note"),
        ("t", "tag_item", "Tag"),
        ("c", "collections_menu", "Collections"),
        ("d", "duplicates_menu", "Duplicates"),
        ("b", "db_menu", "Sem. DB"),
        ("L", "library_menu", "Library"),
        ("y", "copy_key", "Copy key"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, data: ZoteroData | None = None) -> None:
        super().__init__()
        self.data = data or ZoteroData()
        self.current_item: str | None = None
        self.last_status: str = ""
        self._detail_cache: dict[tuple[str, str], str] = {}
        self._rows: dict[str, ItemRow] = {}

    # -- layout -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static("COLLECTIONS", id="sidebar-title")
                tree: Tree[str] = Tree("Library", id="collections")
                tree.show_root = False
                yield tree
            with Vertical(id="main"):
                with Horizontal(id="searchbar"):
                    yield Select(
                        SEARCH_MODES, value="items", id="search-mode", allow_blank=False
                    )
                    yield Input(placeholder="Search…  (press / to focus)", id="query")
                yield DataTable(id="results", cursor_type="row", zebra_stripes=True)
                with TabbedContent(id="detail"):
                    with TabPane("Metadata", id="tab-meta"):
                        yield Markdown("", classes="detail-md", id="md-meta")
                    with TabPane("Full Text", id="tab-fulltext"):
                        yield Markdown("", classes="detail-md", id="md-fulltext")
                    with TabPane("Annotations", id="tab-annotations"):
                        yield Markdown("", classes="detail-md", id="md-annotations")
                    with TabPane("Notes", id="tab-notes"):
                        yield Markdown("", classes="detail-md", id="md-notes")
                    with TabPane("Children", id="tab-children"):
                        yield Markdown("", classes="detail-md", id="md-children")
                    with TabPane("BibTeX", id="tab-bibtex"):
                        yield Markdown("", classes="detail-md", id="md-bibtex")
        yield Static("Starting…", id="status")
        yield Footer()

    # -- lifecycle --------------------------------------------------------

    def on_mount(self) -> None:
        table = self.query_one("#results", DataTable)
        table.add_column("Type", width=12)
        table.add_column("Year", width=5)
        table.add_column("Authors", width=22)
        table.add_column("Title")
        table.add_column("#", width=3)
        self.data.set_log_sink(self._on_tool_log)
        self._connect()

    def set_status(self, message: str) -> None:
        self.last_status = message
        self.query_one("#status", Static).update(message)

    def _on_tool_log(self, level: str, message: str) -> None:
        # Called from worker threads — marshal back to the UI thread.
        self.call_from_thread(self.set_status, f"{message}")

    # -- connection / initial load ---------------------------------------

    @work(thread=True, exclusive=True, group="connect")
    def _connect(self) -> None:
        ok, msg = self.data.check_connection()
        lib = self.data.current_library()
        if not ok:
            self.call_from_thread(
                self.set_status,
                f"⚠ Not connected: {msg}  — is Zotero running with the local API enabled?",
            )
            return
        self.call_from_thread(self.set_status, f"✓ {msg}  ·  library {lib}")
        self.call_from_thread(lambda: setattr(self, "sub_title", f"{msg} · {lib}"))
        cols = self.data.collections()
        self.call_from_thread(self._build_tree, cols)
        rows = self.data.recent(limit=50)
        self.call_from_thread(self._populate_table, rows, "Recent items")

    def _build_tree(self, collections: list[Collection]) -> None:
        tree = self.query_one("#collections", Tree)
        tree.clear()
        root = tree.root
        root.add_leaf("📥  Recent", data=NODE_RECENT)
        root.add_leaf("📚  All Items", data=NODE_ALL)
        # Build parent→children map.
        children: dict[str | None, list[Collection]] = {}
        for c in collections:
            children.setdefault(c.parent, []).append(c)

        def add_children(parent_node: TreeNode, parent_key: str | None) -> None:
            for col in sorted(children.get(parent_key, []), key=lambda c: c.name.lower()):
                has_kids = col.key in children
                node = parent_node.add(
                    f"📁  {col.name}", data=col.key, allow_expand=has_kids
                )
                if has_kids:
                    add_children(node, col.key)

        add_children(root, None)
        root.expand()

    # -- table ------------------------------------------------------------

    def _populate_table(self, rows: list[ItemRow], context: str = "") -> None:
        table = self.query_one("#results", DataTable)
        table.clear()
        self._rows = {}
        for r in rows:
            self._rows[r.key] = r
            score = r.extra.get("score")
            type_cell = f"{r.item_type}"
            year = r.year or "—"
            authors = (r.creators or "—")[:22]
            title = r.title or "(untitled)"
            if score:
                title = f"[{score}] {title}"
            table.add_row(type_cell, year, authors, title, str(r.tags), key=r.key)
        n = len(rows)
        suffix = f" · {context}" if context else ""
        self.set_status(f"{n} item{'s' if n != 1 else ''}{suffix}")
        if rows:
            table.focus()
            # Loading detail for the first row happens via RowHighlighted.

    @on(DataTable.RowHighlighted, "#results")
    def _row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        key = event.row_key.value if event.row_key else None
        if not key or key == self.current_item:
            return
        self.current_item = key
        self._detail_cache.clear()
        self._load_active_tab()

    # -- detail tabs ------------------------------------------------------

    @on(TabbedContent.TabActivated, "#detail")
    def _tab_activated(self, event: TabbedContent.TabActivated) -> None:
        self._load_active_tab()

    _TAB_LOADERS = {
        "tab-meta": ("md-meta", "metadata"),
        "tab-fulltext": ("md-fulltext", "fulltext"),
        "tab-annotations": ("md-annotations", "annotations"),
        "tab-notes": ("md-notes", "notes"),
        "tab-children": ("md-children", "children"),
        "tab-bibtex": ("md-bibtex", "bibtex"),
    }

    def _load_active_tab(self) -> None:
        if not self.current_item:
            return
        tabs = self.query_one("#detail", TabbedContent)
        active = tabs.active
        spec = self._TAB_LOADERS.get(active)
        if not spec:
            return
        md_id, method = spec
        cache_key = (self.current_item, method)
        widget = self.query_one(f"#{md_id}", Markdown)
        if cache_key in self._detail_cache:
            widget.update(self._detail_cache[cache_key])
            return
        widget.update("_Loading…_")
        self._load_detail(self.current_item, method, md_id)

    @work(thread=True, group="detail")
    def _load_detail(self, key: str, method: str, md_id: str) -> None:
        try:
            content = getattr(self.data, method)(key)
        except Exception as e:  # noqa: BLE001
            content = f"**Error loading {method}:** {e}"
        is_bibtex = method == "bibtex"
        text = f"```\n{content}\n```" if is_bibtex else content
        # Ignore if the user moved on to a different item.
        if self.current_item != key:
            return
        self._detail_cache[(key, method)] = text
        self.call_from_thread(self.query_one(f"#{md_id}", Markdown).update, text)

    # -- search -----------------------------------------------------------

    def action_focus_search(self) -> None:
        self.query_one("#query", Input).focus()

    @on(Input.Submitted, "#query")
    def _do_search(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        mode = self.query_one("#search-mode", Select).value
        self.set_status(f"Searching ({mode}): {query}…")
        self._run_search(query, str(mode))

    @work(thread=True, exclusive=True, group="search")
    def _run_search(self, query: str, mode: str) -> None:
        try:
            if mode == "tag":
                rows = self.data.search_by_tag([t.strip() for t in query.split(",")], limit=100)
                ctx = f"tag: {query}"
            elif mode == "semantic":
                rows = self.data.semantic_search(query, limit=30)
                ctx = f"semantic: {query}"
            else:
                rows = self.data.search(query, limit=100)
                ctx = f"keyword: {query}"
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self.set_status, f"⚠ Search failed: {e}")
            return
        self.call_from_thread(self._populate_table, rows, ctx)

    # -- tree navigation --------------------------------------------------

    @on(Tree.NodeSelected, "#collections")
    def _node_selected(self, event: Tree.NodeSelected) -> None:
        data = event.node.data
        if data is None:
            return
        if data == NODE_RECENT:
            self.set_status("Loading recent…")
            self._load_collection(None, "Recent items")
        elif data == NODE_ALL:
            self.set_status("Loading all items…")
            self._load_collection(None, "All items", all_items=True)
        else:
            self.set_status(f"Loading collection {event.node.label}…")
            self._load_collection(str(data), str(event.node.label))

    @work(thread=True, exclusive=True, group="browse")
    def _load_collection(self, key: str | None, context: str, all_items: bool = False) -> None:
        try:
            if key is None and all_items:
                rows = self.data.recent(limit=300)
            elif key is None:
                rows = self.data.recent(limit=50)
            else:
                rows = self.data.collection_items(key, limit=300)
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self.set_status, f"⚠ Load failed: {e}")
            return
        self.call_from_thread(self._populate_table, rows, context)

    # -- misc actions -----------------------------------------------------

    def action_refresh(self) -> None:
        self.set_status("Refreshing…")
        self._connect()

    def action_copy_key(self) -> None:
        if self.current_item:
            try:
                self.copy_to_clipboard(self.current_item)
            except Exception:  # noqa: BLE001
                pass
            self.set_status(f"Item key: {self.current_item} (copied)")

    # The write-action methods (add/edit/note/tag/collections/duplicates/db/
    # library menus) are mixed in from actions.py.

    def selected_row(self) -> ItemRow | None:
        if self.current_item:
            return self._rows.get(self.current_item)
        return None

    def show_result(self, title: str, body: str) -> None:
        """Display a markdown result (e.g. write-action output) in a modal."""
        from zotero_mcp.tui.modals import ResultModal

        self.push_screen(ResultModal(title, body))


def main() -> None:
    """Console entry point for ``zotero-tui``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="zotero-tui",
        description="Interactive terminal UI for your Zotero library.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose tool logging")
    args = parser.parse_args()

    data = ZoteroData(verbose=args.verbose)
    ZoteroTUI(data=data).run()


if __name__ == "__main__":
    main()
