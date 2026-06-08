"""Write-action handlers for the Zotero TUI.

Mixed into :class:`ZoteroTUI`. Each ``action_*`` method opens a form modal,
then runs the corresponding tool function in a worker thread and shows the
markdown result. This is where CLI parity for the mutating commands lives
(add / edit / notes / tags / collections / duplicates / semantic DB / library).
"""

from __future__ import annotations

from collections.abc import Callable

from zotero_mcp.tui.modals import ConfirmModal, Field, FormModal


def _csv(value: str) -> list[str] | None:
    value = (value or "").strip()
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


class ActionsMixin:
    """Provides the write-action key handlers for the app."""

    # -- generic threaded runner -----------------------------------------

    def _run_action(self, title: str, fn: Callable[[], str]) -> None:
        self.set_status(f"{title}…")

        def task() -> None:
            try:
                result = fn()
            except Exception as e:  # noqa: BLE001
                result = f"**Error:** {e}"
            self.call_from_thread(self.show_result, title, result)
            self.call_from_thread(self.set_status, f"{title} — done")
            # Refresh detail/table state that may have changed.
            self.call_from_thread(self._after_write)

        self.run_worker(task, thread=True, group="action")

    def _after_write(self) -> None:
        # Invalidate the detail cache so panes reflect any mutation.
        self._detail_cache.clear()
        self._load_active_tab()

    def _require_item(self) -> str | None:
        row = self.selected_row()
        if not row:
            self.set_status("⚠ Select an item first.")
            self.bell()
            return None
        return row.key

    # -- add --------------------------------------------------------------

    def action_add_menu(self) -> None:
        fields = [
            Field("kind", "Add by", kind="select", required=True, value="doi",
                  options=[("DOI", "doi"), ("URL", "url"), ("Local file", "file")]),
            Field("value", "DOI / URL / file path", required=True,
                  placeholder="10.1234/example  ·  https://arxiv.org/abs/…  ·  /path/to.pdf"),
            Field("collections", "Collections (comma-sep keys)"),
            Field("tags", "Tags (comma-sep)"),
        ]
        self.push_screen(FormModal("Add item", fields, "Add"), self._on_add)

    def _on_add(self, res: dict | None) -> None:
        if not res:
            return
        kind, value = res["kind"], res["value"]
        cols, tags = _csv(res["collections"]), _csv(res["tags"])
        adder = {
            "doi": self.data.add_by_doi,
            "url": self.data.add_by_url,
            "file": self.data.add_from_file,
        }.get(kind, self.data.add_from_file)
        self._run_action(
            f"Add by {kind}",
            lambda: adder(value, collections=cols, tags=tags),
        )

    # -- edit -------------------------------------------------------------

    def action_edit_item(self) -> None:
        key = self._require_item()
        if not key:
            return
        row = self.selected_row()
        fields = [
            Field("title", "Title", value=row.title if row else ""),
            Field("date", "Date"),
            Field("abstract", "Abstract"),
            Field("add_tags", "Add tags (comma-sep)"),
            Field("remove_tags", "Remove tags (comma-sep)"),
            Field("collections", "Add to collections (comma-sep keys)"),
            Field("doi", "DOI"),
            Field("url", "URL"),
        ]
        self.push_screen(FormModal(f"Edit {key}", fields, "Save"),
                         lambda res: self._on_edit(key, res))

    def _on_edit(self, key: str, res: dict | None) -> None:
        if not res:
            return
        payload = {}
        if res["title"]:
            payload["title"] = res["title"]
        if res["date"]:
            payload["date"] = res["date"]
        if res["abstract"]:
            payload["abstract"] = res["abstract"]
        if res["doi"]:
            payload["doi"] = res["doi"]
        if res["url"]:
            payload["url"] = res["url"]
        if _csv(res["add_tags"]):
            payload["add_tags"] = _csv(res["add_tags"])
        if _csv(res["remove_tags"]):
            payload["remove_tags"] = _csv(res["remove_tags"])
        if _csv(res["collections"]):
            payload["collections"] = _csv(res["collections"])
        if not payload:
            self.set_status("Nothing to update.")
            return
        self._run_action("Edit item", lambda: self.data.update_item(key, **payload))

    # -- notes ------------------------------------------------------------

    def action_new_note(self) -> None:
        key = self._require_item()
        if not key:
            return
        fields = [
            Field("title", "Note title", value="TUI Note"),
            Field("text", "Note text", required=True),
            Field("tags", "Tags (comma-sep)"),
        ]
        self.push_screen(FormModal(f"New note on {key}", fields, "Create"),
                         lambda res: self._on_note(key, res))

    def _on_note(self, key: str, res: dict | None) -> None:
        if not res:
            return
        self._run_action(
            "Create note",
            lambda: self.data.create_note(
                key, res["text"], note_title=res["title"] or "TUI Note", tags=_csv(res["tags"])
            ),
        )

    # -- tags -------------------------------------------------------------

    def action_tag_item(self) -> None:
        key = self._require_item()
        if not key:
            return
        fields = [
            Field("add", "Add tags (comma-sep)"),
            Field("remove", "Remove tags (comma-sep)"),
        ]
        self.push_screen(FormModal(f"Tag {key}", fields, "Apply"),
                         lambda res: self._on_tag(key, res))

    def _on_tag(self, key: str, res: dict | None) -> None:
        if not res:
            return
        add, remove = _csv(res["add"]), _csv(res["remove"])
        if not add and not remove:
            self.set_status("No tag changes.")
            return
        payload = {}
        if add:
            payload["add_tags"] = add
        if remove:
            payload["remove_tags"] = remove
        self._run_action("Update tags", lambda: self.data.update_item(key, **payload))

    # -- collections ------------------------------------------------------

    def action_collections_menu(self) -> None:
        fields = [Field("op", "Operation", kind="select", required=True, value="create",
                        options=[("Create", "create"), ("Search", "search"),
                                 ("Add/remove selected item", "manage")])]
        self.push_screen(FormModal("Collections", fields, "Next"), self._on_collections_op)

    def _on_collections_op(self, res: dict | None) -> None:
        if not res:
            return
        op = res["op"]
        if op == "create":
            f = [Field("name", "Collection name", required=True),
                 Field("parent", "Parent collection key (optional)")]
            self.push_screen(FormModal("Create collection", f, "Create"), lambda r: (
                r and self._run_action(
                    "Create collection",
                    lambda: self.data.create_collection(r["name"], parent_collection=r["parent"] or None),
                )
            ))
        elif op == "search":
            f = [Field("query", "Name contains", required=True)]
            self.push_screen(FormModal("Search collections", f, "Search"), lambda r: (
                r and self._run_action("Search collections",
                                       lambda: self.data.search_collections(r["query"]))
            ))
        else:  # manage
            key = self._require_item()
            if not key:
                return
            f = [Field("add_to", "Add to collections (comma-sep keys)"),
                 Field("remove_from", "Remove from collections (comma-sep keys)")]
            self.push_screen(FormModal(f"Manage collections for {key}", f, "Apply"), lambda r: (
                r and self._run_action(
                    "Manage collections",
                    lambda: self.data.manage_collections(
                        [key], add_to=_csv(r["add_to"]), remove_from=_csv(r["remove_from"])
                    ),
                )
            ))

    # -- duplicates -------------------------------------------------------

    def action_duplicates_menu(self) -> None:
        fields = [Field("op", "Operation", kind="select", required=True, value="find",
                        options=[("Find duplicates", "find"), ("Merge", "merge")])]
        self.push_screen(FormModal("Duplicates", fields, "Next"), self._on_duplicates_op)

    def _on_duplicates_op(self, res: dict | None) -> None:
        if not res:
            return
        if res["op"] == "find":
            f = [Field("method", "Method", kind="select", value="both",
                       options=[("Title + DOI", "both"), ("Title", "title"), ("DOI", "doi")]),
                 Field("limit", "Limit", value="50")]
            self.push_screen(FormModal("Find duplicates", f, "Find"), lambda r: (
                r and self._run_action(
                    "Find duplicates",
                    lambda: self.data.find_duplicates(
                        method=r["method"] or "both", limit=int(r["limit"] or 50)
                    ),
                )
            ))
        else:
            f = [Field("keeper", "Keeper item key", required=True),
                 Field("dupes", "Duplicate keys (comma-sep)", required=True),
                 Field("confirm", "Type 'yes' to actually merge (else dry-run)")]
            self.push_screen(FormModal("Merge duplicates", f, "Merge"), lambda r: (
                r and self._run_action(
                    "Merge duplicates",
                    lambda: self.data.merge_duplicates(
                        r["keeper"], _csv(r["dupes"]) or [],
                        confirm=r["confirm"].strip().lower() == "yes",
                    ),
                )
            ))

    # -- semantic DB ------------------------------------------------------

    def action_db_menu(self) -> None:
        fields = [Field("op", "Operation", kind="select", required=True, value="status",
                        options=[("Status", "status"), ("Update (incremental)", "update"),
                                 ("Rebuild (full)", "rebuild")])]
        self.push_screen(FormModal("Semantic search database", fields, "Run"), self._on_db_op)

    def _on_db_op(self, res: dict | None) -> None:
        if not res:
            return
        op = res["op"]
        if op == "status":
            self._run_action("Semantic DB status", self.data.db_status)
        elif op == "update":
            self._run_action("Semantic DB update", lambda: self.data.db_update(force_rebuild=False))
        else:
            self.push_screen(
                ConfirmModal("Rebuild semantic DB",
                             "Full rebuild re-embeds every item and can take minutes. Continue?"),
                lambda ok: ok and self._run_action(
                    "Semantic DB rebuild", lambda: self.data.db_update(force_rebuild=True)
                ),
            )

    # -- library ----------------------------------------------------------

    def action_library_menu(self) -> None:
        fields = [Field("op", "Operation", kind="select", required=True, value="list",
                        options=[("List libraries", "list"), ("Switch", "switch"),
                                 ("Reset to default", "reset")])]
        self.push_screen(FormModal("Library", fields, "Next"), self._on_library_op)

    def _on_library_op(self, res: dict | None) -> None:
        if not res:
            return
        op = res["op"]
        if op == "list":
            self._run_action("Libraries", self.data.list_libraries)
        elif op == "reset":
            self._run_action("Reset library", self.data.reset_library)
            self.action_refresh()
        else:
            f = [Field("library_id", "Library ID", required=True),
                 Field("library_type", "Type", kind="select", value="group",
                       options=[("Group", "group"), ("User", "user")])]
            self.push_screen(FormModal("Switch library", f, "Switch"), lambda r: (
                r and self._switch_library(r)
            ))

    def _switch_library(self, r: dict) -> None:
        self._run_action(
            "Switch library",
            lambda: self.data.switch_library(r["library_id"], r["library_type"] or "group"),
        )
        self.action_refresh()
