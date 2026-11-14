"""The LocalZoteroReader surface the local ``--fulltext`` scan uses (#826).

Test doubles of the reader subclass :class:`ScanReader` and say which files
each item has; the scan then resolves, extracts and keys them by attachment
the way the real reader does, without touching a disk or a database.
"""

from __future__ import annotations

from pathlib import Path

from zotero_mcp.extract import DEFAULT_ATTACHMENT_PRIORITY
from zotero_mcp.local_db import ExtractionTarget


def target(key: str, stamp: str = "1:100", category: str = "pdf") -> ExtractionTarget:
    """A file that does not exist on disk."""
    return ExtractionTarget(
        key=key,
        path=Path("/nonexistent") / f"{key}.pdf",
        content_type="application/pdf",
        category=category,
        file_stamp=stamp,
        size=100,
    )


class ScanReader:
    """Files per item, their text, and standalone files.

    ``targets``: item_id -> files the priority selects for it.
    ``texts``: attachment key -> ``(text, source)``, or None for a file
    that yields nothing. A missing key yields ``("extracted text", "pdf")``.
    ``standalone``: ``(key, path, content_type, date_modified, library_id)``
    rows, each resolving to ``standalone_targets[key]``.
    """

    attachment_priority = DEFAULT_ATTACHMENT_PRIORITY

    def __init__(self, targets=None, texts=None, standalone=(), standalone_targets=None, dates=None, rows=None):
        self.targets = dict(targets or {})
        # item_id -> the item's attachment rows in Zotero's database as [key, path, content type],
        # for files that are not on disk (so have no target); by default the targets'.
        self.rows = dict(rows or {})
        self.texts = dict(texts or {})
        self.standalone = list(standalone)
        self.standalone_targets = dict(standalone_targets or {})
        self.dates = dict(dates or {})
        self.extracted: list[str] = []

    @property
    def extract_calls(self) -> int:
        return len(self.extracted)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_date_modified_by_key(self):
        return dict(self.dates)

    def _resolve_extraction_targets(self, item_id):
        return list(self.targets.get(item_id, []))

    def _iter_standalone_attachments(self, library_ids=None):
        return iter(self.standalone)

    def _resolve_standalone_target(self, key, path, ctype):
        return self.standalone_targets.get(key)

    def get_collection_item_keys(self, collection_keys):
        return set()

    def get_fulltext_meta_for_item(self, item_id):
        if item_id in self.rows:
            return [list(row) for row in self.rows[item_id]]
        return [[t.key, str(t.path), t.content_type] for t in self.targets.get(item_id, [])]

    def extract_targets(self, targets):
        for t in targets:
            self.extracted.append(t.key)
            yield t.key, self.texts.get(t.key, ("extracted text", "pdf"))
