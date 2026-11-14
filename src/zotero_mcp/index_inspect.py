"""``db-inspect`` / ``zotero-cli db inspect`` for an item-keyed index (#826).

New records store no title, creators or item type, so those are read live
from the local Zotero database by the record's key. Without it the command
still lists record kinds, which come from the stored fields, and matches
legacy records (which still carry their fields) on those.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterator
from typing import Any

from .chroma_client import record_key
from .utils import format_creators

logger = logging.getLogger(__name__)

# Each page of ``col.get`` binds one SQLite parameter per row; a few hundred
# stays far under the ceiling whatever the collection's size.
BATCH_SIZE = 500

KIND_BIBLIOGRAPHIC = "bibliographic"
KIND_ATTACHMENT = "attachment"
KIND_PASSAGE = "passage"
KIND_LEGACY = "legacy"
_KINDS = (KIND_BIBLIOGRAPHIC, KIND_ATTACHMENT, KIND_PASSAGE, KIND_LEGACY)

_NEEDS_ZOTERO = "cannot be matched without the local Zotero database"


def record_kind(doc_id: str, meta: dict[str, Any]) -> str:
    """What a record is, from its id and stored fields only.

    A record carrying ``title`` was written by an earlier version, whatever
    its id; new records never store one.
    """
    if meta.get("title") is not None:
        return KIND_LEGACY
    if "#" in doc_id:
        return KIND_PASSAGE
    return KIND_ATTACHMENT if meta.get("file_stamp") else KIND_BIBLIOGRAPHIC


def _iter_records(client: Any, include_documents: bool) -> Iterator[tuple[str, dict[str, Any], str | None]]:
    include = ["metadatas", "documents"] if include_documents else ["metadatas"]
    for col in client.collections():
        total = col.count()
        offset = 0
        while offset < total:
            batch = col.get(limit=BATCH_SIZE, offset=offset, include=include)
            ids = batch.get("ids") or []
            if not ids:
                break
            metas = batch.get("metadatas") or [None] * len(ids)
            docs = (batch.get("documents") or [None] * len(ids)) if include_documents else [None] * len(ids)
            for doc_id, meta, doc in zip(ids, metas, docs):
                yield doc_id, (meta or {}), doc
            offset += BATCH_SIZE


class _Live:
    """Item type, title and creators of record keys, read from Zotero once each.

    ``reachable`` goes False the first time the reader fails, and stays so:
    the command goes on with stored fields only.
    """

    def __init__(self, reader: Any):
        self.reader = reader
        self.reachable = reader is not None
        self.fields: dict[tuple[int, str], dict[str, str]] = {}

    def load(self, records: list[tuple[str, dict[str, Any]]]) -> None:
        if not self.reachable:
            return
        wanted: dict[int, set[str]] = {}
        for doc_id, meta in records:
            if record_kind(doc_id, meta) == KIND_LEGACY:
                continue
            group = int(meta.get("group_id") or 0)
            key = record_key(doc_id)
            if (group, key) not in self.fields:
                wanted.setdefault(group, set()).add(key)
        for group, keys in wanted.items():
            try:
                items = self.reader.get_full_items(sorted(keys), group_id=group)
            except Exception as e:
                logger.warning(f"Could not read items from the local Zotero database: {e}")
                self.reachable = False
                return
            for key in keys:
                data = (items.get(key) or {}).get("data") or {}
                self.fields[(group, key)] = {
                    "item_type": data.get("itemType") or "",
                    "title": data.get("title") or "",
                    "creators": format_creators(data.get("creators") or []) if data.get("creators") else "",
                }

    def of(self, doc_id: str, meta: dict[str, Any]) -> dict[str, str] | None:
        """The record's fields: stored for a legacy record, live for a new one, else None."""
        if record_kind(doc_id, meta) == KIND_LEGACY:
            return {
                "item_type": meta.get("item_type") or "",
                "title": meta.get("title") or "",
                "creators": meta.get("creators") or "",
            }
        return self.fields.get((int(meta.get("group_id") or 0), record_key(doc_id)))


def _resolved(
    client: Any, live: _Live, include_documents: bool
) -> Iterator[tuple[str, dict[str, Any], str | None, dict[str, str] | None]]:
    """Records with their fields, resolving a page of keys at a time."""
    page: list[tuple[str, dict[str, Any], str | None]] = []
    for record in _iter_records(client, include_documents):
        page.append(record)
        if len(page) >= BATCH_SIZE:
            live.load([(d, m) for d, m, _ in page])
            for doc_id, meta, doc in page:
                yield doc_id, meta, doc, live.of(doc_id, meta)
            page = []
    live.load([(d, m) for d, m, _ in page])
    for doc_id, meta, doc in page:
        yield doc_id, meta, doc, live.of(doc_id, meta)


def inspect_index(
    search: Any,
    *,
    stats: bool,
    limit: int,
    filter_text: str | None,
    show_documents: bool,
    out: Callable[[str], None] = print,
) -> None:
    """Print the index's record kinds and, when the local Zotero database answers, what they are."""
    client = search.chroma_client
    try:
        reader = search._open_local_reader()
    except Exception as e:
        logger.warning(f"Could not open the local Zotero database: {e}")
        reader = None
    live = _Live(reader)
    try:
        if stats:
            _print_stats(client, live, out)
        else:
            _print_records(client, live, limit, filter_text, show_documents, out)
    finally:
        close = getattr(reader, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass


def _print_stats(client: Any, live: _Live, out: Callable[[str], None]) -> None:
    out("=== Semantic DB Inspection (Stats) ===")
    info = client.get_collection_info()
    out(f"Collection: {info.get('name')} @ {info.get('persist_directory')}")
    out(f"Count: {info.get('count')}")

    kinds: Counter[str] = Counter()
    legacy_types: Counter[str] = Counter()
    legacy_coverage: dict[str, dict[str, int]] = {}
    legacy_titles: Counter[str] = Counter()
    keys: dict[tuple[int, str], None] = {}
    for doc_id, meta, _doc, _fields in _resolved(client, live, False):
        kind = record_kind(doc_id, meta)
        kinds[kind] += 1
        if kind != KIND_LEGACY:
            keys[(int(meta.get("group_id") or 0), record_key(doc_id))] = None
            continue
        item_type = meta.get("item_type") or "(missing)"
        legacy_types[item_type] += 1
        cov = legacy_coverage.setdefault(item_type, {"total": 0, "with_fulltext": 0, "pdf": 0, "html": 0})
        cov["total"] += 1
        if meta.get("has_fulltext"):
            cov["with_fulltext"] += 1
            src = (meta.get("fulltext_source") or "").lower()
            if src in ("pdf", "html"):
                cov[src] += 1
        if meta.get("title"):
            legacy_titles[meta["title"]] += 1

    out("Record kinds:")
    for kind in _KINDS:
        out(f"  {kind}: {kinds[kind]}")

    if live.reachable:
        types = Counter(live.fields[k]["item_type"] or "(missing)" for k in keys if k in live.fields)
        out("Item types (from the local Zotero database, one per item key):")
        for t, c in types.most_common(20):
            out(f"  {t}: {c}")
        gone = sum(1 for k in keys if k not in live.fields or not live.fields[k]["item_type"])
        if gone:
            out(f"  ({gone} keys are no longer in the local Zotero database)")
    elif keys:
        out(
            "Item types: not shown. New records store no item type, and the local Zotero "
            "database is not reachable (it is read in local mode with Zotero's data directory available)."
        )

    if legacy_types:
        out("Item types (legacy records, as stored):")
        for t, c in legacy_types.most_common(20):
            out(f"  {t}: {c}")
        out("Fulltext coverage (legacy records, by type):")
        for t, cov in legacy_coverage.items():
            out(f"  {t}: {cov['with_fulltext']}/{cov['total']} (pdf:{cov['pdf']}, html:{cov['html']})")
        common = legacy_titles.most_common(10)
        if common:
            out("Common titles (legacy records):")
            for t, c in common:
                out(f"  {t[:80]}{'...' if len(t) > 80 else ''}: {c}")


def _print_records(
    client: Any,
    live: _Live,
    limit: int,
    filter_text: str | None,
    show_documents: bool,
    out: Callable[[str], None],
) -> None:
    out("=== Semantic DB Inspection ===")
    out(f"Total documents: {client.get_collection_info().get('count', 0)}")
    out(f"Showing up to: {limit}")

    needle = filter_text.lower() if filter_text else None
    shown = 0
    unmatchable = 0
    for doc_id, meta, doc, fields in _resolved(client, live, show_documents):
        if needle:
            if fields is None:
                unmatchable += 1
                continue
            if needle not in fields["title"].lower() and needle not in fields["creators"].lower():
                continue
        kind = record_kind(doc_id, meta)
        title = fields["title"] if fields else ""
        creators = fields["creators"] if fields else ""
        out(f"- {title} | {creators} [{kind} {doc_id}]")
        if show_documents:
            full = (doc or "").strip()
            snippet = full[:200].replace("\n", " ")
            if snippet:
                out(f"  doc: {snippet}{'...' if len(full) > 200 else ''}")
        shown += 1
        if shown >= limit:
            break

    if unmatchable:
        out(
            f"{unmatchable} records were not matched: only legacy records store a title and creators, "
            f"so the rest {_NEEDS_ZOTERO}, which is not reachable."
        )
    if shown == 0:
        out("No records matched your filter.")
