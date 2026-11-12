"""Move a chunked index from one collection to two, without re-embedding (#769).

A chunked index built before the split holds its passages in the article
collection. ``zotero-mcp migrate-index`` renames that collection to the passage
collection and creates a new, empty article collection. The rename is one SQLite
transaction, keeps every vector where it is on disk, and needs no extra space,
however many passages there are.

Renaming also matters for memory. Deleting the passages from the article
collection would not shrink its vector index, which ChromaDB loads in full,
deleted records included, so article searches would stay as expensive as before.

Mixed indexes need a little more: records with a bare ``KEY`` id were article
vectors, so they move to the new article collection using their stored vectors.
Each batch is written to its destination before it is deleted from its source,
so a run that dies part way loses nothing and the next run carries on.

This does not embed anything. Article vectors are missing after a migration, and
the next ``update-db`` finds and embeds them through its normal gap fill.

The second step re-keys the passages (#826). Before the item-keyed index a
passage ``KEY#n`` belonged to its parent item and carried the item's whole
metadata; now it belongs to its attachment, ``ATT#n``, and carries only its
own. Passages that carry ``title`` are the old ones. Each item's passages are
written under the attachment's key with the same vectors, offsets shifted past
the metadata text the old chunk 0 began with, then the old ids are deleted.
An item whose attachment cannot be told with certainty loses its passages
instead, and the next ``update-db`` embeds them again: a wrong guess would
leave text searchable under the wrong file.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .extract import select_attachment_rows

logger = logging.getLogger(__name__)

_MOVE_BATCH = 500


class MigrationError(Exception):
    """The index cannot be migrated; the message says why and what to do."""


@dataclass
class MigrationReport:
    """What a migration (or a dry run) found and did."""

    persist_directory: str
    dry_run: bool
    needed: bool = False
    renamed: bool = False
    passage_records: int = 0
    moved_articles: int = 0
    moved_passages: int = 0
    items_needing_articles: int = 0
    # Re-keying legacy passages. ``legacy_*`` is what the index held when the
    # run began; the rest is what was done (what would be done, in a dry run).
    legacy_passages: int = 0
    legacy_items: int = 0
    rekeyed_items: int = 0
    rekeyed_passages: int = 0
    dropped_items: int = 0
    dropped_passages: int = 0
    # Left as they are because Zotero could not be asked (unreachable).
    skipped_items: int = 0
    skipped_passages: int = 0


def _after_rename() -> None:
    """Runs right after the rename, so a test can kill the process at that point."""


def _after_rekey_upsert() -> None:
    """Runs after an item's passages are written under the attachment's key, before the old ones go."""


def _delete_moved(collection: Any, ids: list[str]) -> None:
    collection.delete(ids=ids)


def _ids_in(db: Path, collection_name: str, *, passages: bool) -> list[str]:
    """Record ids of one level in one collection, from SQLite, without loading vectors."""
    condition = "LIKE '%#%'" if passages else "NOT LIKE '%#%'"
    con = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True, timeout=5)
    try:
        rows = con.execute(
            f"""
            SELECT e.embedding_id
            FROM embeddings e
            JOIN segments s ON e.segment_id = s.id
            JOIN collections c ON s.collection = c.id
            WHERE c.name = ? AND s.scope = 'METADATA' AND e.embedding_id {condition}
            """,
            (collection_name,),
        ).fetchall()
    finally:
        con.close()
    return sorted(row[0] for row in rows)


def _move(
    source: Any,
    destination: Any,
    ids: list[str],
    progress: Callable[[str], None],
    *,
    keep_existing: bool = False,
) -> int:
    """Write ``ids`` to ``destination`` with their stored vectors, then delete them from ``source``.

    With ``keep_existing``, a record already in the destination wins and the
    source's copy is only deleted. Returns how many records were handled.
    """
    moved = 0
    for start in range(0, len(ids), _MOVE_BATCH):
        part = ids[start:start + _MOVE_BATCH]
        got = source.get(ids=part, include=["embeddings", "documents", "metadatas"])
        found = got["ids"]
        positions = list(range(len(found)))
        if keep_existing and found:
            existing = set(destination.get(ids=found, include=[])["ids"])
            positions = [i for i in positions if found[i] not in existing]
        if positions:
            destination.upsert(
                ids=[found[i] for i in positions],
                documents=[got["documents"][i] for i in positions],
                metadatas=[got["metadatas"][i] for i in positions],
                embeddings=[got["embeddings"][i] for i in positions],
            )
        if found:
            _delete_moved(source, found)
        moved += len(found)
        progress(f"  moved {moved}/{len(ids)} records")
    return moved


# What the old chunk 0 tells about its item: where its text came from, and the
# text itself (the metadata prefix and, after it, the file's).
_FIRST_PASSAGE_FIELDS = (
    "fulltext_source", "attachment_keys", "attachment_priority", "group_id",
    "title", "creators", "publication", "tags", "chroma:document",
)


class _Unknown:
    """Zotero could not be asked which attachment an item's passages came from."""

    def __repr__(self) -> str:
        return "UNKNOWN"


UNKNOWN = _Unknown()


@dataclass
class LegacyItem:
    """One regular item's legacy passages, as stored."""

    key: str
    passages: list[tuple[str, str]]  # (id, collection name), in chunk order
    first: dict[str, Any]  # stored fields of the first surviving passage (chunk 0 unless it is gone)


def legacy_items(persist_dir: str, collection_name: str) -> list[LegacyItem]:
    """Every item with legacy passages, read-only from SQLite (nothing is opened through ChromaDB)."""
    from .chroma_client import read_legacy_passage_ids, read_stored_fields

    found = read_legacy_passage_ids(persist_dir, collection_name)
    # Every passage carries the item's metadata, so whichever survives says as much as chunk 0.
    firsts = [passages[0] for passages in found.values()]
    stored = read_stored_fields(persist_dir, firsts, _FIRST_PASSAGE_FIELDS)
    return [LegacyItem(key, passages, stored.get(passages[0], {})) for key, passages in sorted(found.items())]


class LocalSources:
    """Tells which attachment a legacy item's passages were cut from.

    ``reader`` is a ``LocalZoteroReader`` (None when there is no local Zotero:
    only web-API indexes can then be re-keyed). The answer is the attachment
    key, or None when it cannot be told for certain.
    """

    def __init__(self, reader: Any = None, config_path: str | None = None) -> None:
        self.reader = reader
        self.config_path = config_path
        self._targets: dict[str, Any] = {}
        self._prefixes: dict[str, str] = {}

    def attachment_for(self, item: LegacyItem) -> str | None | _Unknown:
        """The attachment ``item``'s passages came from, if it can be told for certain.

        A web-API index recorded it (``fulltext_source`` is
        ``web-api:attachment:<key>``). A local index recorded a fingerprint
        of the item's attachments (records of the oldest versions did not) and
        the priority they were chosen under, so the choice is made again and
        trusted only when a recorded fingerprint and priority are unchanged
        and exactly one file is chosen.

        None is a positive finding (the item is gone from Zotero, the choice
        is ambiguous or has changed): its passages cannot be kept.
        ``UNKNOWN`` means Zotero could not be asked, which says nothing about
        the item: its passages must be left as they are.
        """
        from .semantic_search import _WEB_ATTACHMENT_SOURCE

        source = str(item.first.get("fulltext_source") or "")
        if source.startswith(_WEB_ATTACHMENT_SOURCE):
            return source[len(_WEB_ATTACHMENT_SOURCE):] or None
        if self.reader is None or not item.first:
            return UNKNOWN
        try:
            local = self.reader.get_item_by_key(item.key)
            if local is None:
                return None
            rows = self.reader.get_fulltext_meta_for_item(local.item_id)
            fingerprint = ",".join(sorted(row[0] for row in rows))
            stored = item.first.get("attachment_keys")
            if stored is not None and stored != fingerprint:
                return None
            priority = item.first.get("attachment_priority")
            if priority is not None and priority != ",".join(self.reader.attachment_priority):
                return None
            if not rows:
                return None
            targets = self.reader._resolve_extraction_targets(local.item_id)
            # What the database selects, not what happens to be on disk: a file
            # on an unmounted drive, kept on demand, or in a format only
            # Zotero's text cache covers says nothing about where the passages
            # came from, and must not cost the user them.
            selected = select_attachment_rows(rows, self.reader.attachment_priority)
        except Exception as e:
            logger.warning(f"Could not look up {item.key} in Zotero ({e}); its passages are left as they are")
            return UNKNOWN
        if sorted(t.key for t in targets) != selected:
            return UNKNOWN
        if len(targets) != 1:
            return None
        self._targets[targets[0].key] = targets[0]
        self._prefixes[item.key] = _live_prefix(local)
        return targets[0].key

    def live_prefix(self, key: str) -> str | None:
        """The metadata text Zotero's current record of ``key`` gives, or None if it was not looked up."""
        return self._prefixes.get(key)

    def remember_hash(self, attachment_key: str) -> None:
        """Hash the live file and keep the digest in the hash cache, where a later update finds it."""
        from . import file_hash

        target = self._targets.get(attachment_key)
        if target is not None:
            file_hash.get_file_hash(target.key, target.path, target.file_stamp, config_path=self.config_path)


def _live_prefix(item: Any) -> str:
    """The metadata text a local ``--fulltext`` run put before the file's text.

    That recipe is what the legacy records were written with. It lives here
    as it was, not as ``_create_document_text`` is today, so a later change to
    the current recipe cannot misplace the offsets of records already written.
    """
    from .semantic_search import ZoteroSemanticSearch
    from .utils import format_creators

    # Neither method reads the instance.
    creators = ZoteroSemanticSearch._parse_creators_string(None, item.creators or "")  # type: ignore[arg-type]
    return " ".join(filter(None, [item.title or "", format_creators(creators), item.abstract or ""]))


@contextmanager
def open_sources(config_path: str | None, db_path: str | None = None) -> Iterator[LocalSources]:
    """Sources over this install's Zotero database (``db_path`` beats the configured one), or over nothing when there is no local Zotero."""
    from .utils import is_local_mode

    reader = None
    if is_local_mode():
        section: dict[str, Any] = {}
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                section = json.load(f).get("semantic_search", {})
        try:
            from .local_db import LocalZoteroReader

            reader = LocalZoteroReader(
                db_path=db_path or section.get("zotero_db_path"),
                attachment_priority=(section.get("extraction") or {}).get("attachment_priority"),
                config_path=config_path,
            )
        except Exception as e:
            logger.warning(f"Could not open the Zotero database ({e}); only web-API passages can be kept")
    try:
        yield LocalSources(reader, config_path)
    finally:
        close = getattr(reader, "close", None)
        if close:
            close()


def file_text_offset(first_document: str, first: dict[str, Any], live_prefix: str | None) -> int:
    """Where the file's text starts in a legacy item's text (its metadata text, then the file's).

    The legacy chunk 0 begins with that metadata text and a blank line, and the
    passages' offsets count from there. In order of certainty:

    1. the metadata text Zotero gives now, when chunk 0 begins with it;
    2. rebuilt from what chunk 0's metadata stored (title, creators,
       publication, tags): the abstract in between is the unknown, so the text
       ends at the first blank line after the tags, or after the creators when
       there are none;
    3. 0, when nothing identifies a prefix.

    The offsets only place a quote in the file's text, so a miss by an abstract
    paragraph is harmless.
    """
    if live_prefix is not None:
        if not live_prefix.strip():
            return 0
        if first_document.startswith(live_prefix + "\n\n"):
            return len(live_prefix) + 2
    head = " ".join(filter(None, [first.get("title") or "", first.get("creators") or ""]))
    if not head or not first_document.startswith(head):
        return 0
    tail = " ".join(filter(None, [first.get("publication") or "", first.get("tags") or ""]))
    end = first_document.find(f"{tail}\n\n" if tail else "\n\n", len(head))
    return end + len(tail) + 2 if end >= 0 else 0


def _rekeyed(item: LegacyItem, attachment: str, records: list[dict[str, Any]], offset: int) -> list[dict[str, Any]]:
    """The item's passages as ``ATT#n`` records: same vectors, only the passage's own metadata."""
    from .semantic_search import _LEGACY_METADATA_KEYS

    out = []
    for record in records:
        old = record["metadata"]
        number = int(record["id"].rsplit("#", 1)[1])
        meta: dict[str, Any] = {"item_key": attachment}
        if old.get("group_id") is not None:
            meta["group_id"] = int(old["group_id"])
        meta["chunk_index"] = number
        meta["n_chunks"] = old.get("n_chunks")
        for field_name in ("char_start", "char_end"):
            if old.get(field_name) is not None:
                meta[field_name] = max(0, int(old[field_name]) - offset)
        # ChromaDB's upsert merges into a stored record's metadata: None clears
        # what an earlier run left under this id (and is not stored otherwise).
        meta["page"] = old.get("page")
        meta["section"] = old.get("section")
        meta.update(dict.fromkeys((*_LEGACY_METADATA_KEYS, "date_modified")))
        out.append({"id": f"{attachment}#{number}", "document": record["document"],
                    "embedding": record["embedding"], "metadata": meta})
    return out


def rekey_legacy_passages(
    persist_dir: str,
    collection_name: str,
    client: Any,
    sources: LocalSources,
    report: MigrationReport,
    progress: Callable[[str], None],
    *,
    config_path: str | None = None,
) -> None:
    """Re-key the legacy passages of every item to their attachment; fills the report.

    ``client`` is None for a dry run, which only decides what would happen.
    Each item is its own step: its passages are written under the attachment
    key, its article record (if any) marked ``rekeyed_to``, and only then the
    old ids deleted, so an interrupted run finds either both or only the new
    ones, and a rerun carries on. Nothing is embedded.
    """
    from . import file_hash

    items = legacy_items(persist_dir, collection_name)
    report.legacy_items = len(items)
    report.legacy_passages = sum(len(item.passages) for item in items)
    with file_hash.batched(config_path):
        for number, item in enumerate(items, 1):
            attachment = sources.attachment_for(item)
            ids = [doc_id for doc_id, _name in item.passages]
            if attachment is UNKNOWN:
                report.skipped_items += 1
                report.skipped_passages += len(ids)
            elif attachment is None:
                report.dropped_items += 1
                report.dropped_passages += len(ids)
                if client is not None:
                    client.delete_records(ids)
            else:
                report.rekeyed_items += 1
                report.rekeyed_passages += len(ids)
                if client is not None:
                    _rekey_item(client, sources, item, attachment, ids)
            if client is not None and number % 200 == 0:
                progress(f"  re-keyed {number}/{len(items)} items")


def _rekey_item(client: Any, sources: LocalSources, item: LegacyItem, attachment: str, ids: list[str]) -> None:
    # An attachment record exists only when an update has already written the
    # file in the new shape: its passages are the current ones.
    if attachment not in client.get_existing_ids([attachment]):
        records = client.read_records(ids)
        first = next((r for r in records if r["id"].endswith("#0")), None)
        offset = (
            file_text_offset(first["document"], item.first, sources.live_prefix(item.key)) if first else 0
        )
        written = _rekeyed(item, attachment, records, offset)
        client.upsert_embeddings(
            documents=[r["document"] for r in written],
            metadatas=[r["metadata"] for r in written],
            ids=[r["id"] for r in written],
            embeddings=[r["embedding"] for r in written],
        )
        _after_rekey_upsert()
        sources.remember_hash(attachment)
    # The article record says its passages moved, so they are not "missing".
    # The whole metadata goes back: not every ChromaDB version merges.
    stored = client.get_metadatas([item.key]).get(item.key)
    if stored is not None and stored.get("rekeyed_to") != attachment:
        client.update_metadatas([item.key], [{**stored, "rekeyed_to": attachment}])
    client.delete_records(ids)


def _split(
    client: Any,
    report: MigrationReport,
    layout: dict[str, Any],
    unmigrated: bool,
    stray_articles: list[str],
    stray_passages: list[str],
    progress: Callable[[str], None],
) -> None:
    """The split proper: rename an unmigrated index's collection, then move what is in the wrong one."""
    from .chroma_client import passage_collection_name

    collection_name = client.collection_name
    passages_name = passage_collection_name(collection_name)
    if unmigrated:
        if layout["passage_collection_exists"]:
            # Empty, or the index would not count as unmigrated.
            client.client.delete_collection(name=passages_name)
        progress(f"Renaming {collection_name} to {passages_name} ...")
        client.collection.modify(name=passages_name)
        report.renamed = True
        _after_rename()
        progress(f"Creating a new {collection_name} for article vectors ...")
        client.collection = client._open_collection(collection_name)
        client._passage_collection = None
        client._unmigrated = False

    if stray_articles:
        progress(f"Moving {len(stray_articles)} article records to {collection_name} ...")
        report.moved_articles = _move(client._passages(), client.collection, stray_articles, progress)
    if stray_passages:
        progress(f"Moving {len(stray_passages)} passages to {passages_name} ...")
        report.moved_passages = _move(
            client.collection, client._passages(), stray_passages, progress, keep_existing=True
        )


def split_index(client: Any, progress: Callable[[str], None] = lambda _message: None) -> MigrationReport | None:
    """Split ``client``'s index into an article and a passage collection if it needs it, else None.

    What ``migrate-index`` does first, on a client that is already open (an
    explicit ``update-db`` runs this before re-keying, in the plan's order).
    Moves vectors; embeds nothing.
    """
    from .chroma_client import passage_collection_name, read_index_layout

    layout = read_index_layout(client.persist_directory, client.collection_name)
    unmigrated = layout["unmigrated_passages"] > 0
    db = Path(client.persist_directory) / "chroma.sqlite3"
    stray_passages = [] if unmigrated or not layout["passages_in_article_collection"] else _ids_in(
        db, client.collection_name, passages=True
    )
    if not unmigrated and not stray_passages:
        return None
    report = MigrationReport(persist_directory=client.persist_directory, dry_run=False, needed=True)
    report.passage_records = layout["passage_records"]
    _split(client, report, layout, unmigrated, [], stray_passages, progress)
    if unmigrated:
        # Bare records beside the passages were article vectors: they move to the new collection.
        bare = _ids_in(db, passage_collection_name(client.collection_name), passages=False)
        if bare:
            progress(f"Moving {len(bare)} article records to {client.collection_name} ...")
            report.moved_articles = _move(client._passages(), client.collection, bare, progress)
    return report


def migrate_index(
    config_path: str | None = None,
    *,
    dry_run: bool = False,
    progress: Callable[[str], None] = print,
    sources: LocalSources | None = None,
) -> MigrationReport:
    """Split the configured index into an article and a passage collection, then re-key its passages.

    The split comes first, and only an unmigrated or half-moved index needs it;
    the re-key moves every legacy passage ``KEY#n`` to its attachment's key
    (see the module docstring). Safe to run again at any point: it only
    finishes what is left, and finding nothing it does nothing. ``sources``
    answers which attachment a passage came from; by default it reads the
    local Zotero database.
    """
    from .chroma_client import (
        create_chroma_client,
        passage_collection_name,
        read_index_layout,
        read_item_keys_by_level,
    )
    from .config import resolve_chroma_dir

    collection_name = "zotero_library"
    configured_dir = None
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            section = json.load(f).get("semantic_search", {})
        collection_name = section.get("collection_name", collection_name)
        configured_dir = section.get("persist_directory")
    persist_dir = resolve_chroma_dir(configured_dir)
    db = persist_dir / "chroma.sqlite3"
    if not db.exists():
        raise MigrationError(f"no index at {persist_dir}; nothing to migrate. Run: zotero-mcp update-db")

    passages_name = passage_collection_name(collection_name)
    report = MigrationReport(persist_directory=str(persist_dir), dry_run=dry_run)
    layout = read_index_layout(str(persist_dir), collection_name)
    report.passage_records = layout["passage_records"]

    unmigrated = layout["unmigrated_passages"] > 0
    stray_passages = [] if unmigrated else _ids_in(db, collection_name, passages=True)
    # Bare records beside the passages: in the collection about to be renamed,
    # or left in the passage collection by a run that did not finish.
    stray_articles = _ids_in(db, collection_name if unmigrated else passages_name, passages=False)
    report.legacy_passages = layout["legacy_passages"]
    report.legacy_items = layout["legacy_passage_items"]
    report.needed = unmigrated or bool(stray_passages or stray_articles) or report.legacy_passages > 0

    articles, passages = read_item_keys_by_level(str(persist_dir), collection_name)
    if unmigrated:
        passages = {doc_id.split("#", 1)[0] for doc_id in _ids_in(db, collection_name, passages=True)}
        articles = set(stray_articles)
    elif stray_articles:
        articles |= set(stray_articles)
    report.items_needing_articles = len(passages - articles)

    if dry_run:
        report.moved_articles = len(stray_articles)
        report.moved_passages = len(stray_passages)
        if report.legacy_passages:
            with _sources_or(sources, config_path) as src:
                rekey_legacy_passages(str(persist_dir), collection_name, None, src, report, progress)
        return report
    if not report.needed:
        return report

    client = create_chroma_client(config_path, persist_directory=str(persist_dir))
    _split(client, report, layout, unmigrated, stray_articles, stray_passages, progress)

    if report.legacy_passages:
        progress(f"Re-keying the passages of {report.legacy_items} items to their attachments ...")
        with _sources_or(sources, config_path) as src:
            rekey_legacy_passages(
                str(persist_dir), collection_name, client, src, report, progress, config_path=config_path
            )
    return report


@contextmanager
def _sources_or(sources: LocalSources | None, config_path: str | None) -> Iterator[LocalSources]:
    if sources is not None:
        yield sources
    else:
        with open_sources(config_path) as opened:
            yield opened


def rekey_note(legacy_items: int, legacy_passages: int) -> list[str]:
    """What status says about passages still keyed by their parent item: what they are and what to run.

    ``legacy_items`` and ``legacy_passages`` are ``legacy_passage_items`` and
    ``legacy_passages`` of ``chroma_client.read_index_layout``. Empty when
    there are none.
    """
    if not legacy_passages:
        return []
    return [
        f"{legacy_passages:,} passages of {legacy_items:,} items are still stored the way earlier versions "
        "did, under their item's key with the item's metadata. Search keeps working with them.",
        "zotero-mcp migrate-index re-keys them to their attachments without embedding anything "
        "(add --dry-run to see what would change); so does a manual zotero-mcp update-db, "
        "which runs it first. Items whose attachment cannot be told for certain have their "
        "passages embedded again by that update.",
    ]


def status_note(migration: dict[str, Any], update_command: str, *, local_mode: bool) -> list[str]:
    """What status says about an unmigrated chunked index: what is wrong and what to run.

    ``migration`` is the ``migration`` entry of the collection status (see
    ``chroma_client.read_migration_estimate``); ``update_command`` is how the
    caller's users run update-db. In local mode step 4 adds ``--fulltext``
    unless no waiting item was indexed with full text, as migrate-index's own
    report does: without it the update leaves those items waiting for their
    text. ``--fulltext`` fails outside local mode, where the plain update
    fetches the text Zotero has already extracted, so it is left out there.

    Says nothing about whether the run is billed: the settings can't tell a
    paid service from a free one.
    """
    items = migration.get("items_needing_articles")
    tokens = migration.get("estimated_tokens")
    web_fulltext = not local_mode and migration.get("fulltext") is not False
    update = update_command if migration.get("fulltext") is False or not local_mode else f"{update_command} --fulltext"
    those = f"{items:,} items have" if items is not None else "some items have"
    lines = [
        "This index was built by an earlier version: its passages are in the article-level "
        f"ChromaDB collection, and {those} passages but no article record.",
        "Search keeps working as it does now until you migrate. Migrating gives articles their own "
        "ChromaDB collection, so article searches stop loading every passage.",
        "To migrate:",
        "  1. Stop any running zotero-mcp server, and upgrade every other copy of zotero-mcp that "
        "uses this index; an older version would put passages back in the article-level collection.",
        "  2. zotero-mcp migrate-index --dry-run   (shows what would change)",
        "  3. zotero-mcp migrate-index             (moves the existing vectors; embeds nothing, takes seconds)",
        f"  4. {update}   (embeds an article record for each of those items; "
        "add the options you usually use)",
    ]
    if web_fulltext:
        lines.append(
            "     It uses the text Zotero has already extracted; --fulltext needs local mode, "
            "so leave it out here."
        )
    billed = "With a paid embedding API that is billed."
    if tokens is not None and items is not None:
        cap = migration.get("max_tokens")
        limit = f", and at most {cap:,} tokens an item" if cap else ""
        lines.append(
            f"Only step 4 embeds anything: about {tokens:,} tokens for {items:,} items. That is an "
            f"estimate, from the length of each item's indexed text at about "
            f"{migration['chars_per_token']:g} characters a token{limit}. {billed}"
        )
    else:
        lines.append(f"Only step 4 embeds anything; how many tokens could not be estimated. {billed}")
    return lines
