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
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _after_rename() -> None:
    """Runs right after the rename, so a test can kill the process at that point."""


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


def migrate_index(
    config_path: str | None = None,
    *,
    dry_run: bool = False,
    progress: Callable[[str], None] = print,
) -> MigrationReport:
    """Split the configured index into an article and a passage collection.

    Safe to run again at any point: on a split index it only tidies up records
    in the wrong collection, and finding none it does nothing.
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
    report.needed = unmigrated or bool(stray_passages or stray_articles)

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
        return report
    if not report.needed:
        return report

    client = create_chroma_client(config_path, persist_directory=str(persist_dir))

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
    return report


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
