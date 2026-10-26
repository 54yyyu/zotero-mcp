"""Rebuild an unreadable semantic index from the files it left on disk.

ChromaDB keeps a collection in two places: ``chroma.sqlite3`` holds the
documents and metadata, and one HNSW folder per vector segment holds the
vectors. An update that dies part way through can leave ``chroma.sqlite3`` in
a state ChromaDB refuses to read (#747) while the HNSW files, written at a
different time, are intact. Re-embedding a large full-text library costs hours
and API fees, so ``zotero-mcp repair-db`` reads both directly, keeps every item
whose chunks all check out, and writes them into a fresh index through
:func:`create_chroma_client`. Items that fail a check are left out and listed;
``update-db --fulltext`` embeds them again.

Nothing here writes to the damaged index. SQLite is read from a staged copy,
the new index is built in a sibling directory, and the old one is kept, renamed
to ``<dir>.broken-<timestamp>``, when the new one is swapped in.
"""

from __future__ import annotations

import io
import json
import logging
import os
import pickle
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Rows read per query. A query that hits a damaged page is split in half until
# it reads, or until the window is this small and is skipped instead: the rows
# lost there only make their items fail validation, so they get re-embedded.
_SCAN_WINDOW = 20_000
_MIN_SCAN_WINDOW = 64

# Chunks written to the new index per call, grouped by whole items.
_ITEMS_PER_BATCH = 200

_REBUILD_HINT = "Rebuild the index instead: zotero-mcp update-db --force-rebuild"


class RepairError(Exception):
    """The index cannot be repaired; the message says why and what to do."""


@dataclass
class RepairReport:
    """What a repair (or a dry run) found and did."""

    persist_directory: str
    dry_run: bool
    items_kept: int = 0
    chunks_kept: int = 0
    excluded: dict[str, str] = field(default_factory=dict)
    header_page_count_fixed: bool = False
    skipped_rows: int = 0
    broken_directory: str | None = None
    report_path: str | None = None


# --- SQLite ------------------------------------------------------------------


def _clone_or_copy(src: Path, dst: Path) -> None:
    """Copy-on-write clone where the filesystem has it (APFS, Btrfs, XFS), else a copy.

    A clone of a 20 GB database is instant and takes no space until written.
    """
    flag = {"darwin": "-c"}.get(sys.platform, "--reflink=auto" if sys.platform.startswith("linux") else None)
    if flag:
        try:
            subprocess.run(["cp", flag, str(src), str(dst)], check=True, capture_output=True)
            return
        except (OSError, subprocess.CalledProcessError):
            dst.unlink(missing_ok=True)
    shutil.copyfile(src, dst)


def _fix_header_page_count(path: Path) -> bool:
    """Make the header's page count cover the whole file. True if it was short.

    SQLite trusts the page count in the header when the "version-valid-for"
    stamp matches the change counter, and treats any page past it as not part
    of the database. A write that grew the file and was then cut off can leave
    tables pointing at those pages, which SQLite reports as "database disk
    image is malformed". Counting them in again lets the rows be read; pages
    nothing points at are harmless.
    """
    with open(path, "r+b") as f:
        header = f.read(100)
        if len(header) < 100 or not header.startswith(b"SQLite format 3\x00"):
            raise RepairError(f"{path} is not an SQLite database. {_REBUILD_HINT}")
        page_size = struct.unpack_from(">H", header, 16)[0]
        page_size = 65536 if page_size == 1 else page_size
        page_count = struct.unpack_from(">I", header, 28)[0]
        change_counter = struct.unpack_from(">I", header, 24)[0]
        valid_for = struct.unpack_from(">I", header, 92)[0]
        file_pages = os.path.getsize(path) // page_size
        if valid_for != change_counter or page_count >= file_pages:
            return False
        f.seek(28)
        f.write(struct.pack(">I", file_pages))
        return True


def _scan_rows(
    con: sqlite3.Connection, table: str, columns: str, on_skip: Callable[[int], None]
) -> Iterator[tuple]:
    """Yield every readable row of ``table`` by rowid ranges, never through an index.

    Indexes are what an interrupted write damages most, and the new index
    rebuilds them anyway. A range that fails to read is split until it reads
    or is too small to bother, then skipped; ``on_skip`` gets the rowid span.
    """
    try:
        top = con.execute(f"SELECT max(rowid) FROM {table}").fetchone()[0] or 0
    except sqlite3.DatabaseError as e:
        raise RepairError(f"cannot read the {table} table ({e}). {_REBUILD_HINT}") from e
    query = f"SELECT {columns} FROM {table} WHERE rowid BETWEEN ? AND ?"
    pending = [(lo, min(lo + _SCAN_WINDOW - 1, top)) for lo in range(1, top + 1, _SCAN_WINDOW)]
    pending.reverse()
    while pending:
        lo, hi = pending.pop()
        try:
            rows = con.execute(query, (lo, hi)).fetchall()
        except sqlite3.DatabaseError:
            if hi - lo + 1 <= _MIN_SCAN_WINDOW:
                on_skip(hi - lo + 1)
                continue
            mid = (lo + hi) // 2
            pending += [(mid + 1, hi), (lo, mid)]
            continue
        yield from rows


def _seq_id(value: Any) -> int:
    # Older ChromaDB stored sequence ids as big-endian blobs.
    return int.from_bytes(value, "big") if isinstance(value, bytes) else int(value)


# --- HNSW --------------------------------------------------------------------


class _DataOnlyUnpickler(pickle.Unpickler):
    """``index_metadata.pickle`` is a dict of dicts, strs and ints; refuse anything else."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


@dataclass
class _HnswVectors:
    rows: np.ndarray  # (elements, bytes per element), memory-mapped
    data_offset: int
    dim: int
    label_to_row: dict[int, int]
    id_to_label: dict[str, int]

    def get(self, row_numbers: list[int]) -> np.ndarray:
        order = np.argsort(row_numbers)
        picked = np.ascontiguousarray(
            self.rows[np.asarray(row_numbers)[order], self.data_offset:self.data_offset + self.dim * 4]
        )
        out = np.empty((len(row_numbers), self.dim), dtype=np.float32)
        out[order] = picked.view("<f4").reshape(len(row_numbers), self.dim)
        return out


def _open_hnsw(segment_dir: Path, dim: int) -> _HnswVectors:
    """Map the vectors of a persisted hnswlib index, after checking its layout.

    ``header.bin`` is a 4-byte format version followed by hnswlib's own
    header; ``data_level0.bin`` holds one fixed-size record per element: the
    level-0 neighbour list, the vector, then the 8-byte label.
    """
    unsupported = f"unsupported vector index format in {segment_dir}. {_REBUILD_HINT}"
    try:
        header = (segment_dir / "header.bin").read_bytes()
        (version,) = struct.unpack_from("<I", header, 0)
        (offset_level0, max_elements, count, size_per_element, label_offset, data_offset,
         *_rest) = struct.unpack_from("<QQQQQQiIQQQdQ", header, 4)
    except (OSError, struct.error) as e:
        raise RepairError(unsupported) from e
    data_path = segment_dir / "data_level0.bin"
    if (
        version != 1
        or offset_level0 != 0
        or count > max_elements
        or label_offset - data_offset != dim * 4
        or label_offset + 8 != size_per_element
        or not data_path.exists()
        or data_path.stat().st_size < count * size_per_element
    ):
        raise RepairError(unsupported)

    try:
        with open(segment_dir / "index_metadata.pickle", "rb") as f:
            meta = _DataOnlyUnpickler(io.BufferedReader(f)).load()
        id_to_label = meta["id_to_label"]
    except Exception as e:
        raise RepairError(f"cannot read {segment_dir / 'index_metadata.pickle'} ({e}). {_REBUILD_HINT}") from e

    rows = np.memmap(data_path, dtype=np.uint8, mode="r", shape=(count, size_per_element))
    labels = np.ascontiguousarray(rows[:, label_offset:label_offset + 8]).view("<u8").ravel()
    # hnswlib marks a deleted element in byte 2 of its level-0 list header.
    live = np.flatnonzero((rows[:, 2] & 1) == 0)
    label_to_row = {int(labels[i]): int(i) for i in live}
    return _HnswVectors(rows, data_offset, dim, label_to_row, id_to_label)


# --- validation ----------------------------------------------------------------


def _chunk_number(doc_id: str) -> int | None:
    if "#" not in doc_id:
        return None
    tail = doc_id.split("#", 1)[1]
    return int(tail) if tail.isdigit() else -1


def _metadata_value(string_value, int_value, float_value, bool_value):
    if bool_value is not None:
        return bool(bool_value)
    if int_value is not None:
        return int_value
    if float_value is not None:
        return float_value
    return string_value


def _check_item(
    item_key: str,
    doc_ids: list[str],
    metadata: dict[str, dict[str, Any]],
    vectors: _HnswVectors,
) -> tuple[list[str], list[int], list[str], list[dict]] | str:
    """Return (ids, vector rows, documents, metadatas), or why the item is left out.

    The checks are the invariants zotero-mcp writes: the id is ``KEY`` or
    ``KEY#n``, the metadata names the same item and chunk, the chunks run
    0..n_chunks-1 without gaps, and every chunk has a document and a vector.
    """
    numbers = [_chunk_number(d) for d in doc_ids]
    chunked = numbers[0] is not None
    if any((n is not None) != chunked for n in numbers) or (not chunked and len(doc_ids) > 1):
        return "mixed chunked and unchunked ids"
    if chunked:
        order = sorted(range(len(doc_ids)), key=lambda i: numbers[i])
        doc_ids = [doc_ids[i] for i in order]
        if sorted(numbers) != list(range(len(doc_ids))):
            return "chunks missing"

    ids, rows, documents, metadatas = [], [], [], []
    for n, doc_id in enumerate(doc_ids):
        meta = dict(metadata.get(doc_id) or {})
        document = meta.pop("chroma:document", None)
        meta = {k: v for k, v in meta.items() if not k.startswith("chroma:")}
        if not isinstance(document, str) or not document:
            return "chunk without document"
        if meta.get("item_key", item_key) != item_key:
            return "metadata names another item"
        if chunked and meta.get("chunk_index", n) != n:
            return "metadata names another chunk"
        if chunked and n == 0 and meta.get("n_chunks", len(doc_ids)) != len(doc_ids):
            return "chunks missing"
        row = vectors.label_to_row.get(vectors.id_to_label.get(doc_id, -1))
        if row is None:
            return "chunk without vector"
        ids.append(doc_id)
        rows.append(row)
        documents.append(document)
        metadatas.append(meta)
    return ids, rows, documents, metadatas


# --- repair --------------------------------------------------------------------


def _locate(con: sqlite3.Connection, collection_name: str) -> tuple[str, int, str, str]:
    """(collection id, dimension, metadata segment id, vector segment id)."""
    try:
        row = con.execute(
            "SELECT id, dimension FROM collections WHERE name = ?", (collection_name,)
        ).fetchone()
        if not row:
            raise RepairError(f"no collection named {collection_name!r} in the index. {_REBUILD_HINT}")
        collection_id, dim = row
        segments = dict(con.execute(
            "SELECT scope, id FROM segments WHERE collection = ?", (collection_id,)
        ).fetchall())
    except sqlite3.DatabaseError as e:
        raise RepairError(f"cannot read the index catalog ({e}). {_REBUILD_HINT}") from e
    if not dim or "METADATA" not in segments or "VECTOR" not in segments:
        raise RepairError(f"the index catalog is incomplete. {_REBUILD_HINT}")
    return collection_id, int(dim), segments["METADATA"], segments["VECTOR"]


def _pending_item_keys(
    con: sqlite3.Connection, collection_id: str, segment_ids: tuple[str, str], on_skip
) -> set[str]:
    """Items with writes in the queue that some segment has not applied yet.

    Their rows may be half old and half new, so they are left out.
    """
    try:
        applied = [
            _seq_id(v) for (sid, v) in con.execute("SELECT segment_id, seq_id FROM max_seq_id")
            if sid in segment_ids
        ]
    except sqlite3.DatabaseError:
        applied = []
    cutoff = min(applied) if len(applied) == len(segment_ids) else -1
    keys = set()
    for seq_id, topic, doc_id in _scan_rows(con, "embeddings_queue", "seq_id, topic, id", on_skip):
        if topic and topic.endswith(collection_id) and _seq_id(seq_id) > cutoff and doc_id:
            keys.add(doc_id.split("#", 1)[0])
    return keys


def repair_index(
    config_path: str | None = None,
    *,
    dry_run: bool = False,
    progress: Callable[[str], None] = print,
) -> RepairReport:
    """Rebuild the configured semantic index from its own files.

    ``dry_run`` checks everything and reports what would be kept, writing
    nothing but a temporary working copy that is removed afterwards.
    """
    from .chroma_client import create_chroma_client, release_chroma_clients
    from .config import resolve_chroma_dir

    collection_name = "zotero_library"
    configured_dir = None
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            section = json.load(f).get("semantic_search", {})
        collection_name = section.get("collection_name", collection_name)
        configured_dir = section.get("persist_directory")
    persist_dir = resolve_chroma_dir(configured_dir)
    source_db = persist_dir / "chroma.sqlite3"
    if not source_db.exists():
        raise RepairError(f"no index at {persist_dir}; nothing to repair. Run: zotero-mcp update-db")

    report = RepairReport(persist_directory=str(persist_dir), dry_run=dry_run)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    new_dir = persist_dir.with_name(f"{persist_dir.name}.repair-{stamp}")

    def skipped(n: int) -> None:
        report.skipped_rows += n

    with tempfile.TemporaryDirectory(prefix=".repair-work-", dir=persist_dir.parent) as work:
        work_dir = Path(work)
        staged = work_dir / "source.sqlite3"
        progress(f"Staging a copy of {source_db} ...")
        _clone_or_copy(source_db, staged)
        report.header_page_count_fixed = _fix_header_page_count(staged)
        if report.header_page_count_fixed:
            progress("The database header undercounted its pages (an interrupted write); corrected in the copy.")

        src = sqlite3.connect(f"file:{staged}?mode=ro", uri=True)
        collection_id, dim, meta_segment, vector_segment = _locate(src, collection_name)
        segment_dir = persist_dir / vector_segment
        if not (segment_dir / "header.bin").exists():
            raise RepairError(
                "the index has no saved vector files (small indexes keep them in memory "
                f"until 1000 documents). {_REBUILD_HINT}"
            )
        vectors = _open_hnsw(segment_dir, dim)
        progress(f"Vector index: {len(vectors.label_to_row)} live vectors of dimension {dim}.")

        need = int(source_db.stat().st_size * 1.3) + vectors.rows.nbytes
        free = shutil.disk_usage(persist_dir.parent).free
        if not dry_run and free < need:
            raise RepairError(
                f"not enough free disk space: the rebuild needs about {need / 2**30:.1f} GB "
                f"and {free / 2**30:.1f} GB is free on {persist_dir.parent}."
            )

        # Copy what is needed into a clean working database, by full scans.
        work_db = sqlite3.connect(work_dir / "work.sqlite3")
        work_db.execute("CREATE TABLE emb (id INTEGER, doc_id TEXT)")
        work_db.execute("CREATE TABLE meta (id INTEGER, key TEXT, s, i, f, b)")
        progress("Reading documents and metadata ...")
        for batch in _batched(_scan_rows(src, "embeddings", "id, segment_id, embedding_id", skipped)):
            work_db.executemany(
                "INSERT INTO emb VALUES (?, ?)",
                [(rid, doc_id) for rid, seg, doc_id in batch if seg == meta_segment and isinstance(doc_id, str)],
            )
        for batch in _batched(_scan_rows(
            src, "embedding_metadata", "id, key, string_value, int_value, float_value, bool_value", skipped
        )):
            work_db.executemany("INSERT INTO meta VALUES (?, ?, ?, ?, ?, ?)", batch)
        work_db.execute("CREATE INDEX meta_id ON meta(id)")
        pending = _pending_item_keys(src, collection_id, (meta_segment, vector_segment), skipped)
        src.close()

        row_ids: dict[str, int] = {}
        for rid, doc_id in work_db.execute("SELECT id, doc_id FROM emb"):
            if doc_id in row_ids:  # a torn page can repeat a row; trust neither copy
                row_ids[doc_id] = -1
            else:
                row_ids[doc_id] = rid
        by_item: dict[str, list[str]] = defaultdict(list)
        for doc_id in row_ids:
            by_item[doc_id.split("#", 1)[0]].append(doc_id)
        progress(f"Found {len(row_ids)} chunks in {len(by_item)} items.")

        target = None
        if not dry_run:
            target = create_chroma_client(config_path, persist_directory=str(new_dir))

        keys = sorted(by_item)
        for start in range(0, len(keys), _ITEMS_PER_BATCH):
            batch_keys = keys[start:start + _ITEMS_PER_BATCH]
            metadata, conflicted = _load_metadata(
                work_db, [(d, row_ids[d]) for k in batch_keys for d in by_item[k] if row_ids[d] != -1]
            )
            ids, rows, documents, metadatas = [], [], [], []
            for key in batch_keys:
                if key in pending:
                    report.excluded[key] = "write still queued"
                    continue
                if any(row_ids[d] == -1 or d in conflicted for d in by_item[key]):
                    report.excluded[key] = "duplicate or conflicting rows"
                    continue
                checked = _check_item(key, by_item[key], metadata, vectors)
                if isinstance(checked, str):
                    report.excluded[key] = checked
                    continue
                for acc, part in zip((ids, rows, documents, metadatas), checked):
                    acc += part
            if not ids:
                continue
            embeddings = vectors.get(rows)
            norms = np.linalg.norm(embeddings, axis=1)
            bad = {ids[i].split("#", 1)[0] for i in np.flatnonzero(~np.isfinite(norms) | (norms == 0))}
            if bad:
                for key in bad:
                    report.excluded[key] = "invalid vector"
                keep = [i for i, d in enumerate(ids) if d.split("#", 1)[0] not in bad]
                ids = [ids[i] for i in keep]
                documents = [documents[i] for i in keep]
                metadatas = [metadatas[i] for i in keep]
                embeddings = embeddings[keep]
            report.chunks_kept += len(ids)
            if target is not None and ids:
                target.upsert_embeddings(documents, metadatas, ids, embeddings)
            done = min(start + _ITEMS_PER_BATCH, len(keys))
            if done % (_ITEMS_PER_BATCH * 25) == 0 or done == len(keys):
                progress(f"  {done}/{len(keys)} items checked, {report.chunks_kept} chunks kept")
        work_db.close()
        report.items_kept = len(by_item) - len(report.excluded)

    if target is not None:
        written = target.collection.count()
        if written != report.chunks_kept:
            raise RepairError(
                f"the rebuilt index at {new_dir} holds {written} chunks, expected "
                f"{report.chunks_kept}; the old index was left in place."
            )
        del target
        release_chroma_clients()
        broken_dir = persist_dir.with_name(f"{persist_dir.name}.broken-{stamp}")
        os.rename(persist_dir, broken_dir)
        os.rename(new_dir, persist_dir)
        report.broken_directory = str(broken_dir)

    report_path = persist_dir.with_name(f"{persist_dir.name}.repair-{stamp}.json")
    report_path.write_text(json.dumps({
        "dry_run": dry_run,
        "items_kept": report.items_kept,
        "chunks_kept": report.chunks_kept,
        "header_page_count_fixed": report.header_page_count_fixed,
        "skipped_rows": report.skipped_rows,
        "broken_directory": report.broken_directory,
        "excluded": report.excluded,
    }, indent=1))
    report.report_path = str(report_path)
    return report


def _batched(rows: Iterator[tuple], size: int = 50_000) -> Iterator[list[tuple]]:
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_metadata(
    work_db: sqlite3.Connection, docs: list[tuple[str, int]]
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Metadata keyed by document id, and the ids whose metadata contradicts itself.

    A torn page can repeat a row with a different value; neither copy can be
    trusted, so the chunk is reported rather than one value picked.
    """
    by_row: dict[int, dict[str, Any]] = defaultdict(dict)
    conflicted_rows: set[int] = set()
    wanted = [rid for _doc_id, rid in docs]
    for i in range(0, len(wanted), 900):
        part = wanted[i:i + 900]
        query = f"SELECT id, key, s, i, f, b FROM meta WHERE id IN ({','.join('?' * len(part))})"
        for rid, key, s, iv, fv, bv in work_db.execute(query, part):
            value = _metadata_value(s, iv, fv, bv)
            if by_row[rid].setdefault(key, value) != value:
                conflicted_rows.add(rid)
    metadata = {doc_id: by_row[rid] for doc_id, rid in docs if rid in by_row}
    return metadata, {doc_id for doc_id, rid in docs if rid in conflicted_rows}
