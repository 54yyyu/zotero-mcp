"""An attachment out of reach is still selected, by its own text cache (#826, final review C1).

A file that is not on disk (an unmounted drive, files kept on demand) or is in
a format only Zotero's ``.zotero-ft-cache`` covers (an EPUB) keeps its full-text
coverage: its target is its own cache file, never a sibling's. A real
``LocalZoteroReader`` on a small SQLite database in a temp directory.
"""

import sqlite3
from pathlib import Path

import pytest

from zotero_mcp.local_db import LocalZoteroReader

DATE = "2024-01-01 00:00:00"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)


def _reader(tmp_path, rows):
    """Item 1 with the attachment ``rows`` ``(key, path, content type)``; standalone ones have no parent."""
    db = tmp_path / "zotero.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT, libraryID INT, dateModified TEXT);
        CREATE TABLE itemAttachments (itemID INTEGER PRIMARY KEY, parentItemID INT, path TEXT, contentType TEXT);
        CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
        """
    )
    conn.execute("INSERT INTO items VALUES (1, 'ITEM0001', 1, ?)", (DATE,))
    for item_id, (key, path, ctype, parent) in enumerate(rows, 10):
        conn.execute("INSERT INTO items VALUES (?, ?, 1, ?)", (item_id, key, DATE))
        conn.execute("INSERT INTO itemAttachments VALUES (?, ?, ?, ?)", (item_id, 1 if parent else None, path, ctype))
    conn.commit()
    conn.close()
    storage = tmp_path / "storage"
    reader = LocalZoteroReader(db_path=str(db))
    reader._get_storage_dir = lambda: storage
    return reader, storage


def _cache(storage, key, text):
    (storage / key).mkdir(parents=True, exist_ok=True)
    (storage / key / ".zotero-ft-cache").write_text(text, encoding="utf-8")


def test_a_file_missing_on_disk_is_selected_by_its_own_text_cache(tmp_path):
    reader, storage = _reader(tmp_path, [("PDFAAAAA", "storage:a.pdf", "application/pdf", True)])
    _cache(storage, "PDFAAAAA", "text Zotero extracted")
    try:
        (target,) = reader._resolve_extraction_targets(1)
        cache = storage / "PDFAAAAA" / ".zotero-ft-cache"
        st = cache.stat()

        assert (target.key, target.category, target.ft_cache) == ("PDFAAAAA", "pdf", True)
        assert target.path == cache
        assert target.file_stamp == f"ftcache:{st.st_mtime_ns}:{st.st_size}"
        assert dict(reader.extract_targets([target])) == {"PDFAAAAA": ("text Zotero extracted", "zotero-cache")}
    finally:
        reader.close()


def test_an_epub_is_selected_by_its_text_cache_when_nothing_better_exists(tmp_path):
    reader, storage = _reader(tmp_path, [("EPUBAAAA", "storage:b.epub", "application/epub+zip", True)])
    (storage / "EPUBAAAA").mkdir(parents=True)
    (storage / "EPUBAAAA" / "b.epub").write_bytes(b"not parsed here")
    _cache(storage, "EPUBAAAA", "the book's text")
    try:
        (target,) = reader._resolve_extraction_targets(1)

        assert (target.key, target.ft_cache) == ("EPUBAAAA", True)
        assert dict(reader.extract_targets([target]))["EPUBAAAA"] == ("the book's text", "zotero-cache")
    finally:
        reader.close()


def test_a_sibling_is_never_read_for_a_file_without_a_cache(tmp_path):
    reader, storage = _reader(
        tmp_path,
        [
            ("PDFAAAAA", "storage:a.pdf", "application/pdf", True),
            ("EPUBBBBB", "storage:b.epub", "application/epub+zip", True),
        ],
    )
    _cache(storage, "EPUBBBBB", "the sibling's text")
    try:
        targets = reader._resolve_extraction_targets(1)

        # The PDF has neither file nor cache: nothing of it to index. The EPUB
        # stands for itself, under its own key.
        assert [t.key for t in targets] == ["EPUBBBBB"]
    finally:
        reader.close()


def test_a_file_on_disk_is_parsed_as_before(tmp_path):
    reader, storage = _reader(tmp_path, [("PDFAAAAA", "storage:a.pdf", "application/pdf", True)])
    (storage / "PDFAAAAA").mkdir(parents=True)
    (storage / "PDFAAAAA" / "a.pdf").write_bytes(b"%PDF-1.4")
    _cache(storage, "PDFAAAAA", "cache text")
    try:
        (target,) = reader._resolve_extraction_targets(1)

        assert target.ft_cache is False
        assert target.path == storage / "PDFAAAAA" / "a.pdf"
    finally:
        reader.close()


def test_an_empty_cache_is_no_target(tmp_path):
    reader, storage = _reader(tmp_path, [("PDFAAAAA", "storage:a.pdf", "application/pdf", True)])
    _cache(storage, "PDFAAAAA", "")
    try:
        assert reader._resolve_extraction_targets(1) == []
    finally:
        reader.close()


def test_a_standalone_file_missing_on_disk_is_indexed_from_its_cache_but_an_epub_is_not(tmp_path):
    reader, storage = _reader(
        tmp_path,
        [
            ("LONEPDF1", "storage:a.pdf", "application/pdf", False),
            ("LONEEPUB", "storage:b.epub", "application/epub+zip", False),
        ],
    )
    _cache(storage, "LONEPDF1", "lone text")
    _cache(storage, "LONEEPUB", "book text")
    try:
        rows = {row[0]: row for row in reader._iter_standalone_attachments()}
        pdf = reader._resolve_standalone_target(rows["LONEPDF1"][0], rows["LONEPDF1"][1], rows["LONEPDF1"][2])
        epub = reader._resolve_standalone_target(rows["LONEEPUB"][0], rows["LONEEPUB"][1], rows["LONEEPUB"][2])

        assert pdf is not None and pdf.ft_cache
        assert epub is None
    finally:
        reader.close()
