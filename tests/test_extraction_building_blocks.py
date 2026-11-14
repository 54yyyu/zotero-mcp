"""Extraction primitives for the item-keyed index (#826).

Many attachments per item, per-file failure markers
and the stamps a re-key leaves. None of this changes what the indexer writes yet.
"""

import sqlite3
from pathlib import Path

import pytest

from zotero_mcp import fulltext_cache
from zotero_mcp.extract import DEFAULT_ATTACHMENT_PRIORITY, select_by_priority
from zotero_mcp.local_db import LocalZoteroReader


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Nothing here may reach the real ~/.config/zotero-mcp."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))


@pytest.fixture
def config_path(tmp_path):
    return str(tmp_path / "cfg" / "config.json")


# ---------------------------------------------------------------------------
# select_by_priority
# ---------------------------------------------------------------------------


def test_select_by_priority_returns_every_pdf_and_no_snapshot():
    cands = [("pdf", 5, "B"), ("html", 9, "S"), ("pdf", 50, "A")]
    assert select_by_priority(cands, DEFAULT_ATTACHMENT_PRIORITY) == ["A", "B"]


def test_select_by_priority_falls_to_next_category():
    assert select_by_priority([("html", 1, "S")], DEFAULT_ATTACHMENT_PRIORITY) == ["S"]


def test_select_by_priority_empty():
    assert select_by_priority([], DEFAULT_ATTACHMENT_PRIORITY) == []


def test_select_by_priority_other_is_catch_all_of_unnamed():
    cands = [("markdown", 1, "M"), ("text", 1, "T"), ("pdf", 1, "P")]
    assert select_by_priority(cands, ("markdown", "other")) == ["M"]
    assert select_by_priority(cands[1:], ("markdown", "other")) == ["P", "T"]
    # not named and no "other": never chosen
    assert select_by_priority(cands, ("html",)) == []


# ---------------------------------------------------------------------------
# targets (sqlite)
# ---------------------------------------------------------------------------


def _make_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT, libraryID INT,
                            dateModified TEXT);
        CREATE TABLE itemAttachments (itemID INTEGER PRIMARY KEY, parentItemID INT,
                                      path TEXT, contentType TEXT);
        CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
        """
    )
    conn.execute("INSERT INTO items VALUES (1, 'PARENT1', 1, '2024-01-01 00:00:00')")
    conn.commit()
    conn.close()


def _add(db_path, item_id, key, parent, path, ctype, trashed=False, lib=1):
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO items VALUES (?, ?, ?, ?)", (item_id, key, lib, "2024-02-02 00:00:00"))
    conn.execute("INSERT INTO itemAttachments VALUES (?, ?, ?, ?)", (item_id, parent, path, ctype))
    if trashed:
        conn.execute("INSERT INTO deletedItems VALUES (?)", (item_id,))
    conn.commit()
    conn.close()


def _file(storage: Path, key: str, name: str, data: bytes = b"x") -> Path:
    d = storage / key
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(data)
    return p


@pytest.fixture
def lib(tmp_path):
    db = tmp_path / "zotero.sqlite"
    _make_db(db)
    storage = tmp_path / "storage"
    storage.mkdir()
    reader = LocalZoteroReader(db_path=str(db))
    reader.attachment_priority = DEFAULT_ATTACHMENT_PRIORITY
    yield db, storage, reader
    reader.close()


def test_targets_resolve_all_pdfs_with_stamp(lib):
    db, storage, reader = lib
    a = _file(storage, "PDFAAAAA", "a.pdf", b"aaaa")
    b = _file(storage, "PDFBBBBB", "b.pdf", b"bb")
    _file(storage, "HTMLCCCC", "s.html", b"<p>x</p>")
    _add(db, 10, "PDFBBBBB", 1, "storage:b.pdf", "application/pdf")
    _add(db, 11, "PDFAAAAA", 1, "storage:a.pdf", "application/pdf")
    _add(db, 12, "HTMLCCCC", 1, "storage:s.html", "text/html")

    targets = reader._resolve_extraction_targets(1)

    # In the database's order (item id), as 0.17.1 listed them: ties go to the first.
    assert [t.key for t in targets] == ["PDFBBBBB", "PDFAAAAA"]
    t = targets[1]
    assert t.path == a and t.category == "pdf" and t.content_type == "application/pdf"
    st = a.stat()
    assert t.file_stamp == f"{st.st_mtime_ns}:{st.st_size}"
    assert targets[0].path == b


def test_targets_exclude_trashed_and_unresolvable(lib):
    db, storage, reader = lib
    _file(storage, "LIVEAAAA", "a.pdf")
    _file(storage, "TRASHBBB", "b.pdf")
    _add(db, 10, "LIVEAAAA", 1, "storage:a.pdf", "application/pdf")
    _add(db, 11, "TRASHBBB", 1, "storage:b.pdf", "application/pdf", trashed=True)
    _add(db, 12, "MISSINGC", 1, "storage:gone.pdf", "application/pdf")
    assert [t.key for t in reader._resolve_extraction_targets(1)] == ["LIVEAAAA"]


def test_targets_use_filename_drift_fallback(lib):
    db, storage, reader = lib
    _file(storage, "DRIFTAAA", "actual-name.pdf")
    _add(db, 10, "DRIFTAAA", 1, "storage:recorded-name.pdf", "application/pdf")
    targets = reader._resolve_extraction_targets(1)
    assert [t.path.name for t in targets] == ["actual-name.pdf"]


def test_single_target_still_picks_largest_pdf(lib):
    db, storage, reader = lib
    _file(storage, "SMALLAAA", "a.pdf", b"x")
    big = _file(storage, "BIGBBBBB", "b.pdf", b"x" * 100)
    _add(db, 10, "SMALLAAA", 1, "storage:a.pdf", "application/pdf")
    _add(db, 11, "BIGBBBBB", 1, "storage:b.pdf", "application/pdf")
    assert reader._resolve_extraction_target(1) == (big, "BIGBBBBB")


# ---------------------------------------------------------------------------
# ft-cache fallback regression
# ---------------------------------------------------------------------------


def _fallback_reader(storage, config_path):
    class _R(LocalZoteroReader):
        def __init__(self):
            self.db_path = "/dev/null"
            self._connection = None
            self.pdf_max_pages = 10
            self.attachment_priority = DEFAULT_ATTACHMENT_PRIORITY
            self.fulltext_cache_enabled = True
            self.config_path = config_path

        def _iter_parent_attachments(self, _id):
            yield "FIRSTAAA", "storage:f.pdf", "application/pdf"
            yield "CHOSENBB", "storage:c.pdf", "application/pdf"

        def _get_storage_dir(self):
            return storage

    return _R()


def test_ft_cache_fallback_reads_the_chosen_attachments_own_cache(tmp_path, config_path):
    storage = tmp_path / "storage"
    (storage / "FIRSTAAA").mkdir(parents=True)
    (storage / "FIRSTAAA" / ".zotero-ft-cache").write_text("text of FIRST")
    (storage / "CHOSENBB").mkdir(parents=True)
    (storage / "CHOSENBB" / ".zotero-ft-cache").write_text("text of CHOSEN")
    chosen_file = _file(storage, "CHOSENBB", "c.pdf")

    r = _fallback_reader(storage, config_path)
    text, source = r._zotero_ft_cache_fallback(1, (chosen_file, "CHOSENBB"), "PARENT")
    assert text == "text of CHOSEN" and source == "zotero-cache"
    st = chosen_file.stat()
    cached = fulltext_cache.get_cached_text(
        "CHOSENBB",
        st.st_mtime_ns,
        st.st_size,
        profile=r._cache_profile(),
        config_path=config_path,
    )
    assert cached == ("text of CHOSEN", "zotero-cache")


def test_ft_cache_fallback_chosen_without_own_cache_returns_none(tmp_path, config_path):
    storage = tmp_path / "storage"
    (storage / "FIRSTAAA").mkdir(parents=True)
    (storage / "FIRSTAAA" / ".zotero-ft-cache").write_text("text of FIRST")
    chosen_file = _file(storage, "CHOSENBB", "c.pdf")

    r = _fallback_reader(storage, config_path)
    assert r._zotero_ft_cache_fallback(1, (chosen_file, "CHOSENBB"), "P") is None


# ---------------------------------------------------------------------------
# fulltext cache: attachment-level item_key, failure markers
# ---------------------------------------------------------------------------


def test_cache_entry_can_be_keyed_by_attachment_and_evicted(config_path):
    fulltext_cache.put_cached_text("ATTKEY01", 1, 2, "pdf", "ATTKEY01", "hello", config_path=config_path)
    assert fulltext_cache.evict_many(["ATTKEY01"], config_path=config_path) == 1
    assert fulltext_cache.get_cached_text("ATTKEY01", 1, 2, config_path=config_path) is None


def test_failure_marker_is_per_file_stamp(config_path):
    assert not fulltext_cache.failed("ATTKEY01", "1:2", config_path=config_path)
    fulltext_cache.record_failure("ATTKEY01", "1:2", config_path=config_path)
    assert fulltext_cache.failed("ATTKEY01", "1:2", config_path=config_path)
    assert not fulltext_cache.failed("ATTKEY01", "9:2", config_path=config_path)
    assert not fulltext_cache.failed("OTHERKEY", "1:2", config_path=config_path)


def test_failure_marker_coexists_with_cached_text(config_path):
    fulltext_cache.record_failure("ATTKEY01", "1:2", config_path=config_path)
    fulltext_cache.put_cached_text("ATTKEY02", 1, 2, "pdf", None, "t", config_path=config_path)
    assert fulltext_cache.failed("ATTKEY01", "1:2", config_path=config_path)
    assert fulltext_cache.get_cached_text("ATTKEY02", 1, 2, config_path=config_path)


# ---------------------------------------------------------------------------
# stamps left by a re-key
# ---------------------------------------------------------------------------


def test_rekeyed_stamps_are_remembered_per_attachment(config_path):
    assert fulltext_cache.rekeyed_stamps(config_path=config_path) == {}

    fulltext_cache.remember_rekeyed_stamp("ATT1", "1:11000", config_path=config_path)
    fulltext_cache.remember_rekeyed_stamp("ATT2", "2:5", config_path=config_path)
    fulltext_cache.remember_rekeyed_stamp("ATT1", "3:11000", config_path=config_path)

    assert fulltext_cache.rekeyed_stamps(config_path=config_path) == {"ATT1": "3:11000", "ATT2": "2:5"}


def test_rekeyed_stamps_coexist_with_cached_text_and_failures(config_path):
    fulltext_cache.put_cached_text("ATT1", 1, 2, "pdf", None, "t", config_path=config_path)
    fulltext_cache.record_failure("ATT2", "1:2", config_path=config_path)
    fulltext_cache.remember_rekeyed_stamp("ATT3", "1:3", config_path=config_path)

    assert fulltext_cache.get_cached_text("ATT1", 1, 2, config_path=config_path)
    assert fulltext_cache.failed("ATT2", "1:2", config_path=config_path)
    assert fulltext_cache.rekeyed_stamps(config_path=config_path) == {"ATT3": "1:3"}
