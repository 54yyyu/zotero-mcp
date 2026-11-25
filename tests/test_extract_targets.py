"""Extracting attachment files one by one, for the item-keyed scan (#826).

``extract_targets`` reads each selected file under its own attachment key:
its transient-cache entry, its parse, and, failing both, its own
``.zotero-ft-cache``, never a sibling's. Plain .txt files stand in for PDFs,
as in test_multicore_extraction.py.
"""

import sqlite3
from pathlib import Path

import pytest

from zotero_mcp import fulltext_cache
from zotero_mcp.local_db import ExtractionTarget, LocalZoteroReader


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)


class FileReader(LocalZoteroReader):
    """Reader with no sqlite behind it, and Zotero's caches in a dict."""

    def __init__(self, cfg=None, workers=1, ft_cache=False):
        self._connection = None
        self.db_path = None
        self.pdf_max_pages = 50
        self.attachment_priority = ("pdf", "html", "other")
        self.extraction_workers = workers
        self.fulltext_cache_enabled = ft_cache
        self.config_path = cfg
        self.zotero_ft_cache = {}

    def _iter_parent_attachments(self, item_id):
        raise AssertionError("a file is extracted on its own, never through its item")

    def _read_zotero_ft_cache(self, attachment_key):
        return self.zotero_ft_cache.get(attachment_key)


def _target(path: Path, key: str) -> ExtractionTarget:
    st = path.stat() if path.exists() else None
    return ExtractionTarget(
        key=key,
        path=path,
        content_type="text/plain",
        category="text",
        file_stamp=f"{st.st_mtime_ns}:{st.st_size}" if st else "0:0",
        size=st.st_size if st else 0,
    )


@pytest.fixture
def files(tmp_path):
    out = []
    for i in range(6):
        p = tmp_path / f"doc{i}.txt"
        p.write_text(f"contents of document {i}\n" * 20, encoding="utf-8")
        out.append(_target(p, f"ATT{i:05d}"))
    return out


@pytest.mark.parametrize("workers", [1, 3])
def test_every_file_is_reported_under_its_own_key(files, workers):
    got = dict(FileReader(workers=workers).extract_targets(files))

    assert set(got) == {t.key for t in files}
    assert got["ATT00002"][0].startswith("contents of document 2")


@pytest.mark.parametrize("workers", [1, 3])
def test_an_unreadable_file_falls_back_to_its_own_zotero_cache_only(files, tmp_path, workers):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    reader = FileReader(workers=workers)
    reader.zotero_ft_cache["SIBLING1"] = "a sibling's text"
    reader.zotero_ft_cache["OWNCACHE"] = "its own text"

    got = dict(reader.extract_targets([_target(empty, "NOCACHE1"), _target(empty, "OWNCACHE")]))

    assert got["NOCACHE1"] is None
    assert got["OWNCACHE"] == ("its own text", "zotero-cache")


@pytest.mark.parametrize("workers", [1, 3])
def test_text_is_cached_under_the_attachment_key(files, tmp_path, workers):
    cfg = str(tmp_path / "cfg" / "config.json")
    reader = FileReader(cfg=cfg, workers=workers, ft_cache=True)
    list(reader.extract_targets(files[:2]))

    assert fulltext_cache.evict_many(["ATT00000"], config_path=cfg) == 1
    # A cached file is not parsed again.
    files[1].path.chmod(0o000)
    try:
        assert dict(reader.extract_targets([files[1]]))["ATT00001"] is not None
    finally:
        files[1].path.chmod(0o644)


def _db(tmp_path) -> LocalZoteroReader:
    db = tmp_path / "zotero.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT, dateModified TEXT);
        CREATE TABLE collections (collectionID INTEGER PRIMARY KEY, key TEXT, parentCollectionID INT);
        CREATE TABLE collectionItems (collectionID INT, itemID INT);
        INSERT INTO items VALUES (1, 'ITEMAAAA', '2024-01-01 00:00:00'), (2, 'LONEPDF1', '2024-02-02 00:00:00'),
                                 (3, 'OUTSIDE1', '2024-03-03 00:00:00');
        INSERT INTO collections VALUES (10, 'COLLTOP1', NULL), (11, 'COLLSUB1', 10);
        INSERT INTO collectionItems VALUES (10, 1), (11, 2);
        """
    )
    conn.commit()
    conn.close()
    return LocalZoteroReader(db_path=str(db))


def test_dates_by_key_cover_every_item(tmp_path):
    reader = _db(tmp_path)
    try:
        assert reader.get_date_modified_by_key()["LONEPDF1"] == "2024-02-02 00:00:00"
    finally:
        reader.close()


def test_collection_members_include_subcollections(tmp_path):
    reader = _db(tmp_path)
    try:
        assert reader.get_collection_item_keys(["COLLTOP1"]) == {"ITEMAAAA", "LONEPDF1"}
        assert reader.get_collection_item_keys(["NOSUCHCO"]) == set()
    finally:
        reader.close()
