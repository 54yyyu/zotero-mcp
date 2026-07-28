import sqlite3
from pathlib import Path

from zotero_mcp.local_db import PERSONAL_LIBRARY_GROUP_ID, LocalZoteroReader, ZoteroItem


class FakeLocalZoteroReader(LocalZoteroReader):
    """Subclass that skips DB init and allows injecting fake attachment text."""

    def __init__(self, fake_text: str = "", fake_pdf_path: Path | None = None):
        # Skip parent __init__ entirely — no DB needed
        self.db_path = "/dev/null"
        self._connection = None
        self.pdf_max_pages = 10
        self.pdf_timeout = 30
        self._fake_text = fake_text
        self._fake_pdf_path = fake_pdf_path

    def _iter_parent_attachments(self, parent_item_id: int):
        """Yield a single fake PDF attachment."""
        yield "FAKEKEY", "storage:fake.pdf", "application/pdf"

    def _resolve_attachment_path(self, attachment_key: str, zotero_path: str):
        """Return the injected fake path."""
        return self._fake_pdf_path

    def _extract_text_from_file(self, file_path):
        """Return the injected fake text instead of reading a real file."""
        return self._fake_text


def test_extract_fulltext_preserves_long_text(tmp_path):
    """Extracted text longer than 10,000 chars should NOT be truncated."""
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.touch()
    long_text = "x" * 25000
    reader = FakeLocalZoteroReader(fake_text=long_text, fake_pdf_path=fake_pdf)
    result = reader._extract_fulltext_for_item(1)
    assert result is not None
    text, source = result
    assert len(text) == 25000, f"Text was truncated to {len(text)} chars"
    assert source == "pdf"


def test_extract_fulltext_empty_returns_none(tmp_path):
    """Empty extracted text should return None."""
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.touch()
    reader = FakeLocalZoteroReader(fake_text="", fake_pdf_path=fake_pdf)
    result = reader._extract_fulltext_for_item(1)
    assert result is None


def test_get_searchable_text_preserves_long_fulltext():
    """get_searchable_text should not aggressively truncate fulltext."""
    long_fulltext = "y" * 20000
    item = ZoteroItem(item_id=1, key="TEST", item_type_id=1, fulltext=long_fulltext)
    text = item.get_searchable_text()
    # The full 20,000 chars should appear in the output (not truncated to 5,000)
    assert "y" * 20000 in text


def test_get_searchable_text_truncates_at_limit():
    """Fulltext beyond 50,000 chars should be truncated with ellipsis."""
    huge_fulltext = "z" * 60000
    item = ZoteroItem(item_id=1, key="TEST", item_type_id=1, fulltext=huge_fulltext)
    text = item.get_searchable_text()
    # Should contain exactly 50,000 z's plus "..." — not all 60,000
    assert "z" * 50000 in text
    assert "z" * 50001 not in text
    assert "..." in text


class TestResolveAttachmentPath:
    """Tests for _resolve_attachment_path handling of various Zotero path formats."""

    def _make_reader(self, tmp_path):
        """Create a LocalZoteroReader-like object for path resolution tests.

        Uses the real _resolve_attachment_path (not the fake override).
        """
        reader = FakeLocalZoteroReader()
        reader.db_path = str(tmp_path / "zotero.sqlite")
        # Bind the real method so we test actual path resolution
        reader._resolve_attachment_path = LocalZoteroReader._resolve_attachment_path.__get__(reader)
        reader._get_storage_dir = LocalZoteroReader._get_storage_dir.__get__(reader)
        reader._get_base_attachment_path = LocalZoteroReader._get_base_attachment_path.__get__(reader)
        return reader

    def test_storage_path(self, tmp_path):
        """'storage:file.pdf' resolves to <storage_dir>/<key>/file.pdf."""
        reader = self._make_reader(tmp_path)
        (tmp_path / "storage" / "ABC123").mkdir(parents=True)
        result = reader._resolve_attachment_path("ABC123", "storage:paper.pdf")
        assert result == tmp_path / "storage" / "ABC123" / "paper.pdf"

    def test_absolute_path(self, tmp_path):
        """Absolute path passes through unchanged."""
        reader = self._make_reader(tmp_path)
        result = reader._resolve_attachment_path("X", "/home/user/papers/file.pdf")
        assert result == Path("/home/user/papers/file.pdf")

    def test_file_url(self, tmp_path):
        """'file:///path/to/file.pdf' resolves to the decoded path."""
        reader = self._make_reader(tmp_path)
        result = reader._resolve_attachment_path("X", "file:///home/user/my%20paper.pdf")
        assert result == Path("/home/user/my paper.pdf")

    def test_attachments_with_base_path(self, tmp_path):
        """'attachments:rel/path.pdf' resolves against baseAttachmentPath from prefs.js."""
        reader = self._make_reader(tmp_path)
        base_dir = tmp_path / "linked_papers"
        base_dir.mkdir()
        # Write a prefs.js with baseAttachmentPath
        prefs = tmp_path / "prefs.js"
        prefs.write_text(
            f'user_pref("extensions.zotero.baseAttachmentPath", "{base_dir}");\n'
        )
        result = reader._resolve_attachment_path("X", "attachments:subfolder/paper.pdf")
        assert result == base_dir / "subfolder" / "paper.pdf"

    def test_attachments_without_base_path_returns_none(self, tmp_path):
        """'attachments:' path returns None when no baseAttachmentPath is configured."""
        reader = self._make_reader(tmp_path)
        # No prefs.js exists
        result = reader._resolve_attachment_path("X", "attachments:subfolder/paper.pdf")
        assert result is None

    def test_empty_path_returns_none(self, tmp_path):
        """Empty/None path returns None."""
        reader = self._make_reader(tmp_path)
        assert reader._resolve_attachment_path("X", "") is None
        assert reader._resolve_attachment_path("X", None) is None

    def test_unknown_prefix_returns_none(self, tmp_path):
        """Unknown path format returns None."""
        reader = self._make_reader(tmp_path)
        assert reader._resolve_attachment_path("X", "ftp://something") is None


class TestGetAttachmentPaths:
    """Tests for the public get_attachment_paths helper."""

    def _make_reader(self, tmp_path, attachments, item_key="PARENT"):
        reader = FakeLocalZoteroReader()
        reader.db_path = str(tmp_path / "zotero.sqlite")
        reader._resolve_attachment_path = LocalZoteroReader._resolve_attachment_path.__get__(reader)
        reader._get_storage_dir = LocalZoteroReader._get_storage_dir.__get__(reader)
        reader._get_base_attachment_path = LocalZoteroReader._get_base_attachment_path.__get__(reader)

        reader._iter_parent_attachments = lambda parent_id: iter(attachments)
        reader.get_item_by_key = lambda key: ZoteroItem(item_id=1, key=key, item_type_id=1) if key == item_key else None
        return reader

    def test_returns_resolved_paths(self, tmp_path):
        (tmp_path / "storage" / "ABC123").mkdir(parents=True)
        pdf = tmp_path / "storage" / "ABC123" / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        reader = self._make_reader(tmp_path, [("ABC123", "storage:paper.pdf", "application/pdf")])
        result = reader.get_attachment_paths("PARENT")
        assert len(result) == 1
        assert result[0]["key"] == "ABC123"
        assert result[0]["content_type"] == "application/pdf"
        assert result[0]["resolved_path"] == pdf
        assert result[0]["exists"] is True

    def test_marks_missing_files(self, tmp_path):
        reader = self._make_reader(tmp_path, [("X", "storage:gone.pdf", "application/pdf")])
        result = reader.get_attachment_paths("PARENT")
        assert result[0]["exists"] is False

    def test_unknown_item_returns_empty(self, tmp_path):
        reader = self._make_reader(tmp_path, [])
        assert reader.get_attachment_paths("MISSING") == []

    def test_multiple_attachments(self, tmp_path):
        reader = self._make_reader(tmp_path, [
            ("A", "storage:a.pdf", "application/pdf"),
            ("B", "storage:b.html", "text/html"),
        ])
        result = reader.get_attachment_paths("PARENT")
        assert [a["key"] for a in result] == ["A", "B"]


def _create_feed_db(db_path: Path) -> None:
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE feeds (
            libraryID INTEGER PRIMARY KEY,
            name TEXT,
            url TEXT,
            lastCheck TEXT,
            lastUpdate TEXT,
            lastCheckError TEXT,
            refreshInterval INTEGER
        );
        CREATE TABLE feedItems (
            itemID INTEGER PRIMARY KEY,
            readTime TEXT,
            translatedTime TEXT
        );
        CREATE TABLE items (
            itemID INTEGER PRIMARY KEY,
            key TEXT,
            itemTypeID INTEGER,
            libraryID INTEGER,
            dateAdded TEXT
        );
        CREATE TABLE itemTypes (
            itemTypeID INTEGER PRIMARY KEY,
            typeName TEXT
        );
        CREATE TABLE fields (
            fieldID INTEGER PRIMARY KEY,
            fieldName TEXT
        );
        CREATE TABLE itemData (
            itemID INTEGER,
            fieldID INTEGER,
            valueID INTEGER
        );
        CREATE TABLE itemDataValues (
            valueID INTEGER PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE itemCreators (
            itemID INTEGER,
            creatorID INTEGER
        );
        CREATE TABLE creators (
            creatorID INTEGER PRIMARY KEY,
            firstName TEXT,
            lastName TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO fields (fieldID, fieldName) VALUES (?, ?)",
        [
            (1, "title"),
            (2, "abstractNote"),
            (13, "date"),
            (15, "url"),
            (26, "DOI"),
        ],
    )
    conn.execute(
        """
        INSERT INTO feeds (
            libraryID, name, url, lastCheck, lastUpdate, lastCheckError, refreshInterval
        ) VALUES (10, 'Example Feed', 'https://example.test/feed.xml', NULL, NULL, NULL, 60)
        """
    )
    conn.execute("INSERT INTO itemTypes (itemTypeID, typeName) VALUES (7, 'journalArticle')")
    conn.execute(
        """
        INSERT INTO items (itemID, key, itemTypeID, libraryID, dateAdded)
        VALUES (100, 'FEEDKEY1', 7, 10, '2026-06-01 10:00:00')
        """
    )
    conn.execute(
        "INSERT INTO feedItems (itemID, readTime, translatedTime) VALUES (100, NULL, NULL)"
    )
    conn.executemany(
        "INSERT INTO itemDataValues (valueID, value) VALUES (?, ?)",
        [
            (1001, "Feed Item Title"),
            (1002, "Feed item abstract"),
            (1003, "2024-05-15"),
            (1004, "https://example.test/item"),
            (1005, "10.1234/example.doi"),
        ],
    )
    conn.executemany(
        "INSERT INTO itemData (itemID, fieldID, valueID) VALUES (100, ?, ?)",
        [
            (1, 1001),
            (2, 1002),
            (13, 1003),
            (15, 1004),
            (26, 1005),
        ],
    )
    conn.execute(
        "INSERT INTO creators (creatorID, firstName, lastName) VALUES (1, 'Ada', 'Lovelace')"
    )
    conn.execute("INSERT INTO itemCreators (itemID, creatorID) VALUES (100, 1)")
    conn.commit()
    conn.close()


def test_get_feed_items_includes_publication_date(tmp_path):
    db_path = tmp_path / "zotero.sqlite"
    _create_feed_db(db_path)

    reader = LocalZoteroReader(db_path=str(db_path))
    try:
        items = reader.get_feed_items(10, limit=20)
    finally:
        reader.close()

    assert items[0]["date"] == "2024-05-15"


def test_get_feed_items_includes_doi(tmp_path):
    db_path = tmp_path / "zotero.sqlite"
    _create_feed_db(db_path)

    reader = LocalZoteroReader(db_path=str(db_path))
    try:
        items = reader.get_feed_items(10, limit=20)
    finally:
        reader.close()

    assert items[0]["DOI"] == "10.1234/example.doi"


# ---------------------------------------------------------------------------
# get_key_group_map (#163): library attribution for the semantic indexer
# ---------------------------------------------------------------------------

GROUP_ID = 6015547


def _create_multilib_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE libraries (
            libraryID INTEGER PRIMARY KEY,
            type TEXT NOT NULL,
            editable INT NOT NULL,
            filesEditable INT NOT NULL
        );
        CREATE TABLE groups (
            groupID INTEGER PRIMARY KEY,
            libraryID INT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            version INT NOT NULL
        );
        CREATE TABLE items (
            itemID INTEGER PRIMARY KEY,
            key TEXT,
            libraryID INT
        );
        CREATE TABLE deletedItems (
            itemID INTEGER PRIMARY KEY
        );
        """
    )
    # libraryID 1 = personal ("user"), matching Zotero's own convention.
    conn.execute("INSERT INTO libraries VALUES (1, 'user', 1, 1)")
    conn.execute("INSERT INTO libraries VALUES (5, 'group', 1, 1)")
    conn.execute("INSERT INTO libraries VALUES (10, 'feed', 0, 0)")
    conn.execute("INSERT INTO libraries VALUES (20, 'publications', 1, 0)")
    conn.execute(
        f"INSERT INTO groups (groupID, libraryID, name, description, version) "
        f"VALUES ({GROUP_ID}, 5, 'AI in entrepreneurship', '', 1)"
    )

    conn.execute("INSERT INTO items (itemID, key, libraryID) VALUES (1, 'USERKEY1', 1)")
    conn.execute("INSERT INTO items (itemID, key, libraryID) VALUES (2, 'GROUPKEY1', 5)")
    conn.execute("INSERT INTO items (itemID, key, libraryID) VALUES (3, 'FEEDKEY1', 10)")
    conn.execute("INSERT INTO items (itemID, key, libraryID) VALUES (4, 'PUBKEY1', 20)")
    conn.execute("INSERT INTO items (itemID, key, libraryID) VALUES (5, 'DELETEDKEY1', 1)")
    conn.execute("INSERT INTO deletedItems (itemID) VALUES (5)")
    conn.commit()
    conn.close()


def test_get_key_group_map_user_library_maps_to_zero(tmp_path):
    db_path = tmp_path / "zotero.sqlite"
    _create_multilib_db(db_path)
    reader = LocalZoteroReader(db_path=str(db_path))
    try:
        result = reader.get_key_group_map()
    finally:
        reader.close()
    assert result.groups["USERKEY1"] == PERSONAL_LIBRARY_GROUP_ID == 0


def test_get_key_group_map_group_library_maps_to_group_id(tmp_path):
    db_path = tmp_path / "zotero.sqlite"
    _create_multilib_db(db_path)
    reader = LocalZoteroReader(db_path=str(db_path))
    try:
        result = reader.get_key_group_map()
    finally:
        reader.close()
    assert result.groups["GROUPKEY1"] == GROUP_ID


def test_get_key_group_map_feed_items_excluded_not_personal(tmp_path):
    db_path = tmp_path / "zotero.sqlite"
    _create_multilib_db(db_path)
    reader = LocalZoteroReader(db_path=str(db_path))
    try:
        result = reader.get_key_group_map()
    finally:
        reader.close()
    assert "FEEDKEY1" not in result.groups
    assert "FEEDKEY1" in result.excluded_keys


def test_get_key_group_map_publications_library_excluded_not_personal(tmp_path):
    """'My Publications' has no group_id equivalent; must not silently
    collapse into the personal library (0)."""
    db_path = tmp_path / "zotero.sqlite"
    _create_multilib_db(db_path)
    reader = LocalZoteroReader(db_path=str(db_path))
    try:
        result = reader.get_key_group_map()
    finally:
        reader.close()
    assert "PUBKEY1" not in result.groups
    assert "PUBKEY1" in result.excluded_keys


def test_get_key_group_map_deleted_items_omitted_entirely(tmp_path):
    db_path = tmp_path / "zotero.sqlite"
    _create_multilib_db(db_path)
    reader = LocalZoteroReader(db_path=str(db_path))
    try:
        result = reader.get_key_group_map()
    finally:
        reader.close()
    assert "DELETEDKEY1" not in result.groups
    assert "DELETEDKEY1" not in result.excluded_keys


def test_get_key_group_map_full_key_set_partition(tmp_path):
    """Every non-deleted key ends up in exactly one of groups/excluded_keys."""
    db_path = tmp_path / "zotero.sqlite"
    _create_multilib_db(db_path)
    reader = LocalZoteroReader(db_path=str(db_path))
    try:
        result = reader.get_key_group_map()
    finally:
        reader.close()
    assert set(result.groups) | result.excluded_keys == {
        "USERKEY1", "GROUPKEY1", "FEEDKEY1", "PUBKEY1",
    }
    assert not (set(result.groups) & result.excluded_keys)
