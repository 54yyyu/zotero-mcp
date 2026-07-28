"""Regression tests: linked-file attachments must resolve off local storage.

``download_attachment_file`` used to start at ``zot.dump``. For a *linked*
attachment the local Zotero API answers ``/file`` with a 302 to a ``file://``
URL, which httpx refuses to follow ("unsupported protocol"), and a linked file
is by definition never uploaded to WebDAV or Zotero storage — so every source
in the chain failed on a file sitting readable on the same disk.
``zotero_get_annotations`` was the visible casualty: unlike the fulltext and
read_pdf paths it had no local-DB step in front of the downloader.

The fix resolves the path out of zotero.sqlite first, which also covers plain
stored files without a round-trip.
"""

import sqlite3
import types

from zotero_mcp import client as client_module

ATTACHMENT_KEY = "ATT00001"
PARENT_KEY = "PAR00001"


def make_zotero_db(path, *, stored_path):
    """Minimal zotero.sqlite with one item and one attachment at ``stored_path``."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT)"
    )
    conn.execute("INSERT INTO itemTypes VALUES (1, 'journalArticle')")
    conn.execute("INSERT INTO itemTypes VALUES (2, 'attachment')")
    conn.execute(
        """CREATE TABLE items (
            itemID INTEGER PRIMARY KEY, itemTypeID INT, dateAdded TEXT,
            dateModified TEXT, clientDateModified TEXT, libraryID INT,
            key TEXT UNIQUE, version INT, synced INT
        )"""
    )
    for item_id, type_id, key in ((1, 1, PARENT_KEY), (2, 2, ATTACHMENT_KEY)):
        conn.execute(
            "INSERT INTO items VALUES (?, ?, '2026-01-01 00:00:00', "
            "'2026-01-01 00:00:00', '2026-01-01 00:00:00', 1, ?, 1, 0)",
            (item_id, type_id, key),
        )
    conn.execute(
        """CREATE TABLE itemAttachments (
            itemID INTEGER PRIMARY KEY, parentItemID INT, linkMode INT,
            contentType TEXT, charsetID INT, path TEXT
        )"""
    )
    # linkMode 2 == linked_file
    conn.execute(
        "INSERT INTO itemAttachments VALUES (2, 1, 2, 'application/pdf', NULL, ?)",
        (stored_path,),
    )
    conn.execute("CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE itemData (itemID INT, fieldID INT, valueID INT)")
    conn.execute(
        "CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT)"
    )
    conn.execute("INSERT INTO itemDataValues VALUES (11, 'Linked PDF')")
    conn.execute("INSERT INTO itemData VALUES (2, 1, 11)")
    conn.execute("CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT)")
    conn.execute("INSERT INTO fields VALUES (1, 'title')")
    conn.commit()
    conn.close()


def use_local_library(monkeypatch, db_path, *, local=True):
    """Point local mode at ``db_path``.

    Local mode is driven through ZOTERO_LOCAL rather than by patching
    ``is_local_mode``: another test module replaces ``sys.modules
    ["zotero_mcp.utils"]`` with a second copy of utils.py and never restores
    it, so a patch applied by dotted-path name can land on a different module
    object than the one production code resolves through the package
    attribute. The env var is what the function actually reads, so it works
    whichever copy is live.
    """
    monkeypatch.setenv("ZOTERO_LOCAL", "true" if local else "false")
    monkeypatch.setattr(
        "zotero_mcp.config.load_config",
        lambda *a, **k: types.SimpleNamespace(
            resolve_zotero_db_path=lambda: str(db_path)
        ),
    )


class ExplodingClient:
    """Stands in for pyzotero hitting the file:// redirect it cannot follow."""

    def __init__(self):
        self.calls = 0

    def dump(self, *_a, **_k):
        self.calls += 1
        raise Exception("unsupported protocol 'file://'")


def test_linked_file_resolves_without_touching_dump(tmp_path, monkeypatch):
    """The regression: a linked file is served from disk, not via the API."""
    linked = tmp_path / "papers" / "linked.pdf"
    linked.parent.mkdir()
    linked.write_bytes(b"%PDF-1.4 linked payload")

    db_path = tmp_path / "zotero.sqlite"
    make_zotero_db(db_path, stored_path=f"file://{linked}")
    use_local_library(monkeypatch, db_path)

    exploding = ExplodingClient()
    result = client_module.download_attachment_file(
        ATTACHMENT_KEY,
        tmp_path / "out",
        "linked.pdf",
        local_client=exploding,
        web_client=None,
        enable_webdav=False,
    )

    assert result.path is not None, f"download failed: {result.errors}"
    assert result.path.read_bytes() == b"%PDF-1.4 linked payload"
    assert result.source == "Local storage"
    # Before the fix this was the *only* path tried, and it raised.
    assert exploding.calls == 0


def test_returned_file_is_a_copy_not_the_users_original(tmp_path, monkeypatch):
    """Callers delete what they get back; that must not eat the library file."""
    linked = tmp_path / "papers" / "linked.pdf"
    linked.parent.mkdir()
    linked.write_bytes(b"%PDF-1.4 linked payload")

    db_path = tmp_path / "zotero.sqlite"
    make_zotero_db(db_path, stored_path=str(linked))  # bare absolute path form
    use_local_library(monkeypatch, db_path)

    result = client_module.download_attachment_file(
        ATTACHMENT_KEY,
        tmp_path / "out",
        "linked.pdf",
        enable_webdav=False,
    )

    assert result.path is not None, f"download failed: {result.errors}"
    assert result.path.resolve() != linked.resolve()

    result.path.unlink()
    assert linked.exists(), "deleting the scratch copy destroyed the original"


def test_stored_file_also_served_from_disk(tmp_path, monkeypatch):
    """`storage:` attachments skip the API round-trip too."""
    storage_dir = tmp_path / "storage" / ATTACHMENT_KEY
    storage_dir.mkdir(parents=True)
    (storage_dir / "paper.pdf").write_bytes(b"%PDF-1.4 stored payload")

    db_path = tmp_path / "zotero.sqlite"
    make_zotero_db(db_path, stored_path="storage:paper.pdf")
    use_local_library(monkeypatch, db_path)

    exploding = ExplodingClient()
    result = client_module.download_attachment_file(
        ATTACHMENT_KEY,
        tmp_path / "out",
        "paper.pdf",
        local_client=exploding,
        enable_webdav=False,
    )

    assert result.path is not None, f"download failed: {result.errors}"
    assert result.path.read_bytes() == b"%PDF-1.4 stored payload"
    assert exploding.calls == 0


def test_web_mode_does_not_consult_the_local_db(tmp_path, monkeypatch):
    """Guard: a web-API user must not match a same-key row in a local DB."""
    linked = tmp_path / "papers" / "linked.pdf"
    linked.parent.mkdir()
    linked.write_bytes(b"%PDF-1.4 linked payload")

    db_path = tmp_path / "zotero.sqlite"
    make_zotero_db(db_path, stored_path=str(linked))
    use_local_library(monkeypatch, db_path, local=False)

    exploding = ExplodingClient()
    result = client_module.download_attachment_file(
        ATTACHMENT_KEY,
        tmp_path / "out",
        "linked.pdf",
        local_client=exploding,
        enable_webdav=False,
    )

    assert result.path is None
    assert exploding.calls == 1, "web mode must fall through to the API path"


def test_missing_local_db_falls_through_cleanly(tmp_path, monkeypatch):
    """No local DB is not an error — the API chain still runs."""
    use_local_library(monkeypatch, tmp_path / "does-not-exist.sqlite")

    exploding = ExplodingClient()
    result = client_module.download_attachment_file(
        ATTACHMENT_KEY,
        tmp_path / "out",
        "linked.pdf",
        local_client=exploding,
        enable_webdav=False,
    )

    assert result.path is None
    assert exploding.calls == 1
