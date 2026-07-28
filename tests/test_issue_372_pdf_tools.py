"""Regression tests for #372: PDF tools and attachment keys / TOC crashes.

Bug 1 — ``zotero_read_pdf_pages`` given an ATTACHMENT key returned
"No PDF attachment found". ``_get_pdf_path`` only scanned an item's
children for a PDF, and an attachment has no children. The local reader
could not help either: ``get_item_by_key`` is backed by a query that
excludes the 'attachment' item type entirely, so the key resolved to
nothing at all.

Bug 2 — ``zotero_get_pdf_outline`` crashed the whole MCP server. It
downloaded via the legacy ``zot.dump`` path (no WebDAV support) and then
called ``fitz.Document.get_toc()`` in-process, which segfaults on some
born-digital journal PDFs (e.g. doi:10.1038/s41598-022-15627-3). A
segfault cannot be caught in-process — it takes the server down with it,
so the call now runs in a child process that is allowed to die.
"""

import sqlite3
import sys
import types

import pytest
from conftest import DummyContext, FakeZotero

from zotero_mcp import client as client_module
from zotero_mcp import server
from zotero_mcp.local_db import LocalZoteroReader
from zotero_mcp.tools import read_pdf as read_pdf_tools
from zotero_mcp.tools import write as write_tools

ATTACHMENT_KEY = "ATT00001"
PARENT_KEY = "PAR00001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_zotero_db(path, *, stored_filename="paper.pdf"):
    """Create a minimal zotero.sqlite with one item and one PDF attachment."""
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
    conn.execute(
        "INSERT INTO itemAttachments VALUES (2, 1, 0, 'application/pdf', NULL, ?)",
        (f"storage:{stored_filename}",),
    )
    conn.execute("CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE itemData (itemID INT, fieldID INT, valueID INT)")
    conn.execute(
        "CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT)"
    )
    # fieldID 1 is 'title' — held by the parent item and by the attachment.
    conn.execute("INSERT INTO itemDataValues VALUES (10, 'Parent Article')")
    conn.execute("INSERT INTO itemDataValues VALUES (11, 'Full Text PDF')")
    conn.execute("INSERT INTO itemData VALUES (1, 1, 10)")
    conn.execute("INSERT INTO itemData VALUES (2, 1, 11)")
    conn.execute("CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT)")
    conn.execute("INSERT INTO fields VALUES (1, 'title')")
    conn.execute("CREATE TABLE itemNotes (itemID INT, parentItemID INT, note TEXT)")
    conn.execute("CREATE TABLE itemCreators (itemID INT, creatorID INT)")
    conn.execute(
        "CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, "
        "firstName TEXT, lastName TEXT)"
    )
    conn.execute(
        "CREATE TABLE libraries (libraryID INTEGER PRIMARY KEY, type TEXT, "
        "editable INT, filesEditable INT)"
    )
    conn.execute("INSERT INTO libraries VALUES (1, 'user', 1, 1)")
    conn.execute(
        "CREATE TABLE groups (groupID INTEGER PRIMARY KEY, libraryID INT UNIQUE, "
        "name TEXT, description TEXT, version INT)"
    )
    conn.commit()
    conn.close()


def make_library(tmp_path, *, stored_filename="paper.pdf", disk_filename=None):
    """Build a zotero.sqlite plus the attachment file in storage/<KEY>/."""
    db_path = tmp_path / "zotero.sqlite"
    make_zotero_db(db_path, stored_filename=stored_filename)
    attachment_dir = tmp_path / "storage" / ATTACHMENT_KEY
    attachment_dir.mkdir(parents=True)
    pdf_path = attachment_dir / (disk_filename or stored_filename)
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    return db_path, pdf_path


def use_local_library(monkeypatch, db_path, fake_zot):
    """Point local mode at ``db_path`` and stub the Zotero client."""
    monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: True)
    monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)
    monkeypatch.setattr(
        read_pdf_tools,
        "load_config",
        lambda: types.SimpleNamespace(resolve_zotero_db_path=lambda: str(db_path)),
    )


def patch_fitz(monkeypatch, text="Page text."):
    """Fake ``fitz`` so page extraction needs no real PDF."""

    class FakePage:
        def get_text(self):
            return text

    class FakeDocument:
        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return FakePage()

        def close(self):
            pass

    fake_fitz = types.ModuleType("fitz")
    fake_fitz.open = lambda *_a, **_k: FakeDocument()
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)


# ---------------------------------------------------------------------------
# Bug 1: an attachment key IS the PDF
# ---------------------------------------------------------------------------

class TestLocalAttachmentLookup:
    """The local reader can address an attachment by its own key."""

    def test_get_item_by_key_cannot_see_attachments(self, tmp_path):
        """Root cause: the item query excludes the 'attachment' item type."""
        db_path, _ = make_library(tmp_path)

        with LocalZoteroReader(db_path=str(db_path)) as reader:
            assert reader.get_item_by_key(ATTACHMENT_KEY) is None
            assert reader.get_item_by_key(PARENT_KEY) is not None

    def test_get_attachment_by_key_returns_row(self, tmp_path):
        db_path, _ = make_library(tmp_path)

        with LocalZoteroReader(db_path=str(db_path)) as reader:
            attachment = reader.get_attachment_by_key(ATTACHMENT_KEY)

        assert attachment == {
            "key": ATTACHMENT_KEY,
            "content_type": "application/pdf",
            "zotero_path": "storage:paper.pdf",
            "title": "Full Text PDF",
            "parent_key": PARENT_KEY,
        }

    def test_get_attachment_by_key_ignores_non_attachments(self, tmp_path):
        db_path, _ = make_library(tmp_path)

        with LocalZoteroReader(db_path=str(db_path)) as reader:
            assert reader.get_attachment_by_key(PARENT_KEY) is None
            assert reader.get_attachment_by_key("NOSUCHKEY") is None


class TestReadPdfPagesWithAttachmentKey:
    """_get_pdf_path resolves a key that names the PDF attachment itself."""

    def test_local_mode_resolves_attachment_path(self, monkeypatch, tmp_path, fake_zot):
        db_path, pdf_path = make_library(tmp_path)
        use_local_library(monkeypatch, db_path, fake_zot)

        result = read_pdf_tools._get_pdf_path(ATTACHMENT_KEY, DummyContext())

        assert result == (str(pdf_path), "Full Text PDF", False)

    def test_local_mode_still_resolves_parent_key(self, monkeypatch, tmp_path, fake_zot):
        """The parent-key path (the only one that used to work) is unchanged."""
        db_path, pdf_path = make_library(tmp_path)
        use_local_library(monkeypatch, db_path, fake_zot)

        result = read_pdf_tools._get_pdf_path(PARENT_KEY, DummyContext())

        assert result == (str(pdf_path), "Parent Article", False)

    def test_local_mode_survives_filename_drift(self, monkeypatch, tmp_path, fake_zot):
        """Recorded filename no longer on disk -> scan the storage folder (#291)."""
        db_path, pdf_path = make_library(
            tmp_path, stored_filename="paper.pdf", disk_filename="renamed.pdf"
        )
        use_local_library(monkeypatch, db_path, fake_zot)

        result = read_pdf_tools._get_pdf_path(ATTACHMENT_KEY, DummyContext())

        assert result == (str(pdf_path), "Full Text PDF", False)

    def test_tool_reads_pages_from_attachment_key(self, monkeypatch, tmp_path, fake_zot):
        """End to end: the tool no longer answers 'No PDF attachment found'."""
        db_path, _ = make_library(tmp_path)
        use_local_library(monkeypatch, db_path, fake_zot)
        patch_fitz(monkeypatch, text="Attachment page one.")

        result = server.read_pdf_pages(
            item_key=ATTACHMENT_KEY, start_page=1, ctx=DummyContext()
        )

        assert "No PDF attachment found" not in result
        assert "Attachment page one." in result

    def test_web_mode_downloads_the_attachment_itself(self, monkeypatch, tmp_path):
        """Web-API fallback: an attachment key downloads that attachment."""
        downloaded = []

        class AttachmentZotero(FakeZotero):
            def item(self, item_key):
                return {
                    "key": item_key,
                    "data": {
                        "itemType": "attachment",
                        "contentType": "application/pdf",
                        "filename": "paper.pdf",
                        "title": "Full Text PDF",
                    },
                }

            def children(self, item_key, **kwargs):
                raise AssertionError("children() must not be consulted")

            def dump(self, key, filename=None, path=None):
                downloaded.append(key)
                (tmp_path / "unused").mkdir(exist_ok=True)
                with open(f"{path}/{filename}", "wb") as handle:
                    handle.write(b"%PDF-1.4 fake")

        zot = AttachmentZotero()
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: zot)
        monkeypatch.setattr("zotero_mcp.client.get_local_zotero_client", lambda: None)

        result = read_pdf_tools._get_pdf_path(ATTACHMENT_KEY, DummyContext())

        assert downloaded == [ATTACHMENT_KEY]
        assert result is not None
        path, title, is_temp = result
        assert path.endswith("paper.pdf")
        assert title == "Full Text PDF"
        # A downloaded copy is ours to delete; a library file never is.
        assert is_temp is True
        read_pdf_tools._cleanup_path(path)


# ---------------------------------------------------------------------------
# Bug 2: get_toc() must not be able to kill the server
# ---------------------------------------------------------------------------

class TestExtractPdfTocIsolation:
    """_extract_pdf_toc contains child-process death instead of dying with it."""

    def test_reads_toc_from_child_stdout(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            write_tools,
            "_TOC_CHILD_SCRIPT",
            "import sys; sys.stdout.write('[[1, \"Intro\", 1]]')",
        )

        outcome = write_tools._extract_pdf_toc(str(tmp_path / "paper.pdf"))

        assert outcome.status == "ok"
        assert outcome.toc == [[1, "Intro", 1]]

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals only")
    def test_segfaulting_child_is_reported_not_fatal(self, monkeypatch, tmp_path):
        """A child killed by SIGSEGV must come back as a 'crashed' outcome.

        This is the #372 crash simulated hermetically: the real trigger is
        PyMuPDF segfaulting inside get_toc() on certain journal PDFs, which
        would take this test process (and, in production, the MCP server)
        down if the call were not isolated.
        """
        monkeypatch.setattr(
            write_tools,
            "_TOC_CHILD_SCRIPT",
            "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)",
        )

        outcome = write_tools._extract_pdf_toc(str(tmp_path / "paper.pdf"))

        assert outcome.status == "crashed"
        assert "SIGSEGV" in outcome.detail

    def test_timeout_kills_the_child(self, monkeypatch, tmp_path):
        """A hung child is killed and reaped (subprocess.run waits on it)."""
        monkeypatch.setattr(
            write_tools, "_TOC_CHILD_SCRIPT", "import time; time.sleep(30)"
        )

        outcome = write_tools._extract_pdf_toc(str(tmp_path / "paper.pdf"), timeout=1)

        assert outcome.status == "timeout"
        assert outcome.toc == []

    def test_missing_pymupdf_reported_by_exit_code(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            write_tools,
            "_TOC_CHILD_SCRIPT",
            f"import sys; sys.exit({write_tools._TOC_EXIT_NO_PYMUPDF})",
        )

        outcome = write_tools._extract_pdf_toc(str(tmp_path / "paper.pdf"))

        assert outcome.status == "no_pymupdf"

    def test_child_never_imports_zotero_mcp(self):
        """Importing the package in the child re-runs FastMCP init (#178)."""
        assert "zotero_mcp" not in write_tools._TOC_CHILD_SCRIPT

    def test_api_keys_stripped_from_child_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ZOTERO_API_KEY", "secret-key")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-key")
        monkeypatch.setattr(
            write_tools,
            "_TOC_CHILD_SCRIPT",
            "import json, os, sys; sys.stdout.write(json.dumps(sorted(os.environ)))",
        )

        outcome = write_tools._extract_pdf_toc(str(tmp_path / "paper.pdf"))

        assert outcome.status == "ok"
        assert "ZOTERO_API_KEY" not in outcome.toc
        assert "ANTHROPIC_API_KEY" not in outcome.toc


class TestGetPdfOutlineCrashHandling:
    """The tool reports a contained crash instead of disconnecting."""

    def _fake_download(self, monkeypatch, tmp_path, recorder=None):
        pdf_path = tmp_path / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        def _download(attachment_key, destination_dir, filename=None, **kwargs):
            if recorder is not None:
                recorder.append((attachment_key, filename, kwargs))
            return client_module.AttachmentDownloadResult(
                path=pdf_path, source="Local Zotero", errors=[]
            )

        monkeypatch.setattr(client_module, "download_attachment_file", _download)

    def _pdf_child(self, key="ATT00001", filename="paper.pdf"):
        return {
            "key": key,
            "data": {
                "itemType": "attachment",
                "contentType": "application/pdf",
                "filename": filename,
                "parentItem": PARENT_KEY,
            },
        }

    def test_crash_returns_message_and_keeps_server_alive(
        self, monkeypatch, tmp_path, fake_zot
    ):
        fake_zot._children[PARENT_KEY] = [self._pdf_child()]
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)
        self._fake_download(monkeypatch, tmp_path)
        monkeypatch.setattr(
            write_tools,
            "_extract_pdf_toc",
            lambda *_a, **_k: write_tools.TocOutcome("crashed", [], "SIGSEGV"),
        )

        result = server.get_pdf_outline(item_key=PARENT_KEY, ctx=DummyContext())

        assert "crashed" in result.lower()
        assert "SIGSEGV" in result
        assert ATTACHMENT_KEY in result

    def test_timeout_returns_message(self, monkeypatch, tmp_path, fake_zot):
        fake_zot._children[PARENT_KEY] = [self._pdf_child()]
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)
        self._fake_download(monkeypatch, tmp_path)
        monkeypatch.setattr(
            write_tools,
            "_extract_pdf_toc",
            lambda *_a, **_k: write_tools.TocOutcome("timeout", [], "no response after 60s"),
        )

        result = server.get_pdf_outline(item_key=PARENT_KEY, ctx=DummyContext())

        assert "timed out" in result.lower()

    def test_missing_pymupdf_message(self, monkeypatch, tmp_path, fake_zot):
        fake_zot._children[PARENT_KEY] = [self._pdf_child()]
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)
        self._fake_download(monkeypatch, tmp_path)
        monkeypatch.setattr(
            write_tools,
            "_extract_pdf_toc",
            lambda *_a, **_k: write_tools.TocOutcome("no_pymupdf", []),
        )

        result = server.get_pdf_outline(item_key=PARENT_KEY, ctx=DummyContext())

        assert "PyMuPDF" in result

    def test_uses_multi_source_downloader(self, monkeypatch, tmp_path, fake_zot):
        """Downloads go through download_attachment_file so WebDAV works."""
        calls = []
        fake_zot._children[PARENT_KEY] = [self._pdf_child()]
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)
        monkeypatch.setattr("zotero_mcp.client.get_local_zotero_client", lambda: None)
        self._fake_download(monkeypatch, tmp_path, recorder=calls)
        monkeypatch.setattr(
            write_tools,
            "_extract_pdf_toc",
            lambda *_a, **_k: write_tools.TocOutcome("ok", [[1, "Intro", 1]]),
        )

        result = server.get_pdf_outline(item_key=PARENT_KEY, ctx=DummyContext())

        assert "- Intro (p. 1)" in result
        assert len(calls) == 1
        attachment_key, filename, kwargs = calls[0]
        assert attachment_key == ATTACHMENT_KEY
        assert filename == "paper.pdf"
        assert kwargs["web_client"] is fake_zot

    def test_local_mode_uses_the_active_client_for_download(
        self, monkeypatch, tmp_path, fake_zot
    ):
        """In local mode the active client IS the local one — reuse it."""
        calls = []
        fake_zot._children[PARENT_KEY] = [self._pdf_child()]
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: fake_zot)
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: True)
        self._fake_download(monkeypatch, tmp_path, recorder=calls)
        monkeypatch.setattr(
            write_tools,
            "_extract_pdf_toc",
            lambda *_a, **_k: write_tools.TocOutcome("ok", []),
        )

        server.get_pdf_outline(item_key=PARENT_KEY, ctx=DummyContext())

        _key, _filename, kwargs = calls[0]
        assert kwargs["local_client"] is fake_zot
        assert kwargs["web_client"] is None

    def test_accepts_attachment_key_directly(self, monkeypatch, tmp_path):
        """An attachment key names the PDF — do not scan its (empty) children."""
        calls = []

        class AttachmentZotero(FakeZotero):
            def item(self, item_key):
                return {
                    "key": item_key,
                    "data": {
                        "itemType": "attachment",
                        "contentType": "application/pdf",
                        "filename": "paper.pdf",
                    },
                }

            def children(self, item_key, **kwargs):
                raise AssertionError("children() must not be consulted")

        zot = AttachmentZotero()
        monkeypatch.setattr("zotero_mcp.client.get_zotero_client", lambda: zot)
        monkeypatch.setattr("zotero_mcp.utils.is_local_mode", lambda: False)
        monkeypatch.setattr("zotero_mcp.client.get_local_zotero_client", lambda: None)
        self._fake_download(monkeypatch, tmp_path, recorder=calls)
        monkeypatch.setattr(
            write_tools,
            "_extract_pdf_toc",
            lambda *_a, **_k: write_tools.TocOutcome("ok", [[1, "Intro", 1]]),
        )

        result = server.get_pdf_outline(item_key=ATTACHMENT_KEY, ctx=DummyContext())

        assert "- Intro (p. 1)" in result
        assert calls[0][0] == ATTACHMENT_KEY
