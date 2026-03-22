"""Tests for Feature 10: zotero_add_from_file (server.add_from_file)."""

import sys
import types
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from zotero_mcp import server
from conftest import DummyContext, FakeZotero, _FakeResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeZoteroForFile(FakeZotero):
    """FakeZotero extended with attachment_both stub."""

    def __init__(self):
        super().__init__()
        self.attachments = []

    def attachment_both(self, files, parentid=None, **kwargs):
        self.attachments.append({"files": files, "parentid": parentid})
        return {"success": {"0": "ATCH0001"}, "successful": {}, "failed": {}}


class FakeFitzDocument:
    """Stub for a fitz (PyMuPDF) Document object.

    Supports the interface used by server.add_from_file:
    - doc.metadata
    - doc.page_count
    - doc[0].get_text()
    - doc.close()
    """

    def __init__(self, metadata=None, first_page_text=""):
        self._metadata = metadata or {}
        self._first_page_text = first_page_text
        self._pages = [FakeFitzPage(first_page_text)]

    @property
    def metadata(self):
        return self._metadata

    @property
    def page_count(self):
        return len(self._pages)

    def __getitem__(self, index):
        return self._pages[index]

    def load_page(self, page_num):
        if page_num < len(self._pages):
            return self._pages[page_num]
        raise IndexError("page out of range")

    def __len__(self):
        return len(self._pages)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def close(self):
        pass


class FakeFitzPage:
    """Stub for a fitz Page object."""

    def __init__(self, text=""):
        self._text = text

    def get_text(self, *args, **kwargs):
        return self._text


def _make_fake_fitz_module(doc):
    """Create a fake fitz module whose open() returns the given document."""
    fake_fitz = types.ModuleType("fitz")
    fake_fitz.open = lambda *args, **kwargs: doc
    return fake_fitz


def _patch_path_valid(monkeypatch):
    """Patch os.path functions so the file path appears valid."""
    monkeypatch.setattr("os.path.exists", lambda p: True)
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    monkeypatch.setattr("os.path.islink", lambda p: False)
    monkeypatch.setattr("os.path.isabs", lambda p: p.startswith("/"))


def _patch_hybrid_mode(monkeypatch, fake_write_zot):
    """Patch _get_write_client to return (read_zot, write_zot)."""
    read_zot = FakeZoteroForFile()
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._get_write_client",
        lambda ctx: (read_zot, fake_write_zot),
    )
    return read_zot


def _patch_fitz(monkeypatch, doc):
    """Patch fitz in sys.modules so 'import fitz' inside server functions works."""
    fake_fitz = _make_fake_fitz_module(doc)
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)


# ---------------------------------------------------------------------------
# Happy path: PDF file exists, no DOI found, creates document + attachment
# ---------------------------------------------------------------------------

class TestHappyPathNoDoi:
    def test_creates_document_item_and_attachment(self, monkeypatch, dummy_ctx):
        fake_zot = FakeZoteroForFile()
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        # fitz.open returns a doc with no DOI in metadata or text
        fake_doc = FakeFitzDocument(metadata={"subject": "", "keywords": ""}, first_page_text="No doi here.")
        _patch_fitz(monkeypatch, fake_doc)

        result = server.add_from_file(
            file_path="/Users/test/Documents/paper.pdf",
            title="My Paper",
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        # Should have created one item
        assert len(fake_zot.created) == 1
        created_item = fake_zot.created[0]
        assert created_item["itemType"] == "document"
        assert created_item["title"] == "My Paper"

        # Should have called attachment_both
        assert len(fake_zot.attachments) == 1
        att = fake_zot.attachments[0]
        assert att["files"][0] == ("paper.pdf", "/Users/test/Documents/paper.pdf")
        assert att["parentid"] is not None

    def test_uses_filename_as_title_when_none(self, monkeypatch, dummy_ctx):
        fake_zot = FakeZoteroForFile()
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        fake_doc = FakeFitzDocument(metadata={}, first_page_text="Some text without DOI.")
        _patch_fitz(monkeypatch, fake_doc)

        result = server.add_from_file(
            file_path="/Users/test/Documents/report.pdf",
            title=None,
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        assert len(fake_zot.created) == 1
        # When title is None, the function should use the filename (without extension)
        # or the basename as a fallback title
        created_item = fake_zot.created[0]
        assert created_item["title"] != ""


# ---------------------------------------------------------------------------
# DOI extraction from PDF metadata -> delegates to add_by_doi logic
# ---------------------------------------------------------------------------

class TestDoiFromMetadata:
    def test_doi_in_subject_field(self, monkeypatch, dummy_ctx):
        fake_zot = FakeZoteroForFile()
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        fake_doc = FakeFitzDocument(
            metadata={"subject": "doi: 10.1234/test.2024.001", "keywords": ""},
            first_page_text="",
        )
        _patch_fitz(monkeypatch, fake_doc)

        # Mock the add_by_doi function to verify delegation
        doi_called_with = {}

        def mock_add_by_doi(doi, collections=None, tags=None, *, ctx):
            doi_called_with["doi"] = doi
            doi_called_with["collections"] = collections
            doi_called_with["tags"] = tags
            return "Added by DOI: 10.1234/test.2024.001"

        monkeypatch.setattr("zotero_mcp.tools.write.add_by_doi", mock_add_by_doi)

        result = server.add_from_file(
            file_path="/Users/test/Documents/paper.pdf",
            title=None,
            item_type="document",
            collections=["COL001"],
            tags=["tag1"],
            ctx=dummy_ctx,
        )

        assert doi_called_with["doi"] == "10.1234/test.2024.001"
        assert doi_called_with["collections"] == ["COL001"]
        assert doi_called_with["tags"] == ["tag1"]

    def test_doi_in_keywords_field(self, monkeypatch, dummy_ctx):
        fake_zot = FakeZoteroForFile()
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        # _normalize_doi expects the field value to be a bare DOI (not embedded
        # in a longer string), so provide a clean DOI as the keywords value.
        fake_doc = FakeFitzDocument(
            metadata={"subject": "", "keywords": "10.5678/ml.2023.999"},
            first_page_text="",
        )
        _patch_fitz(monkeypatch, fake_doc)

        doi_captured = {}

        def mock_add_by_doi(doi, collections=None, tags=None, *, ctx):
            doi_captured["doi"] = doi
            return "Added by DOI: Item key: `KEY0001`"

        monkeypatch.setattr("zotero_mcp.tools.write.add_by_doi", mock_add_by_doi)

        server.add_from_file(
            file_path="/Users/test/Documents/paper.pdf",
            title=None,
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        assert "10.5678/ml.2023.999" in doi_captured["doi"]


# ---------------------------------------------------------------------------
# DOI extraction from first page text
# ---------------------------------------------------------------------------

class TestDoiFromFirstPageText:
    def test_doi_in_first_page_text(self, monkeypatch, dummy_ctx):
        fake_zot = FakeZoteroForFile()
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        first_page = (
            "Journal of Example Studies, Vol 5\n"
            "DOI: 10.1000/xyz123\n"
            "Abstract: This paper discusses..."
        )
        fake_doc = FakeFitzDocument(
            metadata={"subject": "", "keywords": ""},
            first_page_text=first_page,
        )
        _patch_fitz(monkeypatch, fake_doc)

        doi_captured = {}

        def mock_add_by_doi(doi, collections=None, tags=None, *, ctx):
            doi_captured["doi"] = doi
            return "Added by DOI"

        monkeypatch.setattr("zotero_mcp.tools.write.add_by_doi", mock_add_by_doi)

        server.add_from_file(
            file_path="/Users/test/Documents/paper.pdf",
            title=None,
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        assert doi_captured["doi"] == "10.1000/xyz123"

    def test_no_doi_anywhere_falls_back_to_manual_item(self, monkeypatch, dummy_ctx):
        """When no DOI is found in metadata or text, create a plain document item."""
        fake_zot = FakeZoteroForFile()
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        fake_doc = FakeFitzDocument(
            metadata={"subject": "", "keywords": ""},
            first_page_text="This paper has no DOI anywhere.",
        )
        _patch_fitz(monkeypatch, fake_doc)

        # add_by_doi should NOT be called
        add_by_doi_called = False

        def mock_add_by_doi(doi, collections=None, tags=None, *, ctx):
            nonlocal add_by_doi_called
            add_by_doi_called = True
            return "should not happen"

        monkeypatch.setattr("zotero_mcp.tools.write.add_by_doi", mock_add_by_doi)

        server.add_from_file(
            file_path="/Users/test/Documents/paper.pdf",
            title="Manual Title",
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        assert not add_by_doi_called
        assert len(fake_zot.created) == 1
        assert fake_zot.created[0]["title"] == "Manual Title"


# ---------------------------------------------------------------------------
# Invalid file extension -> error
# ---------------------------------------------------------------------------

class TestInvalidFileExtension:
    def test_rejects_exe_extension(self, monkeypatch, dummy_ctx):
        fake_zot = FakeZoteroForFile()
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        result = server.add_from_file(
            file_path="/Users/test/Documents/malware.exe",
            title="Bad File",
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        assert "error" in result.lower() or "unsupported" in result.lower() or "extension" in result.lower()

    def test_rejects_txt_extension(self, monkeypatch, dummy_ctx):
        fake_zot = FakeZoteroForFile()
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        result = server.add_from_file(
            file_path="/Users/test/Documents/notes.txt",
            title="Text File",
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        assert "error" in result.lower() or "unsupported" in result.lower() or "extension" in result.lower()

    def test_accepts_pdf_extension(self, monkeypatch, dummy_ctx):
        """PDF should be accepted (not rejected at extension check)."""
        fake_zot = FakeZoteroForFile()
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        fake_doc = FakeFitzDocument(metadata={}, first_page_text="No DOI.")
        _patch_fitz(monkeypatch, fake_doc)

        result = server.add_from_file(
            file_path="/Users/test/Documents/paper.pdf",
            title="Good PDF",
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        # Should not contain an extension error
        assert "unsupported" not in result.lower() or "extension" not in result.lower()

    def test_accepts_epub_extension(self, monkeypatch, dummy_ctx):
        """EPUB should be accepted."""
        fake_zot = FakeZoteroForFile()
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        # For EPUB, fitz.open may not be called (DOI extraction is PDF-specific),
        # but the extension should pass validation
        fake_doc = FakeFitzDocument(metadata={}, first_page_text="")
        _patch_fitz(monkeypatch, fake_doc)

        result = server.add_from_file(
            file_path="/Users/test/Documents/book.epub",
            title="Good EPUB",
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        assert "unsupported" not in result.lower()


# ---------------------------------------------------------------------------
# File doesn't exist -> error
# ---------------------------------------------------------------------------

class TestFileDoesNotExist:
    def test_nonexistent_file(self, monkeypatch, dummy_ctx):
        fake_zot = FakeZoteroForFile()
        _patch_hybrid_mode(monkeypatch, fake_zot)
        monkeypatch.setattr("os.path.isabs", lambda p: True)
        monkeypatch.setattr("os.path.exists", lambda p: False)
        monkeypatch.setattr("os.path.isfile", lambda p: False)
        monkeypatch.setattr("os.path.islink", lambda p: False)

        result = server.add_from_file(
            file_path="/Users/test/Documents/nonexistent.pdf",
            title="Ghost File",
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        assert "not found" in result.lower() or "does not exist" in result.lower() or "error" in result.lower()

    def test_path_is_directory_not_file(self, monkeypatch, dummy_ctx):
        fake_zot = FakeZoteroForFile()
        _patch_hybrid_mode(monkeypatch, fake_zot)
        monkeypatch.setattr("os.path.isabs", lambda p: True)
        monkeypatch.setattr("os.path.exists", lambda p: True)
        monkeypatch.setattr("os.path.isfile", lambda p: False)
        monkeypatch.setattr("os.path.islink", lambda p: False)

        result = server.add_from_file(
            file_path="/Users/test/Documents/",
            title="Not a file",
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        assert "error" in result.lower() or "not a file" in result.lower() or "not found" in result.lower()


# ---------------------------------------------------------------------------
# Non-absolute path -> error
# ---------------------------------------------------------------------------

class TestNonAbsolutePath:
    def test_relative_path_rejected(self, monkeypatch, dummy_ctx):
        fake_zot = FakeZoteroForFile()
        _patch_hybrid_mode(monkeypatch, fake_zot)
        monkeypatch.setattr("os.path.isabs", lambda p: False)

        result = server.add_from_file(
            file_path="relative/path/paper.pdf",
            title="Relative",
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        assert "absolute" in result.lower() or "error" in result.lower()

    def test_dot_relative_path_rejected(self, monkeypatch, dummy_ctx):
        fake_zot = FakeZoteroForFile()
        _patch_hybrid_mode(monkeypatch, fake_zot)
        monkeypatch.setattr("os.path.isabs", lambda p: not p.startswith("."))

        result = server.add_from_file(
            file_path="./Documents/paper.pdf",
            title="Dot Relative",
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        assert "absolute" in result.lower() or "error" in result.lower()


# ---------------------------------------------------------------------------
# Path validation: reject symlinks (security)
# ---------------------------------------------------------------------------

class TestSymlinkRejection:
    def test_symlink_rejected(self, monkeypatch, dummy_ctx):
        fake_zot = FakeZoteroForFile()
        _patch_hybrid_mode(monkeypatch, fake_zot)
        monkeypatch.setattr("os.path.isabs", lambda p: True)
        monkeypatch.setattr("os.path.exists", lambda p: True)
        monkeypatch.setattr("os.path.isfile", lambda p: True)
        monkeypatch.setattr("os.path.islink", lambda p: True)

        result = server.add_from_file(
            file_path="/Users/test/Documents/symlink_paper.pdf",
            title="Symlink File",
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        assert "symlink" in result.lower() or "error" in result.lower() or "security" in result.lower()


# ---------------------------------------------------------------------------
# Hybrid mode / local-only rejection
# ---------------------------------------------------------------------------

class TestHybridModeRejection:
    def test_local_only_mode_returns_error(self, monkeypatch, dummy_ctx):
        """In local-only mode (no web credentials), write operations should fail."""
        _patch_path_valid(monkeypatch)

        # _get_write_client raises ValueError in local-only mode
        def raise_local_only(ctx):
            raise ValueError(
                "Cannot perform write operations in local-only mode. "
                "Add ZOTERO_API_KEY and ZOTERO_LIBRARY_ID to enable hybrid mode."
            )

        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", raise_local_only)

        fake_doc = FakeFitzDocument(metadata={}, first_page_text="No DOI.")
        _patch_fitz(monkeypatch, fake_doc)

        result = server.add_from_file(
            file_path="/Users/test/Documents/paper.pdf",
            title="Local Only",
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        assert "local-only" in result.lower() or "write operations" in result.lower()

    def test_hybrid_mode_uses_write_client(self, monkeypatch, dummy_ctx):
        """Verify that attachment_both is called on the write client, not the read client."""
        write_zot = FakeZoteroForFile()
        read_zot = FakeZoteroForFile()
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._get_write_client",
            lambda ctx: (read_zot, write_zot),
        )
        _patch_path_valid(monkeypatch)

        fake_doc = FakeFitzDocument(metadata={}, first_page_text="No DOI.")
        _patch_fitz(monkeypatch, fake_doc)

        server.add_from_file(
            file_path="/Users/test/Documents/paper.pdf",
            title="Hybrid Test",
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        # Write client should have the created item and attachment
        assert len(write_zot.created) == 1
        assert len(write_zot.attachments) == 1
        # Read client should NOT have been used for writes
        assert len(read_zot.created) == 0
        assert len(read_zot.attachments) == 0


# ---------------------------------------------------------------------------
# Tags and collections applied
# ---------------------------------------------------------------------------

class TestTagsAndCollections:
    def test_tags_applied_to_created_item(self, monkeypatch, dummy_ctx):
        fake_zot = FakeZoteroForFile()
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        fake_doc = FakeFitzDocument(metadata={}, first_page_text="No DOI here.")
        _patch_fitz(monkeypatch, fake_doc)

        server.add_from_file(
            file_path="/Users/test/Documents/paper.pdf",
            title="Tagged Paper",
            item_type="document",
            collections=None,
            tags=["machine-learning", "review"],
            ctx=dummy_ctx,
        )

        assert len(fake_zot.created) == 1
        created_item = fake_zot.created[0]
        tag_names = [t["tag"] for t in created_item.get("tags", [])]
        assert "machine-learning" in tag_names
        assert "review" in tag_names

    def test_collections_applied_to_created_item(self, monkeypatch, dummy_ctx):
        fake_zot = FakeZoteroForFile()
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        fake_doc = FakeFitzDocument(metadata={}, first_page_text="No DOI here.")
        _patch_fitz(monkeypatch, fake_doc)

        server.add_from_file(
            file_path="/Users/test/Documents/paper.pdf",
            title="Collected Paper",
            item_type="document",
            collections=["COLKEY01", "COLKEY02"],
            tags=None,
            ctx=dummy_ctx,
        )

        assert len(fake_zot.created) == 1
        created_item = fake_zot.created[0]
        assert "COLKEY01" in created_item.get("collections", [])
        assert "COLKEY02" in created_item.get("collections", [])

    def test_tags_and_collections_together(self, monkeypatch, dummy_ctx):
        fake_zot = FakeZoteroForFile()
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        fake_doc = FakeFitzDocument(metadata={}, first_page_text="No DOI.")
        _patch_fitz(monkeypatch, fake_doc)

        server.add_from_file(
            file_path="/Users/test/Documents/paper.pdf",
            title="Both",
            item_type="document",
            collections=["COL001"],
            tags=["tag1"],
            ctx=dummy_ctx,
        )

        created_item = fake_zot.created[0]
        assert "COL001" in created_item.get("collections", [])
        assert {"tag": "tag1"} in created_item.get("tags", [])

    def test_tags_passed_as_comma_string(self, monkeypatch, dummy_ctx):
        """Tags can arrive as a comma-separated string from LLMs."""
        fake_zot = FakeZoteroForFile()
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        fake_doc = FakeFitzDocument(metadata={}, first_page_text="No DOI.")
        _patch_fitz(monkeypatch, fake_doc)

        server.add_from_file(
            file_path="/Users/test/Documents/paper.pdf",
            title="Comma Tags",
            item_type="document",
            collections=None,
            tags="alpha, beta, gamma",
            ctx=dummy_ctx,
        )

        created_item = fake_zot.created[0]
        tag_names = [t["tag"] for t in created_item.get("tags", [])]
        assert "alpha" in tag_names
        assert "beta" in tag_names
        assert "gamma" in tag_names


# ---------------------------------------------------------------------------
# Uses attachment_both (not attachment_simple)
# ---------------------------------------------------------------------------

class TestAttachmentBoth:
    def test_calls_attachment_both_not_simple(self, monkeypatch, dummy_ctx):
        """Verify attachment_both is called with correct (basename, full_path) tuple."""
        fake_zot = FakeZoteroForFile()
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        fake_doc = FakeFitzDocument(metadata={}, first_page_text="No DOI.")
        _patch_fitz(monkeypatch, fake_doc)

        server.add_from_file(
            file_path="/Users/test/Documents/my_paper.pdf",
            title="Attachment Test",
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        assert len(fake_zot.attachments) == 1
        att_call = fake_zot.attachments[0]
        files_arg = att_call["files"]
        # attachment_both expects [(basename, full_path)]
        assert len(files_arg) == 1
        basename, full_path = files_arg[0]
        assert basename == "my_paper.pdf"
        assert full_path == "/Users/test/Documents/my_paper.pdf"

    def test_attachment_both_receives_parent_item_key(self, monkeypatch, dummy_ctx):
        """The parentid kwarg should be the key of the newly created item."""
        fake_zot = FakeZoteroForFile()
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        fake_doc = FakeFitzDocument(metadata={}, first_page_text="No DOI.")
        _patch_fitz(monkeypatch, fake_doc)

        server.add_from_file(
            file_path="/Users/test/Documents/paper.pdf",
            title="Parent Key Test",
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        att_call = fake_zot.attachments[0]
        parent_id = att_call["parentid"]
        # create_items returns {"success": {"0": "KEY0000"}} from FakeZotero
        assert parent_id == "KEY0000"

    def test_attachment_simple_not_used(self, monkeypatch, dummy_ctx):
        """Ensure attachment_simple is never called (it stores full paths as filenames)."""
        fake_zot = FakeZoteroForFile()
        fake_zot.attachment_simple = MagicMock(side_effect=AssertionError(
            "attachment_simple should not be called"
        ))
        _patch_path_valid(monkeypatch)
        _patch_hybrid_mode(monkeypatch, fake_zot)

        fake_doc = FakeFitzDocument(metadata={}, first_page_text="No DOI.")
        _patch_fitz(monkeypatch, fake_doc)

        # Should complete without raising AssertionError
        server.add_from_file(
            file_path="/Users/test/Documents/paper.pdf",
            title="No Simple",
            item_type="document",
            collections=None,
            tags=None,
            ctx=dummy_ctx,
        )

        fake_zot.attachment_simple.assert_not_called()
