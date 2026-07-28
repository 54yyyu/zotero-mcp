"""Tests for zotero_attach_file (tools/write.attach_file) and its helpers."""

import hashlib

import pytest
from conftest import DummyContext, FakeZotero

from zotero_mcp import server
from zotero_mcp.tools import _helpers


@pytest.fixture
def dummy_ctx():
    return DummyContext()


# ---------------------------------------------------------------------------
# _helpers._attachment_filename_exists
# ---------------------------------------------------------------------------


class TestAttachmentFilenameExists:
    def test_true_when_child_has_same_filename(self):
        zot = FakeZotero()
        zot._children["ITEM1"] = [
            {"data": {"itemType": "attachment", "filename": "paper.pdf"}}
        ]
        assert _helpers._attachment_filename_exists(zot, "ITEM1", "paper.pdf")

    def test_false_when_no_children(self):
        zot = FakeZotero()
        assert not _helpers._attachment_filename_exists(zot, "ITEM1", "paper.pdf")

    def test_false_when_different_filename(self):
        zot = FakeZotero()
        zot._children["ITEM1"] = [{"data": {"filename": "other.pdf"}}]
        assert not _helpers._attachment_filename_exists(zot, "ITEM1", "paper.pdf")

    def test_false_when_children_call_raises(self):
        class Boom(FakeZotero):
            def children(self, item_key, **kwargs):
                raise RuntimeError("api down")

        assert not _helpers._attachment_filename_exists(Boom(), "ITEM1", "paper.pdf")

    def test_tolerates_none_data(self):
        zot = FakeZotero()
        zot._children["ITEM1"] = [{"data": None}]
        assert not _helpers._attachment_filename_exists(zot, "ITEM1", "paper.pdf")


class TestExtractAttachmentKey:
    def test_key_from_success_entry(self):
        result = {"success": [{"key": "ATCH0001"}], "failure": [], "unchanged": []}
        assert _helpers._extract_attachment_key(result) == "ATCH0001"

    def test_key_from_unchanged_entry(self):
        result = {"success": [], "failure": [], "unchanged": [{"key": "ATCH0002"}]}
        assert _helpers._extract_attachment_key(result) == "ATCH0002"

    def test_none_when_no_entries(self):
        result = {"success": [], "failure": [], "unchanged": []}
        assert _helpers._extract_attachment_key(result) is None

    def test_none_on_non_dict(self):
        assert _helpers._extract_attachment_key(None) is None


class TestFindChildAttachment:
    def test_matches_filename(self):
        zot = FakeZotero()
        child = {"key": "ATT1", "data": {"filename": "paper.pdf"}}
        zot._children["ITEM1"] = [child]
        assert (
            _helpers._find_child_attachment(zot, "ITEM1", filename="paper.pdf")
            is child
        )

    def test_matches_md5(self):
        zot = FakeZotero()
        child = {"key": "ATT1", "data": {"filename": "other.pdf", "md5": "abc123"}}
        zot._children["ITEM1"] = [child]
        assert (
            _helpers._find_child_attachment(
                zot, "ITEM1", filename="paper.pdf", file_md5="abc123"
            )
            is child
        )

    def test_no_match_returns_none(self):
        zot = FakeZotero()
        zot._children["ITEM1"] = [
            {"key": "ATT1", "data": {"filename": "other.pdf", "md5": "abc123"}}
        ]
        assert (
            _helpers._find_child_attachment(
                zot, "ITEM1", filename="paper.pdf", file_md5="def456"
            )
            is None
        )

    def test_none_md5_does_not_match_child_without_md5(self):
        zot = FakeZotero()
        zot._children["ITEM1"] = [{"key": "ATT1", "data": {"filename": "other.pdf"}}]
        assert (
            _helpers._find_child_attachment(
                zot, "ITEM1", filename="paper.pdf", file_md5=None
            )
            is None
        )

    def test_children_error_returns_none(self):
        class Boom(FakeZotero):
            def children(self, item_key, **kwargs):
                raise RuntimeError("api down")

        assert (
            _helpers._find_child_attachment(Boom(), "ITEM1", filename="paper.pdf")
            is None
        )

    def test_scans_past_first_api_page(self):
        """The Zotero API caps unpaginated requests (default 25 children) —
        the scan must page through, or a match past the first page is missed."""

        class PagedFake(FakeZotero):
            def children(self, item_key, start=0, limit=25, **kwargs):
                kids = self._children.get(item_key, [])
                return kids[start : start + limit]

        zot = PagedFake()
        match = {"key": "ATT_PG2", "data": {"filename": "paper.pdf"}}
        fillers = [{"key": f"NOTE{i:04d}", "data": {"itemType": "note"}} for i in range(150)]
        zot._children["ITEM1"] = fillers + [match]
        assert (
            _helpers._find_child_attachment(zot, "ITEM1", filename="paper.pdf")
            is match
        )

    def test_skips_trashed_children(self):
        """A child in Zotero's trash must not count as 'already attached'."""
        zot = FakeZotero()
        zot._children["ITEM1"] = [
            {"key": "ATT1", "data": {"filename": "paper.pdf", "deleted": 1}}
        ]
        assert (
            _helpers._find_child_attachment(zot, "ITEM1", filename="paper.pdf")
            is None
        )


class FakeZoteroForAttach(FakeZotero):
    """FakeZotero extended with attachment_both recording."""

    def __init__(self):
        super().__init__()
        self.attachments = []

    def attachment_both(self, files, parentid=None, **kwargs):
        self.attachments.append({"files": files, "parentid": parentid})
        # Shape matches pyzotero Zupload.upload(): status → list of payload
        # dicts, each carrying the registered attachment key.
        return {
            "success": [{"key": "ATCH0001", "title": files[0][0]}],
            "failure": [],
            "unchanged": [],
        }


def _patch_write_client(monkeypatch, fake_zot):
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._get_write_client",
        lambda ctx: (fake_zot, fake_zot),
    )


def _patch_path_valid(monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda p: True)
    monkeypatch.setattr("os.path.islink", lambda p: False)
    monkeypatch.setattr("os.path.isabs", lambda p: p.startswith("/"))
    # The filename-override branch stages the file via shutil.copy2 into a
    # real TemporaryDirectory; the fake source paths above don't exist on
    # disk, so make the copy a no-op here. Target the patch at the module
    # attribute since write.py calls it as `shutil.copy2`.
    monkeypatch.setattr("shutil.copy2", lambda src, dst: None)


# ---------------------------------------------------------------------------
# attach_file: argument and item validation
# ---------------------------------------------------------------------------


class TestAttachFileValidation:
    def test_requires_exactly_one_source_neither(self, monkeypatch, dummy_ctx):
        _patch_write_client(monkeypatch, FakeZoteroForAttach())
        result = server.attach_file(item_key="ITEM1", ctx=dummy_ctx)
        assert "exactly one of file_path or url" in result

    def test_requires_exactly_one_source_both(self, monkeypatch, dummy_ctx):
        _patch_write_client(monkeypatch, FakeZoteroForAttach())
        result = server.attach_file(
            item_key="ITEM1",
            file_path="/tmp/a.pdf",
            url="https://example.org/a.pdf",
            ctx=dummy_ctx,
        )
        assert "exactly one of file_path or url" in result

    def test_local_only_mode_message(self, monkeypatch, dummy_ctx):
        def raise_local(ctx):
            raise ValueError(
                "Cannot perform write operations in local-only mode. "
                "Add ZOTERO_API_KEY and ZOTERO_LIBRARY_ID to enable hybrid mode."
            )

        monkeypatch.setattr("zotero_mcp.tools._helpers._get_write_client", raise_local)
        result = server.attach_file(
            item_key="ITEM1", file_path="/tmp/a.pdf", ctx=dummy_ctx
        )
        assert "local-only mode" in result

    def test_item_not_found(self, monkeypatch, dummy_ctx):
        class NotFound(FakeZoteroForAttach):
            def item(self, item_key, **kwargs):
                raise RuntimeError("404")

        _patch_write_client(monkeypatch, NotFound())
        result = server.attach_file(
            item_key="NOPE", file_path="/tmp/a.pdf", ctx=dummy_ctx
        )
        assert "Error" in result and "not found" in result

    def test_rejects_attachment_key_with_parent_hint(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        fake._items = [
            {
                "key": "ATT1",
                "data": {
                    "itemType": "attachment",
                    "parentItem": "PARENT99",
                    "filename": "x.pdf",
                },
            }
        ]
        _patch_write_client(monkeypatch, fake)
        result = server.attach_file(
            item_key="ATT1", file_path="/tmp/a.pdf", ctx=dummy_ctx
        )
        assert "itemType 'attachment'" in result
        assert "PARENT99" in result

    def test_rejects_note_key(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        fake._items = [{"key": "NOTE1", "data": {"itemType": "note"}}]
        _patch_write_client(monkeypatch, fake)
        result = server.attach_file(
            item_key="NOTE1", file_path="/tmp/a.pdf", ctx=dummy_ctx
        )
        assert "Error" in result


# ---------------------------------------------------------------------------
# attach_file: local-file branch
# ---------------------------------------------------------------------------


class TestAttachFileLocal:
    def test_happy_path_attaches_to_item(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_path_valid(monkeypatch)

        result = server.attach_file(
            item_key="ITEM1",
            file_path="/Users/test/smith-2020.pdf",
            ctx=dummy_ctx,
        )

        assert len(fake.attachments) == 1
        att = fake.attachments[0]
        assert att["parentid"] == "ITEM1"
        assert att["files"][0] == ("smith-2020.pdf", "/Users/test/smith-2020.pdf")
        assert "File attached" in result
        assert "zotero_update_search_database" in result

    def test_filename_override(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_path_valid(monkeypatch)

        server.attach_file(
            item_key="ITEM1",
            file_path="/Users/test/dl (3).pdf",
            filename="smith-2020.pdf",
            ctx=dummy_ctx,
        )
        assert fake.attachments[0]["files"][0][0] == "smith-2020.pdf"

    def test_filename_override_controls_stored_path(self, monkeypatch, dummy_ctx):
        """pyzotero's attachment_both() stores the file under the real file's
        basename, ignoring the title tuple element — so a filename override
        must stage the file under the override name, not just relabel it."""
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_path_valid(monkeypatch)

        copy_calls = []
        monkeypatch.setattr(
            "shutil.copy2",
            lambda src, dst: copy_calls.append((src, dst)),
        )

        server.attach_file(
            item_key="ITEM1",
            file_path="/Users/test/dl (3).pdf",
            filename="smith-2020.pdf",
            ctx=dummy_ctx,
        )

        assert len(fake.attachments) == 1
        stored_path = fake.attachments[0]["files"][0][1]
        assert stored_path.endswith("smith-2020.pdf")
        assert len(copy_calls) == 1
        assert copy_calls[0][0] == "/Users/test/dl (3).pdf"
        assert copy_calls[0][1].endswith("smith-2020.pdf")

    def test_skips_when_same_filename_present(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        fake._children["ITEM1"] = [
            {"data": {"itemType": "attachment", "filename": "smith-2020.pdf"}}
        ]
        _patch_write_client(monkeypatch, fake)
        _patch_path_valid(monkeypatch)

        result = server.attach_file(
            item_key="ITEM1",
            file_path="/Users/test/smith-2020.pdf",
            ctx=dummy_ctx,
        )
        assert len(fake.attachments) == 0
        assert "already present" in result
        assert "not re-uploaded" in result

    def test_result_includes_attachment_key(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_path_valid(monkeypatch)

        result = server.attach_file(
            item_key="ITEM1",
            file_path="/Users/test/smith-2020.pdf",
            ctx=dummy_ctx,
        )
        assert "ATCH0001" in result

    def test_already_present_reports_existing_key(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        fake._children["ITEM1"] = [
            {
                "key": "OLDATT01",
                "data": {"itemType": "attachment", "filename": "smith-2020.pdf"},
            }
        ]
        _patch_write_client(monkeypatch, fake)
        _patch_path_valid(monkeypatch)

        result = server.attach_file(
            item_key="ITEM1",
            file_path="/Users/test/smith-2020.pdf",
            ctx=dummy_ctx,
        )
        assert len(fake.attachments) == 0
        assert "OLDATT01" in result

    def test_rejects_symlink(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_path_valid(monkeypatch)
        monkeypatch.setattr("os.path.islink", lambda p: True)

        result = server.attach_file(
            item_key="ITEM1", file_path="/Users/test/a.pdf", ctx=dummy_ctx
        )
        assert "Symlinks are not allowed" in result

    def test_rejects_relative_path(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_path_valid(monkeypatch)

        result = server.attach_file(
            item_key="ITEM1", file_path="relative/a.pdf", ctx=dummy_ctx
        )
        assert "absolute path" in result

    def test_rejects_bad_extension(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_path_valid(monkeypatch)

        result = server.attach_file(
            item_key="ITEM1", file_path="/Users/test/a.exe", ctx=dummy_ctx
        )
        assert "Unsupported file type" in result

    def test_missing_file(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_path_valid(monkeypatch)
        monkeypatch.setattr("os.path.isfile", lambda p: False)

        result = server.attach_file(
            item_key="ITEM1", file_path="/Users/test/a.pdf", ctx=dummy_ctx
        )
        assert "File not found" in result

    def test_webdav_suffix_passthrough(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_path_valid(monkeypatch)
        monkeypatch.setattr(
            "zotero_mcp.tools._helpers._maybe_upload_to_webdav",
            lambda attach_result, file_path, ctx, write_zot=None: (
                " (uploaded to WebDAV as ATCH0001.zip)"
            ),
        )

        result = server.attach_file(
            item_key="ITEM1", file_path="/Users/test/a.pdf", ctx=dummy_ctx
        )
        assert "uploaded to WebDAV as ATCH0001.zip" in result

    def test_upload_failure_reported_not_raised(self, monkeypatch, dummy_ctx):
        class BoomUpload(FakeZoteroForAttach):
            def attachment_both(self, files, parentid=None, **kwargs):
                raise RuntimeError("quota exceeded")

        _patch_write_client(monkeypatch, BoomUpload())
        _patch_path_valid(monkeypatch)

        result = server.attach_file(
            item_key="ITEM1", file_path="/Users/test/a.pdf", ctx=dummy_ctx
        )
        assert result.startswith("Error")
        assert "quota exceeded" in result

    def test_upload_failure_entries_not_reported_as_success(
        self, monkeypatch, dummy_ctx
    ):
        """Zupload can put the file on the failure list without raising —
        that must not read as 'File attached'."""

        class FailUpload(FakeZoteroForAttach):
            def attachment_both(self, files, parentid=None, **kwargs):
                return {
                    "success": [],
                    "failure": [{"title": files[0][0], "code": 400}],
                    "unchanged": [],
                }

        _patch_write_client(monkeypatch, FailUpload())
        _patch_path_valid(monkeypatch)

        result = server.attach_file(
            item_key="ITEM1", file_path="/Users/test/a.pdf", ctx=dummy_ctx
        )
        assert result.startswith("Error")
        assert "File attached" not in result

    def test_trashed_same_name_child_does_not_block_upload(
        self, monkeypatch, dummy_ctx
    ):
        fake = FakeZoteroForAttach()
        fake._children["ITEM1"] = [
            {
                "key": "OLDATT01",
                "data": {
                    "itemType": "attachment",
                    "filename": "smith-2020.pdf",
                    "deleted": 1,
                },
            }
        ]
        _patch_write_client(monkeypatch, fake)
        _patch_path_valid(monkeypatch)

        result = server.attach_file(
            item_key="ITEM1",
            file_path="/Users/test/smith-2020.pdf",
            ctx=dummy_ctx,
        )
        assert len(fake.attachments) == 1
        assert "File attached" in result

    def test_filename_override_inherits_source_extension(
        self, monkeypatch, dummy_ctx
    ):
        """A local-mode override without an extension must inherit the source
        file's, mirroring the URL branch's .pdf enforcement — otherwise the
        stored file loses its extension (and MIME-type guess)."""
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_path_valid(monkeypatch)

        server.attach_file(
            item_key="ITEM1",
            file_path="/Users/test/dl (3).pdf",
            filename="smith-2020",
            ctx=dummy_ctx,
        )
        assert fake.attachments[0]["files"][0][0] == "smith-2020.pdf"


# ---------------------------------------------------------------------------
# attach_file: content-hash (MD5) dedupe — uses real files on disk
# ---------------------------------------------------------------------------


class TestAttachFileMd5Dedupe:
    def test_same_content_different_filename_skipped(
        self, tmp_path, monkeypatch, dummy_ctx
    ):
        content = b"%PDF-1.4 same-bytes"
        pdf = tmp_path / "new-name.pdf"
        pdf.write_bytes(content)

        fake = FakeZoteroForAttach()
        fake._children["ITEM1"] = [
            {
                "key": "OLDATT01",
                "data": {
                    "itemType": "attachment",
                    "filename": "old-name.pdf",
                    "md5": hashlib.md5(content).hexdigest(),
                },
            }
        ]
        _patch_write_client(monkeypatch, fake)

        result = server.attach_file(
            item_key="ITEM1", file_path=str(pdf), ctx=dummy_ctx
        )
        assert len(fake.attachments) == 0
        assert "old-name.pdf" in result
        assert "OLDATT01" in result
        assert "not re-uploaded" in result

    def test_different_content_still_uploads(self, tmp_path, monkeypatch, dummy_ctx):
        pdf = tmp_path / "new-name.pdf"
        pdf.write_bytes(b"%PDF-1.4 new-bytes")

        fake = FakeZoteroForAttach()
        fake._children["ITEM1"] = [
            {
                "key": "OLDATT01",
                "data": {
                    "itemType": "attachment",
                    "filename": "old-name.pdf",
                    "md5": hashlib.md5(b"%PDF-1.4 other-bytes").hexdigest(),
                },
            }
        ]
        _patch_write_client(monkeypatch, fake)

        result = server.attach_file(
            item_key="ITEM1", file_path=str(pdf), ctx=dummy_ctx
        )
        assert len(fake.attachments) == 1
        assert "File attached" in result

    def test_same_name_different_content_notes_mismatch(
        self, tmp_path, monkeypatch, dummy_ctx
    ):
        """A filename match with different bytes is a replace attempt — the
        skip message must say the stored content differs, not look like a
        no-op re-run."""
        pdf = tmp_path / "smith-2020.pdf"
        pdf.write_bytes(b"%PDF-1.4 corrected-bytes")

        fake = FakeZoteroForAttach()
        fake._children["ITEM1"] = [
            {
                "key": "OLDATT01",
                "data": {
                    "itemType": "attachment",
                    "filename": "smith-2020.pdf",
                    "md5": hashlib.md5(b"%PDF-1.4 old-bytes").hexdigest(),
                },
            }
        ]
        _patch_write_client(monkeypatch, fake)

        result = server.attach_file(
            item_key="ITEM1", file_path=str(pdf), ctx=dummy_ctx
        )
        assert len(fake.attachments) == 0
        assert "content differs" in result
        assert "delete the existing attachment" in result

    def test_child_without_md5_still_uploads(self, tmp_path, monkeypatch, dummy_ctx):
        pdf = tmp_path / "new-name.pdf"
        pdf.write_bytes(b"%PDF-1.4 new-bytes")

        fake = FakeZoteroForAttach()
        fake._children["ITEM1"] = [
            {"key": "OLDATT01", "data": {"itemType": "attachment", "filename": "old-name.pdf"}}
        ]
        _patch_write_client(monkeypatch, fake)

        result = server.attach_file(
            item_key="ITEM1", file_path=str(pdf), ctx=dummy_ctx
        )
        assert len(fake.attachments) == 1
        assert "File attached" in result

    def test_md5_dedupe_applies_to_url_branch(self, monkeypatch, dummy_ctx):
        content = b"%PDF-1.4 " + b"x" * 2000  # FakeResponse default payload

        fake = FakeZoteroForAttach()
        fake._children["ITEM1"] = [
            {
                "key": "OLDATT01",
                "data": {
                    "itemType": "attachment",
                    "filename": "old-name.pdf",
                    "md5": hashlib.md5(content).hexdigest(),
                },
            }
        ]
        _patch_write_client(monkeypatch, fake)
        _patch_guarded_get(monkeypatch, FakeResponse(content=content))

        result = server.attach_file(
            item_key="ITEM1",
            url="https://example.org/papers/smith-2020.pdf",
            ctx=dummy_ctx,
        )
        assert len(fake.attachments) == 0
        assert "old-name.pdf" in result
        assert "not re-uploaded" in result


# ---------------------------------------------------------------------------
# attach_file: URL branch
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(
        self,
        content=b"%PDF-1.4 " + b"x" * 2000,
        content_type="application/pdf",
        status=200,
    ):
        self._content = content
        self.headers = {"Content-Type": content_type}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]


def _patch_guarded_get(monkeypatch, response):
    monkeypatch.setattr(
        "zotero_mcp.tools._helpers._guarded_pdf_get",
        lambda pdf_url, ctx: response,
    )


class TestAttachFileUrl:
    def test_happy_path_downloads_and_attaches(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_guarded_get(monkeypatch, FakeResponse())

        result = server.attach_file(
            item_key="ITEM1",
            url="https://example.org/papers/smith-2020.pdf",
            ctx=dummy_ctx,
        )

        assert len(fake.attachments) == 1
        att = fake.attachments[0]
        assert att["parentid"] == "ITEM1"
        assert att["files"][0][0] == "smith-2020.pdf"
        assert "File attached" in result

    def test_filename_from_url_falls_back_to_item_key(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_guarded_get(monkeypatch, FakeResponse())

        server.attach_file(
            item_key="ITEM1",
            url="https://example.org/download?id=42",
            ctx=dummy_ctx,
        )
        assert fake.attachments[0]["files"][0][0] == "ITEM1.pdf"

    def test_filename_override_gets_pdf_extension(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_guarded_get(monkeypatch, FakeResponse())

        server.attach_file(
            item_key="ITEM1",
            url="https://example.org/download?id=42",
            filename="smith-2020",
            ctx=dummy_ctx,
        )
        assert fake.attachments[0]["files"][0][0] == "smith-2020.pdf"

    def test_rejects_non_http_url(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)

        result = server.attach_file(
            item_key="ITEM1", url="file:///etc/passwd", ctx=dummy_ctx
        )
        assert "http(s)" in result
        assert len(fake.attachments) == 0

    def test_ssrf_rejection_surfaces_error(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_guarded_get(monkeypatch, None)

        result = server.attach_file(
            item_key="ITEM1", url="https://internal.local/a.pdf", ctx=dummy_ctx
        )
        assert result.startswith("Error")
        assert len(fake.attachments) == 0

    def test_non_pdf_content_type_rejected(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_guarded_get(monkeypatch, FakeResponse(content_type="text/html"))

        result = server.attach_file(
            item_key="ITEM1", url="https://example.org/a.pdf", ctx=dummy_ctx
        )
        assert "did not return a PDF" in result
        assert len(fake.attachments) == 0

    def test_tiny_download_rejected(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_guarded_get(monkeypatch, FakeResponse(content=b"tiny"))

        result = server.attach_file(
            item_key="ITEM1", url="https://example.org/a.pdf", ctx=dummy_ctx
        )
        assert "1 KB" in result or "too small" in result
        assert len(fake.attachments) == 0

    def test_http_error_surfaced(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        _patch_write_client(monkeypatch, fake)
        _patch_guarded_get(monkeypatch, FakeResponse(status=403))

        result = server.attach_file(
            item_key="ITEM1", url="https://example.org/a.pdf", ctx=dummy_ctx
        )
        assert result.startswith("Error")
        assert len(fake.attachments) == 0

    def test_dedupe_applies_to_url_branch(self, monkeypatch, dummy_ctx):
        fake = FakeZoteroForAttach()
        fake._children["ITEM1"] = [
            {"data": {"itemType": "attachment", "filename": "smith-2020.pdf"}}
        ]
        _patch_write_client(monkeypatch, fake)
        _patch_guarded_get(monkeypatch, FakeResponse())

        result = server.attach_file(
            item_key="ITEM1",
            url="https://example.org/papers/smith-2020.pdf",
            ctx=dummy_ctx,
        )
        assert len(fake.attachments) == 0
        assert "already present" in result
