import hashlib
import os
import tempfile
import zipfile

from zotero_mcp import pdfannots_downloader


def test_verify_archive_checksum(monkeypatch):
    content = b"test-binary-content"
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(content)
        archive_path = tmp.name

    try:
        expected = hashlib.sha256(content).hexdigest()
        monkeypatch.setitem(
            pdfannots_downloader.EXPECTED_SHA256,
            "asset.bin",
            expected,
        )

        assert pdfannots_downloader._verify_archive_checksum(
            archive_path, "https://example.com/asset.bin"
        )

        monkeypatch.setitem(
            pdfannots_downloader.EXPECTED_SHA256,
            "asset.bin",
            "0" * 64,
        )
        assert not pdfannots_downloader._verify_archive_checksum(
            archive_path, "https://example.com/asset.bin"
        )
    finally:
        os.remove(archive_path)


def test_safe_extract_zip_blocks_path_traversal():
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "bad.zip")
        with zipfile.ZipFile(archive_path, "w") as zip_file:
            zip_file.writestr("../evil.txt", "oops")

        try:
            pdfannots_downloader._safe_extract_zip(archive_path, tmpdir)
            assert False, "Expected ValueError for unsafe zip member path"
        except ValueError as exc:
            assert "Unsafe zip member path" in str(exc)
