"""Tests for direct WebDAV attachment access."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from zotero_mcp import client, webdav


class _FailingZotero:
    def dump(self, *_args, **_kwargs):
        raise RuntimeError("not available")


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def iter_content(self, chunk_size=8192):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start:start + chunk_size]


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self._response = response
        self.auth = None
        self.trust_env = True
        self.requested = []

    def get(self, url, timeout=None, stream=False):
        self.requested.append((url, timeout, stream))
        return self._response

    def close(self):
        return None


def _build_zip_bytes(name: str, content: bytes) -> bytes:
    import io

    buf = io.BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED) as zf:
        zf.writestr(name, content)
    return buf.getvalue()


def test_download_attachment_from_webdav_extracts_expected_file(tmp_path, monkeypatch):
    session = _FakeSession(_FakeResponse(_build_zip_bytes("paper.pdf", b"%PDF-1.4 webdav test")))
    monkeypatch.setenv("ZOTERO_WEBDAV_URL", "https://dav.example.com/zotero")
    monkeypatch.setenv("ZOTERO_WEBDAV_USERNAME", "alice")
    monkeypatch.setenv("ZOTERO_WEBDAV_PASSWORD", "secret")
    monkeypatch.setattr("requests.Session", lambda: session)

    file_path = webdav.download_attachment_from_webdav("ABCD1234", tmp_path, expected_filename="paper.pdf")

    assert file_path == tmp_path / "paper.pdf"
    assert file_path.read_bytes() == b"%PDF-1.4 webdav test"
    assert session.auth == ("alice", "secret")
    assert session.trust_env is True
    assert session.requested == [("https://dav.example.com/zotero/ABCD1234.zip", (10.0, 30.0), True)]


def test_download_attachment_file_falls_back_to_webdav(tmp_path, monkeypatch):
    webdav_path = tmp_path / "nested" / "paper.pdf"
    webdav_path.parent.mkdir()
    webdav_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(
        "zotero_mcp.client.download_attachment_from_webdav",
        lambda attachment_key, destination_dir, expected_filename=None: webdav_path,
    )

    result = client.download_attachment_file(
        "ABCD1234",
        tmp_path,
        "paper.pdf",
        local_client=_FailingZotero(),
        web_client=_FailingZotero(),
    )

    assert result.path == webdav_path
    assert result.source == "WebDAV"
    assert result.errors == [
        "Local Zotero: not available",
        "Web API: not available",
    ]
