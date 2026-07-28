"""Tests for direct WebDAV attachment access."""

from zipfile import ZIP_DEFLATED, ZipFile

from conftest import skip_on_ci
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


@skip_on_ci
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
    assert session.requested == [("https://dav.example.com/zotero/ABCD1234.zip", (10.0, 30.0), True)]


@skip_on_ci
def test_download_attachment_from_webdav_escapes_attachment_key(tmp_path, monkeypatch):
    session = _FakeSession(_FakeResponse(_build_zip_bytes("paper.pdf", b"%PDF-1.4 webdav test")))
    monkeypatch.setenv("ZOTERO_WEBDAV_URL", "https://dav.example.com/zotero")
    monkeypatch.setenv("ZOTERO_WEBDAV_USERNAME", "alice")
    monkeypatch.setenv("ZOTERO_WEBDAV_PASSWORD", "secret")
    monkeypatch.setattr("requests.Session", lambda: session)

    webdav.download_attachment_from_webdav("AB/CD", tmp_path, expected_filename="paper.pdf")

    assert session.requested == [("https://dav.example.com/zotero/AB%2FCD.zip", (10.0, 30.0), True)]


@skip_on_ci
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
    ]


@skip_on_ci
def test_extract_archive_rejects_backslash_path_traversal(tmp_path):
    zip_bytes = _build_zip_bytes("..\\evil.txt", b"oops")

    try:
        webdav._extract_archive(zip_bytes, tmp_path, expected_filename="evil.txt")
    except ValueError as exc:
        assert "Unsafe path" in str(exc)
    else:
        raise AssertionError("Expected _extract_archive() to reject backslash traversal paths")


# ---------------------------------------------------------------------------
# upload_attachment_to_webdav
# ---------------------------------------------------------------------------


class _RecordingPutSession:
    """Captures PUT calls and returns a configurable response."""

    def __init__(self, status_code: int = 201):
        self.status_code = status_code
        self.auth = None
        self.trust_env = True
        self.calls = []  # list of (url, data, headers)
        self.timeouts = []  # parallel list of the timeout passed to each PUT

    def put(self, url, data=None, headers=None, timeout=None):
        self.calls.append((url, data, dict(headers or {})))
        self.timeouts.append(timeout)

        class _Resp:
            def __init__(self_inner, sc):
                self_inner.status_code = sc

            def raise_for_status(self_inner):
                if self_inner.status_code >= 400:
                    raise RuntimeError(f"http {self_inner.status_code}")

        return _Resp(self.status_code)

    def close(self):
        return None


def _setup_webdav_env(monkeypatch):
    monkeypatch.setenv("ZOTERO_WEBDAV_URL", "https://dav.example.com/zotero")
    monkeypatch.setenv("ZOTERO_WEBDAV_USERNAME", "alice")
    monkeypatch.setenv("ZOTERO_WEBDAV_PASSWORD", "secret")


@skip_on_ci
def test_upload_attachment_puts_zip_then_prop(tmp_path, monkeypatch):
    src = tmp_path / "paper.pdf"
    src.write_bytes(b"%PDF-1.4 hello")
    session = _RecordingPutSession()
    _setup_webdav_env(monkeypatch)
    monkeypatch.setattr("requests.Session", lambda: session)

    md5_hex, mtime_ms = webdav.upload_attachment_to_webdav("ABCD1234", src)

    # exactly two PUTs, .zip before .prop (Zotero treats orphan .prop as corruption)
    assert len(session.calls) == 2
    zip_url, zip_data, zip_headers = session.calls[0]
    prop_url, prop_data, prop_headers = session.calls[1]
    assert zip_url == "https://dav.example.com/zotero/ABCD1234.zip"
    assert prop_url == "https://dav.example.com/zotero/ABCD1234.prop"
    assert zip_headers.get("Content-Type") == "application/zip"
    assert prop_headers.get("Content-Type", "").startswith("text/xml")
    assert session.auth == ("alice", "secret")

    # return values match what we'll write to the Zotero attachment item
    import hashlib
    expected_md5 = hashlib.md5(b"%PDF-1.4 hello").hexdigest()  # noqa: S324
    assert md5_hex == expected_md5
    assert mtime_ms == int(src.stat().st_mtime * 1000)

    # prop XML carries the same md5 and mtime
    prop_text = prop_data.decode("utf-8")
    assert f"<hash>{expected_md5}</hash>" in prop_text
    assert f"<mtime>{mtime_ms}</mtime>" in prop_text


@skip_on_ci
def test_upload_attachment_zip_contains_file_under_basename(tmp_path, monkeypatch):
    src = tmp_path / "subdir" / "paper.pdf"
    src.parent.mkdir()
    src.write_bytes(b"%PDF body")
    session = _RecordingPutSession()
    _setup_webdav_env(monkeypatch)
    monkeypatch.setattr("requests.Session", lambda: session)

    webdav.upload_attachment_to_webdav("ABCD1234", src)

    import io
    _zip_url, zip_data, _ = session.calls[0]
    with ZipFile(io.BytesIO(zip_data)) as zf:
        names = zf.namelist()
        # zip stores the BASENAME — not the full path — so Zotero clients
        # can extract it next to other attachments without leaking paths.
        assert names == ["paper.pdf"]
        assert zf.read("paper.pdf") == b"%PDF body"


@skip_on_ci
def test_upload_attachment_raises_when_not_configured(tmp_path, monkeypatch):
    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")
    monkeypatch.delenv("ZOTERO_WEBDAV_URL", raising=False)
    monkeypatch.delenv("ZOTERO_WEBDAV_USERNAME", raising=False)
    monkeypatch.delenv("ZOTERO_WEBDAV_PASSWORD", raising=False)

    try:
        webdav.upload_attachment_to_webdav("ABCD1234", src)
    except webdav.WebDAVNotConfiguredError as exc:
        assert "ZOTERO_WEBDAV_URL" in str(exc)
    else:
        raise AssertionError("Expected WebDAVNotConfiguredError when env vars are missing")


@skip_on_ci
def test_upload_attachment_escapes_key_in_urls(tmp_path, monkeypatch):
    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")
    session = _RecordingPutSession()
    _setup_webdav_env(monkeypatch)
    monkeypatch.setattr("requests.Session", lambda: session)

    webdav.upload_attachment_to_webdav("AB/CD", src)

    assert session.calls[0][0] == "https://dav.example.com/zotero/AB%2FCD.zip"
    assert session.calls[1][0] == "https://dav.example.com/zotero/AB%2FCD.prop"


@skip_on_ci
def test_upload_default_timeout_is_60s(tmp_path, monkeypatch):
    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")
    session = _RecordingPutSession()
    _setup_webdav_env(monkeypatch)
    monkeypatch.delenv("ZOTERO_WEBDAV_TIMEOUT", raising=False)
    monkeypatch.setattr("requests.Session", lambda: session)

    webdav.upload_attachment_to_webdav("ABCD1234", src)

    # Both PUTs use the historic default — (10s connect, 60s read).
    assert session.timeouts == [(10.0, 60.0), (10.0, 60.0)]


@skip_on_ci
def test_upload_env_override_respected(tmp_path, monkeypatch):
    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")
    session = _RecordingPutSession()
    _setup_webdav_env(monkeypatch)
    monkeypatch.setenv("ZOTERO_WEBDAV_TIMEOUT", "300")
    monkeypatch.setattr("requests.Session", lambda: session)

    webdav.upload_attachment_to_webdav("ABCD1234", src)

    assert session.timeouts == [(10.0, 300.0), (10.0, 300.0)]


@skip_on_ci
def test_upload_explicit_timeout_wins_over_env(tmp_path, monkeypatch):
    src = tmp_path / "paper.pdf"
    src.write_bytes(b"x")
    session = _RecordingPutSession()
    _setup_webdav_env(monkeypatch)
    monkeypatch.setenv("ZOTERO_WEBDAV_TIMEOUT", "300")
    monkeypatch.setattr("requests.Session", lambda: session)

    webdav.upload_attachment_to_webdav("ABCD1234", src, timeout=45.0)

    # Explicit caller arg takes precedence over the env var.
    assert session.timeouts == [(10.0, 45.0), (10.0, 45.0)]


# ---------------------------------------------------------------------------
# _webdav_first_attach (shell + PUT orchestration)
# ---------------------------------------------------------------------------


class _NoOpCtx:
    def info(self, *_a, **_kw):
        return None

    def error(self, *_a, **_kw):
        return None

    def warning(self, *_a, **_kw):
        return None


class _ShellZotero:
    """Minimal pyzotero stand-in for the _webdav_first_attach flow.

    Records the template passed to create_items, returns a deterministic
    attachment key, and tracks delete_item calls so cleanup-on-failure can
    be asserted.
    """

    def __init__(self, attachment_key="ATTCHKEY0", attachment_version=42):
        self._attachment_key = attachment_key
        self._attachment_version = attachment_version
        self.created_template = None
        self.deleted_payloads = []
        self.delete_fails = False

    def item_template(self, item_type, **kwargs):
        return {"itemType": item_type, **kwargs}

    def create_items(self, items, **kwargs):
        self.created_template = items[0]
        return {
            "success": {"0": self._attachment_key},
            # Mirrors pyzotero: successVersions is keyed in parallel to
            # success and carries the new item version.
            "successVersions": {"0": self._attachment_version},
            "failed": {},
        }

    def delete_item(self, payload, last_modified=None):
        # Mirror pyzotero's contract: payload is a dict with key/version
        # (or a list of such dicts). A bare string must be rejected so a
        # regression of the call site is caught here rather than in prod.
        if isinstance(payload, str):
            raise TypeError("delete_item expects a dict with 'key'/'version', got str")
        if isinstance(payload, list):
            keys = [p["key"] for p in payload]
            versions = [p["version"] for p in payload]
        else:
            keys = [payload["key"]]
            versions = [payload["version"]]
        self.deleted_payloads.append(payload)
        assert all(v == self._attachment_version for v in versions), f"unexpected version(s): {versions}"
        if self.delete_fails:
            raise RuntimeError("delete refused")
        return keys


def _src_pdf(tmp_path):
    src = tmp_path / "paper.pdf"
    src.write_bytes(b"%PDF-1.4 webdav shell test")
    return src


@skip_on_ci
def test_webdav_first_attach_unconfigured_returns_none(tmp_path, monkeypatch):
    """No WebDAV env -> _webdav_first_attach returns None so the caller falls
    through to its existing attachment_both flow untouched.
    """
    from zotero_mcp.tools import _helpers

    monkeypatch.delenv("ZOTERO_WEBDAV_URL", raising=False)
    monkeypatch.delenv("ZOTERO_WEBDAV_USERNAME", raising=False)
    monkeypatch.delenv("ZOTERO_WEBDAV_PASSWORD", raising=False)

    zot = _ShellZotero()
    src = _src_pdf(tmp_path)
    result = _helpers._webdav_first_attach(zot, "paper.pdf", src, "PARENT0", _NoOpCtx(), content_type="application/pdf")
    assert result is None
    assert zot.created_template is None  # no shell created


@skip_on_ci
def test_webdav_first_attach_configured_creates_shell_and_uploads(tmp_path, monkeypatch):
    """WebDAV configured -> shell created with contentType, PUT issued with the right key."""
    from zotero_mcp.tools import _helpers

    _setup_webdav_env(monkeypatch)
    session = _RecordingPutSession(status_code=201)
    monkeypatch.setattr("requests.Session", lambda: session)

    zot = _ShellZotero(attachment_key="ATTCHKEY0")
    src = _src_pdf(tmp_path)

    result = _helpers._webdav_first_attach(zot, "paper.pdf", src, "PARENT0", _NoOpCtx(), content_type="application/pdf")

    # Shell carries title/filename/parentItem and the contentType.
    assert zot.created_template["title"] == "paper.pdf"
    assert zot.created_template["filename"] == "paper.pdf"
    assert zot.created_template["parentItem"] == "PARENT0"
    assert zot.created_template["contentType"] == "application/pdf"

    # PUT went out under the shell's attachment key.
    assert session.calls[0][0] == "https://dav.example.com/zotero/ATTCHKEY0.zip"
    assert session.calls[1][0] == "https://dav.example.com/zotero/ATTCHKEY0.prop"

    assert result == " (uploaded to WebDAV as ATTCHKEY0.zip)"
    assert zot.deleted_payloads == []  # no cleanup on success


@skip_on_ci
def test_webdav_first_attach_put_failure_deletes_shell_and_warns(tmp_path, monkeypatch):
    """PUT failure -> shell deleted via delete_item and the suffix warns about it."""
    from zotero_mcp.tools import _helpers

    _setup_webdav_env(monkeypatch)
    session = _RecordingPutSession(status_code=500)
    monkeypatch.setattr("requests.Session", lambda: session)

    zot = _ShellZotero(attachment_key="ATTCHKEY0")
    src = _src_pdf(tmp_path)

    result = _helpers._webdav_first_attach(zot, "paper.pdf", src, "PARENT0", _NoOpCtx(), content_type="application/pdf")

    # The orphaned shell was deleted so the Zotero UI/sync stays consistent.
    assert zot.deleted_payloads == [{"key": "ATTCHKEY0", "version": 42}]
    assert "WARNING" in result
    assert "ATTCHKEY0" in result
    assert "was deleted" in result


@skip_on_ci
def test_webdav_first_attach_put_failure_falls_back_to_warning_if_delete_fails(tmp_path, monkeypatch):
    """If the cleanup delete itself fails, the suffix retains the old 'no file bytes' wording."""
    from zotero_mcp.tools import _helpers

    _setup_webdav_env(monkeypatch)
    session = _RecordingPutSession(status_code=500)
    monkeypatch.setattr("requests.Session", lambda: session)

    zot = _ShellZotero(attachment_key="ATTCHKEY0")
    zot.delete_fails = True
    src = _src_pdf(tmp_path)

    result = _helpers._webdav_first_attach(zot, "paper.pdf", src, "PARENT0", _NoOpCtx(), content_type="application/pdf")

    assert zot.deleted_payloads == [{"key": "ATTCHKEY0", "version": 42}]  # delete was attempted
    assert "WARNING" in result
    assert "could not be deleted" in result


# ---------------------------------------------------------------------------
# _maybe_upload_to_webdav (the attachment_both + PUT path)
# ---------------------------------------------------------------------------


class _AttachBothZotero:
    """Fake client for the _maybe_upload_to_webdav flow.

    attachment_both succeeded (so the key/version are in attach_result),
    and delete_item is wired exactly like pyzotero so cleanup-on-PUT-failure
    can be asserted.
    """

    def __init__(self, attachment_key="ATTCH2", attachment_version=7):
        self._key = attachment_key
        self._version = attachment_version
        self.deleted_payloads = []
        self.delete_fails = False

    def item(self, key):
        return {"key": key, "version": self._version}

    def delete_item(self, payload, last_modified=None):
        if isinstance(payload, str):
            raise TypeError("delete_item expects a dict with 'key'/'version', got str")
        if isinstance(payload, list):
            keys = [p["key"] for p in payload]
            versions = [p["version"] for p in payload]
        else:
            keys = [payload["key"]]
            versions = [payload["version"]]
        self.deleted_payloads.append(payload)
        assert all(v == self._version for v in versions), f"unexpected version(s): {versions}"
        if self.delete_fails:
            raise RuntimeError("delete refused")
        return keys

    def attachment_both(self, *_a, **_kw):
        return {
            "success": [{"key": self._key, "version": self._version}],
            "unchanged": [],
            "failed": {},
        }


def _attach_result(key="ATTCH2", version=7):
    return {
        "success": [{"key": key, "version": version}],
        "unchanged": [],
        "failed": {},
    }


@skip_on_ci
def test_maybe_upload_success_returns_suffix(tmp_path, monkeypatch):
    """Successful PUT -> short success suffix, no cleanup."""
    from zotero_mcp.tools import _helpers

    _setup_webdav_env(monkeypatch)
    session = _RecordingPutSession(status_code=201)
    monkeypatch.setattr("requests.Session", lambda: session)

    zot = _AttachBothZotero()
    src = _src_pdf(tmp_path)

    result = _helpers._maybe_upload_to_webdav(_attach_result(), src, _NoOpCtx(), write_zot=zot)

    assert session.calls[0][0] == "https://dav.example.com/zotero/ATTCH2.zip"
    assert result == " (uploaded to WebDAV as ATTCH2.zip)"
    assert zot.deleted_payloads == []  # no cleanup on success


@skip_on_ci
def test_maybe_upload_put_failure_deletes_attachment_and_warns(tmp_path, monkeypatch):
    """PUT failure after attachment_both -> the just-created attachment is deleted."""
    from zotero_mcp.tools import _helpers

    _setup_webdav_env(monkeypatch)
    session = _RecordingPutSession(status_code=500)
    monkeypatch.setattr("requests.Session", lambda: session)

    zot = _AttachBothZotero(attachment_key="ATTCH2", attachment_version=7)
    src = _src_pdf(tmp_path)

    result = _helpers._maybe_upload_to_webdav(_attach_result("ATTCH2", 7), src, _NoOpCtx(), write_zot=zot)

    assert zot.deleted_payloads == [{"key": "ATTCH2", "version": 7}]
    assert "WARNING" in result
    assert "ATTCH2" in result
    assert "was deleted" in result


@skip_on_ci
def test_maybe_upload_put_failure_falls_back_if_delete_fails(tmp_path, monkeypatch):
    """If cleanup delete itself fails, suffix keeps the old 'no file bytes' wording."""
    from zotero_mcp.tools import _helpers

    _setup_webdav_env(monkeypatch)
    session = _RecordingPutSession(status_code=500)
    monkeypatch.setattr("requests.Session", lambda: session)

    zot = _AttachBothZotero(attachment_key="ATTCH2", attachment_version=7)
    zot.delete_fails = True
    src = _src_pdf(tmp_path)

    result = _helpers._maybe_upload_to_webdav(_attach_result("ATTCH2", 7), src, _NoOpCtx(), write_zot=zot)

    assert zot.deleted_payloads == [{"key": "ATTCH2", "version": 7}]  # delete was attempted
    assert "WARNING" in result
    assert "could not be deleted" in result


@skip_on_ci
def test_maybe_upload_no_key_returns_empty(tmp_path, monkeypatch):
    """When no attachment key can be extracted, no PUT and no cleanup happen."""
    from zotero_mcp.tools import _helpers

    _setup_webdav_env(monkeypatch)
    monkeypatch.setattr("requests.Session", lambda: _RecordingPutSession())

    zot = _AttachBothZotero()
    src = _src_pdf(tmp_path)

    result = _helpers._maybe_upload_to_webdav({"success": [], "unchanged": []}, src, _NoOpCtx())

    assert result == ""
    assert zot.deleted_payloads == []
