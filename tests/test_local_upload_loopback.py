"""Attachment uploads to the local API reach 127.0.0.1 too, not only the other requests.

pyzotero sends an attachment's bytes (step 2 of its upload) with its HTTP
library's module-level ``post()``, not with the client we inject, to the URL
the server returned in step 1. Zotero's local server builds that URL from the
request's ``Host`` header, so it names ``localhost``, which resolves to ``::1``
first while Zotero listens on 127.0.0.1 only: on Windows every upload paid two
seconds before falling back. The fake server below listens on 127.0.0.1 only
and builds the upload URL the same way.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from conftest import pyzotero_http_module
from pyzotero import _upload as pyzotero_upload
from pyzotero.zotero import Zotero

from zotero_mcp import client as zclient
from zotero_mcp.tools import _helpers

http = pyzotero_http_module()

ATTACHMENT_KEY = "ATTACH01"


class _FakeLocalZotero(BaseHTTPRequestHandler):
    """Just enough of the local API's upload flow (create, authorize, upload, register)."""

    def log_message(self, *_args):
        pass

    def _reply(self, status, payload=None, headers=None):
        body = json.dumps(payload).encode() if payload is not None else b""
        self.send_response(status)
        for name, value in {"Last-Modified-Version": "7", **(headers or {})}.items():
            self.send_header(name, value)
        if payload is not None:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.seen.append({"path": self.path, "host": self.headers.get("Host"), "body": body})
        if self.path == "/api/users/0/items":
            self._reply(200, {
                "success": {"0": ATTACHMENT_KEY},
                "successful": {"0": {"key": ATTACHMENT_KEY, "version": 5}},
                "failed": {},
            })
        elif self.path == f"/api/users/0/items/{ATTACHMENT_KEY}/file" and b"upload=" in body:
            self._reply(204)
        elif self.path == f"/api/users/0/items/{ATTACHMENT_KEY}/file":
            # Like Zotero: the upload URL is built from the Host header.
            self._reply(200, {
                "url": f"http://{self.headers['Host']}/connector/upload/{ATTACHMENT_KEY}",
                "params": {"key": "file-key", "acl": "private"},
                "uploadKey": "UPLOADKEY",
            })
        elif self.path == f"/connector/upload/{ATTACHMENT_KEY}":
            self._reply(201)
        else:
            self._reply(404, {"error": self.path})


@pytest.fixture
def local_zotero():
    """A fake local Zotero on 127.0.0.1 only, like the real one."""
    server = HTTPServer(("127.0.0.1", 0), _FakeLocalZotero)
    server.seen = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _fresh_shortcut_state(monkeypatch):
    monkeypatch.setattr(zclient, "_loopback_shortcut_disabled_until", 0.0)


@pytest.fixture
def dialled(monkeypatch):
    """Record the host every transport connects to, still sending the request."""
    hosts = []
    real = http.HTTPTransport.handle_request

    def recording(self, request):
        hosts.append(request.url.host)
        return real(self, request)

    monkeypatch.setattr(http.HTTPTransport, "handle_request", recording)
    return hosts


def _local_write_zot(port):
    zot = Zotero(
        library_id="0",
        library_type="user",
        api_key=None,
        local=True,
        client=zclient._make_local_http_client(zclient._LOCAL_WRITE_TIMEOUT),
        server_id="server-id",
        local_api_key="local-key",
    )
    # pyzotero 1.15.2 and later address the local API as 127.0.0.1 themselves;
    # the earlier releases the floor allows use localhost, the case under test.
    zot.endpoint = f"http://localhost:{port}/api"
    return zot


def test_upload_step_reaches_ipv4_server_over_loopback(local_zotero, dialled, tmp_path):
    port = local_zotero.server_address[1]
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake bytes")
    zot = _local_write_zot(port)

    result = _helpers.attach_files(zot, [("Paper", str(pdf))], parentid="PARENT01")

    assert [item["key"] for item in result["success"]] == [ATTACHMENT_KEY]
    assert result["failure"] == []
    paths = [seen["path"] for seen in local_zotero.seen]
    assert f"/connector/upload/{ATTACHMENT_KEY}" in paths
    # Zotero saw the Host header it builds the upload URL from, on every step.
    assert {seen["host"] for seen in local_zotero.seen} == {f"localhost:{port}"}
    # The file's bytes travelled in the multipart body, after the params.
    upload = next(s for s in local_zotero.seen if s["path"].startswith("/connector/upload/"))
    assert b"%PDF-1.4 fake bytes" in upload["body"]
    assert upload["body"].index(b"file-key") < upload["body"].index(b"%PDF-1.4 fake bytes")
    # No connection went to the name (::1 first on Windows): all of them,
    # the upload included, dialled 127.0.0.1 directly.
    assert dialled and set(dialled) == {"127.0.0.1"}


def test_options_pyzotero_passes_keep_the_loopback_route(local_zotero, dialled):
    # pyzotero 1.16 adds trust_env=False for the local upload; earlier releases
    # pass only files, headers and timeout.
    port = local_zotero.server_address[1]
    stand_in = zclient._LoopbackUploadHTTP(http)
    url = f"http://localhost:{port}/connector/upload/{ATTACHMENT_KEY}"

    response = stand_in.post(url, files=[("file", b"x")], headers={"A": "b"}, timeout=5, trust_env=False)

    assert response.status_code == 201
    assert dialled == ["127.0.0.1"]
    assert str(response.request.url) == url  # pyzotero's error messages quote it


def test_other_urls_go_to_the_real_post(monkeypatch):
    calls = []
    monkeypatch.setattr(http, "post", lambda url, **kw: calls.append((url, kw)) or "real")
    stand_in = zclient._LoopbackUploadHTTP(http)

    assert stand_in.post("https://s3.example/bucket", files=[("file", b"x")], timeout=9) == "real"
    assert stand_in.post("http://127.0.0.1:23119/connector/upload/K", files=[]) == "real"
    # An option pyzotero does not pass keeps the library's own semantics.
    assert stand_in.post("http://localhost:23119/x", files=[], verify=False) == "real"
    assert [url for url, _ in calls] == [
        "https://s3.example/bucket",
        "http://127.0.0.1:23119/connector/upload/K",
        "http://localhost:23119/x",
    ]
    assert calls[0][1] == {"files": [("file", b"x")], "timeout": 9}


def test_stand_in_exposes_the_real_library(monkeypatch):
    stand_in = zclient._LoopbackUploadHTTP(http)
    assert stand_in.ConnectError is http.ConnectError
    assert stand_in.HTTPError is http.HTTPError
    assert stand_in.codes.TOO_MANY_REQUESTS == 429


def test_routing_is_installed_once_on_pyzoteros_upload_module(monkeypatch):
    names = [n for n in ("httpx2", "httpx") if hasattr(pyzotero_upload, n)]
    assert names, "pyzotero's upload module no longer names its HTTP library"
    name = names[0]
    monkeypatch.setattr(pyzotero_upload, name, http)

    zclient._route_pyzotero_uploads_over_loopback(http)
    installed = getattr(pyzotero_upload, name)
    zclient._route_pyzotero_uploads_over_loopback(http)

    assert isinstance(installed, zclient._LoopbackUploadHTTP)
    assert getattr(pyzotero_upload, name) is installed  # not wrapped twice


def test_routing_leaves_a_different_library_alone(monkeypatch):
    names = [n for n in ("httpx2", "httpx") if hasattr(pyzotero_upload, n)]
    name = names[0]
    other = object()
    monkeypatch.setattr(pyzotero_upload, name, other)
    zclient._route_pyzotero_uploads_over_loopback(http)
    assert getattr(pyzotero_upload, name) is other
