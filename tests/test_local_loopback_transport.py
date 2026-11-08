"""The local API client connects to 127.0.0.1 instead of resolving localhost.

Zotero's local server listens on the IPv4 loopback only. pyzotero addresses
it as ``localhost``, which resolves to ``::1`` first, and on Windows a
connection to a closed port takes about two seconds to be refused before the
fallback to 127.0.0.1 — so every local API request paid two seconds. The
transport now dials 127.0.0.1 directly while leaving the request itself (URL,
``Host`` header, the request pyzotero quotes in its errors) unchanged.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from conftest import pyzotero_http_module

from zotero_mcp import client as zclient

http = pyzotero_http_module()


class _Recorder(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.server.seen.append({"path": self.path, "host": self.headers.get("Host")})
        body = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def ipv4_only_server():
    """A server on 127.0.0.1 only, like Zotero's local API."""
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
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


def test_localhost_request_reaches_ipv4_server_with_unchanged_request(ipv4_only_server):
    port = ipv4_only_server.server_address[1]
    with zclient._make_local_http_client() as c:
        resp = c.get(f"http://localhost:{port}/api/users/0/items?limit=1")

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    # The server sees the same Host header the name-based request carried.
    assert ipv4_only_server.seen == [
        {"path": "/api/users/0/items?limit=1", "host": f"localhost:{port}"}
    ]
    # pyzotero reads response.request.url for its error messages.
    assert resp.request.url.host == "localhost"
    assert str(resp.request.url) == f"http://localhost:{port}/api/users/0/items?limit=1"


def test_transport_is_still_an_http1_transport_of_pyzoteros_library():
    with zclient._make_local_http_client() as c:
        assert isinstance(c._transport, http.HTTPTransport)
        assert c._transport._pool._http1 is True
        assert c._transport._pool._http2 is False


def _fake_base_transport(monkeypatch, refuse_hosts):
    """Patch the library transport so no socket is opened; record each dial."""
    dialled = []

    def fake_handle_request(self, request):
        dialled.append(request.url.host)
        if request.url.host in refuse_hosts:
            raise http.ConnectError("refused", request=request)
        return http.Response(200, content=b"{}", request=request)

    monkeypatch.setattr(http.HTTPTransport, "handle_request", fake_handle_request)
    return dialled


def test_falls_back_to_the_name_when_ipv4_refuses(monkeypatch):
    dialled = _fake_base_transport(monkeypatch, refuse_hosts={"127.0.0.1"})
    with zclient._make_local_http_client() as c:
        first = c.get("http://localhost:23119/api/")
        second = c.get("http://localhost:23119/api/")

    assert first.status_code == 200
    assert second.status_code == 200
    # First request: shortcut, refused, then the name. Second request skips
    # the shortcut instead of paying for another refused attempt.
    assert dialled == ["127.0.0.1", "localhost", "localhost"]
    assert zclient._loopback_shortcut_disabled_until > 0


def test_shortcut_is_retried_after_the_backoff(monkeypatch):
    dialled = _fake_base_transport(monkeypatch, refuse_hosts=set())
    monkeypatch.setattr(zclient, "_loopback_shortcut_disabled_until", 1.0)  # long past
    with zclient._make_local_http_client() as c:
        c.get("http://localhost:23119/api/")
    assert dialled == ["127.0.0.1"]


def test_stopped_server_fails_with_the_original_error(monkeypatch):
    dialled = _fake_base_transport(monkeypatch, refuse_hosts={"127.0.0.1", "localhost"})
    with zclient._make_local_http_client() as c:
        with pytest.raises(http.ConnectError) as excinfo:
            c.get("http://localhost:23119/api/")

    assert dialled == ["127.0.0.1", "localhost"]
    # The error comes from the request addressed to localhost, as before.
    assert excinfo.value.request.url.host == "localhost"
    # Nothing answered at all, so there is no reason to stop using the shortcut.
    assert zclient._loopback_shortcut_disabled_until == 0.0


def test_other_hosts_are_not_rewritten(monkeypatch):
    dialled = _fake_base_transport(monkeypatch, refuse_hosts=set())
    with zclient._make_local_http_client() as c:
        c.get("http://127.0.0.1:23119/api/")
        c.get("http://example.invalid/x")
    assert dialled == ["127.0.0.1", "example.invalid"]
