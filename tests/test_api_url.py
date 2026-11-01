"""ZOTERO_API_URL points the web API clients at a self-hosted server."""

import http.server
import socketserver
import threading

import pytest

from zotero_mcp import client as _client
from zotero_mcp import semantic_search


@pytest.fixture
def api_server():
    """A stand-in Zotero Web API on 127.0.0.1 that records the paths it serves.

    socketserver.TCPServer, not http.server.HTTPServer (or pytest-httpserver,
    built on it): HTTPServer.server_bind calls socket.getfqdn(), a reverse DNS
    lookup that stalls past the test timeout on macOS CI runners.
    """
    paths = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            paths.append(self.path.split("?")[0])
            body = b"[]"
            self.send_response(200)
            for k, v in (("Content-Type", "application/json"), ("Total-Results", "0"), ("Content-Length", str(len(body)))):
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/", paths
    server.shutdown()
    server.server_close()


@pytest.fixture
def web_mode(monkeypatch):
    monkeypatch.delenv("ZOTERO_LOCAL", raising=False)
    monkeypatch.setenv("ZOTERO_API_KEY", "webkey")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "123")


def test_web_client_defaults_to_zotero_org(web_mode, monkeypatch):
    monkeypatch.delenv("ZOTERO_API_URL", raising=False)
    assert _client.get_zotero_client().endpoint == "https://api.zotero.org"


@pytest.mark.parametrize(
    "factory",
    [
        _client.get_zotero_client,
        _client.get_web_zotero_client,
        lambda: semantic_search._new_scoped_client("123", "user", "webkey", False),
    ],
)
def test_web_clients_send_requests_to_api_url(web_mode, monkeypatch, api_server, factory):
    url, paths = api_server
    monkeypatch.setenv("ZOTERO_API_URL", url)
    assert factory().items(limit=1) == []
    assert paths == ["/users/123/items"]


def test_local_client_ignores_api_url(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_API_URL", "https://zotero.example.org/")
    assert "example.org" not in _client.get_zotero_client().endpoint
