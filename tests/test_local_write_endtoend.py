"""End-to-end checks against a stub of Zotero 10's local API.

The unit tests replace pyzotero with fakes, which means they never exercise
the part most likely to be wrong: whether a real pyzotero client actually puts
Zotero-Server-ID and Zotero-API-Key on the wire, and whether it addresses the
paths the local API expects. This module runs a small HTTP server that speaks
enough of the local API to answer that, and asserts on the requests it saw.

It cannot prove Zotero itself accepts these requests — only a live Zotero 10
can — but it does prove the client is sending what the spec describes.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from zotero_mcp import client as _client
from zotero_mcp.tools import _helpers

SERVER_ID = "stub-server-id"
GRANTED_KEY = "0123456789abcdef0123456789abcdef"


class _Handler(BaseHTTPRequestHandler):
    """Enough of the local API to observe what a client sends."""

    def log_message(self, *_args):
        pass  # keep pytest output clean

    def _record(self, method):
        body = b""
        if length := int(self.headers.get("Content-Length") or 0):
            body = self.rfile.read(length)
        self.server.requests.append(
            {
                "method": method,
                "path": self.path,
                "headers": dict(self.headers),
                "body": body.decode("utf-8") if body else "",
            }
        )
        return body

    def _respond(self, status, payload=None, headers=None):
        body = json.dumps(payload).encode() if payload is not None else b""
        self.send_response(status)
        # Every local API response carries the server id, errors included.
        self.send_header("Zotero-Server-ID", SERVER_ID)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        self._record("GET")
        self._respond(200, {})

    def do_POST(self):
        body = self._record("POST")
        if self.path.endswith("/local/authorize"):
            payload = json.loads(body)
            self._respond(200, {"key": GRANTED_KEY, "remember": bool(payload)})
            return
        # A write: refuse it the way the real server does if the headers the
        # spec requires are missing, so the test can tell them apart.
        if not self.headers.get("Zotero-Server-ID"):
            self._respond(428, {"error": "Zotero-Server-ID not provided"})
            return
        if not self.headers.get("Zotero-API-Key"):
            self._respond(401, {"error": "API key required"})
            return
        self._respond(200, {"success": {"0": "NEWKEY01"}, "unchanged": {}, "failed": {}})

    def do_PATCH(self):
        body = self._record("PATCH")
        if not self.headers.get("Zotero-Server-ID"):
            self._respond(428, {"error": "Zotero-Server-ID not provided"})
            return
        if not self.headers.get("Zotero-API-Key"):
            self._respond(401, {"error": "API key required"})
            return
        # The real local API answers "400 Empty request body" when a PATCH
        # arrives without a JSON content type, even though the bytes are
        # there. The web API does not, so only a strict stub catches it.
        if "json" not in (self.headers.get("Content-Type") or ""):
            self._respond(400, {"error": "Empty request body"})
            return
        if not body:
            self._respond(400, {"error": "Empty request body"})
            return
        self._respond(204)


@pytest.fixture
def stub_zotero():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def local_write_client(stub_zotero, monkeypatch):
    """A real pyzotero client, pointed at the stub instead of port 23119."""
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_LOCAL_API_KEY", GRANTED_KEY)
    monkeypatch.setenv("ZOTERO_LOCAL_SERVER_ID", SERVER_ID)
    zot = _client.get_local_write_client()
    host, port = stub_zotero.server_address
    zot.endpoint = f"http://{host}:{port}/api"
    return zot


def _writes(server):
    return [r for r in server.requests if r["method"] in ("POST", "PATCH")]


class TestHeadersOnTheWire:
    def test_create_items_carries_both_required_headers(self, stub_zotero, local_write_client):
        result = local_write_client.create_items(
            [{"itemType": "journalArticle", "title": "Test"}]
        )
        assert result["success"] == {"0": "NEWKEY01"}

        write = _writes(stub_zotero)[0]
        assert write["headers"]["Zotero-Server-ID"] == SERVER_ID
        assert write["headers"]["Zotero-API-Key"] == GRANTED_KEY
        assert write["path"] == "/api/users/0/items"

    def test_trash_shim_carries_them_too(self, stub_zotero, local_write_client):
        """The old inline client.patch bypassed pyzotero's write dispatcher, so
        it would have sent neither header and earned a 428."""
        ok, detail = _helpers.trash_item(
            local_write_client, {"key": "ITEM01", "version": 3}
        )
        assert (ok, detail) == (True, "")

        write = _writes(stub_zotero)[0]
        assert write["method"] == "PATCH"
        assert write["headers"]["Zotero-Server-ID"] == SERVER_ID
        assert write["headers"]["Zotero-API-Key"] == GRANTED_KEY
        assert write["headers"]["If-Unmodified-Since-Version"] == "3"
        # Without this the local API reports "400 Empty request body" even
        # though the bytes are present.
        assert "json" in write["headers"]["Content-Type"]
        assert json.loads(write["body"]) == {"deleted": 1}

    def test_a_write_without_the_key_is_reported_usefully(
        self, stub_zotero, local_write_client
    ):
        local_write_client.local_api_key = None
        ok, detail = _helpers.trash_item(
            local_write_client, {"key": "ITEM01", "version": 3}
        )
        assert ok is False
        assert "Always Allow" in detail


class TestAuthorizeOverHttp:
    def test_round_trip(self, stub_zotero, monkeypatch):
        monkeypatch.setenv("ZOTERO_LOCAL", "true")
        host, port = stub_zotero.server_address

        real_zotero = _client.zotero.Zotero

        def _pointed_at_stub(*args, **kwargs):
            zot = real_zotero(*args, **kwargs)
            zot.endpoint = f"http://{host}:{port}/api"
            return zot

        monkeypatch.setattr(_client.zotero, "Zotero", _pointed_at_stub)

        result = _client.authorize_local_api("Test App")
        assert result["key"] == GRANTED_KEY
        assert result["server_id"] == SERVER_ID

        request = next(r for r in stub_zotero.requests if "authorize" in r["path"])
        assert request["path"] == "/api/local/authorize"
        assert json.loads(request["body"]) == {"appName": "Test App"}
        # The spec requires the server id on the authorize call as well.
        assert request["headers"]["Zotero-Server-ID"] == SERVER_ID

        # And the granted key is what later writes will pick up.
        assert _client.get_local_write_credentials()[0] == GRANTED_KEY

    def test_server_id_is_discovered_when_not_supplied(self, stub_zotero, monkeypatch):
        """pyzotero reads it off any response; GET /api/ is the cheap one."""
        monkeypatch.setenv("ZOTERO_LOCAL", "true")
        host, port = stub_zotero.server_address

        real_zotero = _client.zotero.Zotero

        def _pointed_at_stub(*args, **kwargs):
            kwargs.pop("server_id", None)
            zot = real_zotero(*args, **kwargs)
            zot.endpoint = f"http://{host}:{port}/api"
            return zot

        monkeypatch.setattr(_client.zotero, "Zotero", _pointed_at_stub)
        result = _client.authorize_local_api("Test App")
        assert result["server_id"] == SERVER_ID
        assert any(r["path"] == "/api/" for r in stub_zotero.requests)
