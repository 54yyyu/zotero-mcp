"""ZOTERO_API_URL points the web API clients at a self-hosted server."""

import pytest

from zotero_mcp import client as _client
from zotero_mcp import semantic_search


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
def test_web_clients_send_requests_to_api_url(web_mode, monkeypatch, httpserver, factory):
    httpserver.expect_request("/users/123/items").respond_with_json([], headers={"Total-Results": "0"})
    monkeypatch.setenv("ZOTERO_API_URL", httpserver.url_for("/"))
    assert factory().items(limit=1) == []
    httpserver.check_assertions()


def test_local_client_ignores_api_url(monkeypatch):
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    monkeypatch.setenv("ZOTERO_API_URL", "https://zotero.example.org/")
    assert "example.org" not in _client.get_zotero_client().endpoint
