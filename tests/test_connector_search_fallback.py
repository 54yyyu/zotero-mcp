"""The ChatGPT connector `search` must work on a core install (#572)."""

import json

from zotero_mcp.tools import connectors


class _Backend:
    def search_items(self, query, **kwargs):
        return [{"key": "ABCD1234", "data": {"title": "Attention Is All You Need"}}]


def test_keyword_fallback_when_semantic_search_is_unavailable(monkeypatch, dummy_ctx):
    def unavailable(_config_path):
        raise ImportError("chromadb is not installed")

    monkeypatch.setattr("zotero_mcp.semantic_search.create_semantic_search", unavailable)
    monkeypatch.setattr(connectors._library, "get_library_backend", lambda: _Backend())

    out = json.loads(connectors.chatgpt_connector_search("attention", ctx=dummy_ctx))

    assert out["results"] == [{
        "id": "ABCD1234",
        "title": "Attention Is All You Need",
        "url": "zotero://select/items/ABCD1234",
    }]
