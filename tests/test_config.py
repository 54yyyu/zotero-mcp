"""Tests for zotero_mcp.config — typed config loading."""

import json

import pytest

from zotero_mcp.config import (
    ExtractionConfig,
    ZoteroMcpConfig,
    _strip_nulls,
    load_config,
)

# -- _strip_nulls ------------------------------------------------------------


@pytest.mark.parametrize(
    "inp, expected",
    [
        ({}, {}),
        ({"a": 1, "b": None}, {"a": 1}),
        ({"a": {"b": None, "c": 2}}, {"a": {"c": 2}}),
        ({"a": None}, {}),
        ("not a dict", "not a dict"),
    ],
    ids=["empty", "top-level-null", "nested-null", "all-null", "non-dict"],
)
def test_strip_nulls(inp, expected):
    assert _strip_nulls(inp) == expected


# -- load_config --------------------------------------------------------------


def test_load_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "zotero_mcp.config.ZOTERO_MCP_CONFIG_PATH",
        tmp_path / "nonexistent.json",
    )
    cfg = load_config()
    assert cfg == ZoteroMcpConfig()


def test_load_empty_file(tmp_path, monkeypatch):
    p = tmp_path / "config.json"
    p.write_text("{}")
    monkeypatch.setattr("zotero_mcp.config.ZOTERO_MCP_CONFIG_PATH", p)
    cfg = load_config()
    assert cfg == ZoteroMcpConfig()


def test_load_normal(tmp_path, monkeypatch):
    p = tmp_path / "config.json"
    p.write_text(
        json.dumps(
            {
                "semantic_search": {
                    "zotero_db_path": "/some/path.sqlite",
                    "extraction": {"pdf_max_pages": 50},
                }
            }
        )
    )
    monkeypatch.setattr("zotero_mcp.config.ZOTERO_MCP_CONFIG_PATH", p)
    cfg = load_config()
    assert cfg.semantic_search.zotero_db_path == "/some/path.sqlite"
    assert cfg.semantic_search.extraction.pdf_max_pages == 50


def test_load_extra_keys_ignored(tmp_path, monkeypatch):
    p = tmp_path / "config.json"
    p.write_text(
        json.dumps(
            {
                "unknown_top_key": True,
                "semantic_search": {
                    "zotero_db_path": "/db.sqlite",
                    "future_setting": [1, 2, 3],
                },
            }
        )
    )
    monkeypatch.setattr("zotero_mcp.config.ZOTERO_MCP_CONFIG_PATH", p)
    cfg = load_config()
    assert cfg.semantic_search.zotero_db_path == "/db.sqlite"
    assert not hasattr(cfg, "unknown_top_key")


def test_load_null_nested_uses_defaults(tmp_path, monkeypatch):
    p = tmp_path / "config.json"
    p.write_text(
        json.dumps(
            {
                "semantic_search": {
                    "extraction": None,
                    "reranker": None,
                }
            }
        )
    )
    monkeypatch.setattr("zotero_mcp.config.ZOTERO_MCP_CONFIG_PATH", p)
    cfg = load_config()
    assert cfg.semantic_search.extraction == ExtractionConfig()
    assert cfg.semantic_search.reranker.enabled is False


def test_load_type_coercion(tmp_path, monkeypatch):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"semantic_search": {"last_sync_version": "42"}}))
    monkeypatch.setattr("zotero_mcp.config.ZOTERO_MCP_CONFIG_PATH", p)
    cfg = load_config()
    assert cfg.semantic_search.last_sync_version == 42


def test_load_invalid_type_falls_back(tmp_path, monkeypatch):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"semantic_search": {"last_sync_version": "not_a_number"}}))
    monkeypatch.setattr("zotero_mcp.config.ZOTERO_MCP_CONFIG_PATH", p)
    cfg = load_config()
    assert cfg == ZoteroMcpConfig()


def test_load_malformed_json_falls_back(tmp_path, monkeypatch):
    p = tmp_path / "config.json"
    p.write_text("{invalid json")
    monkeypatch.setattr("zotero_mcp.config.ZOTERO_MCP_CONFIG_PATH", p)
    cfg = load_config()
    assert cfg == ZoteroMcpConfig()
