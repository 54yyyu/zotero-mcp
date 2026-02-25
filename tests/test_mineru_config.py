from zotero_mcp.semantic_search import ZoteroSemanticSearch


def _resolve(cfg):
    search = ZoteroSemanticSearch.__new__(ZoteroSemanticSearch)
    return search._resolve_mineru_config(cfg)  # type: ignore[misc]


def test_mineru_local_provider_defaults_backend_to_vlm():
    resolved = _resolve(
        {
            "provider": "local_api",
            "enabled": True,
            "local_api": {
                "enabled": True,
                "base_url": "http://localhost:8000",
            },
        }
    )
    assert resolved["provider"] == "local_api"
    assert resolved["local_api"]["backend"] == "vlm"
    assert resolved["fallback_providers"] == ["local_api", "official_upload_batch"]


def test_mineru_legacy_official_config_stays_official_first():
    resolved = _resolve({"enabled": True, "tokens": ["tok"]})
    assert resolved["provider"] == "official_upload_batch"
    assert resolved["fallback_providers"] == ["official_upload_batch"]
    assert resolved["enabled"] is True
