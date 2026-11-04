"""The settings block shared by setup, db-status and setup-info."""

from zotero_mcp.semantic_summary import semantic_config_rows


def _rows(config, **kwargs):
    return dict(semantic_config_rows(config, **kwargs))


def test_rows_never_contain_the_api_key():
    rows = semantic_config_rows(
        {"embedding_model": "openai", "embedding_config": {"api_key": "sk-secret"}}
    )
    assert "sk-secret" not in repr(rows)


def test_rows_for_an_empty_config_use_the_defaults():
    rows = _rows({})
    assert rows["Update frequency"] == "manual"
    assert rows["Full-text indexing"] == "enabled"
    assert rows["Passage chunking"] == "disabled (item-level indexing)"
    assert rows["Reranker"] == "disabled"
    assert rows["Zotero database path"] == "auto-detect"


def test_batch_row_only_for_providers_that_have_one():
    assert _rows({"embedding_model": "openai"})["OpenAI Batch API"] == "disabled"
    assert _rows(
        {"embedding_model": "gemini", "gemini_batch": {"enabled": True}}
    )["Gemini Batch API"] == "enabled"
    assert not any("Batch" in label for label in _rows({"embedding_model": "ollama"}))


def test_chunking_row_shows_the_sizes_when_enabled():
    rows = _rows({"chunking": {"enabled": True, "chunk_size": 800, "overlap": 50,
                               "max_chunks_per_item": 60}})
    assert rows["Passage chunking"] == "enabled (800 chars, overlap 50, up to 60 passages per item)"


def test_chunking_row_says_when_the_batch_api_overrides_it():
    config = {"embedding_model": "openai", "openai_batch": {"enabled": True},
              "chunking": {"enabled": True}}
    assert "NOT applied" in _rows(config)["Passage chunking"]


def test_runtime_batch_state_overrides_the_config():
    config = {"embedding_model": "openai", "openai_batch": {"enabled": True},
              "chunking": {"enabled": True}}
    rows = _rows(config, batch_active=False)
    assert rows["OpenAI Batch API"] == "disabled"
    assert rows["Passage chunking"].startswith("enabled")


def test_short_model_limit_is_reported_when_chunking_is_off():
    assert "first ~256 tokens" in _rows({}, max_tokens=256)["Passage chunking"]


def test_auto_update_off_is_visible():
    rows = _rows({"update_config": {"update_frequency": "daily", "auto_update": False}})
    assert rows["Update frequency"] == "daily (auto update off)"
