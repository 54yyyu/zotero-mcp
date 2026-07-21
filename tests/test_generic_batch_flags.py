"""Tests for the generic --batch/--batch-provider flags (Phase 4).

Covers ZoteroSemanticSearch._resolve_batch_mode's resolution matrix
(explicit provider > generic toggle > legacy per-provider flags > config),
the CLI's flag wiring/validation, and the parallelism-aware realtime slice
sizing.
"""

import json
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb relies on pydantic v1 paths incompatible with Python 3.14+",
        allow_module_level=True,
    )

pytest.importorskip("chromadb")

from types import SimpleNamespace  # noqa: E402

from zotero_mcp import semantic_search  # noqa: E402


class _FakeChroma:
    def __init__(self, embedding_model="openai"):
        self.embedding_model = embedding_model
        self.embedding_config = {"model_name": "m", "api_key": "test"}
        self.embedding_max_tokens = 8000

    def truncate_text(self, text, max_tokens=None):
        return text


def _search(monkeypatch, embedding_model="openai", config: dict | None = None, tmp_path=None):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    config_path = None
    if config is not None:
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"semantic_search": config}), encoding="utf-8")
        config_path = str(cfg)
    return semantic_search.ZoteroSemanticSearch(
        chroma_client=_FakeChroma(embedding_model), config_path=config_path
    )


# ---------------------------------------------------------------------------
# _resolve_batch_mode resolution matrix
# ---------------------------------------------------------------------------


def test_batch_provider_forces_that_provider(monkeypatch):
    search = _search(monkeypatch, "openai")
    assert search._resolve_batch_mode(batch_provider="openai") == (True, "openai")


def test_batch_provider_mismatching_model_raises(monkeypatch):
    search = _search(monkeypatch, "gemini")
    with pytest.raises(ValueError, match="requires embedding_model"):
        search._resolve_batch_mode(batch_provider="openai")


def test_unknown_batch_provider_raises(monkeypatch):
    search = _search(monkeypatch, "openai")
    with pytest.raises(ValueError, match="Unknown batch_provider"):
        search._resolve_batch_mode(batch_provider="nope")


def test_use_batch_false_overrides_explicit_provider(monkeypatch):
    # Realtime forced, but the provider choice is preserved (and no
    # mismatch validation runs — nothing will be submitted).
    search = _search(monkeypatch, "gemini")
    assert search._resolve_batch_mode(use_batch=False, batch_provider="openai") == (False, "openai")


def test_use_batch_true_infers_provider_from_model(monkeypatch):
    search = _search(monkeypatch, "gemini")
    assert search._resolve_batch_mode(use_batch=True) == (True, "gemini")


def test_use_batch_true_with_non_batch_capable_model_raises(monkeypatch):
    search = _search(monkeypatch, "default")
    with pytest.raises(ValueError, match="batch-capable"):
        search._resolve_batch_mode(use_batch=True)


def test_use_batch_false_suppresses_config_enabled_batch(monkeypatch, tmp_path):
    search = _search(
        monkeypatch, "openai", config={"openai_batch": {"enabled": True}}, tmp_path=tmp_path
    )
    enabled, _provider = search._resolve_batch_mode(use_batch=False)
    assert enabled is False


def test_legacy_openai_flag_still_works(monkeypatch):
    search = _search(monkeypatch, "openai")
    assert search._resolve_batch_mode(use_openai_batch=True) == (True, "openai")


def test_legacy_flags_ignored_when_use_batch_given(monkeypatch):
    search = _search(monkeypatch, "openai")
    enabled, _provider = search._resolve_batch_mode(use_batch=False, use_openai_batch=True)
    assert enabled is False


def test_config_driven_default_unchanged(monkeypatch, tmp_path):
    search = _search(
        monkeypatch, "openai", config={"openai_batch": {"enabled": True}}, tmp_path=tmp_path
    )
    assert search._resolve_batch_mode() == (True, "openai")


def test_nothing_requested_resolves_realtime(monkeypatch):
    search = _search(monkeypatch, "openai")
    enabled, _provider = search._resolve_batch_mode()
    assert enabled is False


# ---------------------------------------------------------------------------
# CLI flag wiring
# ---------------------------------------------------------------------------


def _run_cli(args):
    import subprocess

    return subprocess.run(
        [sys.executable, "-m", "zotero_mcp.cli", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_cli_help_shows_generic_batch_flags():
    result = _run_cli(["update-db", "--help"])
    assert result.returncode == 0
    assert "--batch" in result.stdout
    assert "--no-batch" in result.stdout
    assert "--batch-provider" in result.stdout
    assert "openai" in result.stdout and "gemini" in result.stdout


def test_cli_batch_and_no_batch_mutually_exclusive():
    result = _run_cli(["update-db", "--batch", "--no-batch"])
    assert result.returncode != 0
    assert "not allowed with" in result.stderr


def test_cli_batch_conflicts_with_legacy_flags():
    result = _run_cli(["update-db", "--batch", "--openai-batch"])
    assert result.returncode != 0
    assert "cannot be combined" in result.stderr


def test_cli_batch_provider_rejects_unknown():
    result = _run_cli(["update-db", "--batch", "--batch-provider", "nope"])
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


# ---------------------------------------------------------------------------
# Realtime slice sizing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parallel", "expected"),
    [(None, 25), (1, 25), (4, 100), (8, 200), (16, 200)],
)
def test_realtime_slice_scales_with_parallelism_capped(parallel, expected):
    ef = SimpleNamespace(max_parallel_requests=parallel)
    assert semantic_search._realtime_slice_size(ef) == expected


def test_realtime_slice_defaults_without_attribute():
    assert semantic_search._realtime_slice_size(object()) == 25
    assert semantic_search._realtime_slice_size(None) == 25
