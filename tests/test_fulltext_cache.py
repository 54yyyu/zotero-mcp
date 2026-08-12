"""Tests for the transient extracted-fulltext cache."""

import json

import pytest

from zotero_mcp import fulltext_cache


@pytest.fixture
def cfg(tmp_path):
    """A config path whose sibling fulltext_cache/ dir is this test's own."""
    return str(tmp_path / "config.json")


def _put(cfg, key="ATT1", mtime=111, size=22, text="hello", item_key="ITEM1", **kw):
    fulltext_cache.put_cached_text(
        key, mtime, size, source=kw.pop("source", "pdf"), item_key=item_key,
        text=text, config_path=cfg, **kw,
    )


def test_miss_before_anything_is_cached(cfg):
    assert fulltext_cache.get_cached_text("ATT1", 111, 22, config_path=cfg) is None


def test_round_trip_returns_text_and_source(cfg):
    _put(cfg, text="extracted body", source="pdf")
    assert fulltext_cache.get_cached_text("ATT1", 111, 22, config_path=cfg) == (
        "extracted body",
        "pdf",
    )


def test_changed_mtime_or_size_misses(cfg):
    """A replaced attachment must never serve the previous file's text."""
    _put(cfg)
    assert fulltext_cache.get_cached_text("ATT1", 999, 22, config_path=cfg) is None
    assert fulltext_cache.get_cached_text("ATT1", 111, 999, config_path=cfg) is None


def test_profile_mismatch_misses(cfg):
    """Lowering pdf_max_pages must not serve text extracted under a higher cap."""
    _put(cfg, profile="maxpages=50")
    assert fulltext_cache.get_cached_text(
        "ATT1", 111, 22, profile="maxpages=10", config_path=cfg
    ) is None
    assert fulltext_cache.get_cached_text(
        "ATT1", 111, 22, profile="maxpages=50", config_path=cfg
    ) == ("hello", "pdf")


def test_superseded_text_file_is_removed(cfg):
    """Re-caching the same attachment must not leave the old .txt behind."""
    _put(cfg, mtime=111, text="old")
    _put(cfg, mtime=222, text="new")
    root = fulltext_cache.get_fulltext_cache_root(cfg)
    assert len(list(root.glob("*.txt"))) == 1
    assert fulltext_cache.get_cached_text("ATT1", 222, 22, config_path=cfg)[0] == "new"


def test_eviction_removes_every_entry_for_an_item(cfg):
    """Eviction is by item key — an item may own several attachments."""
    _put(cfg, key="ATT1", item_key="ITEM1")
    _put(cfg, key="ATT2", item_key="ITEM1")
    _put(cfg, key="ATT3", item_key="OTHER")
    assert fulltext_cache.evict("ITEM1", config_path=cfg) == 2
    assert fulltext_cache.get_cached_text("ATT1", 111, 22, config_path=cfg) is None
    assert fulltext_cache.get_cached_text("ATT3", 111, 22, config_path=cfg) is not None


def test_evict_many_ignores_unknown_and_empty_keys(cfg):
    _put(cfg, item_key="ITEM1")
    assert fulltext_cache.evict_many([], config_path=cfg) == 0
    assert fulltext_cache.evict_many(["", None], config_path=cfg) == 0
    assert fulltext_cache.evict_many(["NOPE"], config_path=cfg) == 0
    assert fulltext_cache.get_cached_text("ATT1", 111, 22, config_path=cfg) is not None


def test_purge_drops_entries_whose_file_is_gone(cfg, tmp_path):
    backing = tmp_path / "present.pdf"
    backing.write_bytes(b"%PDF-1.4")
    _put(cfg, key="KEEP", path=str(backing))
    _put(cfg, key="GONE", path=str(tmp_path / "missing.pdf"))
    assert fulltext_cache.purge_stale(config_path=cfg) == 1
    assert fulltext_cache.get_cached_text("KEEP", 111, 22, config_path=cfg) is not None
    assert fulltext_cache.get_cached_text("GONE", 111, 22, config_path=cfg) is None


def test_purge_sweeps_orphaned_text_files(cfg):
    """A .txt with no index entry is residue from an interrupted run."""
    root = fulltext_cache.get_fulltext_cache_root(cfg)
    (root / "deadbeef.txt").write_text("orphan", encoding="utf-8")
    _put(cfg)
    fulltext_cache.purge_stale(config_path=cfg)
    assert not (root / "deadbeef.txt").exists()
    assert fulltext_cache.get_cached_text("ATT1", 111, 22, config_path=cfg) is not None


def test_clear_all_empties_the_cache(cfg):
    _put(cfg, key="ATT1")
    _put(cfg, key="ATT2")
    assert fulltext_cache.clear_all(config_path=cfg) == 2
    root = fulltext_cache.get_fulltext_cache_root(cfg)
    assert list(root.glob("*.txt")) == []
    assert fulltext_cache.get_cached_text("ATT1", 111, 22, config_path=cfg) is None


def test_corrupt_index_is_survivable(cfg):
    """A truncated index.json must degrade to a miss, not raise."""
    _put(cfg)
    root = fulltext_cache.get_fulltext_cache_root(cfg)
    (root / "index.json").write_text("{not json", encoding="utf-8")
    assert fulltext_cache.get_cached_text("ATT1", 111, 22, config_path=cfg) is None
    _put(cfg, text="recovered")
    assert fulltext_cache.get_cached_text("ATT1", 111, 22, config_path=cfg)[0] == "recovered"


def test_cache_root_is_private(cfg):
    """Cached text is library content and must not be world-readable."""
    root = fulltext_cache.get_fulltext_cache_root(cfg)
    assert (root.stat().st_mode & 0o077) == 0


def test_index_records_item_key_for_eviction(cfg):
    _put(cfg, item_key="ITEM42")
    root = fulltext_cache.get_fulltext_cache_root(cfg)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    assert index["entries"]["ATT1"]["item_key"] == "ITEM42"
