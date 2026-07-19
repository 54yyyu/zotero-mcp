"""Tests for the transient extracted-fulltext cache (zotero_mcp.fulltext_cache)."""

import json
from concurrent.futures import ThreadPoolExecutor

from zotero_mcp import fulltext_cache


def _config_path(tmp_path):
    # The cache root is derived from the config file's parent directory.
    return str(tmp_path / "config.json")


def test_put_get_roundtrip(tmp_path):
    cfg = _config_path(tmp_path)
    fulltext_cache.put_cached_text(
        "ATTKEY01", 111, 222, source="pdf", item_key="ITEM0001",
        text="hello world", path="/nonexistent.pdf", profile="maxpages=10", config_path=cfg,
    )
    hit = fulltext_cache.get_cached_text("ATTKEY01", 111, 222, profile="maxpages=10", config_path=cfg)
    assert hit == ("hello world", "pdf")


def test_changed_mtime_or_size_misses(tmp_path):
    cfg = _config_path(tmp_path)
    fulltext_cache.put_cached_text(
        "ATTKEY01", 111, 222, source="pdf", item_key="ITEM0001", text="x", config_path=cfg,
    )
    assert fulltext_cache.get_cached_text("ATTKEY01", 999, 222, config_path=cfg) is None
    assert fulltext_cache.get_cached_text("ATTKEY01", 111, 999, config_path=cfg) is None
    assert fulltext_cache.get_cached_text("OTHERKEY", 111, 222, config_path=cfg) is None


def test_profile_mismatch_misses(tmp_path):
    cfg = _config_path(tmp_path)
    fulltext_cache.put_cached_text(
        "ATTKEY01", 111, 222, source="pdf", item_key="ITEM0001",
        text="x", profile="maxpages=10", config_path=cfg,
    )
    assert fulltext_cache.get_cached_text(
        "ATTKEY01", 111, 222, profile="maxpages=50", config_path=cfg
    ) is None


def test_web_roundtrip_and_version_miss(tmp_path):
    cfg = _config_path(tmp_path)
    fulltext_cache.put_cached_web_text("ITEM0001", 42, "web-api:parent", "web text", config_path=cfg)
    assert fulltext_cache.get_cached_web_text("ITEM0001", 42, config_path=cfg) == (
        "web text", "web-api:parent",
    )
    assert fulltext_cache.get_cached_web_text("ITEM0001", 43, config_path=cfg) is None


def test_evict_removes_file_and_index_entry(tmp_path):
    cfg = _config_path(tmp_path)
    fulltext_cache.put_cached_text(
        "ATTKEY01", 111, 222, source="pdf", item_key="ITEM0001", text="x", config_path=cfg,
    )
    fulltext_cache.put_cached_web_text("ITEM0001", 42, "web-api:parent", "y", config_path=cfg)
    fulltext_cache.put_cached_text(
        "ATTKEY02", 111, 222, source="pdf", item_key="ITEM0002", text="z", config_path=cfg,
    )

    removed = fulltext_cache.evict_many(["ITEM0001"], config_path=cfg)
    assert removed == 2  # local + web entries for ITEM0001

    root = fulltext_cache.get_fulltext_cache_root(cfg)
    index = json.loads((root / "index.json").read_text())
    assert set(index["entries"]) == {"ATTKEY02"}
    # Only the surviving entry's text file remains
    assert len(list(root.glob("*.txt"))) == 1
    assert fulltext_cache.get_cached_text("ATTKEY02", 111, 222, config_path=cfg) == ("z", "pdf")


def test_concurrent_puts_keep_index_intact(tmp_path):
    cfg = _config_path(tmp_path)

    def put(i):
        fulltext_cache.put_cached_text(
            f"ATT{i:05d}", i, i, source="pdf", item_key=f"ITEM{i:05d}",
            text=f"text {i}", config_path=cfg,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(put, range(60)))

    root = fulltext_cache.get_fulltext_cache_root(cfg)
    index = json.loads((root / "index.json").read_text())
    assert len(index["entries"]) == 60
    for i in (0, 30, 59):
        assert fulltext_cache.get_cached_text(f"ATT{i:05d}", i, i, config_path=cfg) == (
            f"text {i}", "pdf",
        )


def test_purge_stale_removes_missing_backing_file(tmp_path):
    cfg = _config_path(tmp_path)
    real_pdf = tmp_path / "real.pdf"
    real_pdf.write_bytes(b"%PDF")
    fulltext_cache.put_cached_text(
        "ATTGONE1", 1, 1, source="pdf", item_key="ITEMGONE",
        text="a", path=str(tmp_path / "missing.pdf"), config_path=cfg,
    )
    fulltext_cache.put_cached_text(
        "ATTHERE1", 1, 1, source="pdf", item_key="ITEMHERE",
        text="b", path=str(real_pdf), config_path=cfg,
    )
    removed = fulltext_cache.purge_stale(max_age_days=30, config_path=cfg)
    assert removed == 1
    assert fulltext_cache.get_cached_text("ATTGONE1", 1, 1, config_path=cfg) is None
    assert fulltext_cache.get_cached_text("ATTHERE1", 1, 1, config_path=cfg) == ("b", "pdf")


def test_purge_stale_respects_max_age(tmp_path):
    cfg = _config_path(tmp_path)
    fulltext_cache.put_cached_text(
        "ATTOLD01", 1, 1, source="pdf", item_key="ITEMOLD", text="old", config_path=cfg,
    )
    # Rewrite the entry timestamp far in the past.
    root = fulltext_cache.get_fulltext_cache_root(cfg)
    index = json.loads((root / "index.json").read_text())
    index["entries"]["ATTOLD01"]["cached_at"] = "2000-01-01T00:00:00Z"
    (root / "index.json").write_text(json.dumps(index))

    assert fulltext_cache.purge_stale(max_age_days=30, config_path=cfg) == 1
    assert fulltext_cache.get_cached_text("ATTOLD01", 1, 1, config_path=cfg) is None


def test_clear_all(tmp_path):
    cfg = _config_path(tmp_path)
    for i in range(3):
        fulltext_cache.put_cached_text(
            f"ATT{i}", i, i, source="pdf", item_key=f"ITEM{i}", text="t", config_path=cfg,
        )
    assert fulltext_cache.clear_all(config_path=cfg) == 3
    root = fulltext_cache.get_fulltext_cache_root(cfg)
    assert not (root / "index.json").exists()
    assert list(root.glob("*.txt")) == []


def test_corrupt_index_recovers(tmp_path):
    cfg = _config_path(tmp_path)
    root = fulltext_cache.get_fulltext_cache_root(cfg)
    (root / "index.json").write_text("{not json")
    assert fulltext_cache.get_cached_text("ATTKEY01", 1, 1, config_path=cfg) is None
    fulltext_cache.put_cached_text(
        "ATTKEY01", 1, 1, source="pdf", item_key="ITEM0001", text="ok", config_path=cfg,
    )
    assert fulltext_cache.get_cached_text("ATTKEY01", 1, 1, config_path=cfg) == ("ok", "pdf")
