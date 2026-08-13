"""Tests for the transient extracted-fulltext cache."""

import json
import os

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


def _backdate(cfg, key, cached_at):
    """Rewrite one entry's timestamp — the only way to age a cache on demand."""
    root = fulltext_cache.get_fulltext_cache_root(cfg)
    index_path = root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["entries"][key]["cached_at"] = cached_at
    index_path.write_text(json.dumps(index), encoding="utf-8")


def test_purge_drops_entries_older_than_max_age(cfg):
    """An abandoned run must not leave its extracted text on disk forever.

    This is the branch that makes the cache genuinely transient on the batch
    flow, where per-item eviction only happens at import time and a run that
    is never imported would otherwise leave everything behind.
    """
    _put(cfg, key="OLD")
    _put(cfg, key="NEW")
    _backdate(cfg, "OLD", "2020-01-01T00:00:00Z")

    assert fulltext_cache.purge_stale(max_age_days=30, config_path=cfg) == 1
    assert fulltext_cache.get_cached_text("OLD", 111, 22, config_path=cfg) is None
    assert fulltext_cache.get_cached_text("NEW", 111, 22, config_path=cfg) is not None
    # The text file goes with the entry, not just the index row.
    root = fulltext_cache.get_fulltext_cache_root(cfg)
    assert len(list(root.glob("*.txt"))) == 1


def test_purge_keeps_entries_younger_than_max_age(cfg):
    """The age threshold is honoured, not ignored in favour of purging all."""
    _put(cfg, key="RECENT")
    _backdate(cfg, "RECENT", "2020-01-01T00:00:00Z")
    assert fulltext_cache.purge_stale(max_age_days=99999, config_path=cfg) == 0
    assert fulltext_cache.get_cached_text("RECENT", 111, 22, config_path=cfg) is not None


def test_purge_drops_entries_with_an_unreadable_timestamp(cfg):
    """A truncated or hand-edited entry cannot be aged, so it is not trusted."""
    _put(cfg, key="BROKEN")
    _backdate(cfg, "BROKEN", "not a timestamp")
    assert fulltext_cache.purge_stale(config_path=cfg) == 1
    assert fulltext_cache.get_cached_text("BROKEN", 111, 22, config_path=cfg) is None


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
    """Cached text is library content and must not be world-readable.

    POSIX-only: Windows implements no mode bits beyond the read-only flag, so
    ``os.chmod(path, 0o700)`` succeeds there but ``st_mode`` still reports
    0o777 for the directory. The hardening itself is best-effort on every
    platform (``_private_chmod`` swallows OSError); only the assertion is
    POSIX-specific.
    """
    if os.name != "posix":
        pytest.skip("POSIX file permissions only")
    root = fulltext_cache.get_fulltext_cache_root(cfg)
    assert (root.stat().st_mode & 0o077) == 0


def test_index_records_item_key_for_eviction(cfg):
    _put(cfg, item_key="ITEM42")
    root = fulltext_cache.get_fulltext_cache_root(cfg)
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    assert index["entries"]["ATT1"]["item_key"] == "ITEM42"


# ---------------------------------------------------------------------------
# update-db wiring — purge_stale is only useful if something calls it
# ---------------------------------------------------------------------------

def _run_update_db(monkeypatch, cfg, stats, extra_argv=()):
    """Drive the real ``update-db`` handler with the database work stubbed."""
    import sys

    from zotero_mcp import cli, semantic_search

    class FakeSearch:
        chroma_client = None

        def update_database(self, **kwargs):
            return stats

    monkeypatch.setattr(cli, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(
        semantic_search, "create_semantic_search", lambda *a, **kw: FakeSearch()
    )
    monkeypatch.setattr(
        sys, "argv", ["zotero-mcp", "update-db", "--config-path", cfg, *extra_argv]
    )
    cli.main()


def test_update_db_purges_stale_cache_entries(cfg, monkeypatch, capsys):
    """The gap this closes: nothing shipped ever called ``purge_stale``."""
    _put(cfg, key="OLD")
    _put(cfg, key="FRESH")
    _backdate(cfg, "OLD", "2020-01-01T00:00:00Z")

    _run_update_db(monkeypatch, cfg, {"total_items": 1, "processed_items": 1})

    assert fulltext_cache.get_cached_text("OLD", 111, 22, config_path=cfg) is None
    assert fulltext_cache.get_cached_text("FRESH", 111, 22, config_path=cfg) is not None
    assert "Purged 1 stale fulltext cache entries" in capsys.readouterr().out


def test_update_db_stays_silent_when_nothing_is_stale(cfg, monkeypatch, capsys):
    """A no-op sweep must not add noise to every single run."""
    _put(cfg, key="FRESH")

    _run_update_db(monkeypatch, cfg, {"total_items": 1, "processed_items": 1})

    assert "Purged" not in capsys.readouterr().out
    assert fulltext_cache.get_cached_text("FRESH", 111, 22, config_path=cfg) is not None


def test_update_db_does_not_purge_after_a_failed_run(cfg, monkeypatch):
    """A run that errored may be resumed, so its cached text has to survive.

    Purging here would delete exactly the extraction work the cache exists to
    protect — the failure modes it was built for (rate limits, Ctrl-C) all
    surface as an errored run.
    """
    _put(cfg, key="OLD")
    _backdate(cfg, "OLD", "2020-01-01T00:00:00Z")

    with pytest.raises(SystemExit):
        _run_update_db(monkeypatch, cfg, {"error": "rate limited"})

    assert fulltext_cache.get_cached_text("OLD", 111, 22, config_path=cfg) is not None


def test_update_db_survives_a_broken_cache(cfg, monkeypatch, capsys):
    """The update already succeeded by then; cleanup must never fail the run."""
    def boom(**kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(fulltext_cache, "purge_stale", boom)
    _run_update_db(monkeypatch, cfg, {"total_items": 1, "processed_items": 1})

    out = capsys.readouterr().out
    assert "could not purge stale fulltext cache entries" in out
    assert "Database update completed" in out
