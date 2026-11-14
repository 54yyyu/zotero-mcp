"""A file seen through Zotero's text cache is the same file (#826, fix round 2, issue 3).

The cache is a fallback for an attachment whose file cannot be read (unmounted
drive, kept on demand). Its stamp is its own kind (``ftcache:``), so switching
between the file and its cache re-embeds nothing, a migration done while the
file was missing keeps its passages when the file returns, only a change of the
attachment's own ``date_modified`` is a real change, and a cache-derived record
keeps no file hash and says when Zotero's text is truncated. Real Chroma in a
temp dir, a recording embedder, a fake reader; HOME, Path.home and the config
path point into tmp_path.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 14):
    pytest.skip("chromadb needs pydantic v1 paths on 3.14+", allow_module_level=True)

from _scan_fakes import target  # noqa: E402
from test_item_keyed_local_scan import Item  # noqa: E402
from test_rekeyed_index_updates import ATT, ITEM, NAME, PASSAGES, Reader, World  # noqa: E402

from zotero_mcp import semantic_search  # noqa: E402
from zotero_mcp.chroma_client import level_status_rows, read_index_layout  # noqa: E402
from zotero_mcp.local_db import ExtractionTarget, LocalZoteroReader  # noqa: E402

DATE = "2024-01-01 00:00:00"
REPLACED = "2025-06-06 00:00:00"


@pytest.fixture()
def world(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    world = World(monkeypatch, tmp_path)
    world.tmp = tmp_path
    return world


def _file():
    return target(ATT, stamp="111:100")


def _cache(world, stamp="ftcache:222:50"):
    cache = world.tmp / "storage" / ATT / ".zotero-ft-cache"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("cached text of the file")
    return ExtractionTarget(
        key=ATT,
        path=cache,
        content_type="application/pdf",
        category="pdf",
        file_stamp=stamp,
        size=50,
        ft_cache=True,
    )


class CacheReader(Reader):
    truncated: frozenset = frozenset()

    def get_truncated_fulltext_keys(self, keys):
        return set(keys) & set(self.truncated)


def _reader(found, date=DATE, **kwargs):
    return CacheReader(
        [Item(ITEM, 1)],
        targets={1: [found]},
        dates={ATT: date},
        rows={1: [[ATT, "storage:a.pdf", "application/pdf"]]},
        **kwargs,
    )


def _embeds(world, reader, **kwargs):
    world.embedded.clear()
    world.update(reader, **kwargs)
    return len(world.embedded)


@pytest.mark.parametrize("automatic", [True, False])
def test_mounting_and_unmounting_the_file_re_embeds_nothing(world, automatic):
    assert _embeds(world, _reader(_file()), automatic=automatic) > 0
    stored = world.stored(NAME)[ATT][0]
    assert stored["file_stamp"] == "111:100"

    assert _embeds(world, _reader(_cache(world)), automatic=automatic) == 0, "file -> cache"
    assert _embeds(world, _reader(_file()), automatic=automatic) == 0, "cache -> file"
    assert _embeds(world, _reader(_cache(world, "ftcache:333:60")), automatic=automatic) == 0, "another cache"


def test_a_replaced_file_is_still_a_real_change(world):
    _embeds(world, _reader(_file()))

    assert _embeds(world, _reader(_cache(world), date=REPLACED)) > 0


def test_a_record_made_from_the_cache_has_no_file_hash(world):
    _embeds(world, _reader(_cache(world)))

    stored = world.stored(NAME)[ATT][0]
    assert stored["file_stamp"].startswith("ftcache:")
    assert stored.get("file_hash") is None
    assert not stored.get("text_truncated")


def test_a_truncated_cache_is_flagged_and_counted(world):
    reader = _reader(_cache(world))
    reader.truncated = frozenset({ATT})

    _embeds(world, reader)

    assert world.stored(NAME)[ATT][0]["text_truncated"] is True
    layout = read_index_layout(world.directory, NAME)
    assert layout["truncated_text_records"] == 1
    rows = dict(level_status_rows(layout))
    assert "update-db --force-rebuild" in rows["Records from a cut-short text cache"]


def test_a_migration_done_while_the_file_was_missing_survives_its_return(world):
    world.seed_pre_0_17()
    away = _reader(_cache(world))
    world.migrate(away)
    before = {k: v for k, (_m, v) in world.stored(PASSAGES).items()}
    assert set(before) == {f"{ATT}#{n}" for n in range(3)}

    # The file is back: nothing about it is a change.
    world.update(_reader(_file()), automatic=True)
    assert {k: v for k, (_m, v) in world.stored(PASSAGES).items()} == before

    world.update(_reader(_file()))
    assert {k: v for k, (_m, v) in world.stored(PASSAGES).items()} == before, "an explicit run too"
    assert ATT in world.stored(NAME)


def test_the_reader_names_attachments_zotero_indexed_only_in_part(tmp_path):
    db = tmp_path / "zotero.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE items (itemID INTEGER PRIMARY KEY, key TEXT);
        CREATE TABLE fulltextItems (
            itemID INTEGER PRIMARY KEY, indexedPages INT, totalPages INT, indexedChars INT, totalChars INT);
        INSERT INTO items VALUES (1, 'WHOLEPDF'), (2, 'CUTPAGES'), (3, 'CUTCHARS'), (4, 'NOTAGAIN');
        INSERT INTO fulltextItems VALUES (1, 10, 10, NULL, NULL), (2, 5, 10, NULL, NULL), (3, NULL, NULL, 90, 100);
        """
    )
    conn.commit()
    conn.close()

    with LocalZoteroReader(db_path=str(db)) as reader:
        found = reader.get_truncated_fulltext_keys(["WHOLEPDF", "CUTPAGES", "CUTCHARS", "NOTAGAIN"])

    assert found == {"CUTPAGES", "CUTCHARS"}
