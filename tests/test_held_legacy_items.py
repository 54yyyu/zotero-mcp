"""A legacy item held for an unreachable file is paid for once, not every run (#826, fix round 2).

An edited legacy item whose only file is out of reach is rewritten (its
bibliographic record), its passages kept as the only copy of the text. Its
format is decided by its bare record alone, so the kept passages, which still
carry ``title``, do not make later runs see it as a changed legacy item. The
passages stay until the file is reachable and written, then go. Real Chroma in
a temp dir, a recording embedder, a fake reader; HOME, Path.home and the
config path point into tmp_path.
"""

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 14):
    pytest.skip("chromadb needs pydantic v1 paths on 3.14+", allow_module_level=True)

from _scan_fakes import target  # noqa: E402
from test_item_keyed_local_scan import Item  # noqa: E402
from test_rekeyed_index_updates import ATT, ITEM, NAME, PASSAGES, Reader, World  # noqa: E402

from zotero_mcp import semantic_search  # noqa: E402

EDITED = "2025-05-05 00:00:00"
ROWS = {1: [[ATT, "storage:a.pdf", "application/pdf"]]}


@pytest.fixture()
def world(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)
    return World(monkeypatch, tmp_path)


def _unreachable():
    return Reader([Item(ITEM, 1, date_modified=EDITED)], targets={}, rows=ROWS)


def _reachable():
    return Reader([Item(ITEM, 1, date_modified=EDITED)], targets={1: [target(ATT)]}, rows=ROWS)


def _all_ids(world):
    # An explicit run splits the index first, so passages may be in either collection.
    found = []
    for name in (NAME, PASSAGES):
        try:
            found += list(world.stored(name))
        except Exception:  # the passage collection exists only once split
            pass
    return sorted(found)


def _legacy_passages(world):
    return [k for k in _all_ids(world) if k.startswith(f"{ITEM}#")]


@pytest.mark.parametrize("automatic", [True, False])
def test_held_legacy_item_is_paid_for_once(world, automatic):
    reader = _unreachable()
    world.seed_pre_0_17(bare=True)
    counts = []
    for _ in range(3):
        world.embedded.clear()
        world.update(reader, automatic=automatic)
        counts.append(len(world.embedded))
    assert counts[1:] == [0, 0], counts
    assert len(_legacy_passages(world)) == 3, "the kept passages are still the only copy of the text"


@pytest.mark.parametrize("automatic", [True, False])
def test_held_legacy_item_in_a_split_index_is_paid_for_once(world, automatic):
    reader = _unreachable()
    world.seed_pre_0_17(bare=True)
    world.migrate(reader)
    counts = []
    for _ in range(3):
        world.embedded.clear()
        world.update(reader, automatic=automatic)
        counts.append(len(world.embedded))
    assert counts[1:] == [0, 0], counts


@pytest.mark.parametrize("automatic", [True, False])
def test_kept_passages_go_once_the_file_is_reachable_and_written(world, automatic):
    world.seed_pre_0_17(bare=True)
    world.update(_unreachable(), automatic=automatic)
    world.update(_unreachable(), automatic=automatic)
    assert len(_legacy_passages(world)) == 3

    world.update(_reachable(), automatic=automatic)

    assert ATT in world.stored(NAME)
    assert f"{ATT}#0" in _all_ids(world)
    assert _legacy_passages(world) == [], "the legacy passages go after the file's own are written"


# --- an embed failure of the attachment after the bibliographic record landed (fix round 2, issue 1) ---


def _fail_file_text(world, monkeypatch):
    """The embedder rejects anything that is file text (the item's own record embeds fine)."""
    from chromadb.api.models.CollectionCommon import CollectionCommon

    def embed(collection, input, **kwargs):
        if any("Body sentence" in text for text in input):
            raise RuntimeError("provider is down")
        world.embedded.extend(input)
        return [[1.0, 0.0] for _ in input]

    monkeypatch.setattr(CollectionCommon, "_embed", embed)
    monkeypatch.setattr("time.sleep", lambda _s: None)  # the end-of-run retry pauses
    return CollectionCommon


@pytest.mark.parametrize("automatic", [True, False])
def test_a_failed_attachment_embed_keeps_the_legacy_passages_until_a_later_run_writes_it(world, monkeypatch, automatic):
    world.seed_pre_0_17(bare=True)
    from chromadb.api.models.CollectionCommon import CollectionCommon

    real = CollectionCommon._embed
    _fail_file_text(world, monkeypatch)

    world.update(_reachable(), automatic=automatic)

    stored = world.stored(NAME)
    assert "title" not in stored[ITEM][0], "the bibliographic record landed in the new shape"
    assert ATT not in stored, "the file's own record did not"
    if automatic:
        assert len(_legacy_passages(world)) == 3, "so the old passages are still the only copy of its text"
    else:
        # An explicit run re-keys the paid passages to the file first, no embedding needed.
        assert f"{ATT}#0" in _all_ids(world)

    monkeypatch.setattr(CollectionCommon, "_embed", real)
    world.update(_reachable(), automatic=automatic)

    assert ATT in world.stored(NAME)
    assert _legacy_passages(world) == [], "dropped once the new passages have landed"
    assert f"{ATT}#0" in _all_ids(world)


def test_an_untouched_legacy_item_is_not_registered_for_settling(world, monkeypatch):
    """Settling reads the index per registered item; items that keep their legacy records need none."""
    registered = []
    original = semantic_search.ZoteroSemanticSearch._expect_replacement
    monkeypatch.setattr(
        semantic_search.ZoteroSemanticSearch,
        "_expect_replacement",
        lambda self, key, group, wanted: registered.append(key) or original(self, key, group, wanted),
    )
    world.seed_pre_0_17(bare=True)
    unchanged = Reader([Item(ITEM, 1)], targets={1: [target(ATT)]}, rows=ROWS)

    world.update(unchanged, automatic=True)
    assert registered == []

    world.update(_reachable(), automatic=True)  # edited: rewritten, so registered
    assert registered == [ITEM]
