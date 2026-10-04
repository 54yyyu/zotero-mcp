"""A failed manifest save must leave the previous manifest readable.

The manifest is the only local record of a paid provider batch. Written in
place, a crash or full disk mid-write left truncated JSON that
``_sorted_manifests`` skips as unreadable, so the run vanished from status
and import, and an older run became "newest".
"""

import json

import pytest

from zotero_mcp import batch_common


def _manifest(tmp_path, **extra):
    path = tmp_path / "run1" / "manifest.json"
    return {"manifest_path": str(path), "run_id": "run1",
            "created_at": "2026-10-04T00:00:00Z", **extra}


def test_failed_save_keeps_the_previous_manifest(tmp_path, monkeypatch):
    m = _manifest(tmp_path, batches=[{"id": "batch_paid_1"}])
    batch_common.save_manifest(m)

    real_dump = json.dump

    def torn_dump(obj, f, **kw):
        f.write('{"run_id": "run1", "batc')
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(batch_common.json, "dump", torn_dump)
    with pytest.raises(OSError):
        batch_common.save_manifest({**m, "status": "refreshed"})
    monkeypatch.setattr(batch_common.json, "dump", real_dump)

    loaded = batch_common.load_manifest(tmp_path / "run1" / "manifest.json")
    assert loaded["batches"] == [{"id": "batch_paid_1"}]
    assert batch_common.newest_run_path(tmp_path) is not None
    assert [p.name for p in (tmp_path / "run1").iterdir()] == ["manifest.json"]


def test_save_round_trips_and_is_owner_only(tmp_path):
    m = _manifest(tmp_path, batches=[])
    batch_common.save_manifest(m)
    path = tmp_path / "run1" / "manifest.json"
    assert batch_common.load_manifest(path)["run_id"] == "run1"
    assert path.stat().st_mode & 0o077 == 0
