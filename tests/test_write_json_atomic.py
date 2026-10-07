"""utils.write_json_atomic: the one atomic JSON writer (config.json, manifests)."""

import json
import sys

import pytest

from zotero_mcp.utils import write_json_atomic


def test_round_trip_leaves_no_temp_file(tmp_path):
    path = tmp_path / "sub" / "data.json"
    write_json_atomic(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert [p.name for p in path.parent.iterdir()] == ["data.json"]


def test_replaces_an_existing_file(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"old": true}')
    write_json_atomic(path, {"new": True})
    assert json.loads(path.read_text()) == {"new": True}


def test_failed_write_keeps_the_old_file_and_cleans_up(tmp_path):
    path = tmp_path / "data.json"
    path.write_text('{"keep": 1}')
    with pytest.raises(TypeError):
        write_json_atomic(path, {"bad": object()})
    assert json.loads(path.read_text()) == {"keep": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["data.json"]


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra privileges on Windows")
def test_symlink_is_written_through(tmp_path):
    real = tmp_path / "dotfiles" / "data.json"
    real.parent.mkdir()
    real.write_text("{}")
    link = tmp_path / "home" / "data.json"
    link.parent.mkdir()
    link.symlink_to(real)
    write_json_atomic(link, {"x": 2})
    assert link.is_symlink()
    assert json.loads(real.read_text()) == {"x": 2}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes do not exist on Windows")
def test_file_is_owner_only(tmp_path):
    path = tmp_path / "data.json"
    write_json_atomic(path, {"secret": "k"})
    assert path.stat().st_mode & 0o077 == 0
