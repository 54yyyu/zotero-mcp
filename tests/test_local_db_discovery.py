"""Tests for Zotero database auto-discovery in LocalZoteroReader."""

from pathlib import Path

import pytest
from conftest import skip_on_ci

import zotero_mcp.local_db as local_db
from zotero_mcp.local_db import LocalZoteroReader, _read_string_pref


def _make_profile(tmp_path: Path, prefs: dict[str, str]) -> Path:
    """Create a fake Zotero profiles dir with one profile's prefs.js."""
    profiles_dir = tmp_path / "Profiles"
    profile = profiles_dir / "abcd1234.default"
    profile.mkdir(parents=True)
    lines = [f'user_pref("{k}", "{v}");' for k, v in prefs.items()]
    (profile / "prefs.js").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return profiles_dir


def _make_data_dir(tmp_path: Path, name: str) -> Path:
    """Create a fake Zotero data dir containing a zotero.sqlite file."""
    data_dir = tmp_path / name
    data_dir.mkdir(parents=True)
    (data_dir / "zotero.sqlite").write_bytes(b"")
    return data_dir


@pytest.fixture
def isolated_discovery(monkeypatch, tmp_path):
    """Point discovery at an empty home and no profiles; clear env override."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr(local_db, "_zotero_profiles_dirs", lambda: [])
    monkeypatch.delenv("ZOTERO_DB_PATH", raising=False)
    return fake_home


@skip_on_ci
def test_env_var_is_honored(monkeypatch, tmp_path, isolated_discovery):
    data_dir = _make_data_dir(tmp_path, "custom")
    monkeypatch.setenv("ZOTERO_DB_PATH", str(data_dir / "zotero.sqlite"))

    reader = LocalZoteroReader()
    assert reader.db_path == str(data_dir / "zotero.sqlite")


@skip_on_ci
def test_env_var_pointing_at_missing_file_raises(monkeypatch, tmp_path, isolated_discovery):
    missing = tmp_path / "nowhere" / "zotero.sqlite"
    monkeypatch.setenv("ZOTERO_DB_PATH", str(missing))

    with pytest.raises(FileNotFoundError, match="ZOTERO_DB_PATH"):
        LocalZoteroReader()


@skip_on_ci
def test_default_location_is_found(isolated_discovery):
    default_dir = _make_data_dir(isolated_discovery, "Zotero")

    reader = LocalZoteroReader()
    assert reader.db_path == str(default_dir / "zotero.sqlite")


@skip_on_ci
def test_custom_data_dir_from_prefs_is_found(monkeypatch, tmp_path, isolated_discovery):
    data_dir = _make_data_dir(tmp_path, "Documents/Zotero")
    profiles_dir = _make_profile(
        tmp_path, {"extensions.zotero.dataDir": str(data_dir)}
    )
    monkeypatch.setattr(local_db, "_zotero_profiles_dirs", lambda: [profiles_dir])

    reader = LocalZoteroReader()
    assert reader.db_path == str(data_dir / "zotero.sqlite")


@skip_on_ci
def test_prefs_data_dir_beats_default_location(monkeypatch, tmp_path, isolated_discovery):
    stale_default = _make_data_dir(isolated_discovery, "Zotero")
    configured = _make_data_dir(tmp_path, "configured")
    profiles_dir = _make_profile(
        tmp_path, {"extensions.zotero.dataDir": str(configured)}
    )
    monkeypatch.setattr(local_db, "_zotero_profiles_dirs", lambda: [profiles_dir])

    reader = LocalZoteroReader()
    assert reader.db_path == str(configured / "zotero.sqlite")
    assert reader.db_path != str(stale_default / "zotero.sqlite")


@skip_on_ci
def test_error_lists_candidates_and_remedy(isolated_discovery):
    with pytest.raises(FileNotFoundError) as excinfo:
        LocalZoteroReader()

    message = str(excinfo.value)
    assert "ZOTERO_DB_PATH" in message
    assert "zotero-mcp setup" in message
    assert str(isolated_discovery / "Zotero" / "zotero.sqlite") in message


@skip_on_ci
def test_read_string_pref_unescapes_windows_paths(tmp_path):
    prefs_path = tmp_path / "prefs.js"
    prefs_path.write_text(
        'user_pref("extensions.zotero.dataDir", "C:\\\\Users\\\\dan\\\\Zotero");\n',
        encoding="utf-8",
    )

    value = _read_string_pref(prefs_path, "extensions.zotero.dataDir")
    assert value == "C:\\Users\\dan\\Zotero"


@skip_on_ci
def test_read_string_pref_absent_returns_none(tmp_path):
    prefs_path = tmp_path / "prefs.js"
    prefs_path.write_text('user_pref("some.other.pref", "x");\n', encoding="utf-8")

    assert _read_string_pref(prefs_path, "extensions.zotero.dataDir") is None
    assert _read_string_pref(tmp_path / "missing.js", "any.pref") is None


@skip_on_ci
def test_base_attachment_path_from_profile_prefs(monkeypatch, tmp_path, isolated_discovery):
    data_dir = _make_data_dir(tmp_path, "data")
    profiles_dir = _make_profile(
        tmp_path,
        {
            "extensions.zotero.dataDir": str(data_dir),
            "extensions.zotero.baseAttachmentPath": str(tmp_path / "attachments"),
        },
    )
    monkeypatch.setattr(local_db, "_zotero_profiles_dirs", lambda: [profiles_dir])

    reader = LocalZoteroReader()
    assert reader._get_base_attachment_path() == tmp_path / "attachments"
