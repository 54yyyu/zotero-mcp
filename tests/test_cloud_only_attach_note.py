"""An upload that will never reach local storage must say so (#463).

In hybrid mode (local reads, web writes) an attached file goes to cloud
storage and reaches this computer only when Zotero syncs files. With file
syncing off, the upload reported "File attached" while the file never appeared
in ~/Zotero/storage, and every local tool failed on it later.
"""

import pytest
from conftest import DummyContext
from test_attach_verification import _AttachFake

from zotero_mcp import local_db
from zotero_mcp.tools import _helpers

FILE_SYNC_OFF = 'user_pref("extensions.zotero.sync.storage.enabled", false);\n'


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    """Point prefs discovery at profiles this test writes."""
    files = []
    monkeypatch.setattr(local_db, "_profile_prefs_files", lambda: list(files))

    def add(text):
        path = tmp_path / f"profile{len(files)}" / "prefs.js"
        path.parent.mkdir()
        path.write_text(text)
        files.append(path)
        return path

    return add


@pytest.fixture
def hybrid(monkeypatch):
    monkeypatch.setattr(_helpers._utils, "is_local_mode", lambda: True)
    monkeypatch.setattr(_helpers, "_maybe_upload_to_webdav", lambda *a, **k: "")


def test_bool_pref_reader(tmp_path):
    prefs = tmp_path / "prefs.js"
    prefs.write_text(FILE_SYNC_OFF + 'user_pref("extensions.zotero.sync.autoSync", true);\n')
    assert local_db._read_bool_pref(prefs, "extensions.zotero.sync.storage.enabled") is False
    assert local_db._read_bool_pref(prefs, "extensions.zotero.sync.autoSync") is True
    assert local_db._read_bool_pref(prefs, "extensions.zotero.absent") is None
    assert local_db._read_bool_pref(tmp_path / "missing.js", "x") is None


def test_hybrid_upload_with_file_sync_off_says_so(profiles, hybrid):
    profiles(FILE_SYNC_OFF)
    ok, suffix, key = _helpers._attach_and_verify(
        _AttachFake(), "paper.pdf", "/abs/paper.pdf", "ITEM1", DummyContext()
    )
    assert ok is True and key == "ATCH0001"
    assert suffix.count("file syncing is off") == 1


def test_no_note_when_file_sync_is_on(profiles, hybrid):
    profiles('user_pref("extensions.zotero.sync.autoSync", true);\n')
    _ok, suffix, _key = _helpers._attach_and_verify(
        _AttachFake(), "paper.pdf", "/abs/paper.pdf", "ITEM1", DummyContext()
    )
    assert "file syncing" not in suffix


def test_one_profile_still_syncing_means_no_note(profiles, hybrid):
    profiles(FILE_SYNC_OFF)
    profiles("")
    assert _helpers._zotero_file_sync_disabled() is False


def test_local_write_client_needs_no_note(profiles, hybrid):
    profiles(FILE_SYNC_OFF)
    zot = _AttachFake()
    zot.local = True
    assert _helpers._cloud_only_note(zot) == ""


def test_web_only_setup_needs_no_note(profiles, monkeypatch):
    """Without local reads nothing here resolves local file paths."""
    profiles(FILE_SYNC_OFF)
    monkeypatch.setattr(_helpers._utils, "is_local_mode", lambda: False)
    assert _helpers._cloud_only_note(_AttachFake()) == ""


def test_webdav_upload_gets_the_note_once(profiles, hybrid, monkeypatch):
    from zotero_mcp import webdav

    profiles(FILE_SYNC_OFF)
    monkeypatch.setattr(webdav, "is_webdav_configured", lambda: True)
    monkeypatch.setattr(webdav, "upload_attachment_to_webdav", lambda **kw: None)

    suffix = _helpers._webdav_first_attach(
        _AttachFake(), "paper.pdf", "/abs/paper.pdf", "ITEM1", DummyContext()
    )
    assert "uploaded to WebDAV" in suffix
    assert suffix.count("file syncing is off") == 1
