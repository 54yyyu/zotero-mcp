"""The pre-update backup follows a moved index (semantic_search.persist_directory, #617)."""

import json
import shutil
from pathlib import Path

from zotero_mcp import updater


def test_backup_and_restore_use_the_configured_index_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    moved = tmp_path / "elsewhere" / "index"
    moved.mkdir(parents=True)
    (moved / "chroma.sqlite3").write_text("rows")
    cfg = tmp_path / ".config" / "zotero-mcp"
    cfg.mkdir(parents=True)
    (cfg / "config.json").write_text(json.dumps({"semantic_search": {"persist_directory": str(moved)}}))

    backup = updater.backup_configurations()
    assert (backup / "chroma_db" / "chroma.sqlite3").read_text() == "rows"

    shutil.rmtree(moved)
    assert updater.restore_configurations(backup)
    assert (moved / "chroma.sqlite3").read_text() == "rows"
    assert not (cfg / "chroma_db").exists(), "the default folder is not created"


def test_restore_leaves_an_existing_index_and_its_folder_alone(tmp_path, monkeypatch):
    """A package update never writes to the index, so restore must not replace
    it. Replacing means deleting first, and a backup that stopped partway (disk
    full, an unreadable file) would take the rest of the folder with it."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    shared = tmp_path / "Documents"
    shared.mkdir()
    (shared / "chroma.sqlite3").write_text("rows")
    (shared / "thesis.docx").write_text("not part of the index")
    cfg = tmp_path / ".config" / "zotero-mcp"
    cfg.mkdir(parents=True)
    (cfg / "config.json").write_text(json.dumps({"semantic_search": {"persist_directory": str(shared)}}))

    backup = updater.backup_configurations()
    (backup / "chroma_db" / "thesis.docx").unlink()  # the file the copy failed on
    (shared / "chroma.sqlite3").write_text("rows indexed during the update")

    assert updater.restore_configurations(backup)
    assert (shared / "thesis.docx").read_text() == "not part of the index"
    assert (shared / "chroma.sqlite3").read_text() == "rows indexed during the update"
