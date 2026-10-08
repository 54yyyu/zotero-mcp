"""The pre-update backup follows a moved index (semantic_search.persist_directory, #617)."""

import json
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

    (moved / "chroma.sqlite3").write_text("broken")
    assert updater.restore_configurations(backup)
    assert (moved / "chroma.sqlite3").read_text() == "rows"
    assert not (cfg / "chroma_db").exists(), "the default folder is not created"
