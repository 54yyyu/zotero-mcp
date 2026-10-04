"""Background config writers must never replace a config they cannot read.

config.json holds the web API key, the local write key and the embedding
settings next to the index bookkeeping. The index-state savers and
``--db-path`` used to treat an unparseable file as ``{}`` and write that back,
so one hand-editing typo, or a reader that caught another process mid-write,
erased everything else in the file.
"""

import json
import threading

import pytest

from zotero_mcp import cli
from zotero_mcp.semantic_search import ZoteroSemanticSearch

FULL = {
    "client_env": {"ZOTERO_API_KEY": "SECRET", "ZOTERO_LIBRARY_ID": "1"},
    "local_api": {"key": "LOCALKEY"},
    "semantic_search": {"embedding_model": "openai"},
}


def _searcher(path):
    s = ZoteroSemanticSearch.__new__(ZoteroSemanticSearch)
    s.config_path = str(path)
    s.update_config = {"auto_update": False}
    s._active_library_key = lambda: "0"
    return s


SAVERS = {
    "update_config": lambda s: s._save_update_config(last_sync_version=7),
    "index_schema_version": lambda s: s._save_index_schema_version(3),
    "backfill_unattributed": lambda s: s._save_backfill_unattributed(2),
}


@pytest.mark.parametrize("save", SAVERS.values(), ids=SAVERS.keys())
def test_invalid_config_is_left_untouched(tmp_path, save):
    path = tmp_path / "config.json"
    broken = json.dumps(FULL)[:-1]  # a missing closing brace
    path.write_text(broken)
    save(_searcher(path))
    assert path.read_text() == broken


@pytest.mark.parametrize("save", SAVERS.values(), ids=SAVERS.keys())
def test_valid_config_keeps_the_other_sections(tmp_path, save):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(FULL))
    save(_searcher(path))
    saved = json.loads(path.read_text())
    assert saved["client_env"] == FULL["client_env"]
    assert saved["local_api"] == FULL["local_api"]
    assert saved["semantic_search"]["embedding_model"] == "openai"


def test_saves_are_owner_only(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(FULL))
    _searcher(path)._save_index_schema_version(3)
    assert path.stat().st_mode & 0o077 == 0


def test_concurrent_saves_keep_the_credentials(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(FULL))
    s = _searcher(path)
    stop = threading.Event()

    def loop(save):
        while not stop.is_set():
            save(s)

    threads = [threading.Thread(target=loop, args=(f,)) for f in SAVERS.values()]
    for t in threads:
        t.start()
    stop.wait(1.5)
    stop.set()
    for t in threads:
        t.join()
    saved = json.loads(path.read_text())
    assert saved["client_env"] == FULL["client_env"]
    assert saved["local_api"] == FULL["local_api"]


def test_db_path_refuses_an_invalid_config(tmp_path, capsys):
    path = tmp_path / "config.json"
    broken = json.dumps(FULL)[:-1]
    path.write_text(broken)
    cli._save_zotero_db_path_to_config(path, "/x/zotero.sqlite")
    assert path.read_text() == broken
    assert "Could not save db_path" in capsys.readouterr().out


def test_db_path_is_added_alongside_existing_settings(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(FULL))
    cli._save_zotero_db_path_to_config(path, "/x/zotero.sqlite")
    saved = json.loads(path.read_text())
    assert saved["zotero_db_path"] == "/x/zotero.sqlite"
    assert saved["client_env"] == FULL["client_env"]
