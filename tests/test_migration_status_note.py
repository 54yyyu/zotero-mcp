"""Status explains an unmigrated chunked index, and counts items per level (#769).

Real persisted indexes in temp directories, with given vectors, so nothing embeds.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

if importlib.util.find_spec("chromadb") is None:
    pytest.skip("chromadb not installed", allow_module_level=True)

import numpy as np

from zotero_mcp import chroma_client
from zotero_mcp.chroma_client import ChromaClient, read_collection_status
from zotero_mcp.index_migration import migrate_index

NAME = "zotero_library"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("ZOTERO_EMBEDDING_MODEL", raising=False)
    # Local mode, where update-db --fulltext works; the web-API tests switch it off.
    monkeypatch.setenv("ZOTERO_LOCAL", "true")
    return home


@pytest.fixture()
def index(home):
    """(index directory, config path) for an OpenAI index: 3 characters a token, 8,000 at most."""
    directory = home / "chroma"
    config = home / ".config" / "zotero-mcp" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"semantic_search": {
        "persist_directory": str(directory),
        "embedding_model": "openai",
    }}))
    return str(directory), str(config)


def _client(directory, *, legacy):
    client = ChromaClient(collection_name=NAME, persist_directory=directory)
    client.route_passages = not legacy
    return client


def _write(client, records):
    """records: (id, metadata) pairs."""
    rng = np.random.default_rng(5)
    ids = [doc_id for doc_id, _ in records]
    client.upsert_embeddings(
        documents=[f"text of {i}" for i in ids],
        metadatas=[{"group_id": 0, "item_key": i.split("#")[0], **meta} for i, meta in records],
        ids=ids,
        embeddings=[rng.normal(size=4).astype(float).tolist() for _ in ids],
    )


def _passages(key, lengths, fulltext=True):
    """An item's passages; ``lengths`` are the char_end of each, the last being the text's length."""
    return [
        (f"{key}#{n}", {"parent_item_key": key, "chunk_index": n, "n_chunks": len(lengths),
                        "char_start": max(0, end - 1500), "char_end": end,
                        **({"has_fulltext": True} if fulltext else {})})
        for n, end in enumerate(lengths)
    ]


def _seed_unmigrated(directory, extra=(), fulltext=True):
    """AAA (3,000 characters), BBB (90,000), and OLD, an article record from before chunking."""
    records = (_passages("AAA", [1500, 3000], fulltext) + _passages("BBB", [1500, 3000, 90000], fulltext)
               + [("OLD", {})])
    _write(_client(directory, legacy=True), records + list(extra))


def _migrate(directory):
    """migrate-index with the model the index was built with, so no OpenAI client is made."""
    config = Path(directory).parent / "migrate-config.json"
    config.write_text(json.dumps({"semantic_search": {
        "persist_directory": directory, "embedding_model": "default",
    }}))
    migrate_index(str(config), progress=lambda _m: None)


# ---- the counts and the estimate -------------------------------------------


def test_unmigrated_index_reports_what_update_db_would_embed(index):
    directory, config = index
    _seed_unmigrated(directory)

    info = read_collection_status(config)

    assert info["article_items"] == 1
    assert info["passage_items"] == 2
    migration = info["migration"]
    assert migration["items_needing_articles"] == 2
    # AAA: 3,000 / 3 = 1,000 tokens. BBB: 30,000, capped at 8,000.
    assert migration["estimated_tokens"] == 9000
    assert migration["fulltext"] is True


def test_an_item_whose_first_passage_is_missing_still_counts(index):
    """Counted like migrate-index counts it; its tokens are the average of the others."""
    directory, config = index
    _seed_unmigrated(directory, extra=_passages("CCC", [1500, 6000])[1:])

    migration = read_collection_status(config)["migration"]

    assert migration["items_needing_articles"] == 3
    assert migration["estimated_tokens"] == 9000 + 4500
    assert migration["items_needing_articles"] == migrate_index(
        config, dry_run=True, progress=lambda _m: None
    ).items_needing_articles


def test_estimate_failure_keeps_the_note_and_status(index, monkeypatch):
    directory, config = index
    _seed_unmigrated(directory)

    def broken(*_a, **_k):
        raise chroma_client.sqlite3.OperationalError("no such table: embedding_metadata")

    monkeypatch.setattr(chroma_client, "_unmigrated_items", broken)
    info = read_collection_status(config)

    assert info["initialized"] is True
    assert info["passage_items"] is None
    assert info["migration"]["estimated_tokens"] is None


def test_migrated_index_has_no_note_and_counts_both_levels(index):
    directory, config = index
    _seed_unmigrated(directory)
    _migrate(directory)

    info = read_collection_status(config)

    assert "migration" not in info
    assert info["article_items"] == 1
    assert info["passage_items"] == 2


def test_item_level_index_has_no_note(index):
    directory, config = index
    _write(_client(directory, legacy=False), [("AAA", {}), ("BBB", {})])

    info = read_collection_status(config)

    assert "migration" not in info
    assert info["article_items"] == 2
    assert info["passage_items"] == 0


def test_split_passage_only_index_has_no_note(index):
    directory, config = index
    _write(_client(directory, legacy=False), _passages("AAA", [1500, 3000]))

    info = read_collection_status(config)

    assert "migration" not in info
    assert info["article_items"] == 0
    assert info["passage_items"] == 1


def test_client_collection_info_matches_the_model_free_read(index):
    directory, config = index
    _seed_unmigrated(directory)

    info = _client(directory, legacy=False).get_collection_info()

    # This client uses the default local model: 4 characters a token, 256 at most.
    assert info["article_items"] == 1
    assert info["passage_items"] == 2
    assert info["migration"]["items_needing_articles"] == 2
    assert info["migration"]["estimated_tokens"] == 512


def test_both_status_paths_give_the_same_estimate(index, monkeypatch):
    """The MCP tool reads the config, the CLI builds a client; env settings count for both."""
    directory, config = index
    Path(config).write_text(json.dumps({"semantic_search": {
        "persist_directory": directory, "embedding_model": "default",
    }}))
    _seed_unmigrated(directory)

    assert read_collection_status(config)["migration"] == (
        _client(directory, legacy=False).get_collection_info()["migration"]
    )


def test_model_from_the_environment_sets_the_cap(index, monkeypatch):
    directory, config = index
    Path(config).write_text(json.dumps({"semantic_search": {
        "persist_directory": directory, "embedding_model": "gemini",
    }}))
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-key")
    monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    _seed_unmigrated(directory)

    assert read_collection_status(config)["migration"]["max_tokens"] == 7980


def test_any_failure_in_the_estimate_keeps_status(index, monkeypatch):
    directory, config = index
    _seed_unmigrated(directory)

    def broken(*_a, **_k):
        raise ValueError("invalid literal for int()")

    monkeypatch.setattr(chroma_client, "article_token_budget", broken)
    info = read_collection_status(config)

    assert info["initialized"] is True
    assert "error" not in info
    assert info["migration"]["estimated_tokens"] is None


def test_empty_or_missing_index_has_no_note(index):
    directory, config = index
    assert "migration" not in read_collection_status(config)
    _client(directory, legacy=False)
    info = read_collection_status(config)
    assert "migration" not in info
    assert info["article_items"] == 0


def test_unmigrated_index_with_chunking_off_still_gets_the_note(index):
    """Migrating still pays off: the next update fills the articles either way."""
    directory, config = index
    Path(config).write_text(json.dumps({"semantic_search": {
        "persist_directory": directory, "embedding_model": "openai", "chunking": {"enabled": False},
    }}))
    _seed_unmigrated(directory)

    assert read_collection_status(config)["migration"]["items_needing_articles"] == 2


# ---- the three status outputs ----------------------------------------------


class _Ctx:
    def info(self, *_):
        pass

    def error(self, *_):
        pass


def _status_tool():
    from zotero_mcp.tools import search as search_tools

    tool = search_tools.get_search_database_status
    fn = tool.fn if hasattr(tool, "fn") else tool
    return fn(ctx=_Ctx())


class _Search:
    def __init__(self, directory):
        self.info = _client(directory, legacy=False).get_collection_info()

    def get_database_status(self):
        return {"collection_info": self.info, "update_config": {}, "openai_batch": {}, "chunking": {}}


def _cli_status(monkeypatch, capsys, config, directory, which):
    import zotero_mcp.cli as cli
    import zotero_mcp.cli_standalone as standalone
    import zotero_mcp.semantic_search as semantic_search

    search = _Search(directory)
    monkeypatch.setattr(cli, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(standalone, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(semantic_search, "create_semantic_search", lambda *a, **k: search)
    if which == "db-status":
        monkeypatch.setattr(sys, "argv", ["zotero-mcp", "db-status", "--config-path", config])
        cli.main()
    else:
        monkeypatch.setattr(sys, "argv", ["zotero-cli", "db", "status", "--config-path", config])
        standalone.main()
    return capsys.readouterr().out


def _assert_note(out, update_command):
    assert f"{update_command}   (" in out
    assert "local model" not in out
    assert "With a paid embedding API" in out
    assert "earlier version" in out
    assert "Search keeps working" in out
    assert "zotero-mcp migrate-index --dry-run" in out
    assert out.index("migrate-index --dry-run") < out.index(update_command)
    assert "upgrade" in out
    assert "2 items" in out
    assert "estimate" in out


def test_status_tool_explains_an_unmigrated_index(index):
    directory, _config = index
    _seed_unmigrated(directory)

    out = _status_tool()

    _assert_note(out, "zotero-mcp update-db --fulltext")
    assert "9,000 tokens" in out
    assert "**Items with Article Records:** 1" in out
    assert "**Items with Passages:** 2" in out


@pytest.mark.parametrize("which, update", [("db-status", "zotero-mcp update-db --fulltext"),
                                           ("zotero-cli", "zotero-cli db update --fulltext")])
def test_cli_status_explains_an_unmigrated_index(index, monkeypatch, capsys, which, update):
    directory, config = index
    _seed_unmigrated(directory)

    out = _cli_status(monkeypatch, capsys, config, directory, which)

    _assert_note(out, update)
    assert "512 tokens" in out
    assert "Items with article records: 1" in out
    assert "Items with passages: 2" in out


def test_status_tool_says_nothing_about_migrating_a_migrated_index(index):
    directory, config = index
    _seed_unmigrated(directory)
    _migrate(directory)

    out = _status_tool()

    assert "migrate-index" not in out
    assert "**Items with Passages:** 2" in out


@pytest.mark.parametrize("which", ["db-status", "zotero-cli"])
def test_cli_status_says_nothing_about_migrating_an_item_level_index(index, monkeypatch, capsys, which):
    directory, config = index
    _write(_client(directory, legacy=False), [("AAA", {})])

    out = _cli_status(monkeypatch, capsys, config, directory, which)

    assert "migrate-index" not in out
    assert "Items with article records: 1" in out


def test_step_four_is_the_update_migrate_index_itself_names(index, capsys):
    """Plain update-db leaves items with full text waiting for it; the note must not."""
    from zotero_mcp.cli import _print_migration_report

    directory, config = index
    _seed_unmigrated(directory)
    note = _status_tool()
    capsys.readouterr()

    _print_migration_report(_dry_report(config))
    command = next(line.split("#")[0].strip() for line in capsys.readouterr().out.splitlines()
                   if "update-db" in line)

    assert command == "zotero-mcp update-db --fulltext"
    assert f"4. {command}   (" in note


@pytest.mark.parametrize("which", ["tool", "db-status", "zotero-cli"])
def test_step_four_with_the_web_api_leaves_out_fulltext(index, monkeypatch, capsys, which):
    """--fulltext needs local mode; the web API's update fetches the text Zotero extracted."""
    monkeypatch.setenv("ZOTERO_LOCAL", "false")
    directory, config = index
    _seed_unmigrated(directory)

    out = _status_tool() if which == "tool" else _cli_status(monkeypatch, capsys, config, directory, which)

    update = "zotero-cli db update" if which == "zotero-cli" else "zotero-mcp update-db"
    _assert_note(out, update)
    assert "--fulltext" not in out.replace("--fulltext needs local mode", "")
    assert "--fulltext needs local mode" in out


def test_migrate_index_with_the_web_api_names_the_same_update_as_status(index, monkeypatch, capsys):
    from zotero_mcp.cli import _print_migration_report

    monkeypatch.setenv("ZOTERO_LOCAL", "false")
    directory, config = index
    _seed_unmigrated(directory)
    note = _status_tool()
    capsys.readouterr()

    _print_migration_report(_dry_report(config))
    out = capsys.readouterr().out
    command = next(line.split("#")[0].strip() for line in out.splitlines() if "update-db" in line)

    assert command == "zotero-mcp update-db"
    assert f"4. {command}   (" in note
    assert "--fulltext needs local mode" in out


def _dry_report(config):
    """migrate-index's report of a real run, without building a client for the configured model."""
    report = migrate_index(config, dry_run=True, progress=lambda _m: None)
    report.dry_run = False
    return report


def test_index_without_full_text_is_updated_without_it(index):
    directory, _config = index
    _seed_unmigrated(directory, fulltext=False)

    out = _status_tool()

    assert "4. zotero-mcp update-db   (" in out
    assert "--fulltext" not in out


@pytest.mark.parametrize("model, extra", [
    ("openai", {"base_url": "http://127.0.0.1:9/v1"}),
    ("gemini", {}),
    ("ollama", {}),
    ("default", {}),
])
def test_cost_wording_never_guesses_whether_it_is_billed(index, model, extra):
    directory, config = index
    Path(config).write_text(json.dumps({"semantic_search": {
        "persist_directory": directory, "embedding_model": model, "embedding_config": extra,
    }}))
    _seed_unmigrated(directory)

    out = _status_tool()

    assert "With a paid embedding API" in out
    assert "local model" not in out
    assert "costs money" not in out


@pytest.mark.parametrize("which", ["db-status", "zotero-cli"])
def test_status_counts_items_not_attachment_records_on_an_item_keyed_index(index, monkeypatch, capsys, which):
    """An item-keyed index (#826) holds a record per item and one per file: status counts the items."""
    directory, config = index
    _write(_client(directory, legacy=False), [
        ("BIB1", {"date_modified": "d"}),
        ("BIB2", {"date_modified": "d"}),
        ("ATT1", {"date_modified": "d", "file_stamp": "1:1"}),
        ("ATT1#0", {"chunk_index": 0, "n_chunks": 1}),
    ])

    out = _cli_status(monkeypatch, capsys, config, directory, which)

    assert "Document count: 4" in out
    assert "Items with article records: 2" in out
    assert "Items with passages: 1" in out
    assert "**Items with Article Records:** 2" in _status_tool()
