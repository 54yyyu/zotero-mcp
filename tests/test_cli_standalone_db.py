"""`zotero-cli db` runs the same handlers as `zotero-mcp update-db`,
`batch-status` and `batch-import`.

It used to have its own copies: `db batch-status` / `db batch-import`
always read OpenAI manifests, so a Gemini batch run could not be watched or
imported from zotero-cli, and `db update` had no --batch-provider,
--extraction-workers or --auto-loop.
"""

from unittest.mock import MagicMock

import pytest

from zotero_mcp import cli, cli_standalone


@pytest.fixture
def fake_search(monkeypatch):
    search = MagicMock()
    search.chroma_client.embedding_model = "gemini"
    search.update_database.return_value = {"total_items": 0}
    search._get_batch_status.return_value = {"batches": []}
    search._import_batch.return_value = {}
    created = {}

    def create(config_path, **kwargs):
        created.update(kwargs, config_path=config_path)
        return search

    monkeypatch.setattr("zotero_mcp.semantic_search.create_semantic_search", create)
    monkeypatch.setattr(cli, "setup_zotero_environment", lambda: None)
    monkeypatch.setattr(cli_standalone, "setup_zotero_environment", lambda: None)
    # A successful update sweeps the fulltext cache; keep it off the real one.
    monkeypatch.setattr("zotero_mcp.fulltext_cache.purge_stale", lambda **_k: 0)
    search.created = created
    return search


def _run(argv):
    args = cli_standalone.build_parser().parse_args(argv)
    cli_standalone.cmd_db(args)


def test_batch_status_reads_the_configured_providers_manifests(fake_search):
    _run(["db", "batch-status"])

    fake_search._get_batch_status.assert_called_once_with("gemini", batch_ids=None)
    fake_search.get_openai_batch_status.assert_not_called()


def test_batch_import_takes_an_explicit_provider(fake_search):
    fake_search.chroma_client.embedding_model = "openai"

    _run(["db", "batch-import", "--provider", "gemini", "--batch-id", "b1"])

    fake_search._import_batch.assert_called_once_with("gemini", batch_ids=["b1"])
    fake_search.import_openai_batch.assert_not_called()


def test_db_update_takes_the_update_db_flags(fake_search):
    _run([
        "db", "update", "--batch", "--batch-provider", "gemini",
        "--extraction-workers", "4", "--auto-loop", "--poll-interval", "5",
    ])

    assert fake_search.created["extraction_workers"] == 4
    kwargs = fake_search.update_database.call_args.kwargs
    assert kwargs["use_batch"] is True
    assert kwargs["batch_provider"] == "gemini"
    assert kwargs["auto_loop"] is True
    assert kwargs["batch_poll_interval"] == 5


def test_db_update_keeps_its_old_flags(fake_search):
    fake_search.chroma_client.embedding_model = "openai"

    _run(["db", "update", "--force-rebuild", "--limit", "3", "--openai-batch",
          "--allow-mass-deletion"])

    kwargs = fake_search.update_database.call_args.kwargs
    assert kwargs["force_full_rebuild"] is True
    assert kwargs["limit"] == 3
    assert kwargs["use_openai_batch"] is True
    assert kwargs["allow_mass_deletion"] is True
