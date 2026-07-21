from unittest.mock import patch

import pytest

from zotero_mcp.cli import main


def test_zotero_mcp_help_mentions_batch_indexing(capsys):
    with patch("sys.argv", ["zotero-mcp", "help"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Batch API indexing" in out
    assert "zotero-mcp update-db --openai-batch" in out
    assert "zotero-mcp update-db --gemini-batch" in out
    assert "batch-status" in out
    assert "batch-import" in out
    # Backward-compatible aliases must still be discoverable via subcommands.
    assert "openai-batch-status" in out
    assert "openai-batch-import" in out


def test_zotero_mcp_help_update_db_shows_batch_flags(capsys):
    with patch("sys.argv", ["zotero-mcp", "help", "update-db"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--openai-batch" in out
    assert "--no-openai-batch" in out
    assert "--gemini-batch" in out
    assert "--no-gemini-batch" in out
    assert "--extraction-workers" in out
    assert "--clear-fulltext-cache" in out
