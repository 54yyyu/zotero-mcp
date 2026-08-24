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
    assert "zotero-mcp update-db --batch" in out
    assert "batch-status" in out
    assert "batch-import" in out


def test_zotero_mcp_help_still_lists_the_deprecated_openai_commands(capsys):
    """The openai-batch-* commands shipped in 0.10.0 and stay callable."""
    with patch("sys.argv", ["zotero-mcp", "help"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "openai-batch-status" in out
    assert "openai-batch-import" in out


def test_zotero_mcp_help_update_db_shows_batch_flags(capsys):
    with patch("sys.argv", ["zotero-mcp", "help", "update-db"]):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--batch" in out
    assert "--no-batch" in out
    assert "--batch-provider" in out
    # Deprecated but still accepted, so scripts written against 0.10.0 work.
    assert "--openai-batch" in out
    assert "--no-openai-batch" in out
