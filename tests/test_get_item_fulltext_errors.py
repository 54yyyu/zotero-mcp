from zotero_mcp import cli_standalone
from zotero_mcp.tools import retrieval


def test_fulltext_failure_puts_error_before_metadata():
    result = retrieval._fulltext_error("Paper metadata", "accessing attachment: boom")

    assert result.startswith("Error: accessing attachment: boom")
    assert "Paper metadata" in result
    assert cli_standalone._reports_failure(result)
