"""Library code must never write diagnostics to stdout (#529).

stdout is a protocol channel twice over: `zotero-cli --json` promises exactly
one JSON envelope on it, and the MCP server's stdio transport carries JSON-RPC
on it. A bare print() from a helper lands in the middle of either. The Better
BibTeX client printed its fallback diagnostics, which made
`zotero-cli --json annotations list` emit invalid JSON even though the
annotation lookup itself succeeded; the pdfannots helpers printed the same way
on the zotero_get_annotations path.
"""

from unittest.mock import patch

from zotero_mcp import pdfannots_downloader, pdfannots_helper
from zotero_mcp.better_bibtex_client import ZoteroBetterBibTexAPI


def _bbt():
    api = ZoteroBetterBibTexAPI.__new__(ZoteroBetterBibTexAPI)
    return api


def test_better_bibtex_search_failure_does_not_touch_stdout(capsys):
    api = _bbt()
    with patch.object(ZoteroBetterBibTexAPI, "_make_request",
                      side_effect=RuntimeError("Invalid condition 'blockStart'")):
        assert api.search_citekeys("A title") == []
    out, err = capsys.readouterr()
    assert out == ""


def test_better_bibtex_attachment_failure_does_not_touch_stdout(capsys):
    api = _bbt()
    with patch.object(ZoteroBetterBibTexAPI, "_make_request",
                      side_effect=RuntimeError("boom")):
        assert api.get_attachments("citekey2020", 1) == []
    assert capsys.readouterr().out == ""


def test_pdfannots_not_installed_does_not_touch_stdout(capsys):
    with patch.object(pdfannots_helper, "ensure_pdfannots_installed", return_value=False):
        assert pdfannots_helper.extract_annotations_from_pdf("/nonexistent.pdf") == []
    assert capsys.readouterr().out == ""


def test_pdfannots_download_without_url_does_not_touch_stdout(capsys):
    with patch.object(pdfannots_downloader, "get_download_url", return_value=None):
        assert pdfannots_downloader.download_and_install() is False
    assert capsys.readouterr().out == ""
