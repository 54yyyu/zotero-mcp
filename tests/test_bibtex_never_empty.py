"""``format="bibtex"`` must never answer with an empty string.

Reported against a real 15k-item library: ``zotero_get_item_metadata`` on
item ``KGBSSZ3W`` returned ``""`` for ``format="bibtex"`` while
``format="markdown"`` rendered the full record. The item existed and was
readable; it was a deduplicated duplicate sitting in the *trash*.

Better BibTeX does not index trashed items, so ``item.citationkey`` answers
``{"KGBSSZ3W": null}`` — a successful RPC carrying no key. ``export_bibtex``
caught its own resulting exception and returned ``""``, and
``generate_bibtex`` handed that straight back because Zotero "was running",
never reaching the local generator sitting directly below it.

An empty string is the one answer a caller cannot act on: it reads the same
as "this item has no BibTeX" and as "this item does not exist". BBT is an
enhancement here, not a prerequisite — every one of these paths must fall
through to local generation.
"""

from unittest.mock import patch

import pytest

from zotero_mcp.better_bibtex_client import BetterBibTexError
from zotero_mcp.client import generate_bibtex

# A trashed journal article, shaped like the real Zotero API response.
TRASHED_ITEM = {
    "key": "KGBSSZ3W",
    "data": {
        "key": "KGBSSZ3W",
        "itemType": "journalArticle",
        "title": "Eight Simple Guidelines for Improved Understanding",
        "publicationTitle": "Organizational Research Methods",
        "volume": "25",
        "issue": "1",
        "pages": "48-87",
        "DOI": "10.1177/1094428121991907",
        "date": "2021-3-11",
        "deleted": True,
        "creators": [
            {"creatorType": "author", "firstName": "Mikko", "lastName": "Rönkkö"},
        ],
    },
}


def _bbt(*, running=True, export=None, error=None):
    """Patch the BBT client: is_zotero_running -> *running*, export_bibtex
    either returns *export* or raises *error*."""
    running_patch = patch(
        "zotero_mcp.better_bibtex_client.ZoteroBetterBibTexAPI.is_zotero_running",
        return_value=running,
    )
    kwargs = {"side_effect": error} if error is not None else {"return_value": export}
    export_patch = patch(
        "zotero_mcp.better_bibtex_client.ZoteroBetterBibTexAPI.export_bibtex",
        **kwargs,
    )
    return running_patch, export_patch


class TestNeverEmpty:
    def test_null_citekey_falls_back_to_local_generation(self):
        """The reported case: BBT has no citekey for a trashed item."""
        running, export = _bbt(
            error=BetterBibTexError("Better BibTeX has no citation key for item")
        )
        with running, export:
            out = generate_bibtex(TRASHED_ITEM)

        assert out.strip(), "bibtex format returned an empty string"
        assert out.startswith("@article{")
        # The fallback must carry the metadata, not just an empty shell.
        assert "Organizational Research Methods" in out
        assert "10.1177/1094428121991907" in out
        assert "Rönkkö" in out

    def test_blank_bbt_export_falls_back(self):
        """Belt and braces: even if BBT hands back a blank string rather than
        raising, that must not become the answer."""
        running, export = _bbt(export="   \n  ")
        with running, export:
            out = generate_bibtex(TRASHED_ITEM)
        assert out.strip()
        assert "Organizational Research Methods" in out

    def test_bbt_transport_error_falls_back(self):
        running, export = _bbt(error=OSError("connection refused"))
        with running, export:
            out = generate_bibtex(TRASHED_ITEM)
        assert out.strip()

    def test_good_bbt_export_is_preferred(self):
        """The fallback must not displace a working BBT export — BBT carries
        the user's real pinned citekeys."""
        running, export = _bbt(export="@article{ronkkoEight2022,\n  title = {T}\n}")
        with running, export:
            out = generate_bibtex(TRASHED_ITEM)
        assert out.startswith("@article{ronkkoEight2022,")

    def test_zotero_not_running_uses_local_generation(self):
        running, export = _bbt(running=False, export="unused")
        with running, export:
            out = generate_bibtex(TRASHED_ITEM)
        assert out.strip()
        assert out.startswith("@article{")

    def test_item_without_key_still_generates(self):
        """A locally-assembled item may carry no key at all; the citekey must
        not render the literal string "None"."""
        item = {"data": dict(TRASHED_ITEM["data"])}
        del item["data"]["key"]
        running, export = _bbt(running=False, export="unused")
        with running, export:
            out = generate_bibtex(item)
        assert out.strip()
        assert "None" not in out.splitlines()[0]

    def test_attachment_still_raises(self):
        """Types that genuinely have no BibTeX must keep raising — the point is
        distinguishability, not that every input yields an entry."""
        item = {"key": "X", "data": {"key": "X", "itemType": "attachment"}}
        running, export = _bbt(running=False, export="unused")
        with running, export, pytest.raises(ValueError):
            generate_bibtex(item)
