"""A conference paper's proceedings title exports as ``booktitle``.

Zotero keeps a conference paper's venue in ``proceedingsTitle``. The BibTeX
field table mapped ``publicationTitle`` and ``bookTitle`` but not it, so every
``@inproceedings`` entry came out without saying where it was published.
"""

from unittest.mock import patch

import pytest

from zotero_mcp.client import generate_bibtex


@pytest.fixture(autouse=True)
def _no_better_bibtex():
    """Exercise the local generator, not a Better BibTeX that may be running."""
    with patch(
        "zotero_mcp.better_bibtex_client.ZoteroBetterBibTexAPI.is_zotero_running",
        return_value=False,
    ):
        yield


def _conference_paper():
    return {"key": "CONF0001", "data": {
        "key": "CONF0001", "itemType": "conferencePaper",
        "title": "Feature Prediction Diffusion Model for Video Anomaly Detection",
        "proceedingsTitle": "2023 IEEE/CVF International Conference on Computer Vision (ICCV)",
        "date": "2023-10", "pages": "5504-5514",
        "creators": [{"creatorType": "author", "firstName": "Cheng", "lastName": "Yan"}],
        "tags": [],
    }}


def test_proceedings_title_becomes_booktitle():
    bib = generate_bibtex(_conference_paper())
    assert bib.startswith("@inproceedings{")
    assert "booktitle = {2023 IEEE/CVF International Conference on Computer Vision (ICCV)}" in bib


def test_conference_paper_is_not_given_a_journal():
    assert "journal = {" not in generate_bibtex(_conference_paper())
