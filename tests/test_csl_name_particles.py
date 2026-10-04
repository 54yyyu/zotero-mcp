"""CSL name particles and suffixes survive CSL JSON import.

Better BibTeX and citation.js split "van der Maaten" into
``non-dropping-particle`` + ``family``; reading only ``family`` stored the
author as "Maaten".
"""

from zotero_mcp.citation_import import _csl_names_to_creators


def _one(name):
    (c,) = _csl_names_to_creators([name], "author")
    return c["firstName"], c["lastName"]


def test_non_dropping_particle_kept_in_last_name():
    assert _one({"given": "Laurens", "family": "Maaten",
                 "non-dropping-particle": "van der"}) == ("Laurens", "van der Maaten")


def test_apostrophe_particle_joins_without_space():
    assert _one({"given": "Pierre", "family": "Alembert",
                 "non-dropping-particle": "d'"}) == ("Pierre", "d'Alembert")


def test_dropping_particle_kept_with_given_name():
    assert _one({"given": "Alexander", "family": "Humboldt",
                 "dropping-particle": "von"}) == ("Alexander von", "Humboldt")


def test_suffix_kept():
    assert _one({"given": "Henry Louis", "family": "Gates",
                 "suffix": "Jr."}) == ("Henry Louis, Jr.", "Gates")
