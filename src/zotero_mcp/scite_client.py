"""Thin HTTP client for the Scite public API (api.scite.ai).

Provides citation tallies and paper metadata for Zotero library items.
No API key required — uses Scite's public endpoints.

This is the MCP counterpart of the `Scite Zotero Plugin`_ which adds
tally columns to Zotero's desktop interface.  Instead of columns, Scite
data appears inline in MCP search results and tool outputs.

.. _Scite Zotero Plugin: https://github.com/scitedotai/scite-zotero-plugin
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

_BASE = "https://api.scite.ai"
_TIMEOUT = 15  # seconds
_HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Tallies
# ---------------------------------------------------------------------------


def get_tally(doi: str) -> dict | None:
    """Fetch citation tally for a single DOI.

    Returns dict with keys: ``doi``, ``total``, ``supporting``,
    ``contradicting``, ``mentioning``, ``unclassified``,
    ``citingPublications``.  Returns ``None`` on any failure.
    """
    try:
        resp = requests.get(
            f"{_BASE}/tallies/{doi}",
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except requests.RequestException as exc:
        logger.debug("Scite tally request failed for %s: %s", doi, exc)
        return None


def get_tallies_batch(dois: list[str]) -> dict[str, dict]:
    """Fetch tallies for up to 500 DOIs.

    Returns ``{doi: tally_dict}``; empty dict on failure.
    """
    if not dois:
        return {}
    try:
        resp = requests.post(
            f"{_BASE}/tallies",
            json=dois[:500],
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json().get("tallies", {})
        return {}
    except requests.RequestException as exc:
        logger.debug("Scite batch tallies request failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Paper metadata (includes editorialNotices / retraction info)
# ---------------------------------------------------------------------------


def get_paper(doi: str) -> dict | None:
    """Fetch paper metadata including ``editorialNotices``.

    Returns the full paper dict or ``None`` on failure.
    """
    try:
        resp = requests.get(
            f"{_BASE}/papers/{doi}",
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except requests.RequestException as exc:
        logger.debug("Scite paper request failed for %s: %s", doi, exc)
        return None


def get_papers_batch(dois: list[str]) -> dict[str, dict]:
    """Fetch paper metadata for up to 500 DOIs.

    Returns ``{doi: paper_dict}``; empty dict on failure.
    """
    if not dois:
        return {}
    try:
        resp = requests.post(
            f"{_BASE}/papers",
            json={"dois": dois[:500]},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json().get("papers", {})
        return {}
    except requests.RequestException as exc:
        logger.debug("Scite batch papers request failed: %s", exc)
        return {}
