"""Shared fixtures for Zotero MCP concise mode tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_item() -> dict:
    """A full Zotero item dict with all common fields populated."""
    return {
        "key": "ABC12345",
        "data": {
            "key": "ABC12345",
            "itemType": "journalArticle",
            "title": "Deep Learning for NLP",
            "date": "2023-06-15",
            "creators": [
                {
                    "creatorType": "author",
                    "firstName": "John",
                    "lastName": "Smith",
                },
                {
                    "creatorType": "author",
                    "firstName": "Jane",
                    "lastName": "Doe",
                },
            ],
            "abstractNote": (
                "This paper presents a comprehensive survey of deep learning "
                "techniques applied to natural language processing tasks "
                "including sentiment analysis, machine translation, and more."
            ),
            "tags": [
                {"tag": "deep-learning"},
                {"tag": "nlp"},
            ],
            "publicationTitle": "Journal of AI Research",
            "volume": "42",
            "issue": "3",
            "pages": "100-150",
            "DOI": "10.1234/example",
            "url": "https://example.com/paper",
        },
    }


@pytest.fixture
def sample_item_minimal() -> dict:
    """Item with only title — no date, creators, or key."""
    return {
        "data": {
            "title": "Orphan Paper",
        },
    }


@pytest.fixture
def sample_item_single_author() -> dict:
    """Item with a single creator."""
    return {
        "key": "SINGLE01",
        "data": {
            "key": "SINGLE01",
            "title": "Solo Author Work",
            "date": "2021",
            "creators": [
                {
                    "creatorType": "author",
                    "firstName": "Alice",
                    "lastName": "Wong",
                },
            ],
        },
    }


@pytest.fixture
def sample_item_name_only_creator() -> dict:
    """Item whose creator uses 'name' instead of firstName/lastName."""
    return {
        "key": "NAME0001",
        "data": {
            "key": "NAME0001",
            "title": "Institutional Report",
            "date": "2020-01",
            "creators": [
                {
                    "creatorType": "author",
                    "name": "World Health Organization",
                },
            ],
        },
    }


@pytest.fixture
def sample_notes() -> list[dict]:
    """List of note items with HTML content and parent keys."""
    return [
        {
            "key": "NOTE0001",
            "data": {
                "key": "NOTE0001",
                "itemType": "note",
                "note": "<p>Key insight about the methodology.</p>",
                "parentItem": "ABC12345",
                "tags": [],
            },
        },
        {
            "key": "NOTE0002",
            "data": {
                "key": "NOTE0002",
                "itemType": "note",
                "note": "<p>Follow-up notes on experiments.</p>",
                "parentItem": "XYZ99999",
                "tags": [{"tag": "review"}],
            },
        },
    ]


@pytest.fixture
def mock_ctx() -> MagicMock:
    """MagicMock standing in for fastmcp.Context."""
    ctx = MagicMock()
    ctx.info = MagicMock()
    ctx.error = MagicMock()
    ctx.warn = MagicMock()
    return ctx


@pytest.fixture
def concise_env(monkeypatch: pytest.MonkeyPatch):
    """Enable concise mode via environment variable."""
    monkeypatch.setenv("ZOTERO_CONCISE_MODE", "true")


@pytest.fixture
def normal_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure concise mode is OFF."""
    monkeypatch.delenv("ZOTERO_CONCISE_MODE", raising=False)
