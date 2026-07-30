"""Tests for online scholarly-repository MCP tools."""

from __future__ import annotations

import asyncio

from zotero_mcp._app import mcp
from zotero_mcp.tools import research


class DummyContext:
    def __init__(self):
        self.info_messages = []
        self.error_messages = []

    def info(self, message):
        self.info_messages.append(message)

    def error(self, message):
        self.error_messages.append(message)


class FakeService:
    def __init__(self):
        self.search_call = None
        self.lookup_call = None

    def search(self, query, provider_names, **kwargs):
        self.search_call = (query, provider_names, kwargs)
        return {
            "operation": "search",
            "query": query,
            "total": 1,
            "papers": [{"title": "A result"}],
        }

    def lookup_doi(self, doi, provider_names, **kwargs):
        self.lookup_call = (doi, provider_names, kwargs)
        return {
            "operation": "lookup_doi",
            "doi": doi,
            "total": 1,
            "papers": [{"title": "A result"}],
        }


def test_research_tools_are_registered_with_mcp():
    tool_names = {tool.name for tool in asyncio.run(mcp.list_tools())}

    assert {
        "zotero_search_online_papers",
        "zotero_lookup_online_paper",
        "zotero_get_research_provider_status",
    } <= tool_names


def test_search_tool_forwards_normalized_arguments(monkeypatch):
    service = FakeService()
    monkeypatch.setattr(
        research,
        "_service_and_provider_names",
        lambda value: (service, ["crossref", "arxiv"]),
    )
    ctx = DummyContext()

    result = research.search_online_papers(
        "  multilingual retrieval  ",
        providers="crossref,arxiv",
        limit_per_provider="4",
        max_results="7",
        from_year="2022",
        to_year=2026,
        require_abstract=True,
        ctx=ctx,
    )

    assert result["total"] == 1
    assert service.search_call == (
        "multilingual retrieval",
        ["crossref", "arxiv"],
        {
            "limit_per_provider": 4,
            "max_results": 7,
            "year_from": 2022,
            "year_to": 2026,
            "require_abstract": True,
        },
    )
    assert ctx.info_messages
    assert not ctx.error_messages


def test_search_tool_returns_structured_validation_error(monkeypatch):
    monkeypatch.setattr(
        research,
        "_service_and_provider_names",
        lambda value: (_ for _ in ()).throw(AssertionError("should not build service")),
    )
    ctx = DummyContext()

    result = research.search_online_papers(
        "query",
        from_year=2026,
        to_year=2020,
        ctx=ctx,
    )

    assert result == {
        "operation": "search",
        "error": "from_year must be less than or equal to to_year",
        "total": 0,
        "papers": [],
    }
    assert ctx.error_messages


def test_lookup_tool_forwards_doi_and_provider_selection(monkeypatch):
    service = FakeService()
    monkeypatch.setattr(
        research,
        "_service_and_provider_names",
        lambda value: (service, ["europe_pmc"]),
    )
    ctx = DummyContext()

    result = research.lookup_online_paper(
        "https://doi.org/10.1234/Example",
        providers="europe_pmc",
        require_abstract=True,
        ctx=ctx,
    )

    assert result["total"] == 1
    assert service.lookup_call == (
        "https://doi.org/10.1234/Example",
        ["europe_pmc"],
        {"require_abstract": True},
    )


def test_provider_status_does_not_expose_secret_values(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "secret-semantic-key")
    monkeypatch.setenv("OPENALEX_API_KEY", "secret-openalex-key")
    monkeypatch.setenv("SCHOLAR_API_EMAIL", "researcher@example.com")
    ctx = DummyContext()

    result = research.get_research_provider_status(ctx=ctx)

    assert result["environment"] == {
        "contact_email_set": True,
        "semantic_scholar_key_set": True,
        "openalex_key_set": True,
    }
    assert "secret-semantic-key" not in repr(result)
    assert "secret-openalex-key" not in repr(result)
