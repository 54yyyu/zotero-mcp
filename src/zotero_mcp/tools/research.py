"""Online scholarly-repository discovery tools."""

from __future__ import annotations

import os
from typing import Any

from zotero_mcp._app import mcp
from zotero_mcp._context import Context
from zotero_mcp.scholarly import (
    HttpClient,
    PaperSearchService,
    build_providers,
    parse_provider_names,
    provider_status,
)

_DEFAULT_LIMIT_PER_PROVIDER = 5
_MAX_LIMIT_PER_PROVIDER = 25
_DEFAULT_MAX_RESULTS = 20
_MAX_RESULTS = 100


def _bounded_int(
    value: int | str | None,
    *,
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Normalize an MCP integer argument and enforce a safe response bound."""
    if value is None or value == "":
        return default
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= normalized <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return normalized


def _normalize_year(value: int | str | None, name: str) -> int | None:
    if value is None or value == "":
        return None
    return _bounded_int(value, name=name, default=0, minimum=1000, maximum=9999)


def _service_and_provider_names(provider_selection: str | None):
    """Build a request-scoped service so environment changes take effect."""
    email = os.getenv("SCHOLAR_API_EMAIL") or None
    client = HttpClient(email=email)
    providers = build_providers(client, email=email)
    names = parse_provider_names(provider_selection or "auto", providers)
    if not names:
        raise ValueError("At least one research provider is required")
    return PaperSearchService(providers), names


def _error_payload(operation: str, error: Exception) -> dict[str, Any]:
    return {"operation": operation, "error": str(error), "total": 0, "papers": []}


@mcp.tool(
    name="zotero_search_online_papers",
    description=(
        "Search scholarly repositories outside the user's Zotero library and return "
        "normalized, deduplicated paper metadata with source provenance. Use this for "
        "literature discovery, then pass a selected DOI to zotero_add_by_doi if the user "
        "wants it saved. Searches Crossref, Semantic Scholar, Europe PMC, and arXiv by "
        "default; OpenAlex is included automatically when OPENALEX_API_KEY is configured. "
        "providers accepts 'auto', 'all', or a comma-separated subset of crossref, "
        "semantic_scholar, europe_pmc, arxiv, and openalex. Optional year bounds are "
        "inclusive. require_abstract removes records without abstracts. Results are merged "
        "by DOI/provider identifiers/title and ranked by reciprocal-rank fusion. Individual "
        "provider failures are reported without discarding successful results. Example: "
        "zotero_search_online_papers(query='multilingual retrieval augmented generation', "
        "from_year=2023, max_results=15, require_abstract=True)."
    ),
)
def search_online_papers(
    query: str,
    providers: str = "auto",
    limit_per_provider: int | str | None = _DEFAULT_LIMIT_PER_PROVIDER,
    max_results: int | str | None = _DEFAULT_MAX_RESULTS,
    from_year: int | str | None = None,
    to_year: int | str | None = None,
    require_abstract: bool = False,
    *,
    ctx: Context,
) -> dict[str, Any]:
    """Search multiple online scholarly indexes concurrently."""
    try:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if len(normalized_query) > 500:
            raise ValueError("query must be 500 characters or fewer")

        per_provider = _bounded_int(
            limit_per_provider,
            name="limit_per_provider",
            default=_DEFAULT_LIMIT_PER_PROVIDER,
            minimum=1,
            maximum=_MAX_LIMIT_PER_PROVIDER,
        )
        total_limit = _bounded_int(
            max_results,
            name="max_results",
            default=_DEFAULT_MAX_RESULTS,
            minimum=1,
            maximum=_MAX_RESULTS,
        )
        year_from = _normalize_year(from_year, "from_year")
        year_to = _normalize_year(to_year, "to_year")
        if year_from is not None and year_to is not None and year_from > year_to:
            raise ValueError("from_year must be less than or equal to to_year")

        service, provider_names = _service_and_provider_names(providers)
        ctx.info(
            f"Searching {len(provider_names)} research repositories for: {normalized_query}"
        )
        return service.search(
            normalized_query,
            provider_names,
            limit_per_provider=per_provider,
            max_results=total_limit,
            year_from=year_from,
            year_to=year_to,
            require_abstract=bool(require_abstract),
        )
    except Exception as exc:
        ctx.error(f"Online paper search failed: {exc}")
        return _error_payload("search", exc)


@mcp.tool(
    name="zotero_lookup_online_paper",
    description=(
        "Look up and reconcile one paper by DOI across online scholarly repositories. "
        "Use this when a DOI is known and complete metadata, an abstract, open-access PDF "
        "link, citation count, or cross-provider provenance is needed before adding or "
        "updating a Zotero item. Crossref, Semantic Scholar, Europe PMC, and arXiv are used "
        "by default; configured OpenAlex can be selected with providers='auto' or forced "
        "with providers='all'. providers also accepts a comma-separated subset. "
        "require_abstract suppresses a merged record if no provider supplies an abstract. "
        "A provider's 404 is treated as no match, while other failures appear in errors. "
        "Example: zotero_lookup_online_paper(doi='10.18653/v1/N18-3011')."
    ),
)
def lookup_online_paper(
    doi: str,
    providers: str = "auto",
    require_abstract: bool = False,
    *,
    ctx: Context,
) -> dict[str, Any]:
    """Enrich a DOI using all selected repository providers concurrently."""
    try:
        service, provider_names = _service_and_provider_names(providers)
        ctx.info(f"Looking up DOI across {len(provider_names)} research repositories")
        return service.lookup_doi(
            doi,
            provider_names,
            require_abstract=bool(require_abstract),
        )
    except Exception as exc:
        ctx.error(f"Online DOI lookup failed: {exc}")
        return _error_payload("lookup_doi", exc)


@mcp.tool(
    name="zotero_get_research_provider_status",
    description=(
        "List the online scholarly repositories available to Zotero MCP and whether each "
        "is configured. Use this before a search when provider coverage or API-key setup "
        "matters. Crossref, Semantic Scholar, Europe PMC, and arXiv work without keys; a "
        "Semantic Scholar key improves rate limits, and OpenAlex requires OPENALEX_API_KEY. "
        "The response reports configuration booleans only and never exposes credential "
        "values. Example: zotero_get_research_provider_status()."
    ),
)
def get_research_provider_status(*, ctx: Context) -> dict[str, Any]:
    """Return provider capabilities without making network requests."""
    try:
        email = os.getenv("SCHOLAR_API_EMAIL") or None
        providers = build_providers(HttpClient(email=email), email=email)
        return provider_status(providers)
    except Exception as exc:
        ctx.error(f"Could not inspect research providers: {exc}")
        return _error_payload("provider_status", exc)
