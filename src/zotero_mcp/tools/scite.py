"""Scite citation intelligence tools for the Zotero MCP server.

The MCP counterpart of the Scite Zotero Plugin
(https://github.com/scitedotai/scite-zotero-plugin).  Where the desktop
plugin adds tally columns to the Zotero UI, these tools surface the same
data — supporting/contrasting/mentioning citation counts and retraction
alerts — inside MCP-powered AI assistants (Claude, ChatGPT, etc.).

No API key or Scite account is required.  All data comes from Scite's
public endpoints.

For Smart Citations, gap analysis, and full-text search, see the
Scite MCP server: https://scite.ai/mcp
"""

from __future__ import annotations

import logging

from zotero_mcp._context import Context
from zotero_mcp import client as _client
from zotero_mcp import scite_client as _scite
from zotero_mcp import utils as _utils
from zotero_mcp._app import mcp
from zotero_mcp.tools import _helpers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_doi(item: dict) -> str | None:
    """Extract and normalize DOI from a Zotero item."""
    doi = item.get("data", {}).get("DOI", "")
    if doi:
        return _helpers._normalize_doi(doi)
    extra = item.get("data", {}).get("extra", "")
    for line in extra.splitlines():
        if line.lower().strip().startswith("doi:"):
            candidate = line.split(":", 1)[1].strip()
            return _helpers._normalize_doi(candidate)
    return None


def _format_tally_line(tally: dict) -> str:
    """Format a tally dict as a compact inline string."""
    s = tally.get("supporting", 0)
    c = tally.get("contradicting", 0)
    m = tally.get("mentioning", 0)
    total = tally.get("citingPublications", tally.get("total", s + c + m))
    return f"Supporting: {s} | Contrasting: {c} | Mentioning: {m} (total citing: {total})"


def _format_editorial_notices(notices: list[dict]) -> list[str]:
    """Format editorial notices as warning lines."""
    lines = []
    for notice in notices:
        ntype = notice.get("type", notice.get("editorialNoticeType", "notice"))
        ntype = ntype.replace("_", " ").title()
        source_doi = notice.get("sourceDoi", notice.get("source", ""))
        lines.append(f"**{ntype}**: https://doi.org/{source_doi}")
    return lines


def enrich_items(items: list[dict]) -> dict[str, dict[str, str]]:
    """Batch-enrich a list of Zotero items with Scite data.

    Returns ``{doi: extra_fields_dict}`` suitable for passing to
    ``format_item_result(extra_fields=...)``.
    """
    doi_map: dict[str, str] = {}
    for item in items:
        doi = _extract_doi(item)
        if doi:
            doi_map[doi] = doi

    if not doi_map:
        return {}

    dois = list(doi_map.keys())
    tallies = _scite.get_tallies_batch(dois)
    papers = _scite.get_papers_batch(dois)

    result: dict[str, dict[str, str]] = {}
    for doi in dois:
        fields: dict[str, str] = {}
        tally = tallies.get(doi)
        if tally:
            fields["Scite"] = _format_tally_line(tally)

        paper = papers.get(doi)
        if paper:
            notices = paper.get("editorialNotices", [])
            if notices:
                notice_strs = _format_editorial_notices(notices)
                fields["Editorial Notices"] = "; ".join(notice_strs)

        if fields:
            result[doi] = fields

    return result


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool(
    name="scite_enrich_item",
    description=(
        "Get a Scite citation report for a paper — supporting, contrasting, "
        "and mentioning citation counts plus retraction/correction alerts. "
        "Provide either a DOI or a Zotero item key. No Scite account needed."
    ),
)
def enrich_item(
    doi: str | None = None,
    item_key: str | None = None,
    *,
    ctx: Context,
) -> str:
    """Enrich a single item with Scite citation data."""
    try:
        if not doi and not item_key:
            return "Error: provide either a DOI or a Zotero item_key"

        # Resolve DOI from Zotero item if needed
        if not doi and item_key:
            ctx.info(f"Looking up DOI for Zotero item {item_key}")
            zot = _client.get_zotero_client()
            item = zot.item(item_key)
            if not item:
                return f"Error: Zotero item '{item_key}' not found"
            doi = _extract_doi(item)
            if not doi:
                return f"Error: no DOI found for Zotero item '{item_key}'"

        doi = _helpers._normalize_doi(doi) or doi
        ctx.info(f"Fetching Scite data for {doi}")

        # Fetch tally + paper metadata in parallel-ish (same thread, two requests)
        tally = _scite.get_tally(doi)
        paper = _scite.get_paper(doi)

        if tally is None and paper is None:
            return f"No Scite data found for DOI: {doi}"

        title = (paper or {}).get("title", doi)
        output = [f"# Scite Report: {title}", f"**DOI:** https://doi.org/{doi}", ""]

        # Tally
        if tally:
            output.append("## Citation Tally")
            output.append(f"- **Supporting:** {tally.get('supporting', 0)}")
            output.append(f"- **Contrasting:** {tally.get('contradicting', 0)}")
            output.append(f"- **Mentioning:** {tally.get('mentioning', 0)}")
            output.append(
                f"- **Total citing publications:** "
                f"{tally.get('citingPublications', 'N/A')}"
            )
            output.append("")

        # Editorial notices
        notices = (paper or {}).get("editorialNotices", [])
        if notices:
            output.append("## Editorial Notices")
            for notice in notices:
                ntype = notice.get("type", notice.get("editorialNoticeType", "notice"))
                ntype = ntype.replace("_", " ").title()
                source = notice.get("sourceDoi", notice.get("source", ""))
                output.append(f"- **{ntype}**: https://doi.org/{source}")
            output.append("")

        output.append(
            "---\n"
            "*Powered by [scite.ai](https://scite.ai). "
            "For Smart Citations and gap analysis, see the "
            "[Scite MCP](https://scite.ai/mcp).*"
        )

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error enriching item: {e}")
        return f"Error enriching item: {e}"


@mcp.tool(
    name="scite_enrich_search",
    description=(
        "Search your Zotero library and enrich results with Scite citation "
        "data. Each result shows supporting/contrasting/mentioning tallies "
        "and retraction alerts alongside standard Zotero metadata. "
        "No Scite account needed."
    ),
)
def enrich_search(
    query: str,
    limit: int | str = 10,
    *,
    ctx: Context,
) -> str:
    """Search Zotero and enrich results with Scite tallies."""
    try:
        if not query.strip():
            return "Error: search query cannot be empty"

        zot = _client.get_zotero_client()
        limit_int = _helpers._normalize_limit(limit, default=10)

        ctx.info(f"Searching Zotero for '{query}' and enriching with Scite data")
        zot.add_parameters(
            q=query,
            qmode="titleCreatorYear",
            itemType="-attachment",
            limit=limit_int,
        )
        results = zot.items()

        if not results:
            return f"No items found matching query: '{query}'"

        # Batch-enrich with Scite
        enrichment = enrich_items(results)

        output = [f"# Search Results for '{query}' (Scite-enriched)", ""]

        for i, item in enumerate(results, 1):
            doi = _extract_doi(item)
            extra = enrichment.get(doi, {}) if doi else {}
            output.extend(
                _utils.format_item_result(item, index=i, extra_fields=extra)
            )

        items_with_scite = sum(
            1 for item in results if _extract_doi(item) in enrichment
        )
        output.append(
            f"---\n*Scite data shown for {items_with_scite}/{len(results)} items "
            f"— powered by [scite.ai](https://scite.ai).*"
        )

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error in enriched search: {e}")
        return f"Error in enriched search: {e}"


@mcp.tool(
    name="scite_check_retractions",
    description=(
        "Scan items in your Zotero library for retractions, corrections, "
        "and other editorial notices using Scite data. Filter by collection, "
        "tag, or check recent items. No Scite account needed."
    ),
)
def check_retractions(
    collection: str | None = None,
    tag: str | None = None,
    limit: int | str = 50,
    *,
    ctx: Context,
) -> str:
    """Check Zotero items for editorial notices (retractions, corrections)."""
    try:
        zot = _client.get_zotero_client()
        limit_int = _helpers._normalize_limit(limit, default=50, max_val=500)

        # Fetch items
        if collection:
            ctx.info(f"Checking collection '{collection}' for retractions")
            keys = _helpers._resolve_collection_names(zot, [collection], ctx)
            if not keys:
                return f"Collection '{collection}' not found"
            items = zot.collection_items(
                keys[0], limit=limit_int, itemType="-attachment"
            )
        elif tag:
            ctx.info(f"Checking items tagged '{tag}' for retractions")
            zot.add_parameters(tag=tag, itemType="-attachment", limit=limit_int)
            items = zot.items()
        else:
            ctx.info("Checking recent items for retractions")
            items = zot.items(
                sort="dateModified",
                direction="desc",
                limit=limit_int,
                itemType="-attachment",
            )

        if not items:
            return "No items found to check."

        # Extract DOIs
        doi_items: dict[str, dict] = {}
        for item in items:
            doi = _extract_doi(item)
            if doi:
                doi_items[doi] = item

        if not doi_items:
            return f"None of the {len(items)} items have DOIs — cannot check Scite."

        ctx.info(f"Checking {len(doi_items)} DOIs against Scite editorial notices")
        papers = _scite.get_papers_batch(list(doi_items.keys()))

        if not papers:
            return "Could not reach Scite API — try again later."

        # Find items with notices
        flagged: list[tuple[dict, list[dict]]] = []
        for doi, item in doi_items.items():
            paper = papers.get(doi, {})
            notices = paper.get("editorialNotices", [])
            if notices:
                flagged.append((item, notices))

        if not flagged:
            return (
                f"All clear! Checked {len(doi_items)} items with DOIs — "
                "no retractions or editorial notices found."
            )

        output = [
            "# Editorial Notice Alerts",
            f"Found **{len(flagged)}** item(s) with editorial notices "
            f"(out of {len(doi_items)} checked):",
            "",
        ]

        for i, (item, notices) in enumerate(flagged, 1):
            data = item.get("data", {})
            title = data.get("title", "Untitled")
            item_doi = data.get("DOI", "")
            output.append(f"## {i}. {title}")
            output.append(f"**DOI:** {item_doi}")
            for notice in notices:
                ntype = notice.get("type", notice.get("editorialNoticeType", "notice"))
                ntype = ntype.replace("_", " ").title()
                source = notice.get("sourceDoi", notice.get("source", ""))
                output.append(f"- **{ntype}**: https://doi.org/{source}")
            output.append("")

        output.append(
            "---\n*Powered by [scite.ai](https://scite.ai).*"
        )

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error checking retractions: {e}")
        return f"Error checking retractions: {e}"
