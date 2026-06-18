"""Discovery tools: find related papers via OpenAlex and assess library coverage."""

import re
from typing import Literal

import requests

from zotero_mcp import client as _client
from zotero_mcp import utils as _utils  # noqa: F401  (kept for module-level conventions)
from zotero_mcp._app import mcp
from zotero_mcp._context import Context
from zotero_mcp.client import with_zotero_api_lock
from zotero_mcp.tools import _helpers

_OPENALEX_BASE = "https://api.openalex.org"
_MAILTO = "zotero-mcp@users.noreply.github.com"
_ITEM_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")
_HTTP_TIMEOUT = 30


def _doi_in_library(zot, doi: str) -> bool:
    """Best-effort membership check: is a paper with this DOI already in Zotero?

    Tolerates any pyzotero error by returning False (treat as "not in library").
    """
    if not doi:
        return False
    try:
        zot.add_parameters(q=doi, qmode="everything", itemType="-attachment", limit=5)
        results = zot.items()
    except Exception:
        try:
            results = zot.items(q=doi, qmode="everything", itemType="-attachment", limit=5)
        except Exception:
            return False
    norm = doi.strip().lower()
    for item in results or []:
        item_doi = str(item.get("data", {}).get("DOI", "")).strip().lower()
        if item_doi and item_doi == norm:
            return True
    return False


def _short_id(openalex_id: str) -> str:
    """Reduce a full OpenAlex URL/ID to its bare 'Wxxxx' form."""
    if not openalex_id:
        return ""
    return openalex_id.rstrip("/").rsplit("/", 1)[-1]


def _work_summary(work: dict) -> dict:
    """Extract a compact summary from an OpenAlex work object."""
    title = work.get("title") or work.get("display_name") or "Untitled"
    year = work.get("publication_year")
    doi = _helpers._normalize_doi(work.get("doi")) or ""
    cited_by = work.get("cited_by_count", 0) or 0

    authors = []
    for authorship in (work.get("authorships") or [])[:3]:
        author = authorship.get("author") or {}
        name = author.get("display_name")
        if name:
            authors.append(name)
    if len(work.get("authorships") or []) > 3:
        authors.append("et al.")

    return {
        "title": title,
        "year": year,
        "doi": doi,
        "cited_by": cited_by,
        "authors": authors,
    }


def _openalex_get(url: str, params: dict | None = None) -> dict | None:
    """GET an OpenAlex endpoint, returning parsed JSON or None on any failure."""
    p = {"mailto": _MAILTO}
    if params:
        p.update(params)
    try:
        resp = requests.get(url, params=p, timeout=_HTTP_TIMEOUT)
    except Exception:
        return None
    if getattr(resp, "status_code", None) != 200:
        return None
    try:
        return resp.json()
    except Exception:
        return None


def _resolve_doi(identifier: str, zot) -> str | None:
    """Resolve an identifier (Zotero item key or DOI/URL) to a normalized DOI."""
    ident = (identifier or "").strip()
    if not ident:
        return None
    if _ITEM_KEY_RE.match(ident):
        try:
            item = zot.item(ident)
        except Exception:
            return None
        raw_doi = (item or {}).get("data", {}).get("DOI")
        return _helpers._normalize_doi(raw_doi)
    return _helpers._normalize_doi(ident)


def _render_related(papers: list[dict], heading: str) -> list[str]:
    """Render a list of related-paper summaries as markdown lines."""
    lines = [f"## {heading} ({len(papers)})", ""]
    if not papers:
        lines.append("_None found._")
        lines.append("")
        return lines
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p["authors"]) if p["authors"] else "Unknown authors"
        year = p["year"] if p["year"] else "n.d."
        marker = "in library ✓" if p.get("in_library") else "not in library"
        lines.append(f"{i}. **{p['title']}** ({year})")
        lines.append(f"   - Authors: {authors}")
        if p["doi"]:
            lines.append(f"   - DOI: {p['doi']}")
        lines.append(f"   - Cited by: {p['cited_by']}")
        lines.append(f"   - {marker}")
        lines.append("")
    return lines


@mcp.tool(
    name="zotero_find_related_papers",
    description=(
        "Discover papers related to a known work by following its citation "
        "graph via OpenAlex (a free scholarly index). Use this to expand a "
        "literature review: find what a paper CITES (its references) and what "
        "CITES it (newer follow-up work). Each related paper is flagged as "
        "already in your Zotero library or not, so you can quickly spot gaps "
        "to fetch (e.g. via zotero_add_by_doi). "
        "identifier: either an 8-char Zotero item key (its DOI is looked up) "
        "or a DOI / DOI-URL directly (e.g. '10.1038/nature12373' or "
        "'https://doi.org/10.1038/nature12373'). "
        "direction: 'references' (works this paper cites), 'citations' (works "
        "citing this paper), or 'both' (default). "
        "limit: max related papers per direction (default 20, max 50). "
        "Citations are sorted by citation count (most-cited first); references "
        "keep their original order. Requires the work to have a resolvable "
        "DOI present in OpenAlex. "
        "Example: zotero_find_related_papers(identifier='10.1038/nature12373', "
        "direction='citations', limit=10)."
    ),
)
@with_zotero_api_lock
def find_related_papers(
    identifier: str,
    direction: Literal["references", "citations", "both"] = "both",
    limit: int | str | None = 20,
    *,
    ctx: Context,
) -> str:
    """Find references and/or citing works for a paper via OpenAlex."""
    try:
        if direction not in {"references", "citations", "both"}:
            return "Error: direction must be 'references', 'citations', or 'both'."

        limit = _helpers._normalize_limit(limit, default=20, max_val=50)
        zot = _client.get_zotero_client()

        ctx.info(f"Resolving identifier to DOI: {identifier}")
        doi = _resolve_doi(identifier, zot)
        if not doi:
            return (
                f"Could not resolve a DOI for '{identifier}'. Provide a valid "
                "DOI / DOI-URL, or an 8-char Zotero item key whose item has a "
                "DOI in its metadata."
            )

        ctx.info(f"Querying OpenAlex for DOI {doi}")
        work = _openalex_get(f"{_OPENALEX_BASE}/works/https://doi.org/{doi}")
        if not work:
            return f"OpenAlex has no record for DOI '{doi}', or the lookup failed."

        want_refs = direction in {"references", "both"}
        want_cites = direction in {"citations", "both"}

        references: list[dict] = []
        citations: list[dict] = []

        if want_refs:
            ref_ids = [_short_id(r) for r in (work.get("referenced_works") or [])]
            ref_ids = [r for r in ref_ids if r][:limit]
            if ref_ids:
                filter_val = "openalex_id:" + "|".join(ref_ids)
                data = _openalex_get(
                    f"{_OPENALEX_BASE}/works",
                    {"filter": filter_val, "per-page": min(len(ref_ids), 50)},
                )
                results = (data or {}).get("results", []) or []
                # Preserve referenced_works order.
                by_id = {_short_id(w.get("id")): w for w in results}
                for rid in ref_ids:
                    if rid in by_id:
                        references.append(_work_summary(by_id[rid]))

        if want_cites:
            cited_by_url = work.get("cited_by_api_url")
            if cited_by_url:
                data = _openalex_get(cited_by_url, {"per-page": min(limit, 50)})
                results = (data or {}).get("results", []) or []
                citations = [_work_summary(w) for w in results]
                citations.sort(key=lambda p: p["cited_by"], reverse=True)
                citations = citations[:limit]

        # Flag library membership for every related paper.
        for p in references + citations:
            p["in_library"] = _doi_in_library(zot, p["doi"]) if p["doi"] else False

        src_title = work.get("title") or work.get("display_name") or doi
        output = [
            f"# Related Papers for: {src_title}",
            "",
            f"Source DOI: {doi}",
        ]
        summary_bits = []
        if want_refs:
            summary_bits.append(f"{len(references)} references")
        if want_cites:
            summary_bits.append(f"{len(citations)} citations")
        in_lib = sum(1 for p in references + citations if p.get("in_library"))
        summary_bits.append(f"{in_lib} already in library")
        output.append("Found " + ", ".join(summary_bits) + ".")
        output.append("")

        if want_refs:
            output.extend(_render_related(references, "References (works this paper cites)"))
        if want_cites:
            output.extend(_render_related(citations, "Citations (works citing this paper)"))

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error finding related papers: {e}")
        return f"Error finding related papers: {e}"


def _item_has_pdf(zot, item: dict) -> bool:
    """Return True if the item is/has a PDF attachment.

    A standalone PDF attachment counts directly; otherwise we inspect the
    item's children for any PDF attachment. Tolerant of children() errors.
    """
    data = item.get("data", {})
    if data.get("itemType") == "attachment" and data.get("contentType") == "application/pdf":
        return True
    key = item.get("key") or data.get("key")
    if not key:
        return False
    try:
        children = zot.children(key)
    except Exception:
        return False
    for child in children or []:
        cdata = child.get("data", {})
        if cdata.get("itemType") == "attachment" and cdata.get("contentType") == "application/pdf":
            return True
    return False


@mcp.tool(
    name="zotero_library_coverage",
    description=(
        "Audit PDF coverage across your Zotero library (or one collection): "
        "which items have a downloaded PDF attachment and which are missing "
        "one. Use this to find papers you can still fetch full text for — the "
        "missing list includes each item's DOI so you can pass it to "
        "zotero_add_by_doi's open-access download cascade. "
        "collection_key: optional 8-char key to scope the audit to one "
        "collection; omit to scan the whole library. "
        "limit: max top-level items to scan (default 200). "
        "Attachments, notes, and annotations are skipped as scan targets; an "
        "item counts as covered if it (or any child) is a PDF attachment. "
        "Reports total scanned, covered count, missing count, coverage "
        "percentage, and a capped list (first 50) of missing items with "
        "title, year, key, and DOI. "
        "Example: zotero_library_coverage(collection_key='ABCD1234', "
        "limit=100)."
    ),
)
@with_zotero_api_lock
def library_coverage(
    collection_key: str | None = None,
    limit: int | str | None = 200,
    *,
    ctx: Context,
) -> str:
    """Report PDF-attachment coverage across the library or a collection."""
    try:
        limit = _helpers._normalize_limit(limit, default=200, max_val=2000)
        zot = _client.get_zotero_client()

        skip_types = {"attachment", "note", "annotation"}

        ctx.info("Scanning library for PDF coverage...")
        if collection_key:
            items = _helpers._paginate(
                zot.collection_items,
                collection_key,
                max_items=limit,
                itemType="-attachment",
            )
        else:
            items = _helpers._paginate(
                zot.items,
                max_items=limit,
                itemType="-attachment",
            )

        scanned = 0
        with_pdf = 0
        missing: list[dict] = []

        for item in items or []:
            data = item.get("data", {})
            item_type = data.get("itemType")
            # A standalone PDF attachment is a valid scan target even though
            # we excluded attachments at the API level (defensive).
            is_standalone_pdf = item_type == "attachment" and data.get("contentType") == "application/pdf"
            if item_type in skip_types and not is_standalone_pdf:
                continue

            scanned += 1
            if _item_has_pdf(zot, item):
                with_pdf += 1
            else:
                title = data.get("title") or data.get("filename") or "Untitled"
                year = str(data.get("date", ""))[:4]
                missing.append(
                    {
                        "title": title,
                        "year": year,
                        "key": item.get("key", ""),
                        "doi": data.get("DOI", ""),
                    }
                )

        missing_count = len(missing)
        pct = (with_pdf / scanned * 100) if scanned else 0.0

        scope = f"collection {collection_key}" if collection_key else "entire library"
        output = [
            f"# PDF Coverage Report ({scope})",
            "",
            f"- Items scanned: {scanned}",
            f"- With PDF: {with_pdf}",
            f"- Missing PDF: {missing_count}",
            f"- Coverage: {pct:.1f}%",
            "",
        ]

        if missing:
            shown = missing[:50]
            output.append(f"## Items Missing a PDF (showing {len(shown)} of {missing_count})")
            output.append("")
            for i, m in enumerate(shown, 1):
                year = m["year"] if m["year"] else "n.d."
                line = f"{i}. **{m['title']}** ({year}) — key: {m['key']}"
                if m["doi"]:
                    line += f" — DOI: {m['doi']}"
                output.append(line)
            output.append("")
        else:
            output.append("All scanned items have a PDF attachment. ✓")

        return "\n".join(output)

    except Exception as e:
        ctx.error(f"Error computing library coverage: {e}")
        return f"Error computing library coverage: {e}"
