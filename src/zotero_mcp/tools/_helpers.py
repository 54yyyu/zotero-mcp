"""Shared private helpers used across tool modules."""

import json
import os
import re
import tempfile

import requests

from zotero_mcp import client as _client
from zotero_mcp import utils as _utils


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CROSSREF_TYPE_MAP = {
    "journal-article": "journalArticle",
    "book": "book",
    "book-chapter": "bookSection",
    "proceedings-article": "conferencePaper",
    "report": "report",
    "dissertation": "thesis",
    "posted-content": "preprint",
    "monograph": "book",
    "reference-entry": "encyclopediaArticle",
    "dataset": "document",
    "peer-review": "document",
    "edited-book": "book",
    "standard": "document",
}


# ---------------------------------------------------------------------------
# Write-operation helpers
# ---------------------------------------------------------------------------

def _get_write_client(ctx):
    """Return (read_client, write_client) for hybrid-mode operations.

    In web-only mode: both are the web client.
    In local mode with web credentials: read from local, write to web.
    In local-only mode: raises ValueError with clear message.
    """
    read_zot = _client.get_zotero_client()
    if not _utils.is_local_mode():
        return read_zot, read_zot
    web_zot = _client.get_web_zotero_client()
    if web_zot is not None:
        override = _client.get_active_library()
        if override:
            web_zot.library_id = override.get("library_id", web_zot.library_id)
            web_zot.library_type = override.get("library_type", web_zot.library_type)
        return read_zot, web_zot
    raise ValueError(
        "Cannot perform write operations in local-only mode. "
        "Add ZOTERO_API_KEY and ZOTERO_LIBRARY_ID to enable hybrid mode."
    )


def _handle_write_response(response, ctx=None):
    """Check if a pyzotero write operation succeeded."""
    if hasattr(response, "status_code"):
        ok = response.status_code in (200, 204)
        if not ok and ctx is not None:
            ctx.error(f"Write failed ({response.status_code}): {response.text[:500]}")
        return ok
    if isinstance(response, dict):
        return bool(response.get("success"))
    return bool(response)


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

def _normalize_limit(limit: int | str | None, default: int = 10, max_val: int = 100) -> int:
    """Coerce *limit* to a bounded int."""
    if limit is None:
        return default
    if isinstance(limit, str):
        limit = int(limit)
    return max(1, min(limit, max_val))


def _normalize_str_list_input(value, field_name="value"):
    """Normalize list-like user input into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
            if isinstance(parsed, str):
                s = parsed.strip()
                return [s] if s else []
            raise ValueError(
                f"{field_name} must be a list of strings or a string, "
                f"got JSON {type(parsed).__name__}"
            )
        except json.JSONDecodeError:
            pass
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) > 1:
            return parts
        return [raw]
    raise ValueError(f"{field_name} must be a list of strings or a string")


def _resolve_collection_names(zot, names, ctx=None):
    """Resolve collection names to keys (case-insensitive)."""
    if not names:
        return []
    all_collections = zot.collections()
    results = []
    for name in names:
        name_lower = name.lower()
        matches = [
            c["key"] for c in all_collections
            if c.get("data", {}).get("name", "").lower() == name_lower
        ]
        if not matches:
            raise ValueError(f"No collection found matching name '{name}'")
        if len(matches) > 1 and ctx is not None:
            ctx.warn(
                f"Multiple collections match '{name}': {matches}. "
                "Using all. Pass collection keys directly to disambiguate."
            )
        results.extend(matches)
    return results


def _normalize_doi(raw):
    """Normalize a DOI string from various input formats."""
    if not raw:
        return None
    s = raw.strip()
    if s.lower().startswith("doi:"):
        s = s[4:].strip()
    if s.lower().startswith("http://") or s.lower().startswith("https://"):
        m = re.search(r"doi\.org/(10\.\d{4,9}/[^\s?#]+)", s, flags=re.IGNORECASE)
        if not m:
            return None
        s = m.group(1)
    s = s.rstrip(".,);]")
    if re.match(r"^10\.\d{4,9}/\S+$", s):
        return s
    return None


def _normalize_arxiv_id(raw):
    """Normalize an arXiv ID from various input formats."""
    if not raw:
        return None
    s = raw.strip()
    if s.lower().startswith("arxiv:"):
        s = s[6:].strip()
    if s.lower().startswith("http://") or s.lower().startswith("https://"):
        m = re.search(
            r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z\-]+/\d{7}(?:v\d+)?)(?:\.pdf)?",
            s, flags=re.IGNORECASE,
        )
        if not m:
            return None
        s = m.group(1)
    if re.match(r"^[0-9]{4}\.[0-9]{4,5}(?:v\d+)?$", s):
        return s
    if re.match(r"^[a-z\-]+/\d{7}(?:v\d+)?$", s, flags=re.IGNORECASE):
        return s
    return None


# ---------------------------------------------------------------------------
# PDF / open-access helpers
# ---------------------------------------------------------------------------

def _download_and_attach_pdf(write_zot, item_key, pdf_url, doi, ctx):
    """Download a PDF from a URL and attach it to a Zotero item."""
    try:
        pdf_resp = requests.get(pdf_url, timeout=30, stream=True)
        pdf_resp.raise_for_status()

        content_type = pdf_resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and "octet-stream" not in content_type:
            ctx.info(f"URL did not return a PDF (Content-Type: {content_type})")
            return False

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f"{doi.replace('/', '_')}.pdf"
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, "wb") as f:
                for chunk in pdf_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            if os.path.getsize(filepath) < 1000:
                ctx.info("Downloaded file too small, likely not a real PDF")
                return False

            write_zot.attachment_both(
                [(filename, filepath)],
                parentid=item_key,
            )
        return True
    except Exception as e:
        ctx.info(f"PDF download/attach failed: {e}")
        return False


def _attach_pdf_linked_url(write_zot, pdf_url, parent_key, ctx):
    """Create a linked-URL attachment (bookmarks the PDF URL without downloading)."""
    try:
        template = write_zot.item_template("attachment", "linked_url")
        template["url"] = pdf_url
        template["title"] = "PDF (linked URL)"
        template["contentType"] = "application/pdf"
        template["parentItem"] = parent_key
        result = write_zot.create_items([template])
        if result.get("success"):
            ctx.info(f"Linked URL attachment created for {pdf_url}")
            return True
        return False
    except Exception as e:
        ctx.info(f"Linked URL attachment failed: {e}")
        return False


def _try_unpaywall(doi, ctx):
    """Try Unpaywall API for open-access PDF URLs."""
    try:
        resp = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": "zotero-mcp@users.noreply.github.com"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None

        oa_data = resp.json()

        best = oa_data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf")
        if pdf_url:
            ctx.info("Unpaywall: found PDF via best_oa_location")
            return pdf_url

        for loc in oa_data.get("oa_locations", []):
            pdf_url = loc.get("url_for_pdf")
            if pdf_url:
                ctx.info("Unpaywall: found PDF via alternate oa_location")
                return pdf_url

        landing = best.get("url")
        if landing:
            ctx.info("Unpaywall: no direct PDF URL, trying landing page")
            return landing

        return None
    except Exception as e:
        ctx.info(f"Unpaywall lookup failed: {e}")
        return None


def _try_arxiv_from_crossref(crossref_metadata, ctx):
    """Check CrossRef metadata for an arXiv ID and return a PDF URL."""
    if not crossref_metadata:
        return None
    try:
        relations = crossref_metadata.get("relation", {})
        for rel_type in ("has-preprint", "is-preprint-of", "is-identical-to",
                         "is-version-of", "has-version"):
            for rel in relations.get(rel_type, []):
                rel_id = rel.get("id", "")
                if rel.get("id-type") == "arxiv" and rel_id:
                    ctx.info(f"CrossRef relation contains arXiv ID: {rel_id}")
                    return f"https://arxiv.org/pdf/{rel_id}.pdf"
                if rel.get("id-type") == "doi" and "arxiv" in rel_id.lower():
                    m = re.search(r"arXiv\.(\d{4}\.\d{4,5}(?:v\d+)?)", rel_id, re.IGNORECASE)
                    if m:
                        arxiv_id = m.group(1)
                        ctx.info(f"CrossRef relation contains arXiv DOI: {rel_id} -> {arxiv_id}")
                        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        for alt_id in crossref_metadata.get("alternative-id", []):
            if re.match(r"\d{4}\.\d{4,5}", str(alt_id)):
                ctx.info(f"CrossRef alternative-id looks like arXiv: {alt_id}")
                return f"https://arxiv.org/pdf/{alt_id}.pdf"

        for link in crossref_metadata.get("link", []):
            url = link.get("URL", "")
            if "arxiv.org" in url:
                m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)", url)
                if m:
                    ctx.info("CrossRef link contains arXiv URL")
                    return f"https://arxiv.org/pdf/{m.group(1)}.pdf"

        return None
    except Exception as e:
        ctx.info(f"arXiv-from-CrossRef check failed: {e}")
        return None


def _try_semantic_scholar(doi, ctx):
    """Try Semantic Scholar API for an open-access PDF URL."""
    try:
        resp = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
            params={"fields": "openAccessPdf"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        oa_pdf = data.get("openAccessPdf") or {}
        pdf_url = oa_pdf.get("url")
        if pdf_url:
            ctx.info("Semantic Scholar: found OA PDF")
            return pdf_url
        return None
    except Exception as e:
        ctx.info(f"Semantic Scholar lookup failed: {e}")
        return None


def _try_pmc(doi, ctx):
    """Try PubMed Central for a free PDF via DOI-to-PMCID conversion."""
    try:
        conv_resp = requests.get(
            "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/",
            params={"ids": doi, "format": "json", "tool": "zotero-mcp",
                    "email": "zotero-mcp@users.noreply.github.com"},
            timeout=10,
        )
        if conv_resp.status_code != 200:
            return None

        records = conv_resp.json().get("records", [])
        if not records:
            return None

        pmcid = records[0].get("pmcid")
        if not pmcid:
            return None

        ctx.info(f"PMC: found PMCID {pmcid}")
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"

    except Exception as e:
        ctx.info(f"PMC lookup failed: {e}")
        return None


def _try_attach_oa_pdf(write_zot, item_key, doi, ctx, crossref_metadata=None,
                       attach_mode="auto"):
    """Attempt to find and attach an open-access PDF for a DOI."""
    sources = [
        ("Unpaywall", lambda: _try_unpaywall(doi, ctx)),
        ("arXiv (via CrossRef)", lambda: _try_arxiv_from_crossref(crossref_metadata, ctx)),
        ("Semantic Scholar", lambda: _try_semantic_scholar(doi, ctx)),
        ("PubMed Central", lambda: _try_pmc(doi, ctx)),
    ]

    for source_name, find_url in sources:
        try:
            pdf_url = find_url()
            if pdf_url:
                ctx.info(f"Trying PDF from {source_name}: {pdf_url}")

                if attach_mode == "linked_url":
                    if _attach_pdf_linked_url(write_zot, pdf_url, item_key, ctx):
                        return f"PDF linked (source: {source_name})"
                elif attach_mode == "import_file":
                    if _download_and_attach_pdf(write_zot, item_key, pdf_url, doi, ctx):
                        return f"PDF attached (source: {source_name})"
                else:  # "auto"
                    if _download_and_attach_pdf(write_zot, item_key, pdf_url, doi, ctx):
                        return f"PDF attached (source: {source_name})"
                    if _attach_pdf_linked_url(write_zot, pdf_url, item_key, ctx):
                        return f"PDF linked as URL (source: {source_name})"

                ctx.info(f"{source_name} URL didn't yield a valid PDF, trying next source")
        except Exception as e:
            ctx.info(f"{source_name} failed: {e}")

    return "no open-access PDF found (checked Unpaywall, arXiv, Semantic Scholar, PMC)"


# ---------------------------------------------------------------------------
# Citation key helpers
# ---------------------------------------------------------------------------

def _extra_has_citekey(extra: str, citekey: str) -> bool:
    """Check if the Extra field contains the given citation key."""
    for line in extra.splitlines():
        lower = line.lower().strip()
        if lower.startswith("citation key:") or lower.startswith("citationkey:"):
            value = line.split(":", 1)[1].strip()
            if value == citekey:
                return True
    return False


def _format_citekey_result(item: dict, citekey: str) -> str:
    """Format a Zotero item found by citation key as markdown."""
    extra = {"Citation Key": citekey}
    if doi := item.get("data", {}).get("DOI"):
        extra["DOI"] = doi
    lines = [f"# Citation Key: {citekey}", ""]
    lines.extend(_utils.format_item_result(item, extra_fields=extra))
    return "\n".join(lines)


def _format_bbt_result(bbt_item: dict, citekey: str) -> str:
    """Format a BetterBibTeX search result."""
    title = bbt_item.get("title", "Untitled")
    year = bbt_item.get("year", "N/A")
    creators_str = _utils.format_creators(bbt_item.get("creators", []))

    output = [
        f"# Citation Key: {citekey}",
        "",
        f"## {title}",
        f"**Citation Key:** {citekey}",
        f"**Year:** {year}",
        f"**Authors:** {creators_str}",
        "",
        "*Note: Item found via BetterBibTeX. Use the citation key with other tools for full details.*",
        "",
    ]
    return "\n".join(output)
