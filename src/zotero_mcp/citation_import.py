"""Parsers and mappers that convert BibTeX / CSL JSON into Zotero item dicts.

Reference crosswalk for type and field mappings:
https://aurimasv.github.io/z2csl/typeMap.xml

Entry point functions:
- ``parse_bibtex(text)`` -> list of parsed entry dicts
- ``bibtex_entry_to_zotero(entry, template_fn)`` -> Zotero item dict
- ``csl_json_to_zotero(csl, template_fn)`` -> Zotero item dict

The converters never mutate ``template_fn``'s output; they return a fresh dict
populated only with fields valid for the resolved Zotero item type. Unmapped
source fields are preserved as labelled lines in ``extra`` so nothing is lost
silently.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode

# ---------------------------------------------------------------------------
# Type maps
# ---------------------------------------------------------------------------

BIBTEX_TYPE_MAP = {
    "article": "journalArticle",
    "book": "book",
    "booklet": "book",
    "inbook": "bookSection",
    "incollection": "bookSection",
    "suppbook": "bookSection",
    "suppcollection": "bookSection",
    "inproceedings": "conferencePaper",
    "conference": "conferencePaper",
    "proceedings": "book",
    "phdthesis": "thesis",
    "mastersthesis": "thesis",
    "thesis": "thesis",
    "techreport": "report",
    "report": "report",
    "manual": "document",
    "misc": "document",
    "unpublished": "manuscript",
    "patent": "patent",
    "online": "webpage",
    "electronic": "webpage",
    "webpage": "webpage",
    "software": "computerProgram",
    "dataset": "document",
    "artwork": "artwork",
    "audio": "audioRecording",
    "video": "videoRecording",
    "letter": "letter",
    "standard": "document",
    "periodical": "journalArticle",
    "collection": "book",
}

CSL_TYPE_MAP = {
    "article-journal": "journalArticle",
    "article-magazine": "magazineArticle",
    "article-newspaper": "newspaperArticle",
    "article": "preprint",
    "book": "book",
    "chapter": "bookSection",
    "paper-conference": "conferencePaper",
    "thesis": "thesis",
    "report": "report",
    "webpage": "webpage",
    "post-weblog": "blogPost",
    "post": "forumPost",
    "patent": "patent",
    "manuscript": "manuscript",
    "dataset": "document",
    "entry-encyclopedia": "encyclopediaArticle",
    "entry-dictionary": "dictionaryEntry",
    "speech": "presentation",
    "interview": "interview",
    "personal_communication": "letter",
    "broadcast": "radioBroadcast",
    "motion_picture": "film",
    "song": "audioRecording",
    "map": "map",
    "legal_case": "case",
    "legislation": "statute",
    "bill": "bill",
    "software": "computerProgram",
    "figure": "artwork",
    "graphic": "artwork",
    "pamphlet": "document",
    "review": "journalArticle",
    "review-book": "journalArticle",
    "treaty": "document",
}


# ---------------------------------------------------------------------------
# Author / creator parsing
# ---------------------------------------------------------------------------

_CORP_SUFFIXES = (
    " inc", " inc.", " llc", " ltd", " ltd.", " corp", " corp.",
    " gmbh", " ag", " plc", " co.", " co ", " s.a.", " s.a",
    " university", " universität", " universite", " universidad",
    " institute", " institut", " academy", " laboratory", " labs",
    " consortium", " foundation", " association", " society",
    " organization", " organisation", " committee", " council",
    " group", " team",
)


def _looks_corporate(name: str) -> bool:
    lower = " " + name.lower() + " "
    return any(suffix in lower for suffix in _CORP_SUFFIXES)


def _parse_bibtex_author_list(raw: str) -> list[dict[str, str]]:
    """Split a BibTeX authors string on ' and ' and structure each name."""
    if not raw:
        return []
    # Preserve braces inside { ... } groups: split on " and " only at top level.
    parts = _split_bibtex_authors(raw)
    creators = []
    for part in parts:
        name = part.strip().strip("{}").strip()
        if not name:
            continue
        if "," in name and name.count(",") == 1 and not _looks_corporate(name):
            last, first = [x.strip() for x in name.split(",", 1)]
            creators.append({
                "creatorType": "author",
                "firstName": first,
                "lastName": last,
            })
        elif " " in name and not _looks_corporate(name):
            first, last = name.rsplit(" ", 1)
            creators.append({
                "creatorType": "author",
                "firstName": first.strip(),
                "lastName": last.strip(),
            })
        else:
            creators.append({"creatorType": "author", "name": name})
    return creators


def _split_bibtex_authors(raw: str) -> list[str]:
    """Split on ' and ' while respecting brace groups."""
    out = []
    buf = []
    depth = 0
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif depth == 0 and raw[i:i + 5].lower() == " and ":
            out.append("".join(buf))
            buf = []
            i += 5
            continue
        else:
            buf.append(ch)
        i += 1
    if buf:
        out.append("".join(buf))
    return out


def _csl_names_to_creators(names: list[dict], creator_type: str) -> list[dict]:
    out = []
    for n in names or []:
        if not isinstance(n, dict):
            continue
        if "literal" in n and n["literal"]:
            out.append({"creatorType": creator_type, "name": str(n["literal"]).strip()})
        elif "family" in n or "given" in n:
            entry = {"creatorType": creator_type}
            given = (n.get("given") or "").strip()
            family = (n.get("family") or "").strip()
            if given:
                entry["firstName"] = given
            if family:
                entry["lastName"] = family
            if "firstName" not in entry and "lastName" not in entry:
                continue
            # pyzotero expects both firstName and lastName — fill blanks
            entry.setdefault("firstName", "")
            entry.setdefault("lastName", "")
            out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

_MONTH_NAMES = {
    "jan": "01", "january": "01", "feb": "02", "february": "02",
    "mar": "03", "march": "03", "apr": "04", "april": "04",
    "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
    "aug": "08", "august": "08", "sep": "09", "september": "09",
    "oct": "10", "october": "10", "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}


def _format_bibtex_date(year: str, month: str, day: str, iso_date: str) -> str:
    """Build an ISO-ish date string from bibtex fields."""
    if iso_date:
        return iso_date.strip()
    year = (year or "").strip()
    month = (month or "").strip()
    day = (day or "").strip()
    if not year:
        return ""
    month_num = _MONTH_NAMES.get(month.lower()[:3]) if month and not month.isdigit() else (
        month.zfill(2) if month else ""
    )
    parts = [year]
    if month_num:
        parts.append(month_num)
        if day and day.isdigit():
            parts.append(day.zfill(2))
    return "-".join(parts)


def _format_csl_date(issued: Any) -> str:
    if not isinstance(issued, dict):
        return ""
    if "literal" in issued and issued["literal"]:
        return str(issued["literal"]).strip()
    dp = issued.get("date-parts")
    if isinstance(dp, list) and dp and isinstance(dp[0], list):
        parts = [str(p).strip() for p in dp[0] if str(p).strip()]
        # Zero-pad month/day
        if len(parts) >= 2:
            parts[1] = parts[1].zfill(2)
        if len(parts) >= 3:
            parts[2] = parts[2].zfill(2)
        return "-".join(parts)
    if "raw" in issued and issued["raw"]:
        return str(issued["raw"]).strip()
    return ""


# ---------------------------------------------------------------------------
# BibTeX parsing
# ---------------------------------------------------------------------------

def parse_bibtex(text: str) -> list[dict]:
    """Parse a BibTeX string and return a list of structured entries.

    Each entry: ``{"entry_type": str, "citekey": str, "fields": dict}``.
    Applies ``convert_to_unicode`` so LaTeX accents become unicode.
    """
    parser = BibTexParser(common_strings=True)
    parser.customization = convert_to_unicode
    parser.ignore_nonstandard_types = False
    db = bibtexparser.loads(text or "", parser=parser)
    results = []
    for e in db.entries:
        entry_type = (e.get("ENTRYTYPE") or "").lower()
        citekey = e.get("ID") or ""
        fields = {k: v for k, v in e.items() if k not in ("ENTRYTYPE", "ID")}
        results.append({"entry_type": entry_type, "citekey": citekey, "fields": fields})
    return results


# ---------------------------------------------------------------------------
# Container-field routing
# ---------------------------------------------------------------------------

def _pick_container_field(zot_type: str, template: dict) -> str | None:
    """Return the Zotero field name that holds the "container title" for this type."""
    candidates = {
        "journalArticle": "publicationTitle",
        "magazineArticle": "publicationTitle",
        "newspaperArticle": "publicationTitle",
        "preprint": "publicationTitle",
        "bookSection": "bookTitle",
        "conferencePaper": "proceedingsTitle",
        "encyclopediaArticle": "encyclopediaTitle",
        "dictionaryEntry": "dictionaryTitle",
        "blogPost": "blogTitle",
        "forumPost": "forumTitle",
        "radioBroadcast": "programTitle",
        "tvBroadcast": "programTitle",
        "podcast": "seriesTitle",
    }
    candidate = candidates.get(zot_type)
    if candidate and candidate in template:
        return candidate
    # Fallback: first of a few common names present in template
    for name in ("publicationTitle", "bookTitle", "proceedingsTitle", "seriesTitle"):
        if name in template:
            return name
    return None


# ---------------------------------------------------------------------------
# Extra-field helpers
# ---------------------------------------------------------------------------

def _append_extra(template: dict, line: str) -> None:
    if "extra" not in template:
        return
    existing = template.get("extra", "") or ""
    if existing:
        template["extra"] = existing.rstrip() + "\n" + line
    else:
        template["extra"] = line


def _set_if_in_template(template: dict, field: str, value: str) -> bool:
    """Assign ``value`` to ``field`` iff the template has that field. Returns True if set."""
    if value is None or value == "":
        return False
    if field in template:
        template[field] = value
        return True
    return False


# ---------------------------------------------------------------------------
# BibTeX -> Zotero
# ---------------------------------------------------------------------------

def bibtex_entry_to_zotero(
    entry: dict,
    template_fn: Callable[[str], dict],
) -> dict:
    """Convert one parsed BibTeX entry dict into a Zotero item dict.

    ``template_fn(item_type)`` must return a pyzotero-style item template for
    the given Zotero item type (i.e. ``write_zot.item_template``).
    """
    entry_type = (entry.get("entry_type") or "").lower()
    fields = entry.get("fields") or {}
    citekey = entry.get("citekey") or ""

    # Thesis subtype detection before type lookup
    zot_type = BIBTEX_TYPE_MAP.get(entry_type, "document")
    template = dict(template_fn(zot_type))

    # Thesis "type" field (biblatex) distinguishes phd / masters
    if zot_type == "thesis" and "thesisType" in template:
        if entry_type == "phdthesis":
            template["thesisType"] = "PhD thesis"
        elif entry_type == "mastersthesis":
            template["thesisType"] = "Master's thesis"
        elif fields.get("type"):
            template["thesisType"] = fields["type"]

    # Report type
    if zot_type == "report" and "reportType" in template and fields.get("type"):
        template["reportType"] = fields["type"]

    # Title
    title = (fields.get("title") or "").strip().strip("{}").strip()
    _set_if_in_template(template, "title", title)

    # Short title
    _set_if_in_template(template, "shortTitle", (fields.get("shorttitle") or "").strip())

    # Authors / editors / translators
    authors = _parse_bibtex_author_list(fields.get("author", ""))
    editors = _parse_bibtex_author_list(fields.get("editor", ""))
    translators = _parse_bibtex_author_list(fields.get("translator", ""))
    for e in editors:
        e["creatorType"] = "editor"
    for t in translators:
        t["creatorType"] = "translator"
    creators = authors + editors + translators
    if creators and "creators" in template:
        template["creators"] = creators

    # Date
    date = _format_bibtex_date(
        fields.get("year", ""),
        fields.get("month", ""),
        fields.get("day", ""),
        fields.get("date", ""),
    )
    _set_if_in_template(template, "date", date)

    # Container title (journal / booktitle / proceedings)
    container_field = _pick_container_field(zot_type, template)
    container_value = (
        fields.get("journaltitle")
        or fields.get("journal")
        or fields.get("booktitle")
        or fields.get("maintitle")
        or ""
    ).strip()
    if container_field and container_value:
        template[container_field] = container_value

    # Series
    _set_if_in_template(template, "series", (fields.get("series") or "").strip())
    _set_if_in_template(template, "seriesNumber", (fields.get("seriesnumber") or "").strip())

    # Standard fields
    _set_if_in_template(template, "volume", (fields.get("volume") or "").strip())
    _set_if_in_template(
        template,
        "issue",
        (fields.get("issue") or fields.get("number") or "").strip(),
    )
    pages = (fields.get("pages") or "").strip().replace("--", "-")
    _set_if_in_template(template, "pages", pages)
    _set_if_in_template(template, "publisher", (fields.get("publisher") or "").strip())
    _set_if_in_template(
        template,
        "place",
        (fields.get("address") or fields.get("location") or "").strip(),
    )
    _set_if_in_template(template, "edition", (fields.get("edition") or "").strip())
    _set_if_in_template(template, "ISBN", (fields.get("isbn") or "").strip())
    _set_if_in_template(template, "ISSN", (fields.get("issn") or "").strip())
    _set_if_in_template(template, "language", (fields.get("language") or "").strip())
    _set_if_in_template(template, "url", (fields.get("url") or "").strip())
    _set_if_in_template(template, "DOI", (fields.get("doi") or "").strip())
    _set_if_in_template(template, "abstractNote", (fields.get("abstract") or "").strip())
    _set_if_in_template(template, "numPages", (fields.get("pagetotal") or "").strip())
    _set_if_in_template(template, "numberOfVolumes", (fields.get("volumes") or "").strip())

    # Institution / school fall back to publisher for report / thesis
    if "publisher" in template and not template["publisher"]:
        fallback = (
            fields.get("school")
            or fields.get("institution")
            or fields.get("organization")
            or ""
        ).strip()
        if fallback:
            template["publisher"] = fallback

    # Report/thesis: institution often belongs in "institution" or "university"
    _set_if_in_template(template, "institution", (fields.get("institution") or "").strip())
    _set_if_in_template(template, "university", (fields.get("school") or "").strip())

    # Patent fields
    if zot_type == "patent":
        _set_if_in_template(template, "patentNumber", (fields.get("number") or "").strip())

    # Keywords -> tags
    kw = fields.get("keywords") or fields.get("keyword") or ""
    source_tags = _split_keywords(kw)
    if source_tags and "tags" in template:
        template["tags"] = [{"tag": t} for t in source_tags]

    # Citation key preservation
    if citekey:
        _append_extra(template, f"Citation Key: {citekey}")

    # Arxiv eprint
    eprint = (fields.get("eprint") or "").strip()
    eprint_type = (fields.get("eprinttype") or fields.get("archiveprefix") or "").strip().lower()
    if eprint and eprint_type in ("arxiv", ""):
        _append_extra(template, f"arXiv: {eprint}")
        if "url" in template and not template.get("url"):
            template["url"] = f"https://arxiv.org/abs/{eprint}"

    # PMID / PMCID
    for src_key, label in (("pmid", "PMID"), ("pmcid", "PMCID")):
        val = (fields.get(src_key) or "").strip()
        if val:
            _append_extra(template, f"{label}: {val}")

    # Note -> extra
    note = (fields.get("note") or "").strip()
    if note:
        _append_extra(template, note)

    # Preserve anything we didn't map
    handled = {
        "title", "shorttitle", "author", "editor", "translator",
        "year", "month", "day", "date",
        "journal", "journaltitle", "booktitle", "maintitle",
        "series", "seriesnumber",
        "volume", "issue", "number", "pages",
        "publisher", "address", "location", "edition",
        "isbn", "issn", "language", "url", "doi", "abstract",
        "keywords", "keyword", "note",
        "school", "institution", "organization",
        "eprint", "eprinttype", "archiveprefix", "primaryclass",
        "pmid", "pmcid", "type", "pagetotal", "volumes",
    }
    for k, v in fields.items():
        if k.lower() in handled:
            continue
        if not v:
            continue
        _append_extra(template, f"{k}: {v}")

    return template


def _split_keywords(raw: str) -> list[str]:
    if not raw:
        return []
    # Try semicolon first, then comma
    for sep in (";", ","):
        if sep in raw:
            return [p.strip() for p in raw.split(sep) if p.strip()]
    s = raw.strip()
    return [s] if s else []


# ---------------------------------------------------------------------------
# CSL JSON -> Zotero
# ---------------------------------------------------------------------------

def csl_json_to_zotero(
    csl: dict,
    template_fn: Callable[[str], dict],
) -> dict:
    """Convert one CSL JSON object into a Zotero item dict."""
    csl_type = (csl.get("type") or "").lower()
    zot_type = CSL_TYPE_MAP.get(csl_type, "document")
    template = dict(template_fn(zot_type))

    # Title
    _set_if_in_template(template, "title", (csl.get("title") or "").strip())
    _set_if_in_template(template, "shortTitle", (csl.get("title-short") or "").strip())

    # Creators
    creators = []
    creators += _csl_names_to_creators(csl.get("author"), "author")
    creators += _csl_names_to_creators(csl.get("editor"), "editor")
    creators += _csl_names_to_creators(csl.get("translator"), "translator")
    if creators and "creators" in template:
        template["creators"] = creators

    # Date
    date = _format_csl_date(csl.get("issued"))
    _set_if_in_template(template, "date", date)

    # Container title
    container_field = _pick_container_field(zot_type, template)
    container_value = (csl.get("container-title") or "").strip()
    if container_field and container_value:
        template[container_field] = container_value

    # Series
    _set_if_in_template(template, "series", (csl.get("collection-title") or "").strip())
    _set_if_in_template(template, "seriesNumber", str(csl.get("collection-number") or "").strip())

    # Common fields
    _set_if_in_template(template, "volume", str(csl.get("volume") or "").strip())
    _set_if_in_template(template, "issue", str(csl.get("issue") or "").strip())
    _set_if_in_template(template, "pages", str(csl.get("page") or "").strip())
    _set_if_in_template(template, "publisher", (csl.get("publisher") or "").strip())
    _set_if_in_template(template, "place", (csl.get("publisher-place") or "").strip())
    _set_if_in_template(template, "edition", str(csl.get("edition") or "").strip())
    _set_if_in_template(template, "ISBN", (csl.get("ISBN") or "").strip())
    _set_if_in_template(template, "ISSN", (csl.get("ISSN") or "").strip())
    _set_if_in_template(template, "DOI", (csl.get("DOI") or "").strip())
    _set_if_in_template(template, "url", (csl.get("URL") or "").strip())
    _set_if_in_template(template, "language", (csl.get("language") or "").strip())
    _set_if_in_template(template, "abstractNote", (csl.get("abstract") or "").strip())
    _set_if_in_template(template, "numPages", str(csl.get("number-of-pages") or "").strip())

    # Type-specific number fields
    number = str(csl.get("number") or "").strip()
    if number:
        if zot_type == "report":
            _set_if_in_template(template, "reportNumber", number)
        elif zot_type == "patent":
            _set_if_in_template(template, "patentNumber", number)
        elif zot_type == "bill":
            _set_if_in_template(template, "billNumber", number)
        elif zot_type == "case":
            _set_if_in_template(template, "docketNumber", number)
        elif zot_type == "statute":
            _set_if_in_template(template, "publicLawNumber", number)

    # Thesis genre
    if zot_type == "thesis" and "thesisType" in template:
        genre = (csl.get("genre") or "").strip()
        if genre:
            template["thesisType"] = genre

    # Keywords -> tags
    keywords = csl.get("keyword") or csl.get("keywords")
    if isinstance(keywords, list):
        source_tags = [str(k).strip() for k in keywords if str(k).strip()]
    elif isinstance(keywords, str):
        source_tags = _split_keywords(keywords)
    else:
        source_tags = []
    if source_tags and "tags" in template:
        template["tags"] = [{"tag": t} for t in source_tags]

    # Citation key from `id`
    citekey = str(csl.get("id") or "").strip()
    if citekey:
        _append_extra(template, f"Citation Key: {citekey}")

    # Note
    note = (csl.get("note") or "").strip()
    if note:
        _append_extra(template, note)

    # Preserve unmapped fields
    handled = {
        "type", "id", "title", "title-short",
        "author", "editor", "translator",
        "issued", "container-title", "collection-title", "collection-number",
        "volume", "issue", "page", "publisher", "publisher-place",
        "edition", "ISBN", "ISSN", "DOI", "URL", "language",
        "abstract", "number-of-pages", "number",
        "genre", "keyword", "keywords", "note",
    }
    for k, v in csl.items():
        if k in handled:
            continue
        if v is None or v == "" or v == [] or v == {}:
            continue
        if isinstance(v, (dict, list)):
            try:
                v = json.dumps(v, ensure_ascii=False)
            except (TypeError, ValueError):
                v = str(v)
        _append_extra(template, f"{k}: {v}")

    return template


# ---------------------------------------------------------------------------
# Input coercion helpers
# ---------------------------------------------------------------------------

def coerce_csl_json_input(value: Any) -> list[dict]:
    """Accept a CSL JSON string, single object, or list; return a list of dicts."""
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    raise ValueError("csl_json must be a JSON string, object, or array of objects")


def merge_tags(source_tags: list[str], extra_tags: list[str]) -> list[str]:
    """Merge two tag lists preserving order and deduplicating case-insensitively."""
    seen: set[str] = set()
    out: list[str] = []
    for t in list(source_tags or []) + list(extra_tags or []):
        t = (t or "").strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out
