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
import re
from collections.abc import Callable
from typing import Any

import bibtexparser

from zotero_mcp import schema

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
    "standard": "standard",
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
    "dataset": "dataset",
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
    # Crossref's /transform/application/vnd.citationstyles.csl+json endpoint
    # is the most common source of CSL JSON in practice, but it keeps
    # Crossref's own type vocabulary rather than the CSL spec's:
    # "journal-article" for "article-journal", "book-chapter" for "chapter".
    # Without these aliases every one of them falls through to "document",
    # whose template has no publicationTitle/volume/issue/pages — so the
    # journal name, volume, issue and page range are dropped silently.
    # Only "book" and "report" spell the same in both vocabularies.
    "journal-article": "journalArticle",
    "proceedings-article": "conferencePaper",
    "book-chapter": "bookSection",
    "book-part": "bookSection",
    "book-section": "bookSection",
    "reference-entry": "dictionaryEntry",
    "monograph": "book",
    "edited-book": "book",
    "reference-book": "book",
    "book-set": "book",
    "book-series": "book",
    "book-track": "bookSection",
    "proceedings": "book",
    "proceedings-series": "book",
    "report-component": "report",
    "report-series": "report",
    "dissertation": "thesis",
    "posted-content": "preprint",
    # Container types. Publishers do register ordinary articles as these — a
    # "journal-issue" carrying a title, volume, issue and page range is an
    # article — and journalArticle has somewhere to put those fields where
    # "document", the fallthrough, does not.
    "journal-issue": "journalArticle",
    "journal-volume": "journalArticle",
    "journal": "journalArticle",
    "database": "dataset",
    "standard": "standard",
    "component": "document",
    "grant": "document",
    "peer-review": "document",
    "other": "document",
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
        # "and others" is BibTeX for et al., not a person named Others.
        if not name or name.lower() == "others":
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


# BibTeX separates names with "and" between any whitespace, newlines
# included: exported .bib files wrap long author lists mid-field.
_AND_SEPARATOR = re.compile(r"\s+and\s+", re.IGNORECASE)


def _split_bibtex_authors(raw: str) -> list[str]:
    """Split on whitespace-delimited ``and`` while respecting brace groups."""
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
        elif depth == 0 and (sep := _AND_SEPARATOR.match(raw, i)):
            out.append("".join(buf))
            buf = []
            i = sep.end()
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
            # Particles and suffix are separate CSL keys; Zotero has only
            # two name parts, so fold them back in ("van der" + "Maaten").
            ndp = (n.get("non-dropping-particle") or "").strip()
            if ndp and family:
                family = ndp + ("" if ndp[-1] in "'’-" else " ") + family
            dp = (n.get("dropping-particle") or "").strip()
            if dp and given:
                given = f"{given} {dp}"
            suffix = (n.get("suffix") or "").strip()
            if suffix and given:
                given = f"{given}, {suffix}"
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

# Fields whose value is not LaTeX: decoding a URL turns its ``&`` into spaces,
# and a page range's ``--`` into a dash the converter no longer recognises.
_BIBTEX_VERBATIM_FIELDS = frozenset({"url", "doi", "file", "eprint", "pages"})


def _parse_bibtex_v1(text: str) -> list[tuple[str, str, dict]]:
    from bibtexparser.bparser import BibTexParser
    from bibtexparser.customization import convert_to_unicode

    parser = BibTexParser(common_strings=True)
    parser.customization = convert_to_unicode
    parser.ignore_nonstandard_types = False
    db = bibtexparser.loads(text, parser=parser)
    return [
        (e.get("ENTRYTYPE") or "", e.get("ID") or "",
         {k: v for k, v in e.items() if k not in ("ENTRYTYPE", "ID")})
        for e in db.entries
    ]


def _parse_bibtex_v2(text: str) -> list[tuple[str, str, dict]]:
    from bibtexparser.model import Entry
    from pylatexenc.latex2text import LatexNodes2Text

    decode = LatexNodes2Text(math_mode="verbatim").latex_to_text
    out = []
    for block in bibtexparser.parse_string(text).blocks:
        # A repeated citekey or field name is set aside as a failed block that
        # still carries the parsed entry; 1.x kept those entries, so do we.
        entry = block if isinstance(block, Entry) else getattr(block, "ignore_error_block", None)
        if not isinstance(entry, Entry):
            continue
        fields = {}
        for f in entry.fields:
            key, value = f.key.lower(), f.value
            if isinstance(value, str) and key not in _BIBTEX_VERBATIM_FIELDS:
                try:
                    value = decode(value)
                except Exception:
                    pass  # keep the raw text over losing the field
            fields[key] = value
        out.append((entry.entry_type, entry.key, fields))
    return out


def parse_bibtex(text: str) -> list[dict]:
    """Parse a BibTeX string and return a list of structured entries.

    Each entry: ``{"entry_type": str, "citekey": str, "fields": dict}``, with
    LaTeX accents decoded to unicode. Works on bibtexparser 1.x and 2.x, whose
    APIs share nothing: pyzotero requires 1.x up to 1.16 and 2.x from 1.17.
    """
    parse = _parse_bibtex_v2 if hasattr(bibtexparser, "parse_string") else _parse_bibtex_v1
    return [
        {"entry_type": entry_type.lower(), "citekey": citekey, "fields": fields}
        for entry_type, citekey, fields in parse(text or "")
    ]


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
    # BibTeX ``number`` is the type's own number where it has one (a report's
    # reportNumber, a patent's patentNumber, a standard's number: all on the
    # schema's ``number`` base field) and the issue of a journal otherwise.
    number = (fields.get("number") or "").strip()
    if _set_if_in_template(template, schema.resolve_field(zot_type, "number"), number):
        number = ""
    _set_if_in_template(template, "issue", (fields.get("issue") or number).strip())
    # The body that issued it: a standard's organization, a patent's issuing
    # authority (the schema's ``authority`` base field).
    _set_if_in_template(
        template, schema.resolve_field(zot_type, "authority"), (fields.get("organization") or "").strip()
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
        # Zotero 7.1+ has native PMID and PMCID fields; Extra only where it has not.
        if val and not _set_if_in_template(template, label, val):
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

def _csl_text(value: Any) -> str:
    """One trimmed string out of a CSL field that may arrive as an array.

    The CSL spec types these fields as strings, but real producers disagree.
    Crossref's transform endpoint sends ``ISSN`` and ``ISBN`` as arrays, and
    its plain ``/works/{doi}`` payload does the same for ``title`` and
    ``container-title``; other exporters wrap ``container-title`` too. Take
    the first non-empty element in that case — the further ISSNs are the same
    journal's other media forms — so a provider response converts instead of
    raising ``AttributeError`` on ``.strip()``.

    Numbers pass through ``str`` as well: ``volume`` and ``issue`` arrive as
    integers often enough to matter.
    """
    if isinstance(value, list):
        value = next((v for v in value if v is not None and str(v).strip()), "")
    # An object is not text: Crossref's ``license`` is a list of them, and its
    # repr must not be written to the field CSL's ``license`` maps to.
    if value is None or isinstance(value, (dict, list)):
        return ""
    return str(value).strip()


# Zotero's CSL crosswalk name -> the CSL 1.0.2 names producers (Crossref,
# citeproc exports) send for the same variable.
_CSL_SPEC_NAMES = {"journalAbbreviation": ("container-title-short",)}


# A CSL ``id`` usable as a citation key: one token with a letter in it and
# none of the characters a path or URL would bring.
_CITEKEY_LIKE = re.compile(r"^(?=.*[^\W\d_])[^\s/\\{}%#~,]+$")
# ...and not a reference manager's placeholder (Mendeley and Word: "ITEM-1").
_RECORD_ID = re.compile(r"^ITEM-\d+$", re.IGNORECASE)

# Plain-valued keys of a Crossref record that describe the record, not the work.
_PROVIDER_KEYS = frozenset({
    "reference-count", "references-count", "is-referenced-by-count",
    "member", "prefix", "score",
})


def _csl_variables() -> set[str]:
    """Every CSL variable Zotero's crosswalk knows, under either spelling."""
    names = {*schema.csl_text_fields(), *schema.csl_date_fields(), *schema.csl_name_variables()}
    for aliases in _CSL_SPEC_NAMES.values():
        names.update(aliases)
    return names


def csl_json_to_zotero(
    csl: dict,
    template_fn: Callable[[str], dict],
) -> dict:
    """Convert one CSL JSON object into a Zotero item dict.

    Fields, dates and creators are mapped through Zotero's own CSL crosswalk
    (``schema.csl_*``) and resolved to the item type's actual field, so a
    variable lands wherever that type keeps it (``number`` is ``reportNumber``
    on a report and ``number`` on a standard) and a type or field Zotero adds
    is covered by the next schema refresh, not a code change. Every variable
    that is not written to a field is kept in Extra as ``name: value``.
    """
    csl_type = (csl.get("type") or "").lower()
    # The importer's own choices first (they also take Crossref's non-CSL
    # types), then Zotero's crosswalk, so a CSL type added later still maps.
    zot_type = CSL_TYPE_MAP.get(csl_type) or schema.csl_item_type(csl_type) or "document"
    template = dict(template_fn(zot_type))
    consumed = {"type"}  # CSL variables written somewhere; the rest go to Extra

    # Creators: each CSL name variable to the creator type this item type
    # uses for it; author falls back to the type's primary creator type
    # (a computer program's programmer).
    creators = []
    if "creators" in template:
        primary = (schema.creator_types(zot_type) or ["author"])[0]
        for var in dict.fromkeys(["author", "editor", "translator", *schema.csl_name_variables()]):
            creator_type = schema.creator_type_for(zot_type, var) or (primary if var == "author" else None)
            converted = _csl_names_to_creators(csl.get(var), creator_type) if creator_type else []
            if converted:
                creators += converted
                consumed.add(var)
        if creators:
            template["creators"] = creators

    # Container title: the importer's own per-type choice first; the
    # crosswalk below fills it for any type the picker does not know.
    container_field = _pick_container_field(zot_type, template)
    container_value = _csl_text(csl.get("container-title"))
    if container_field and container_value:
        template[container_field] = container_value
        consumed.add("container-title")

    # PubMed ids are not in Zotero's CSL crosswalk.
    for label in ("PMID", "PMCID"):
        val = _csl_text(csl.get(label))
        if val and not _set_if_in_template(template, label, val):
            _append_extra(template, f"{label}: {val}")
        consumed.add(label)

    # Text variables, in Zotero's own order of target base fields. A
    # variable may arrive under its CSL 1.0.2 name, which Zotero's crosswalk
    # still lists by its older CSL-JSON name.
    for var, bases in schema.csl_text_fields().items():
        source = next((k for k in (var, *_CSL_SPEC_NAMES.get(var, ())) if _csl_text(csl.get(k))), var)
        value = _csl_text(csl.get(source))
        if source in consumed or var == "note" or not value:
            continue
        for base in bases:
            field = schema.resolve_field(zot_type, base)
            if field in template and field != "extra" and not template[field]:
                template[field] = value
                consumed.add(source)
                break

    # Date variables (issued -> date, accessed -> accessDate, ...).
    for var, base in schema.csl_date_fields().items():
        if _set_if_in_template(template, schema.resolve_field(zot_type, base), _format_csl_date(csl.get(var))):
            consumed.add(var)

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
        consumed.update({"keyword", "keywords"})

    # Citation key from `id`, when it is one. Most producers put their own
    # record identifier there (a DOI URL, a zotero.org item URI, "ITEM-1", a
    # bare number), and a "Citation Key:" line in Extra is read as a key.
    citekey = _csl_text(csl.get("id"))
    if _CITEKEY_LIKE.match(citekey) and not _RECORD_ID.match(citekey):
        _append_extra(template, f"Citation Key: {citekey}")
    consumed.add("id")

    # Note
    note = _csl_text(csl.get("note"))
    if note:
        _append_extra(template, note)
    consumed.add("note")

    # Preserve everything not written above, including a known variable this
    # item type has no field for (a standard's ``volume``), so nothing is lost.
    # A key that is not a CSL variable is kept only as a plain value: a
    # provider's record carries its own bookkeeping beside the citation
    # (Crossref's ``reference`` is the whole cited-works list, with
    # ``indexed``, ``license``, ``content-domain`` ...), and all of it used to
    # be dumped into Extra as JSON, kilobytes per item.
    known = _csl_variables()
    text_variables = schema.csl_text_fields()
    for k, v in csl.items():
        if k in consumed:
            continue
        if v is None or v == "" or v == [] or v == {}:
            continue
        if k not in known and (isinstance(v, (dict, list)) or k in _PROVIDER_KEYS):
            continue
        if k in text_variables and not _csl_text(v):
            continue  # a text variable holding objects (Crossref's ``license``)
        if isinstance(v, (dict, list)):
            try:
                v = json.dumps(v, ensure_ascii=False)
            except (TypeError, ValueError):
                v = str(v)
        _append_extra(template, f"{k}: {v}")

    return template


# ---------------------------------------------------------------------------
# PubMed XML -> CSL JSON
# ---------------------------------------------------------------------------

_MONTHS = {m: i for i, m in enumerate(
    "jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
_SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in",
                "into", "nor", "of", "on", "or", "the", "to", "with", "und",
                "de", "des", "du", "la", "le", "et", "y"}
# Correction notices a reader must see on the item itself.
_NOTICE_TYPES = ("RetractionIn", "RetractionOf", "ErratumIn", "ErratumFor",
                 "ExpressionOfConcernIn", "UpdateIn", "RepublishedIn")


def _xml_text(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _title_case(s: str) -> str:
    """Journal titles as Zotero's PubMed translator writes them
    ("Statistics in medicine" -> "Statistics in Medicine")."""
    words = s.split(" ")
    return " ".join(
        w if (i and w.lower() in _SMALL_WORDS) or w[:1].isupper() or not w
        else w[:1].upper() + w[1:]
        for i, w in enumerate(words))


def _expand_pages(p: str) -> str:
    """MEDLINE abbreviates the last page: 2409-19 -> 2409-2419."""
    m = re.fullmatch(r"(\d+)-(\d+)", p)
    if m and len(m.group(2)) < len(m.group(1)):
        a, b = m.groups()
        return f"{a}-{a[:len(a) - len(b)]}{b}"
    return p


def _pubmed_date(el) -> list[int] | None:
    if el is None:
        return None
    year = el.findtext("Year")
    if not year:  # MedlineDate: "2009 May-Jun", "1998 Dec-1999 Jan"
        m = re.match(r"(\d{4})(?:\s+([A-Za-z]{3}))?", el.findtext("MedlineDate") or "")
        if not m:
            return None
        parts = [int(m.group(1))]
        if m.group(2) and m.group(2).lower() in _MONTHS:
            parts.append(_MONTHS[m.group(2).lower()])
        return parts
    parts = [int(year)]
    month = (el.findtext("Month") or "").strip()
    if month:
        parts.append(int(month) if month.isdigit() else _MONTHS.get(month[:3].lower(), 0))
        if not parts[-1]:
            parts.pop()
        elif el.findtext("Day"):
            parts.append(int(el.findtext("Day")))
    return parts


def pubmed_xml_to_csl(article) -> dict:
    """One <PubmedArticle> as CSL JSON, the shape Zotero's PubMed translator
    gives: native PMID/PMCID, abstract with its section labels, MeSH headings
    and author keywords as tags, language, journal abbreviation, expanded
    pages. What Zotero has no field for (publication types, the e-pub date,
    correction and retraction notices) goes to Extra as `Label: value`.
    """
    mc = article.find("MedlineCitation")
    art = mc.find("Article")
    journal = art.find("Journal")
    ji = journal.find("JournalIssue")
    ids = {a.get("IdType"): (a.text or "").strip()
           for a in article.findall("PubmedData/ArticleIdList/ArticleId")}

    title = _xml_text(art.find("ArticleTitle")) or _xml_text(art.find("VernacularTitle"))
    title = title.rstrip(".")
    if title.startswith("[") and title.endswith("]"):  # translated title
        title = title[1:-1].rstrip(".")
    csl = {"type": "article-journal", "title": title, "source": "PubMed",
           "PMID": (mc.findtext("PMID") or "").strip()}
    if ids.get("pmc"):
        csl["PMCID"] = ids["pmc"]
    if ":" in title:
        csl["title-short"] = title.split(":", 1)[0].strip()

    authors = []
    for a in art.findall("AuthorList/Author"):
        if a.findtext("CollectiveName"):
            authors.append({"literal": _xml_text(a.find("CollectiveName"))})
        elif a.findtext("LastName"):
            authors.append({"family": a.findtext("LastName").strip(),
                            "given": (a.findtext("ForeName") or a.findtext("Initials") or "").strip()})
    if authors:
        csl["author"] = authors

    def put(key, value):
        if value:
            csl[key] = value

    put("container-title", _title_case(_xml_text(journal.find("Title"))))
    put("container-title-short", (journal.findtext("ISOAbbreviation") or "").replace(".", "").strip())
    put("ISSN", (journal.findtext("ISSN") or "").strip())
    put("volume", (ji.findtext("Volume") or "").strip())
    put("issue", (ji.findtext("Issue") or "").strip())
    put("page", _expand_pages((art.findtext("Pagination/MedlinePgn") or "").strip()))
    put("language", (art.findtext("Language") or "").strip())
    issued = _pubmed_date(ji.find("PubDate"))
    if issued:
        csl["issued"] = {"date-parts": [issued]}
    doi = ids.get("doi") or next((e.text for e in art.findall("ELocationID")
                                  if e.get("EIdType") == "doi"), "")
    put("DOI", (doi or "").strip())

    sections = []
    for t in art.findall("Abstract/AbstractText"):
        label, text = t.get("Label"), _xml_text(t)
        if label and label.upper() != "UNLABELLED":
            label = label.capitalize() if label.isupper() else label
            sections.append(f"{label}: {text}")
        elif text:
            sections.append(text)
    put("abstract", "\n".join(sections))

    tags = [_xml_text(d) for d in mc.findall("MeshHeadingList/MeshHeading/DescriptorName")]
    tags += [_xml_text(k) for k in mc.findall("KeywordList/Keyword")]
    tags = list(dict.fromkeys(t for t in tags if t))
    if tags:
        csl["keyword"] = tags

    # Extra: what PubMed records and Zotero has no field for.
    put("Publication Type", "; ".join(_xml_text(p) for p in art.findall(
        "PublicationTypeList/PublicationType")))
    epub = _pubmed_date(art.find("ArticleDate"))
    if epub:
        csl["Epub Date"] = "-".join(f"{x:02d}" for x in epub)
    for c in mc.findall("CommentsCorrectionsList/CommentsCorrections"):
        if c.get("RefType") in _NOTICE_TYPES:
            ref = _xml_text(c.find("RefSource"))
            pmid = (c.findtext("PMID") or "").strip()
            key = re.sub(r"(?<!^)(?=[A-Z])", " ", c.get("RefType"))
            prev = csl.get(key)
            val = ref + (f" (PMID {pmid})" if pmid else "")
            csl[key] = f"{prev}; {val}" if prev else val
    return csl


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
