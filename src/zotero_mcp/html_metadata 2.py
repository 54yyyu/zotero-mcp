"""Read bibliographic metadata that a publisher page embeds in its ``<head>``.

Journal platforms — OJS/PKP, Atypon, Silverchair, Highwire, ScienceDirect,
SpringerLink — publish an article's citation on the landing page itself, as
Highwire ``citation_*`` meta tags and/or Dublin Core ``DC.*`` tags. Zotero's
own connector reads exactly these, which is why saving a paper from the
browser produces a full record while saving the same URL through the API
produced a bare ``webpage`` with only a URL on it.

Parsed with the stdlib HTML parser rather than a new dependency: only the
``<head>``'s meta tags are wanted, and the input is untrusted markup from an
arbitrary publisher, so the parser is told to stop reading at ``</head>``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

# Highwire tag -> our field name. Both the ``citation_`` and the bare
# Dublin Core spellings appear in the wild; DC is the weaker source and is
# only consulted for fields Highwire did not supply.
_HIGHWIRE = {
    "citation_title": "title",
    "citation_journal_title": "publication",
    "citation_conference_title": "publication",
    "citation_inbook_title": "book_title",
    "citation_book_title": "book_title",
    "citation_publisher": "publisher",
    "citation_volume": "volume",
    "citation_issue": "issue",
    "citation_firstpage": "first_page",
    "citation_lastpage": "last_page",
    "citation_doi": "doi",
    "citation_issn": "issn",
    "citation_isbn": "isbn",
    "citation_language": "language",
    "citation_abstract": "abstract",
    "citation_pdf_url": "pdf_url",
    "citation_dissertation_institution": "institution",
    "citation_technical_report_institution": "institution",
}

_DUBLIN_CORE = {
    "dc.title": "title",
    "dc.publisher": "publisher",
    "dc.identifier": "doi",          # only used when it looks like a DOI
    "dc.language": "language",
    "dc.description": "abstract",
    "dc.source": "publication",
    "dcterms.abstract": "abstract",
    "dcterms.issued": "date",
}

# Dates arrive as 2020/07/07, 2020-07-07, or a bare year.
_DATE_TAGS = (
    "citation_publication_date",
    "citation_date",
    "citation_online_date",
    "citation_cover_date",
    "citation_year",
)

_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE)


@dataclass
class EmbeddedMetadata:
    """Bibliographic fields lifted from a page's meta tags.

    Every field is optional; a page may carry none of them. ``authors`` holds
    ``(first, last)`` pairs, with ``first`` empty for single-token or
    corporate names.
    """

    title: str = ""
    authors: list[tuple[str, str]] = field(default_factory=list)
    publication: str = ""
    book_title: str = ""
    publisher: str = ""
    volume: str = ""
    issue: str = ""
    first_page: str = ""
    last_page: str = ""
    date: str = ""
    doi: str = ""
    issn: str = ""
    isbn: str = ""
    language: str = ""
    abstract: str = ""
    pdf_url: str = ""
    institution: str = ""

    @property
    def pages(self) -> str:
        """``first_page``–``last_page``, or whichever end exists."""
        if self.first_page and self.last_page:
            if self.first_page == self.last_page:
                return self.first_page
            return f"{self.first_page}-{self.last_page}"
        return self.first_page or self.last_page

    def is_usable(self) -> bool:
        """True when there is enough here to beat a bare URL.

        A title alone qualifies: an item called "Determinants of Public
        Trust..." is strictly more findable than one called
        "https://ojs.example/article/view/5155".
        """
        return bool(self.title or self.doi)

    def looks_like_article(self) -> bool:
        """True when the page describes a journal article rather than a
        generic web page."""
        return bool(self.publication or self.volume or self.issue
                    or self.first_page or self.issn)

    def looks_like_chapter(self) -> bool:
        return bool(self.book_title or self.isbn) and not self.publication


class _HeadMetaParser(HTMLParser):
    """Collect ``<meta>`` name/content pairs, stopping at ``</head>``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.metas: list[tuple[str, str]] = []
        self._done = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._done:
            return
        if tag == "body":
            # Some pages never close <head>; the body opening is the
            # reliable end of the metadata block.
            self._done = True
            return
        if tag != "meta":
            return
        a = {k.lower(): (v or "") for k, v in attrs}
        # Highwire uses name=, Open Graph and some platforms use property=.
        key = a.get("name") or a.get("property") or a.get("http-equiv") or ""
        content = a.get("content", "")
        if key and content:
            self.metas.append((key.strip().lower(), content.strip()))

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            self._done = True


def parse_meta_tags(html: str) -> dict[str, list[str]]:
    """Return ``{meta name: [content, ...]}`` for the document head.

    Repeated names (``citation_author`` in particular) keep every value, in
    document order. Malformed markup yields whatever was parsed before the
    error rather than raising — a publisher's broken page should cost us the
    metadata, not the call.
    """
    parser = _HeadMetaParser()
    try:
        parser.feed(html)
    except Exception:
        pass
    out: dict[str, list[str]] = {}
    for name, content in parser.metas:
        out.setdefault(name, []).append(content)
    return out


def _split_name(raw: str) -> tuple[str, str]:
    """Split one ``citation_author`` value into ``(first, last)``.

    Both orders occur: "Haning, Mohamad Thahir" and "Mohamad Thahir Haning".
    A comma is the only reliable signal of which is which; without one, the
    last whitespace-separated token is taken as the surname.
    """
    name = " ".join(raw.split())
    if not name:
        return ("", "")
    if "," in name:
        last, _, first = name.partition(",")
        return (first.strip(), last.strip())
    parts = name.split(" ")
    if len(parts) == 1:
        return ("", parts[0])
    return (" ".join(parts[:-1]), parts[-1])


def _normalize_date(raw: str) -> str:
    """``2020/07/07`` -> ``2020-07-07``; anything else passes through."""
    value = raw.strip()
    if re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", value):
        return value.replace("/", "-")
    if re.fullmatch(r"\d{4}/\d{1,2}", value):
        return value.replace("/", "-")
    return value


def _clean_doi(raw: str) -> str:
    """Pull a bare DOI out of a tag value that may be a URL or prefixed."""
    match = _DOI_RE.search(raw or "")
    return match.group(1).rstrip(".,;)") if match else ""


def extract_embedded_metadata(html: str) -> EmbeddedMetadata:
    """Build an :class:`EmbeddedMetadata` from a page's meta tags."""
    tags = parse_meta_tags(html)
    meta = EmbeddedMetadata()

    def first(name: str) -> str:
        values = tags.get(name)
        return values[0] if values else ""

    for tag, attr in _HIGHWIRE.items():
        value = first(tag)
        if value and not getattr(meta, attr):
            setattr(meta, attr, value)

    for tag, attr in _DUBLIN_CORE.items():
        value = first(tag)
        if not value or getattr(meta, attr):
            continue
        if attr == "doi":
            value = _clean_doi(value)
            if not value:
                continue
        setattr(meta, attr, value)

    # Open Graph is the last resort for a title, and only that: og:type and
    # friends say nothing bibliographic.
    if not meta.title:
        meta.title = first("og:title") or first("twitter:title")

    for tag in _DATE_TAGS:
        value = first(tag)
        if value:
            meta.date = _normalize_date(value)
            break
    if not meta.date:
        meta.date = _normalize_date(first("dcterms.issued") or first("dc.date"))

    meta.doi = _clean_doi(meta.doi)

    # Authors: Highwire repeats the tag, DC uses DC.creator. Preserve order
    # and drop duplicates, which some platforms emit for both spellings.
    raw_authors: list[str] = []
    for tag in ("citation_author", "citation_authors", "dc.creator",
                "dc.contributor", "citation_editor"):
        raw_authors.extend(tags.get(tag, []))
    seen: set[str] = set()
    for raw in raw_authors:
        # citation_authors packs several names into one tag, separated by
        # semicolons.
        for piece in (raw.split(";") if ";" in raw else [raw]):
            first_name, last_name = _split_name(piece)
            if not last_name:
                continue
            key = f"{first_name}|{last_name}".lower()
            if key in seen:
                continue
            seen.add(key)
            meta.authors.append((first_name, last_name))

    return meta
