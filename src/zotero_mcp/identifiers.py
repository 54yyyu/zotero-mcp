"""Public, dependency-free normalisation of scholarly identifiers.

``tools/_helpers.py`` has carried ``_normalize_doi`` since the batch-import
work, and it is the only canonical DOI handling this package exposes. Two
problems with leaving it there:

* it is private, in a private module, so downstream consumers that want the
  same canonicalisation the server uses have to import
  ``zotero_mcp.tools._helpers._normalize_doi`` and pin themselves to an
  implementation detail;
* ``tools/_helpers`` pulls in ``requests``, ``pyzotero`` and the rest of the
  tool layer, which is a heavy price for ten lines of string handling.

This module is stdlib-only and imports nothing from the rest of the
package, so ``from zotero_mcp.identifiers import normalize_doi`` stays
cheap. ``tools/_helpers`` re-exports the private name, so existing callers
and tests are unaffected.
"""

from __future__ import annotations

import html
import re
import unicodedata

__all__ = [
    "normalize_doi",
    "doi_match_key",
    "normalize_isbn",
    "isbn_match_keys",
    "normalize_arxiv_id",
    "arxiv_identity",
    "arxiv_identity_from_extra",
    "normalize_title_for_matching",
    "metadata_match_keys",
]

#: A well-formed DOI: the ``10.NNNN`` registrant prefix plus a suffix.
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")

_DOI_IN_URL_RE = re.compile(
    r"doi\.org/(10\.\d{4,9}/[^\s?#]+)", flags=re.IGNORECASE
)

#: Punctuation that trails a DOI copied out of prose or a reference list.
_TRAILING_PUNCT = ".,);]"

#: Closing brackets and the opener each one belongs to. A closer is prose
#: punctuation only when the DOI holds no matching opener.
_CLOSERS = {")": "(", "]": "["}


def _strip_trailing_punct(s):
    """Strip prose punctuation from the end of a DOI, keeping brackets that
    the DOI itself opened.

    ``rstrip(_TRAILING_PUNCT)`` cannot tell "(see 10.1234/foo)" from TAO's
    ``10.3319/TAO.2009.05.25.02(IWNOP)``, where the parentheses are part of
    the suffix and CrossRef 404s without the closer.
    """
    while s and s[-1] in _TRAILING_PUNCT:
        opener = _CLOSERS.get(s[-1])
        if opener is not None and s.count(opener) >= s.count(s[-1]):
            break
        s = s[:-1]
    return s


def normalize_doi(raw):
    """Normalize a DOI string from various input formats.

    Accepts a bare DOI, a ``doi:`` prefixed form, or a ``doi.org`` /
    ``dx.doi.org`` URL, and strips trailing punctuation picked up from
    surrounding prose (brackets the DOI itself opened are kept). Returns
    the canonical bare DOI, or ``None`` when the input is not a DOI.

    Case is preserved: DOIs are case-insensitive for resolution, but some
    consumers (Scite among them) echo back what they were given.
    """
    if not raw:
        return None
    s = str(raw).strip()
    if s.lower().startswith("doi:"):
        s = s[4:].strip()
    if s.lower().startswith("http://") or s.lower().startswith("https://"):
        m = _DOI_IN_URL_RE.search(s)
        if not m:
            return None
        s = m.group(1)
    s = _strip_trailing_punct(s)
    if DOI_RE.match(s):
        return s
    return None


def doi_match_key(raw):
    """Case-folded canonical DOI for equality tests, or ``None``.

    ``normalize_doi`` stays case-preserving because some consumers echo a
    DOI back to the user; this is the case-folded counterpart for callers
    that only need to answer "is this the same DOI?".
    """
    doi = normalize_doi(raw)
    return doi.lower() if doi else None


def normalize_isbn(raw):
    """Normalize an ISBN string and validate the checksum.

    Accepts ISBN-10, ISBN-13, and prefixed/URL forms (isbn:, https://isbndb.com/...).
    Strips hyphens, spaces, and any prefix. Returns the canonical digits-only
    form (13-digit preferred — ISBN-10 inputs are converted to ISBN-13).
    Returns None on invalid input or failing checksum.
    """
    if not raw:
        return None
    s = str(raw).strip()
    if s.lower().startswith("isbn:"):
        s = s[5:].strip()
    if s.lower().startswith("isbn-") or s.lower().startswith("isbn "):
        s = s[5:].strip()
    if s.lower().startswith("http://") or s.lower().startswith("https://"):
        m = re.search(r"/(97[89][\- ]?\d[\- ]?\d{3}[\- ]?\d{5}[\- ]?\d|\d{9}[\dX])",
                      s, flags=re.IGNORECASE)
        if not m:
            return None
        s = m.group(1)
    digits = re.sub(r"[\s\-]", "", s)
    if re.match(r"^\d{9}[\dXx]$", digits):
        if not _isbn10_checksum_valid(digits):
            return None
        return _isbn10_to_isbn13(digits)
    if re.match(r"^97[89]\d{10}$", digits):
        if not _isbn13_checksum_valid(digits):
            return None
        return digits
    return None


def _isbn10_checksum_valid(s):
    total = 0
    for i, ch in enumerate(s):
        v = 10 if ch in ("X", "x") else int(ch)
        total += v * (10 - i)
    return total % 11 == 0


def _isbn13_checksum_valid(s):
    total = 0
    for i, ch in enumerate(s):
        v = int(ch)
        total += v if i % 2 == 0 else v * 3
    return total % 10 == 0


def _isbn10_to_isbn13(isbn10):
    core = "978" + isbn10[:9]
    total = 0
    for i, ch in enumerate(core):
        total += int(ch) * (1 if i % 2 == 0 else 3)
    check = (10 - total % 10) % 10
    return core + str(check)


def isbn_match_keys(raw):
    """Every valid ISBN in a Zotero ISBN field (space/comma/semicolon separated), as ISBN-13."""
    if not raw:
        return frozenset()
    return frozenset(n for tok in re.split(r"[,;\s]+", str(raw)) if tok and (n := normalize_isbn(tok)))


_ARXIV_LEGACY_RE = r"[a-z][a-z\-]*(?:\.[a-z][a-z\-]*)?/\d{7}(?:v\d+)?"


def normalize_arxiv_id(raw):
    """Normalize an arXiv ID from various input formats."""
    if not raw:
        return None
    s = raw.strip()
    if s.lower().startswith("arxiv:"):
        s = s[6:].strip()
    if s.lower().startswith("http://") or s.lower().startswith("https://"):
        m = re.search(
            r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|"
            + _ARXIV_LEGACY_RE + r")(?:\.pdf)?",
            s, flags=re.IGNORECASE,
        )
        if not m:
            return None
        s = m.group(1)
    if re.match(r"^[0-9]{4}\.[0-9]{4,5}(?:v\d+)?$", s):
        return s
    if re.match(rf"^{_ARXIV_LEGACY_RE}$", s, flags=re.IGNORECASE):
        return s
    return None


# arXiv's DataCite DOIs are minted as 10.48550/arXiv.<id>, which is what
# Zotero puts in the DOI field for a preprint imported from arXiv.
_ARXIV_DOI_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/)?10\.48550/arxiv\.(.+)$",
                           re.IGNORECASE)
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)


def arxiv_identity(raw):
    """The version-independent arXiv identity of an ID, URL, DOI or archiveID.

    ``normalize_arxiv_id`` deliberately keeps the ``v2`` suffix: callers use
    its result to fetch a specific version from arXiv. Deduplication wants the
    opposite — 2401.00001v1 and 2401.00001v2 are the same paper and must not
    become two library items — so identity comparison goes through here
    instead. This also accepts arXiv's DataCite DOI form, so an item added by
    DOI is recognized by a later add of the same paper's arXiv ID.

    Returns the bare, unversioned ID, or None if ``raw`` isn't an arXiv
    identifier in any of those forms.
    """
    if not raw:
        return None
    s = str(raw).strip()
    m = _ARXIV_DOI_RE.match(s)
    if m:
        s = m.group(1)
    ident = normalize_arxiv_id(s)
    if not ident:
        return None
    return _ARXIV_VERSION_RE.sub("", ident)


#: Matches an ``arXiv:<id>`` line in a Zotero item's Extra field.
_ARXIV_EXTRA_RE = re.compile(r"^\s*arxiv:\s*(\S+)", re.IGNORECASE | re.MULTILINE)


def arxiv_identity_from_extra(extra):
    """arXiv identity parsed from an ``arXiv:<id>`` line in an Extra field, or ``None``."""
    m = _ARXIV_EXTRA_RE.search(extra or "")
    return arxiv_identity(m.group(1)) if m else None


#: An HTML/XML start or end tag, e.g. ``<i>`` or ``</sub>``. Stripped before
#: entities are unescaped, so an escaped tag like ``&lt;i&gt;`` survives the
#: strip and is unescaped into literal ``<i>`` text, not removed.
_TITLE_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")

_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+")


def normalize_title_for_matching(title):
    """Fold a title down to a whitespace-normalised, case-folded key.

    Strips markup tags, unescapes HTML entities, decomposes accents
    (NFKD) and drops combining marks, maps punctuation/symbol/separator
    characters to spaces, collapses whitespace, case-folds, and drops a
    single leading English article. Returns ``""`` for falsy input.
    """
    if not title:
        return ""
    s = unicodedata.normalize("NFKD", html.unescape(_TITLE_TAG_RE.sub(" ", str(title))))
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat[0] == "M":
            continue
        out.append(" " if cat[0] in "PSZ" or ch.isspace() else ch)
    s = " ".join("".join(out).casefold().split())
    return _LEADING_ARTICLE_RE.sub("", s)


def _fields_of(item_like) -> dict:
    """pyzotero item {"data": {...}}, bare data dict, or an object such as local_db.ZoteroItem
    (attributes doi, title, extra; no url/ISBN/archiveID)."""
    if isinstance(item_like, dict):
        data = item_like.get("data")
        return data if isinstance(data, dict) else item_like
    fields = {}
    for name, attrs in (
        ("DOI", ("doi", "DOI")),
        ("ISBN", ("isbn", "ISBN")),
        ("title", ("title",)),
        ("url", ("url",)),
        ("archiveID", ("archive_id", "archiveID")),
        ("extra", ("extra",)),
    ):
        for attr in attrs:
            if v := getattr(item_like, attr, None):
                fields[name] = v
                break
    return fields


def metadata_match_keys(item_like) -> frozenset[tuple[str, str]]:
    """Every (kind, value) this item could match under. Kinds: doi, arxiv, isbn, title.
    Which kinds decide is the caller's policy."""
    f = _fields_of(item_like)
    keys: set[tuple[str, str]] = set()
    if d := doi_match_key(f.get("DOI")):
        keys.add(("doi", d))
    for field in ("url", "archiveID", "DOI"):
        if a := arxiv_identity(f.get(field)):
            keys.add(("arxiv", a))
    if a := arxiv_identity_from_extra(f.get("extra")):
        keys.add(("arxiv", a))
    keys.update(("isbn", i) for i in isbn_match_keys(f.get("ISBN")))
    if t := normalize_title_for_matching(f.get("title")):
        keys.add(("title", t))
    return frozenset(keys)
