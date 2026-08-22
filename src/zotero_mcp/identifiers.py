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

import re

__all__ = ["normalize_doi"]

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
