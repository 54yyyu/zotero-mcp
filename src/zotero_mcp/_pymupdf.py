"""PyMuPDF under its current module name.

``import fitz`` prints a deprecation notice to stdout on recent PyMuPDF, which
breaks ``zotero-cli --json`` output (#642). Import from here instead:
``from zotero_mcp._pymupdf import fitz``. 1.24.2, our floor, only has ``fitz``.
"""

try:
    import pymupdf as fitz
except ImportError:  # PyMuPDF 1.24.2
    import fitz  # noqa: F401
