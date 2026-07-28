"""Tool for reading specific page ranges from PDF attachments."""

import os
import tempfile

from fastmcp import Context

from zotero_mcp import client as _client
from zotero_mcp import utils as _utils
from zotero_mcp._app import mcp
from zotero_mcp.config import load_config
from zotero_mcp.tools import _helpers


_TMPDIR_PREFIX = "zotero_pdf_"


def _cleanup_path(file_path: str) -> None:
    """Remove a PDF this module downloaded, along with the directory it made.

    Deletes the file's *parent directory*, so it must only ever be handed a
    path inside a directory this module created with ``mkdtemp``. Two things
    are checked before removing anything, both of which have bitten:

    - The directory's name must carry our ``zotero_pdf_`` prefix. A bare
      "is it under the temp dir" test is not enough: on Linux
      ``gettempdir()`` is ``/tmp``, so a path like ``/tmp/paper.pdf`` has
      ``/tmp`` as its parent and passes that test, and the call then wipes
      the entire system temp directory. (This is not hypothetical; a test
      stub returning ``/tmp/test.pdf`` did exactly that on CI, which
      presented as unrelated tests failing with FileNotFoundError on
      pytest's own temp root.) macOS hides the bug, because there
      ``gettempdir()`` is under ``/var/folders`` and the prefix never
      matches ``/tmp``.
    - The directory must still be a strict subdirectory of the temp root, so
      the root itself can never be the target.

    A file resolved out of the user's Zotero storage must never be passed
    here: deleting its parent takes the user's own copy of the PDF with it.
    """
    try:
        parent = os.path.dirname(os.path.abspath(file_path))
        temp_root = os.path.abspath(tempfile.gettempdir())
        if not os.path.isdir(parent):
            return
        if os.path.samefile(parent, temp_root):
            return
        if os.path.commonpath([parent, temp_root]) != temp_root:
            return
        if not os.path.basename(parent).startswith(_TMPDIR_PREFIX):
            return
        import shutil

        shutil.rmtree(parent, ignore_errors=True)
    except Exception:
        pass


def _get_pdf_path(item_key: str, ctx: Context) -> tuple[str, str, bool] | None:
    """Resolve a PDF attachment and return ``(file_path, title, is_temp)``.

    Tries local storage first (via LocalZoteroReader), then downloads via API.
    Returns None if no PDF attachment is found.

    ``is_temp`` says whether the caller owns the file. It is True only for a
    file downloaded into a directory this function created, which the caller
    must clean up. It is False for a file resolved out of the user's Zotero
    storage, which must be left alone: those paths point into the real
    library, and deleting one takes the user's copy of the PDF with it.
    """
    zot = _client.get_zotero_client()
    item = zot.item(item_key)

    # Try local storage first (persists on disk — no cleanup needed)
    try:
        from zotero_mcp.local_db import LocalZoteroReader

        if _utils.is_local_mode():
            with LocalZoteroReader(db_path=load_config().resolve_zotero_db_path()) as reader:
                # The key may name the PDF attachment itself. Attachments have
                # no children, so the parent scan below comes up empty and we
                # would wrongly report "No PDF attachment found" (#372).
                attachment = reader.get_attachment_by_key(item_key)
                if attachment and "pdf" in (attachment["content_type"] or "").lower():
                    resolved = reader._resolve_attachment_path(
                        item_key, attachment["zotero_path"] or ""
                    )
                    if not (resolved and resolved.exists()):
                        # Recorded filename drifted on disk — scan the folder (#291)
                        resolved = reader._scan_storage_for_attachment(
                            item_key, attachment["content_type"]
                        )
                    if resolved and resolved.exists():
                        return str(resolved), attachment["title"] or item_key, False

                local_item = reader.get_item_by_key(item_key)
                if local_item:
                    for att_key, path, ctype in reader._iter_parent_attachments(local_item.item_id):
                        if ctype == "application/pdf":
                            resolved = reader._resolve_attachment_path(att_key, path or "")
                            if resolved and resolved.exists():
                                return str(resolved), local_item.title or item_key, False
    except Exception:
        pass

    # Fallback: resolve via the multi-source downloader (local -> WebDAV ->
    # Zotero cloud) so WebDAV-backed attachments work, not just cloud storage.
    attachment = _client.get_attachment_details(zot, item)
    if not attachment:
        return None

    pdf_extensions = {".pdf", ".PDF"}
    filename = attachment.filename or f"{attachment.key}.pdf"
    if not any(filename.endswith(ext) for ext in pdf_extensions):
        content_type = attachment.content_type or ""
        if "pdf" not in content_type.lower():
            return None

    tmpdir = tempfile.mkdtemp(prefix="zotero_pdf_")
    probe = os.path.join(tmpdir, os.path.basename(filename))
    try:
        download = _client.download_attachment_file(
            attachment.key,
            tmpdir,
            os.path.basename(filename),
            local_client=_client.get_local_zotero_client(),
            web_client=None if _utils.is_local_mode() else zot,
        )
    except Exception:
        _cleanup_path(probe)
        raise

    if download.path and download.path.exists() and download.path.stat().st_size > 0:
        return str(download.path), attachment.title, True

    _cleanup_path(probe)
    return None


@mcp.tool(
    name="zotero_read_pdf_pages",
    description="Read specific page range(s) from a PDF attachment of a Zotero item. "
    "Use this when you know which pages to read — for example after getting the PDF "
    "outline via zotero_get_pdf_outline. Pages are 1-indexed. "
    "Requires PyMuPDF (the [pdf] extra).",
)
def read_pdf_pages(
    item_key: str,
    start_page: int,
    end_page: int | None = None,
    *,
    ctx: Context,
) -> str:
    """Extract and return text from a specific page range of a PDF.

    Args:
        item_key: Zotero item key/ID of the paper or its PDF attachment.
        start_page: First page to read (1-indexed).
        end_page: Last page to read (1-indexed). If omitted, reads only start_page.
        ctx: MCP context.

    Returns:
        Markdown-formatted page content with metadata header.
    """
    try:
        if not item_key or not item_key.strip():
            return "Error: item_key cannot be empty."

        if end_page is not None and end_page < start_page:
            return "Error: end_page must be greater than or equal to start_page."

        ctx.info(f"Reading PDF pages {start_page}-{end_page or start_page} for item {item_key}")

        result = _get_pdf_path(item_key, ctx)
        if result is None:
            return f"No PDF attachment found for item: {item_key}"

        pdf_path, title, is_temp = result

        def _release() -> None:
            """Drop the working copy, but never a file in the user's library."""
            if is_temp:
                _cleanup_path(pdf_path)

        try:
            import fitz
        except ImportError:
            return f"PyMuPDF is required for PDF page reading. {_utils.install_hint('pdf')}"

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        actual_end = end_page if end_page is not None else start_page

        if start_page < 1 or start_page > total_pages:
            doc.close()
            _release()
            return f"Start page {start_page} is out of range. PDF has {total_pages} pages (1-{total_pages})."
        if end_page is not None and end_page > total_pages:
            doc.close()
            _release()
            return f"End page {end_page} is out of range. PDF has {total_pages} pages (1-{total_pages})."

        # Zero-indexed page numbers for PyMuPDF
        zstart = start_page - 1
        zend = actual_end - 1

        output = [
            f"# PDF Pages {start_page}-{actual_end} of {title}",
            f"**Item Key:** {item_key}",
            f"**Total pages in PDF:** {total_pages}",
            "",
        ]

        page_count = zend - zstart + 1
        if page_count > 50:
            doc.close()
            _release()
            return f"Requested {page_count} pages (max 50). Please narrow your page range."

        for page_num in range(zstart, zend + 1):
            page = doc[page_num]
            text = page.get_text()
            output.append(f"## Page {page_num + 1}")
            output.append("")
            if text.strip():
                output.append(text.strip())
            else:
                output.append("*[No extractable text on this page]*")
            output.append("")

        doc.close()
        _release()
        return _helpers._prepend_size_warning(
            "\n".join(output),
            "Consider using zotero_semantic_search to find specific content instead of reading full pages.",
        )

    except Exception as e:
        ctx.error(f"Error reading PDF pages: {str(e)}")
        return f"Error reading PDF pages: {str(e)}"
