"""Tool for reading specific page ranges from PDF attachments."""

import os
import tempfile
from contextlib import contextmanager
from typing import Literal

from fastmcp import Context
from fastmcp.utilities.types import Image
from fastmcp.exceptions import ToolError

from zotero_mcp import client as _client
from zotero_mcp import library as _library
from zotero_mcp import utils as _utils
from zotero_mcp._app import mcp
from zotero_mcp.config import load_config
from zotero_mcp.extract import extract_pdf, pdf_page_count
from zotero_mcp.tools import _helpers

_TMPDIR_PREFIX = "zotero_pdf_"


class PdfReadError(ToolError):
    """A page read that did not produce pages.

    This tool used to *return* its failures as prose -- "No PDF attachment
    found for item: ...", "Could not read PDF for item ...". A return value is
    indistinguishable from content, so every caller treated a failed read as a
    successful one: ``zotero-cli --json read`` wrapped the message in an
    ``ok: true`` envelope and exited 0, and the MCP tool answered with
    ``isError: false``. A pipeline consuming either could not tell "here are
    the pages" from "there are no pages" without parsing English.

    Raising fixes both surfaces at once, because both already know how to
    report an exception: FastMCP marks the tool result as an error, and
    ``cli_standalone.main`` turns it into an ``ok: false`` envelope with a
    nonzero exit code. Neither needed a change.

    Subclasses ``ToolError`` so FastMCP treats it as a tool error rather than
    an internal crash, and carries ``code`` so the envelope's ``error.code``
    is a stable value a caller can branch on instead of the class name.
    """

    def __init__(self, message: str, code: str = "pdf_read_failed"):
        super().__init__(message)
        self.code = code


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
    item = _library.get_library_backend().get_item(item_key)

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
                    resolved = reader.resolve_attachment_file(item_key)
                    if resolved:
                        return str(resolved), attachment["title"] or item_key, False

                local_item = reader.get_item_by_key(item_key)
                if local_item:
                    for att_key, _path, ctype in reader._iter_parent_attachments(local_item.item_id):
                        if ctype == "application/pdf":
                            resolved = reader.resolve_attachment_file(att_key)
                            if resolved:
                                return str(resolved), local_item.title or item_key, False
    except Exception:
        pass

    # Fallback: resolve via the multi-source downloader (local -> WebDAV ->
    # Zotero cloud) so WebDAV-backed attachments work, not just cloud storage.
    # PDF only: this tool renders page ranges, so a markdown-first
    # attachment_priority must not hand it a file it cannot paginate.
    # Everything below is the download fallback, reached only when the file
    # is not in local storage. The API client is built here so a PDF already
    # on disk never needs one.
    zot = _client.get_zotero_client()
    attachment = _client.get_attachment_details(zot, item, priority=("pdf",))
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


#: Most pages one text read returns.
_TEXT_MAX_PAGES = 50
#: Most pages one image read returns. A page image costs a vision model a few
#: thousand tokens, so an image read is for the pages that need one.
_IMAGE_MAX_PAGES = 10
#: Long edge of a rendered image in pixels. Vision models downscale anything
#: larger, so rendering past it only makes the response heavier.
_IMAGE_MAX_EDGE = 1568
#: Cap on magnification, so a crop of a few words is not blown up to mush.
_IMAGE_MAX_ZOOM = 4.0
#: Inline math characters on a page before its text is flagged as garbled.
#: A few variable names survive extraction; dense notation does not.
_INLINE_MATH_FLAG = 25

_IMAGE_HINTS = {
    "mcp": (
        "*Flagged pages have math, figures or tables that text extraction garbles. "
        "Read them with format='image'; add rect=[x, y, width, height] from "
        "zotero_get_page_layout to zoom into one of them.*"
    ),
    "cli": (
        "*Flagged pages have math, figures or tables that text extraction garbles. "
        "View them with `zotero-cli read {item_key} --start-page N --format image`; add "
        "--rect x,y,width,height from `zotero-cli layout ATTACHMENT_KEY` to zoom into one of them.*"
    ),
}


@mcp.tool(
    name="zotero_read_pdf_pages",
    description="Read specific page range(s) from a PDF attachment of a Zotero item. "
    "Use this when you know which pages to read — for example after getting the PDF "
    "outline via zotero_get_pdf_outline. Pages are 1-indexed. "
    "format='text' (default) returns Markdown with the heading structure preserved and "
    "flags pages whose equations, figures or tables the text garbles. "
    "format='image' returns the pages as PNG images (up to 10) so those can be read "
    "exactly; rect=[x, y, width, height] (normalized 0-1, e.g. from "
    "zotero_get_page_layout) returns just that region of start_page, zoomed in.",
    # Text or a list of text and images, so no single structured schema fits.
    output_schema=None,
)
def read_pdf_pages(
    item_key: str,
    start_page: int,
    end_page: int | None = None,
    format: Literal["text", "image"] = "text",
    rect: list[float] | str | None = None,
    *,
    ctx: Context,
) -> str | list:
    """Read a page range of an item's PDF as Markdown, or as page images.

    Args:
        item_key: Zotero item key/ID of the paper or its PDF attachment.
        start_page: First page to read (1-indexed).
        end_page: Last page to read (1-indexed). If omitted, reads only start_page.
        format: "text" for Markdown, "image" for PNG page images.
        rect: With format="image", crop start_page to [x, y, width, height].
        ctx: MCP context.
    """
    if format == "image":
        header, pages = render_pdf_pages(item_key, start_page, end_page, rect=rect, ctx=ctx)
        return [header, *(Image(data=page["png"], format="png") for page in pages)]
    if format != "text":
        raise PdfReadError(f"format must be 'text' or 'image', got {format!r}", code="bad_format")
    return read_pdf_text(item_key, start_page, end_page, ctx=ctx)


@contextmanager
def _page_range(
    item_key: str,
    start_page: int,
    end_page: int | None,
    *,
    max_pages: int,
    ctx: Context,
):
    """Validate a page range against an item's PDF.

    Yields ``(pdf_path, title, total_pages, end_page, clamped_note)`` and
    removes a downloaded working copy afterwards, never a file in the user's
    library. The range is checked before anything is resolved (#528).
    """
    if not item_key or not item_key.strip():
        raise PdfReadError("Error: item_key cannot be empty.", code="empty_item_key")
    if end_page is not None and end_page < start_page:
        raise PdfReadError(
            "Error: end_page must be greater than or equal to start_page.",
            code="invalid_page_range",
        )

    ctx.info(f"Reading PDF pages {start_page}-{end_page or start_page} for item {item_key}")

    result = _get_pdf_path(item_key, ctx)
    if result is None:
        raise PdfReadError(
            f"No PDF attachment found for item: {item_key}",
            code="no_pdf_attachment",
        )
    pdf_path, title, is_temp = result

    try:
        try:
            total_pages = pdf_page_count(pdf_path)
        except Exception as exc:
            raise PdfReadError(
                f"Could not read PDF for item {item_key}: {exc}",
                code="pdf_unreadable",
            ) from exc

        if start_page < 1 or start_page > total_pages:
            raise PdfReadError(
                f"Start page {start_page} is out of range. PDF has {total_pages} pages (1-{total_pages}).",
                code="page_out_of_range",
            )
        # A caller rarely knows the page count before the first read, and
        # "read to the end" is the usual intent behind an end page that is too
        # large. Clamp and say so instead of failing the whole read.
        actual_end = end_page if end_page is not None else start_page
        clamped_note = None
        if actual_end > total_pages:
            clamped_note = (
                f"*End page {actual_end} is past the last page; "
                f"read through page {total_pages}.*"
            )
            actual_end = total_pages

        requested = actual_end - start_page + 1
        if requested > max_pages:
            raise PdfReadError(
                f"Requested {requested} pages (max {max_pages}). Please narrow your page range.",
                code="page_limit_exceeded",
            )

        yield pdf_path, title, total_pages, actual_end, clamped_note
    finally:
        if is_temp:
            _cleanup_path(pdf_path)


def read_pdf_text(
    item_key: str,
    start_page: int,
    end_page: int | None = None,
    *,
    ctx: Context,
    surface: Literal["mcp", "cli"] = "mcp",
) -> str:
    """Markdown for a page range, with pages the text garbles flagged.

    ``surface`` picks whether the closing advice names the MCP tool's image
    format or the zotero-cli flag.
    """
    try:
        with _page_range(item_key, start_page, end_page,
                         max_pages=_TEXT_MAX_PAGES, ctx=ctx) as (pdf_path, title, total_pages,
                                                                 actual_end, clamped_note):
            try:
                # extract_pdf takes 0-indexed pages; the tool's API is 1-indexed.
                doc = extract_pdf(pdf_path, pages=list(range(start_page - 1, actual_end)))
            except Exception as exc:
                raise PdfReadError(
                    f"Could not read PDF for item {item_key}: {exc}",
                    code="pdf_unreadable",
                ) from exc
            flags = _garbled_content_flags(pdf_path, doc)

        output = [
            f"# PDF Pages {start_page}-{actual_end} of {title}",
            f"**Item Key:** {item_key}",
            f"**Total pages in PDF:** {total_pages}",
            "",
        ]
        if clamped_note:
            output.extend([clamped_note, ""])

        for page_index, markdown in zip(doc.page_numbers, doc.pages):
            output.append(f"## Page {page_index + 1}")
            output.append("")
            if markdown.strip():
                output.append(markdown.strip())
            elif page_index in doc.needs_ocr:
                output.append("*[No text layer on this page — it is a scanned image]*")
            else:
                output.append("*[No extractable text on this page]*")
            output.append("")
            if page_index in flags:
                output.extend([flags[page_index], ""])
        if flags:
            output.append(_IMAGE_HINTS[surface].format(item_key=item_key))
        return _helpers._prepend_size_warning(
            "\n".join(output),
            "Consider using zotero_semantic_search to find specific content instead of reading full pages.",
        )

    except PdfReadError:
        # Already carries the specific code; re-wrapping it here would bury
        # that under the generic one and repeat the message.
        raise
    except Exception as e:
        ctx.error(f"Error reading PDF pages: {str(e)}")
        raise PdfReadError(f"Error reading PDF pages: {str(e)}") from e


def _garbled_content_flags(pdf_path: str, doc) -> dict[int, str]:
    """A note per page (0-indexed) naming what its extracted text cannot carry.

    Display equations and dense inline math come from the page's fonts
    (``pdf_layout.scan_math``); figures and tables from their captions in the
    extracted Markdown, which is already in hand. A page with none of these
    gets no note. Best effort: without PyMuPDF, or on a file it cannot open,
    the read simply carries no flags.
    """
    try:
        import fitz

        from zotero_mcp.pdf_layout import _parse_caption_block, scan_math

        pdf = fitz.open(pdf_path)
    except Exception:
        return {}

    flags: dict[int, str] = {}
    try:
        for page_index, markdown in zip(doc.page_numbers, doc.pages):
            equations, inline_math = scan_math(pdf[page_index])
            numbered = [eq["label"] for eq in equations if eq["label"]]
            unnumbered = len(equations) - len(numbered)
            items = []
            if numbered:
                items.append(("Equation " if len(numbered) == 1 else "Equations ") + ", ".join(numbered))
            if unnumbered:
                items.append(f"{unnumbered} unnumbered equation{'s' if unnumbered > 1 else ''}")
            for line in markdown.splitlines():
                caption = _parse_caption_block(line.strip().lstrip("#*_ ").strip())
                if caption and caption["label"] not in items:
                    items.append(caption["label"])
            if inline_math >= _INLINE_MATH_FLAG:
                items.append("inline math")
            if items:
                flags[page_index] = f"> **Garbled in this text:** {', '.join(items)}"
    except Exception:
        return {}
    finally:
        pdf.close()
    return flags


def render_pdf_pages(
    item_key: str,
    start_page: int,
    end_page: int | None = None,
    *,
    rect: list[float] | str | None = None,
    ctx: Context,
) -> tuple[str, list[dict]]:
    """Render a page range, or one region of ``start_page``, to PNG.

    Returns:
        (header, pages): a Markdown header naming what was rendered, and one
        ``{"page", "png", "width", "height"}`` per image.
    """
    box = None
    if rect is not None:
        box = _helpers._normalize_float_list_input(rect, 4, "rect")
        if (
            box is None
            or not all(0 <= value <= 1 for value in box)
            or box[2] <= 0 or box[3] <= 0
            or box[0] + box[2] > 1.0001 or box[1] + box[3] > 1.0001
        ):
            raise PdfReadError(
                f"rect must be [x, y, width, height] within the page, normalized to 0-1; got {rect!r}",
                code="bad_rect",
            )
        if end_page not in (None, start_page):
            raise PdfReadError(
                "rect crops a single page; omit end_page or set it to start_page.",
                code="invalid_page_range",
            )
    try:
        import fitz
    except ImportError as exc:
        raise PdfReadError(
            f"Rendering pages requires PyMuPDF. {_utils.install_hint('pdf')}",
            code="missing_dependency",
        ) from exc

    with _page_range(item_key, start_page, end_page,
                     max_pages=_IMAGE_MAX_PAGES, ctx=ctx) as (pdf_path, title, total_pages,
                                                              actual_end, clamped_note):
        pdf = fitz.open(pdf_path)
        try:
            pages = []
            for number in range(start_page, actual_end + 1):
                page = pdf[number - 1]
                clip = page.rect
                if box is not None:
                    x, y, w, h = box
                    clip = fitz.Rect(
                        clip.x0 + x * clip.width, clip.y0 + y * clip.height,
                        clip.x0 + (x + w) * clip.width, clip.y0 + (y + h) * clip.height,
                    )
                zoom = min(_IMAGE_MAX_EDGE / max(clip.width, clip.height), _IMAGE_MAX_ZOOM)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
                pages.append({"page": number, "png": pixmap.tobytes("png"),
                              "width": pixmap.width, "height": pixmap.height})
        finally:
            pdf.close()

    what = (
        f"Region [{', '.join(f'{v:.4f}' for v in box)}] of page {start_page}"
        if box is not None else f"Pages {start_page}-{actual_end}"
    )
    header = [f"# {what} of {title}", f"**Item Key:** {item_key}",
              f"**Total pages in PDF:** {total_pages}"]
    if clamped_note:
        header.append(clamped_note)
    return "\n".join(header), pages
