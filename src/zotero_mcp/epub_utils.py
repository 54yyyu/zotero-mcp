"""
EPUB utility functions for Zotero annotation creation.

This module finds a quote in an EPUB and builds the position Zotero stores
for a highlight on it: an EPUB CFI (Canonical Fragment Identifier) range in a
WADM (Web Annotation Data Model) FragmentSelector, and the sort index.

Zotero's reader resolves CFIs with epub.js, so the steps written here follow
epub.js rather than the letter of the CFI spec: a text step counts the
parent's text nodes one by one, whitespace-only ones included, and the offset
is a position in that text node's raw characters.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

# =============================================================================
# Text Normalization
# =============================================================================


def normalize_text_for_search(text: str) -> str:
    """
    Normalize text for searching.

    - Collapse all whitespace to single spaces
    - Normalize smart quotes to ASCII equivalents
    - Trim leading/trailing whitespace
    """
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize smart single quotes
    text = text.replace('‘', "'").replace('’', "'")
    # Normalize smart double quotes
    text = text.replace('“', '"').replace('”', '"')
    return text.strip()


# =============================================================================
# CFI Text Parser
# =============================================================================


@dataclass
class TextNodeInfo:
    """
    One text node of a parsed document that has something to search.

    This simulates DOM text nodes for CFI generation without requiring
    an actual DOM implementation.
    """

    # Normalized text content of this node (used for searching)
    text: str
    # Original text content (used for offset calculations)
    original_text: str
    # Position in accumulated document text (normalized)
    doc_start: int
    doc_end: int
    # Path of element indices from body to parent element (0-indexed)
    element_path: list[int]
    # Index of this node among ALL text nodes of its parent, as epub.js counts
    text_node_index: int
    # Raw characters of body text before this node (for the sort index)
    raw_start: int


@dataclass
class TextSearchResult:
    """Result of a text search in a document."""

    # Start position in accumulated text
    start_pos: int
    # End position in accumulated text
    end_pos: int
    # The text node containing the start
    start_node: TextNodeInfo
    # Offset within start node's raw text
    start_offset: int
    # The text node containing the end
    end_node: TextNodeInfo
    # Offset within end node's raw text
    end_offset: int
    # Matched text
    matched_text: str


class CFITextParser(HTMLParser):
    """
    HTML parser that extracts text while tracking precise positions for CFI generation.

    This parser:
    - Tracks element indices among element siblings only
    - Tracks text node indices among text node siblings only, counting every
      text node the way a DOM has them, whitespace-only ones included
    - Builds accumulated text with space separators between text nodes
    - Maintains proper position mappings for CFI path generation

    Important: The root element (html) is NOT included in the CFI path.
    The path starts from body's children (matching Zotero's internal format).

    Named and numeric character references are decoded by HTMLParser itself,
    so a text node arrives as the characters a DOM would hold.
    """

    def __init__(self):
        super().__init__()
        # Accumulated text (normalized with space separators)
        self.accumulated_text = ""
        # Text nodes that carry text
        self.text_nodes: list[TextNodeInfo] = []
        # Raw characters of body text seen so far
        self.raw_length = 0

        # Parsing state
        # Stack of open element tags
        self.element_stack: list[str] = []
        # Current element path (0-indexed element indices) - excludes root html
        self.element_path: list[int] = []
        # Element child counts at each nesting level (for calculating sibling index)
        self.element_child_counts: list[int] = [0]
        # Text node counts at each nesting level
        self.text_child_counts: list[int] = [0]

        # Elements to skip entirely (their content is skipped)
        self.skip_elements = {'script', 'style', 'head', 'meta', 'link'}
        self.skip_depth = 0

        # Track nesting depth to know if we're at the root level
        # The root element (html) should not be added to the path
        self.root_elements = {'html'}
        self.at_root = True

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        tag = tag.lower()

        # If we're already inside a skipped element, don't count children
        if self.skip_depth > 0:
            if tag in self.skip_elements:
                self.skip_depth += 1
            return

        if tag in self.skip_elements:
            self.skip_depth += 1
            # Still increment element child count so sibling indices are correct
            if self.element_child_counts and not self.at_root:
                self.element_child_counts[-1] += 1
            return

        # Always increment element child count at current level
        # (this tracks siblings correctly even for skipped elements)
        if self.element_child_counts:
            self.element_child_counts[-1] += 1

        # Element index among element siblings (0-indexed)
        element_index = self.element_child_counts[-1] - 1

        self.element_stack.append(tag)

        # Don't add root element (html) to the path
        # The CFI path starts from children of the document, not including html
        if tag not in self.root_elements:
            self.element_path.append(element_index)
        else:
            # We're leaving root level
            self.at_root = False

        # New level for children
        self.element_child_counts.append(0)
        self.text_child_counts.append(0)

    def handle_endtag(self, tag: str):
        tag = tag.lower()

        if tag in self.skip_elements:
            self.skip_depth = max(0, self.skip_depth - 1)
            return

        if self.skip_depth > 0:
            return

        if self.element_stack and self.element_stack[-1] == tag:
            self.element_stack.pop()

            # Only pop from element_path if we added to it
            if tag not in self.root_elements and self.element_path:
                self.element_path.pop()

            if self.element_child_counts:
                self.element_child_counts.pop()
            if self.text_child_counts:
                self.text_child_counts.pop()

    def handle_data(self, data: str):
        if self.skip_depth > 0:
            return

        # Every text node takes an index, also one that is only the line
        # break and indentation between two tags: epub.js resolves a text
        # step by counting the parent's text nodes, so leaving those out
        # points every later sibling one node too early.
        text_node_index = self.text_child_counts[-1]
        self.text_child_counts[-1] += 1
        raw_start = self.raw_length
        if "body" in self.element_stack:
            self.raw_length += len(data)

        normalized = normalize_text_for_search(data)
        if not normalized:
            return

        # Calculate start position in accumulated text
        # Add space separator if not at beginning
        doc_start = len(self.accumulated_text)
        if doc_start > 0:
            self.accumulated_text += " "
            doc_start += 1

        self.accumulated_text += normalized
        doc_end = len(self.accumulated_text)

        self.text_nodes.append(TextNodeInfo(
            text=normalized,
            original_text=data,  # Keep original for offset mapping
            doc_start=doc_start,
            doc_end=doc_end,
            element_path=list(self.element_path),
            text_node_index=text_node_index,
            raw_start=raw_start,
        ))

    def find_text_nodes_for_range(
        self,
        start_pos: int,
        end_pos: int
    ) -> tuple[TextNodeInfo, int, TextNodeInfo, int] | None:
        """
        Find the text nodes and offsets for a character range.

        Args:
            start_pos: Start position in accumulated (normalized) text
            end_pos: End position in accumulated (normalized) text

        Returns:
            Tuple of (start_node, start_offset, end_node, end_offset) or None
            Offsets are mapped back to original text positions.
        """
        start_node = None
        start_offset_normalized = 0
        end_node = None
        end_offset_normalized = 0

        for node in self.text_nodes:
            # Find start node
            if start_node is None and node.doc_start <= start_pos < node.doc_end:
                start_node = node
                start_offset_normalized = start_pos - node.doc_start

            # Find end node
            if node.doc_start < end_pos <= node.doc_end:
                end_node = node
                end_offset_normalized = end_pos - node.doc_start

            if start_node and end_node:
                break

        if not start_node or not end_node:
            return None

        # Map offsets from normalized text back to original text
        start_offset = _map_normalized_to_original_offset(
            start_node.original_text, start_offset_normalized
        )
        end_offset = _map_normalized_to_original_offset(
            end_node.original_text, end_offset_normalized
        )

        return (start_node, start_offset, end_node, end_offset)


def _map_normalized_to_original_offset(original: str, normalized_offset: int) -> int:
    """
    Map a position in a node's normalized text back to its raw text.

    Normalizing strips the whitespace a node starts with and collapses every
    later run of whitespace to one space, so the raw position is past all of
    the leading whitespace and past the whole of each run before it.

    Args:
        original: The raw text of the node
        normalized_offset: Offset in the node's normalized text

    Returns:
        Corresponding offset in the raw text
    """
    orig_pos = len(original) - len(original.lstrip())
    norm_pos = 0

    while norm_pos < normalized_offset and orig_pos < len(original):
        if original[orig_pos].isspace():
            while orig_pos < len(original) and original[orig_pos].isspace():
                orig_pos += 1
        else:
            orig_pos += 1
        norm_pos += 1

    return orig_pos


def find_text_in_document(
    parser: CFITextParser,
    search_text: str,
    use_fuzzy_match: bool = False,
) -> TextSearchResult | None:
    """
    Find text in a parsed document and return position information.

    Args:
        parser: A CFITextParser that has parsed the document
        search_text: Text to search for
        use_fuzzy_match: If True, ignore whitespace differences

    Returns:
        TextSearchResult or None if not found
    """
    accumulated_text = parser.accumulated_text
    normalized_search = normalize_text_for_search(search_text)

    search_start_pos = -1
    search_end_pos = -1

    if not use_fuzzy_match:
        # Exact match (case-insensitive)
        search_start_pos = accumulated_text.lower().find(normalized_search.lower())
        if search_start_pos != -1:
            search_end_pos = search_start_pos + len(normalized_search)
    else:
        # Fuzzy match: compare with all whitespace removed
        text_no_spaces = re.sub(r'\s+', '', accumulated_text).lower()
        search_no_spaces = re.sub(r'\s+', '', normalized_search).lower()

        fuzzy_pos_no_spaces = text_no_spaces.find(search_no_spaces) if search_no_spaces else -1

        if fuzzy_pos_no_spaces != -1:
            # Map position back to original text
            non_spaces_seen = 0
            for i, char in enumerate(accumulated_text):
                if not char.isspace():
                    if non_spaces_seen == fuzzy_pos_no_spaces:
                        search_start_pos = i
                        break
                    non_spaces_seen += 1

            # Find end: count search_no_spaces.length non-space chars from start
            if search_start_pos != -1:
                non_spaces_seen = 0
                for i in range(search_start_pos, len(accumulated_text)):
                    if not accumulated_text[i].isspace():
                        non_spaces_seen += 1
                        if non_spaces_seen == len(search_no_spaces):
                            search_end_pos = i + 1
                            break

    if search_start_pos == -1 or search_end_pos == -1:
        return None

    # Find the text nodes for this range
    result = parser.find_text_nodes_for_range(search_start_pos, search_end_pos)
    if not result:
        return None

    start_node, start_offset, end_node, end_offset = result

    return TextSearchResult(
        start_pos=search_start_pos,
        end_pos=search_end_pos,
        start_node=start_node,
        start_offset=start_offset,
        end_node=end_node,
        end_offset=end_offset,
        matched_text=accumulated_text[search_start_pos:search_end_pos],
    )


def build_cfi(result: TextSearchResult, spine_index: int) -> str:
    """
    The ``epubcfi(...)`` range for a text search result.

    Format: ``epubcfi(/6/<spine>!<common path>,<start>,<end>)``. Element
    steps are even, ``(index + 1) * 2``; text steps are odd,
    ``1 + 2 * index``; ``/6`` is the spine in the package document.

    Args:
        result: The text search result
        spine_index: 0-indexed spine position
    """
    start_path, end_path = result.start_node.element_path, result.end_node.element_path
    shared = 0
    while shared < min(len(start_path), len(end_path)) and start_path[shared] == end_path[shared]:
        shared += 1

    def steps(element_indices: list[int]) -> str:
        return "".join(f"/{(index + 1) * 2}" for index in element_indices)

    def end_point(node: TextNodeInfo, offset: int) -> str:
        return f"{steps(node.element_path[shared:])}/{1 + 2 * node.text_node_index}:{offset}"

    return (
        f"epubcfi(/6/{(spine_index + 1) * 2}!{steps(start_path[:shared])},"
        f"{end_point(result.start_node, result.start_offset)},"
        f"{end_point(result.end_node, result.end_offset)})"
    )


# =============================================================================
# EPUB Parsing
# =============================================================================


def _read_spine(archive: zipfile.ZipFile) -> list[str]:
    """Archive paths of an EPUB's spine documents, in reading order."""
    container_root = ET.fromstring(archive.read('META-INF/container.xml'))
    rootfile = container_root.find('.//{*}rootfile')
    if rootfile is None:
        raise ValueError("Could not find rootfile in container.xml")

    opf_path = rootfile.get('full-path')
    opf_dir = opf_path.rsplit('/', 1)[0] + '/' if '/' in opf_path else ''
    opf_root = ET.fromstring(archive.read(opf_path))

    manifest = {
        item.get('id'): opf_dir + item.get('href')
        for item in opf_root.findall('.//{*}item')
        if item.get('id') and item.get('href')
    }
    return [
        manifest[itemref.get('idref')]
        for itemref in opf_root.findall('.//{*}itemref')
        if itemref.get('idref') in manifest
    ]


class EpubBook:
    """An EPUB's spine, each document parsed the first time it is searched.

    Finding a quote used to parse every document of the book twice, and a
    batch did that again for each of its highlights. One EpubBook serves all
    the lookups of a call and parses a chapter at most once.
    """

    def __init__(self, epub_path: str):
        self.path = epub_path
        with zipfile.ZipFile(epub_path) as archive:
            self.spine = _read_spine(archive)
        self._parsers: dict[int, CFITextParser | None] = {}

    def parser(self, spine_index: int) -> CFITextParser | None:
        """The parsed document at a spine position, or None when it cannot be read."""
        if spine_index not in self._parsers:
            try:
                with zipfile.ZipFile(self.path) as archive:
                    html = archive.read(self.spine[spine_index]).decode('utf-8')
                parser = CFITextParser()
                parser.feed(html)
                parser.close()
            except Exception:
                parser = None
            self._parsers[spine_index] = parser
        return self._parsers[spine_index]


def build_epub_annotation_position(cfi: str) -> str:
    """
    Build the annotationPosition JSON for Zotero EPUB annotations.

    Zotero uses WADM FragmentSelector format.
    """
    position = {
        "type": "FragmentSelector",
        "conformsTo": "http://www.idpf.org/epub/linking/cfi/epub-cfi.html",
        "value": cfi,
    }
    return json.dumps(position)


# =============================================================================
# Public API
# =============================================================================

def verify_epub_attachment(file_path: str) -> bool:
    """
    Whether a file is an EPUB: a zip with an OCF container and, if present,
    the EPUB mimetype entry.

    A structural check only. Parsing the whole book here, as this used to,
    repeated the parse that searching it does anyway.
    """
    try:
        with zipfile.ZipFile(file_path) as archive:
            names = set(archive.namelist())
            if "META-INF/container.xml" not in names:
                return False
            return "mimetype" not in names or archive.read("mimetype").strip() == b"application/epub+zip"
    except (OSError, zipfile.BadZipFile):
        return False


def find_text_in_epub(
    epub_path: str,
    chapter_num: int,
    search_text: str,
    cache: dict | None = None,
) -> dict:
    """
    Find text in an EPUB and return position data for annotation.

    The chapter asked for is searched first, then the others from the nearest
    outwards, so a phrase that is also in the table of contents or an earlier
    chapter is highlighted where the caller pointed.

    Args:
        epub_path: Path to the EPUB file
        chapter_num: 1-indexed chapter number (spine position) to look in first
        search_text: Text to find
        cache: A dict kept by the caller across lookups in the same files;
            it holds each parsed book

    Returns:
        On success: 'cfi', 'annotation_position', 'matched_text',
        'chapter_found' (1-indexed), 'spine_index' and 'char_position' (the
        raw characters of the chapter before the match), which together make
        the sort index Zotero's reader writes. On failure: 'error'.
    """
    book = cache.get(epub_path) if cache is not None else None
    if book is None:
        try:
            book = EpubBook(epub_path)
        except Exception as e:
            return {'error': f'Failed to parse EPUB: {e}'}
        if cache is not None:
            cache[epub_path] = book

    hint = chapter_num - 1
    for spine_index in sorted(range(len(book.spine)), key=lambda i: (abs(i - hint), i)):
        parser = book.parser(spine_index)
        if parser is None:
            continue
        # As written first, then ignoring whitespace (text split by inline tags).
        result = find_text_in_document(parser, search_text) or find_text_in_document(
            parser, search_text, use_fuzzy_match=True)
        if result is None:
            continue
        cfi = build_cfi(result, spine_index)
        return {
            'cfi': cfi,
            'annotation_position': build_epub_annotation_position(cfi),
            'matched_text': result.matched_text,
            'chapter_found': spine_index + 1,
            'spine_index': spine_index,
            'char_position': result.start_node.raw_start + result.start_offset,
            'score': 1.0,
        }

    return {'error': "Text not found in EPUB"}
