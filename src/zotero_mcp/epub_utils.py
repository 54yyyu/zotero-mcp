"""
EPUB utility functions for Zotero annotation creation.

This module provides text search capabilities for EPUBs to extract position data
needed for creating Zotero highlight annotations. It generates EPUB CFI
(Canonical Fragment Identifiers) for text locations.

EPUB annotations in Zotero use the WADM (Web Annotation Data Model) format
with FragmentSelector containing EPUB CFI values.

The CFI generation logic was ported from foliate-js:
https://github.com/johnfactotum/foliate-js
(MIT License)
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Literal
from xml.etree import ElementTree as ET

if TYPE_CHECKING:
    from typing import Any


# =============================================================================
# EPUB CFI Data Structures
# =============================================================================


@dataclass
class EPUBCFIStep:
    """
    Represents a single step in a CFI path.

    In CFI notation:
    - Elements are even numbers: (index + 1) * 2
    - Text nodes are odd numbers: 1 + (2 * index)
    """

    type: Literal["element", "text"]
    index: int
    id: str | None = None

    def to_cfi(self) -> str:
        """Convert this step to CFI notation."""
        if self.type == "element":
            num = (self.index + 1) * 2
        else:  # text
            num = 1 + (2 * self.index)

        if self.id:
            return f"{num}[{self.id}]"
        return str(num)


@dataclass
class EPUBCFISegment:
    """
    A segment of a CFI path, consisting of steps and an optional terminal offset.
    """

    steps: list[EPUBCFIStep] = field(default_factory=list)
    terminal_offset: int | None = None

    def to_cfi(self) -> str:
        """Convert this segment to CFI notation."""
        if not self.steps:
            return ""

        path = "/" + "/".join(step.to_cfi() for step in self.steps)

        if self.terminal_offset is not None:
            path += f":{self.terminal_offset}"

        return path


@dataclass
class EPUBCFI:
    """
    Complete EPUB CFI representation.

    Format: epubcfi(/6/<spine>!<path>,<start>,<end>)

    For ranges:
    - base: /6/<spine_index> (spine reference)
    - path: common path to divergence point
    - start: path from divergence to start + offset
    - end: path from divergence to end + offset
    """

    base: EPUBCFISegment = field(default_factory=EPUBCFISegment)
    path: EPUBCFISegment = field(default_factory=EPUBCFISegment)
    start: EPUBCFISegment | None = None
    end: EPUBCFISegment | None = None
    is_range: bool = False

    def to_string(self) -> str:
        """Convert to epubcfi(...) string format."""
        cfi = "epubcfi("
        cfi += self.base.to_cfi()
        cfi += "!"
        cfi += self.path.to_cfi()

        if self.is_range and self.start and self.end:
            cfi += ","
            cfi += self.start.to_cfi()
            cfi += ","
            cfi += self.end.to_cfi()

        cfi += ")"
        return cfi


@dataclass
class TextNodeInfo:
    """
    Information about a text node in the parsed HTML document.

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
    # Element ID if parent has one
    element_id: str | None = None
    # Index of this text node among text node siblings
    text_node_index: int = 0


@dataclass
class TextSearchResult:
    """Result of a text search in a document."""

    # Start position in accumulated text
    start_pos: int
    # End position in accumulated text
    end_pos: int
    # The text node containing the start
    start_node: TextNodeInfo
    # Offset within start node's text
    start_offset: int
    # The text node containing the end
    end_node: TextNodeInfo
    # Offset within end node's text
    end_offset: int
    # Matched text
    matched_text: str



# =============================================================================
# Text Normalization
# =============================================================================

# HTML entities that need to be replaced before parsing
HTML_ENTITY_REPLACEMENTS = {
    "&nbsp;": "\u00A0",    # Non-breaking space
    "&mdash;": "\u2014",   # Em dash
    "&ndash;": "\u2013",   # En dash
    "&lsquo;": "\u2018",   # Left single quote
    "&rsquo;": "\u2019",   # Right single quote
    "&ldquo;": "\u201C",   # Left double quote
    "&rdquo;": "\u201D",   # Right double quote
    "&hellip;": "\u2026",  # Ellipsis
}

def replace_html_entities(html: str) -> str:
    """Replace HTML entities with their Unicode equivalents before parsing."""
    for entity, char in HTML_ENTITY_REPLACEMENTS.items():
        html = html.replace(entity, char)
    return html


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
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    # Normalize smart double quotes
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    return text.strip()


# =============================================================================
# CFI Text Parser
# =============================================================================


class CFITextParser(HTMLParser):
    """
    HTML parser that extracts text while tracking precise positions for CFI generation.

    This parser:
    - Tracks element indices among element siblings only
    - Tracks text node indices among text node siblings only
    - Builds accumulated text with space separators between text nodes
    - Maintains proper position mappings for CFI path generation

    Important: The root element (html) is NOT included in the CFI path.
    The path starts from body's children (matching Zotero's internal format).
    """

    def __init__(self):
        super().__init__()
        # Accumulated text (normalized with space separators)
        self.accumulated_text = ""
        # List of TextNodeInfo objects
        self.text_nodes: list[TextNodeInfo] = []

        # Parsing state
        # Stack of (tag, element_index, element_id, text_child_count)
        # element_index is index among element siblings at parent level
        self.element_stack: list[tuple[str, int, str | None, int]] = []
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

        # Get element id if present
        element_id = None
        for name, value in attrs:
            if name.lower() == 'id' and value:
                element_id = value
                break

        # Track text child count at this level
        text_child_count = 0

        self.element_stack.append((tag, element_index, element_id, text_child_count))

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

        if self.element_stack and self.element_stack[-1][0] == tag:
            popped_tag, _, _, _ = self.element_stack.pop()

            # Only pop from element_path if we added to it
            if popped_tag not in self.root_elements and self.element_path:
                self.element_path.pop()

            if self.element_child_counts:
                self.element_child_counts.pop()
            if self.text_child_counts:
                self.text_child_counts.pop()

    def handle_data(self, data: str):
        if self.skip_depth > 0:
            return

        # Skip pure whitespace text nodes
        if not data.strip():
            return

        # Normalize the text content
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

        # Get text node index among text node siblings at current level
        if self.text_child_counts:
            text_node_index = self.text_child_counts[-1]
            self.text_child_counts[-1] += 1
        else:
            text_node_index = 0

        # Get parent element id
        element_id = None
        if self.element_stack:
            element_id = self.element_stack[-1][2]

        # Record this text node with both normalized and original text
        self.text_nodes.append(TextNodeInfo(
            text=normalized,
            original_text=data,  # Keep original for offset mapping
            doc_start=doc_start,
            doc_end=doc_end,
            element_path=list(self.element_path),
            element_id=element_id,
            text_node_index=text_node_index,
        ))

    def get_accumulated_text(self) -> str:
        """Get the full accumulated text."""
        return self.accumulated_text

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
            start_node.text, start_node.original_text, start_offset_normalized
        )
        end_offset = _map_normalized_to_original_offset(
            end_node.text, end_node.original_text, end_offset_normalized
        )

        return (start_node, start_offset, end_node, end_offset)


def _map_normalized_to_original_offset(
    normalized: str,
    original: str,
    normalized_offset: int,
) -> int:
    """
    Map a position in normalized text back to position in original text.

    This handles the case where multiple whitespace characters in the
    original text are collapsed to a single space in normalized text.

    Args:
        normalized: The normalized text
        original: The original text
        normalized_offset: Offset in the normalized text

    Returns:
        Corresponding offset in the original text
    """
    norm_pos = 0
    orig_pos = 0

    while norm_pos < normalized_offset and orig_pos < len(original):
        orig_char = original[orig_pos]

        # Handle whitespace sequences
        if orig_char.isspace():
            # Count consecutive whitespace in original
            ws_count = 0
            temp_pos = orig_pos
            while temp_pos < len(original) and original[temp_pos].isspace():
                ws_count += 1
                temp_pos += 1

            if ws_count > 1:
                # Multiple whitespace chars normalized to one space
                orig_pos += ws_count
                norm_pos += 1
            else:
                # Single whitespace
                orig_pos += 1
                norm_pos += 1
        else:
            orig_pos += 1
            norm_pos += 1

    return orig_pos


def find_text_in_document(
    parser: CFITextParser,
    search_text: str,
    use_fuzzy_match: bool = False,
    skip_chars: int = 0,
) -> TextSearchResult | None:
    """
    Find text in a parsed document and return position information.

    Args:
        parser: A CFITextParser that has parsed the document
        search_text: Text to search for
        use_fuzzy_match: If True, ignore whitespace differences
        skip_chars: Number of characters to skip (for finding later occurrences)

    Returns:
        TextSearchResult or None if not found
    """
    accumulated_text = parser.get_accumulated_text()
    normalized_search = normalize_text_for_search(search_text)

    search_start_pos = -1
    search_end_pos = -1

    if not use_fuzzy_match:
        # Exact match (case-insensitive)
        search_lower = accumulated_text.lower()
        normalized_search_lower = normalized_search.lower()
        search_start_pos = search_lower.find(normalized_search_lower, skip_chars)
        if search_start_pos != -1:
            search_end_pos = search_start_pos + len(normalized_search)
    else:
        # Fuzzy match: compare with all whitespace removed
        text_no_spaces = re.sub(r'\s+', '', accumulated_text).lower()
        search_no_spaces = re.sub(r'\s+', '', normalized_search).lower()

        # Map skip_chars to position in no-spaces text
        skip_no_spaces = 0
        for i in range(min(skip_chars, len(accumulated_text))):
            if not accumulated_text[i].isspace():
                skip_no_spaces += 1

        fuzzy_pos_no_spaces = text_no_spaces.find(search_no_spaces, skip_no_spaces)

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

    if search_start_pos == -1:
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


def build_cfi_from_search_result(
    result: TextSearchResult,
    spine_index: int,
) -> EPUBCFI:
    """
    Build an EPUBCFI object from a text search result.

    Args:
        result: The text search result
        spine_index: 0-indexed spine position

    Returns:
        EPUBCFI object
    """
    # Build base segment: /6/<spine_pos>
    # /6 references the spine in the package document
    # spine_pos = (spine_index + 1) * 2
    base = EPUBCFISegment(
        steps=[
            EPUBCFIStep(type="element", index=2),  # /6 in CFI notation
            EPUBCFIStep(type="element", index=spine_index),
        ]
    )

    # Build path to start and end text nodes
    start_path = _build_element_path_steps(result.start_node.element_path)
    end_path = _build_element_path_steps(result.end_node.element_path)

    # Find common path (steps that are identical)
    common_steps: list[EPUBCFIStep] = []
    min_len = min(len(start_path), len(end_path))

    for i in range(min_len):
        if (start_path[i].type == end_path[i].type and
                start_path[i].index == end_path[i].index):
            common_steps.append(start_path[i])
        else:
            break

    # Path is the common portion
    path = EPUBCFISegment(steps=common_steps)

    # Start segment: remaining steps from start path + text node
    start_remaining = start_path[len(common_steps):]
    start_steps = start_remaining + [
        EPUBCFIStep(type="text", index=result.start_node.text_node_index)
    ]
    start_segment = EPUBCFISegment(
        steps=start_steps,
        terminal_offset=result.start_offset,
    )

    # End segment: remaining steps from end path + text node
    end_remaining = end_path[len(common_steps):]
    end_steps = end_remaining + [
        EPUBCFIStep(type="text", index=result.end_node.text_node_index)
    ]
    end_segment = EPUBCFISegment(
        steps=end_steps,
        terminal_offset=result.end_offset,
    )

    return EPUBCFI(
        base=base,
        path=path,
        start=start_segment,
        end=end_segment,
        is_range=True,
    )


def _build_element_path_steps(element_path: list[int]) -> list[EPUBCFIStep]:
    """Convert a list of element indices to CFI steps."""
    return [EPUBCFIStep(type="element", index=idx) for idx in element_path]


# =============================================================================
# EPUB Parsing and CFI Generation
# =============================================================================


def parse_epub_for_cfi(epub_path: str) -> tuple[Any, list[dict]]:
    """
    Parse an EPUB file and extract spine information.

    Uses zipfile directly (no ebooklib dependency for this function).

    Returns:
        Tuple of (zip_file, spine_items)
        where spine_items is a list of dicts with 'id', 'href', 'media_type'
    """
    zf = zipfile.ZipFile(epub_path, 'r')

    # Read container.xml to find OPF file
    container_xml = zf.read('META-INF/container.xml').decode('utf-8')
    container_root = ET.fromstring(container_xml)

    # Get the OPF path
    ns = {'container': 'urn:oasis:names:tc:opendocument:xmlns:container'}
    rootfile = container_root.find('.//container:rootfile', ns)
    if rootfile is None:
        # Try without namespace
        rootfile = container_root.find('.//{*}rootfile')

    if rootfile is None:
        raise ValueError("Could not find rootfile in container.xml")

    opf_path = rootfile.get('full-path')
    opf_dir = opf_path.rsplit('/', 1)[0] + '/' if '/' in opf_path else ''

    # Parse OPF file
    opf_xml = zf.read(opf_path).decode('utf-8')
    opf_root = ET.fromstring(opf_xml)

    # Build manifest map
    manifest = {}
    ns_opf = {'opf': 'http://www.idpf.org/2007/opf'}

    for item in opf_root.findall('.//{*}item'):
        item_id = item.get('id')
        href = item.get('href')
        media_type = item.get('media-type')
        if item_id and href:
            manifest[item_id] = {
                'href': opf_dir + href,
                'media_type': media_type,
            }

    # Build spine
    spine = []
    for itemref in opf_root.findall('.//{*}itemref'):
        idref = itemref.get('idref')
        if idref and idref in manifest:
            spine.append({
                'id': idref,
                'href': manifest[idref]['href'],
                'media_type': manifest[idref]['media_type'],
            })

    return zf, spine


# Heuristic: typical book page has ~250-300 words ≈ 1500-2000 characters
# Using 1800 as a reasonable middle ground
CHARS_PER_PAGE = 1800


def generate_cfi_python(
    epub_path: str,
    search_text: str,
) -> dict | None:
    """
    Generate EPUB CFI for a text search match.

    Args:
        epub_path: Path to the EPUB file
        search_text: Text to find

    Returns:
        Dict with 'cfi', 'spineIndex', 'charPosition', etc. on success, or None if not found
    """
    try:
        zf, spine = parse_epub_for_cfi(epub_path)
    except Exception as e:
        return {'error': f'Failed to parse EPUB: {e}'}

    try:
        # Track cumulative character count for pseudo-page calculation
        cumulative_chars = 0
        spine_char_counts: list[int] = []

        # First pass: count characters in each spine item
        for spine_item in spine:
            try:
                html_content = zf.read(spine_item['href']).decode('utf-8')
                html_content = replace_html_entities(html_content)
                parser = CFITextParser()
                parser.feed(html_content)
                char_count = len(parser.get_accumulated_text())
                spine_char_counts.append(char_count)
            except Exception:
                spine_char_counts.append(0)

        # Second pass: search for text
        for spine_index, spine_item in enumerate(spine):
            try:
                # Load HTML content
                html_content = zf.read(spine_item['href']).decode('utf-8')
                html_content = replace_html_entities(html_content)

                # Parse the document
                parser = CFITextParser()
                parser.feed(html_content)

                accumulated_text = parser.get_accumulated_text()
                normalized_search = normalize_text_for_search(search_text)

                # Check for exact match
                exact_match = normalized_search.lower() in accumulated_text.lower()

                # Check for fuzzy match
                text_no_spaces = re.sub(r'\s+', '', accumulated_text).lower()
                search_no_spaces = re.sub(r'\s+', '', normalized_search).lower()
                fuzzy_match = not exact_match and search_no_spaces in text_no_spaces

                if exact_match or fuzzy_match:
                    # Try exact match first
                    result = None
                    if exact_match:
                        result = find_text_in_document(parser, search_text, False)

                    # If exact match failed, try fuzzy
                    if not result:
                        result = find_text_in_document(parser, search_text, True)

                    if result:
                        # Build CFI
                        cfi = build_cfi_from_search_result(result, spine_index)
                        cfi_string = cfi.to_string()

                        # Calculate pseudo-page number
                        # Sum chars from all previous spine items + position in current
                        chars_before = sum(spine_char_counts[:spine_index])
                        char_position = chars_before + result.start_pos
                        pseudo_page = (char_position // CHARS_PER_PAGE) + 1  # 1-indexed

                        return {
                            'cfi': cfi_string,
                            'spineIndex': spine_index,
                            'spineId': spine_item['id'],
                            'href': spine_item['href'],
                            'foundText': result.matched_text[:100],
                            'pseudoPage': pseudo_page,
                            'charPosition': char_position,
                        }

            except Exception:
                # Continue to next spine item
                continue

        return None

    finally:
        zf.close()


# =============================================================================
# Helper Functions
# =============================================================================


def _get_epub_spine(epub_path: str) -> list[dict]:
    """
    Extract spine items (chapters) from an EPUB file.

    Returns list of dicts with 'idref', 'href', and 'content'.
    """
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError:
        raise ImportError("ebooklib is required for EPUB support. Install with: pip install ebooklib")

    book = epub.read_epub(epub_path)
    spine_items = []

    # Get items from spine in reading order
    for item_id, linear in book.spine:
        item = book.get_item_with_id(item_id)
        if item and item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content()
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')

            spine_items.append({
                'idref': item_id,
                'href': item.get_name(),
                'content': content,
            })

    return spine_items


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
    Verify that a file is a valid EPUB.
    """
    try:
        import ebooklib
        from ebooklib import epub
        book = epub.read_epub(file_path)
        return book is not None and len(list(book.spine)) > 0
    except Exception:
        return False


def find_text_in_epub(
    epub_path: str,
    chapter_num: int,
    search_text: str,
    fuzzy: bool = True,
) -> dict:
    """
    Find text in an EPUB and return position data for annotation.

    Args:
        epub_path: Path to the EPUB file
        chapter_num: 1-indexed chapter number (spine position) - used for sortIndex
        search_text: Text to find
        fuzzy: Enable fuzzy matching (currently always searches entire EPUB)

    Returns:
        Dict with 'cfi', 'annotation_position', 'matched_text' on success,
        or 'error' key on failure with debug info.
    """
    # Use pure Python CFI generation
    result = generate_cfi_python(epub_path, search_text)

    if result and result.get('cfi'):
        # Success - build the annotation position
        cfi = result['cfi']
        spine_index = result.get('spineIndex', 0)
        char_position = result.get('charPosition', 0)

        return {
            'cfi': cfi,
            'annotation_position': build_epub_annotation_position(cfi),
            'matched_text': result.get('foundText', search_text),
            'chapter_found': spine_index + 1,  # Convert to 1-indexed
            'char_position': char_position,  # Character offset for sortIndex
            'score': 1.0,
        }

    # Search failed - return error
    try:
        spine = _get_epub_spine(epub_path)
        total_chapters = len(spine) if spine else 0
    except Exception:
        total_chapters = 0

    error_msg = "Text not found in EPUB"
    if result and result.get('error'):
        error_msg = result['error']

    return {
        'error': error_msg,
        'total_chapters': total_chapters,
        'search_text': search_text[:100],
    }


