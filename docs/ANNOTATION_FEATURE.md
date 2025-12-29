# Zotero Annotation Feature - Code Overview

## Summary

This feature adds the ability to create highlight annotations on PDF attachments in Zotero via the MCP interface. It handles the complexity of PDF text search, coordinate systems, and multiple storage configurations (Zotero Cloud, WebDAV).

## Commits

| Commit | Description |
|--------|-------------|
| `b6e7842` | Initial `zotero_create_annotation` tool |
| `999ab2a` | Add fuzzy text matching |
| `8ef202e` | Fix for PDFs with missing word spaces |
| `9d093ea` | Enhanced fuzzy matching (normalization, thresholds) |
| `555d8c8` | Debug info on search failures |
| `df41245` | **Anchor-based matching** for long passages |
| `6c7655f` | Major refactor for maintainability |
| `e361cc7` | "Did you mean" suggestions |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Tool: zotero_create_annotation           │
│                         (server.py:1700-1939)                   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         PDF Download                             │
│  1. Try local Zotero (WebDAV/local storage)                     │
│  2. Fallback to Web API (Zotero Cloud)                          │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Text Search (pdf_utils.py)                    │
│                                                                  │
│  Strategy Order:                                                 │
│  1. Anchor-based (for text >100 chars)                          │
│  2. Exact match (PyMuPDF search_for)                            │
│  3. Fuzzy match (normalized text comparison)                    │
│                                                                  │
│  Features:                                                       │
│  • Neighboring page search (±2 pages)                           │
│  • Debug info on failure                                         │
│  • "Did you mean" suggestions                                    │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Coordinate Conversion                          │
│  PyMuPDF (top-left origin) → Zotero (bottom-left origin)        │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                Create Annotation via Web API                     │
│  POST to Zotero API with annotationPosition JSON                │
└─────────────────────────────────────────────────────────────────┘
```

---

## File: `pdf_utils.py` (816 lines)

### Module Structure

```python
# Configuration Constants (lines 28-47)
ANCHOR_MIN_TEXT_LENGTH = 100      # Use anchor matching for text > 100 chars
ANCHOR_TARGET_LENGTH = 40         # Length of start/end anchors
ANCHOR_MATCH_THRESHOLD = 0.75     # Fuzzy threshold for anchors
FUZZY_THRESHOLD_SHORT = 0.85      # For text < 50 chars
FUZZY_THRESHOLD_MEDIUM = 0.75     # For text 50-150 chars
FUZZY_THRESHOLD_LONG = 0.65       # For text > 150 chars
DEFAULT_NEIGHBOR_PAGES = 2        # Pages to search on either side

# Character Replacement Maps
DASH_REPLACEMENTS = {...}         # em-dash, en-dash → hyphen
QUOTE_REPLACEMENTS = {...}        # curly quotes → straight
LIGATURE_REPLACEMENTS = {...}     # fi, fl, ff → expanded
```

### Function Groups

#### Text Normalization
| Function | Purpose |
|----------|---------|
| `normalize_text()` | Handle hyphens, dashes, quotes, ligatures |
| `normalize_for_matching()` | Aggressive: remove ALL spaces, lowercase |

#### Page Text Extraction
| Function | Purpose |
|----------|---------|
| `_extract_page_spans()` | Get text spans with bounding boxes |
| `_build_normalized_text_index()` | Create position-to-span mapping |
| `_get_spans_in_range()` | Find spans overlapping a position range |

#### Coordinate Conversion
| Function | Purpose |
|----------|---------|
| `_convert_rects_to_zotero()` | Transform PyMuPDF → Zotero coords |
| `_build_sort_index()` | Create annotation sort key |
| `_build_search_result()` | Assemble result dict |

#### Search Strategies
| Function | Purpose |
|----------|---------|
| `_anchor_based_search()` | Match start/end, highlight between |
| `_fuzzy_search_page()` | Normalized text comparison |
| `_sliding_window_match()` | SequenceMatcher-based fuzzy search |
| `_search_single_page()` | Orchestrates all strategies |

#### Public API
| Function | Purpose |
|----------|---------|
| `find_text_position()` | Main entry point for text search |
| `get_page_label()` | Get PDF page label (e.g., "i", "ii") |
| `verify_pdf_attachment()` | Check if file is valid PDF |
| `build_annotation_position()` | Create Zotero position JSON |

---

## File: `server.py` Changes

### New MCP Tool

```python
@mcp.tool(name="zotero_create_annotation")
def create_annotation(
    attachment_key: str,   # PDF attachment key
    page: int,             # 1-indexed page number
    text: str,             # Text to highlight
    comment: str = None,   # Optional comment
    color: str = "#ffd400" # Highlight color
) -> str:
```

### Error Handling

When text search fails, the tool provides helpful feedback:

```
Error: Could not find text on page 21

Text searched: "16.27% savings rate"

==================================================
DID YOU MEAN (score: 76%):

  "we find a couple must save 16.27% (i.e., 63% more) to achieve
  the same expected utility with the TDF..."

  (Found on page 21)
==================================================

TIP: Copy the exact text from the PDF instead of paraphrasing.
```

---

## Key Design Decisions

### 1. Anchor-Based Matching
For passages >100 characters, instead of matching the entire text:
- Extract first ~40 chars as **start anchor**
- Extract last ~40 chars as **end anchor**
- Find both in PDF, highlight everything between

**Why?** Long passages often have line breaks, hyphenation, or character variations in the middle. Only the start/end need to match.

### 2. Aggressive Text Normalization
PDF text extraction produces inconsistent output:
- Words without spaces: `"Wechallengetwotenets"`
- Special characters: em-dashes, curly quotes, ligatures

Solution: `normalize_for_matching()` removes ALL spaces and lowercases, making comparison robust.

### 3. Neighboring Page Search
PDF page numbers often don't match document page numbers (due to front matter). The tool automatically searches ±2 pages if text isn't found on the specified page.

### 4. Hybrid Storage Support
Works with both:
- **Zotero Cloud Storage**: Downloads via Web API
- **WebDAV Storage**: Downloads via local Zotero (port 23119)

Annotations are always created via Web API (local API is read-only).

---

## Testing

Run the test suite:
```bash
.venv/bin/python -c "
from zotero_mcp.pdf_utils import (
    normalize_text, normalize_for_matching,
    find_text_position, _extract_anchor
)
# Tests run automatically on import
print('All imports successful')
"
```

---

## Usage Example

```python
# Via MCP tool
zotero_create_annotation(
    attachment_key="NHZFE5A7",
    page=1,
    text="We challenge two tenets of lifecycle investing",
    comment="Key thesis of the paper",
    color="#ffd400"
)
```

---

## Future Improvements

1. **Auto-retry on "Did you mean"**: Claude could automatically retry with suggested text
2. **Multi-page highlights**: Support annotations spanning multiple pages
3. **Other annotation types**: Underline, strikethrough, notes
4. **Batch annotations**: Create multiple highlights in one call
