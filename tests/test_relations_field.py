"""Tests for relations field output in format_item_metadata.

The relations field (dc:relation) is surfaced in markdown output only.
BibTeX output intentionally omits it — the `related` field is non-standard
and would pollute citation databases.
"""

import importlib.util
import pathlib
import sys
from unittest.mock import MagicMock

# Stub out heavy optional dependencies so client.py can be imported in isolation
for _mod_name in (
    "markitdown", "pyzotero", "pyzotero.zotero",
    "dotenv", "fastmcp", "mcp", "mcp.server",
    "zotero_mcp", "zotero_mcp.utils", "zotero_mcp._app",
):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# Import client directly to avoid heavy optional dependencies (fastmcp etc.)
_client_path = pathlib.Path(__file__).parent.parent / "src" / "zotero_mcp" / "client.py"
_spec = importlib.util.spec_from_file_location("zotero_mcp.client", _client_path)
_client_mod = importlib.util.module_from_spec(_spec)
sys.modules["zotero_mcp.client"] = _client_mod
_spec.loader.exec_module(_client_mod)
format_item_metadata = _client_mod.format_item_metadata
generate_bibtex = _client_mod.generate_bibtex


def _make_item(relations=None, item_type="journalArticle", extra_data=None):
    """Build a minimal Zotero item dict for testing."""
    data = {
        "key": "TESTKEY1",
        "itemType": item_type,
        "title": "Test Article",
        "creators": [{"creatorType": "author", "lastName": "Smith", "firstName": "J."}],
        "date": "2024",
        "publicationTitle": "Test Journal",
        "relations": relations if relations is not None else {},
    }
    if extra_data:
        data.update(extra_data)
    return {"data": data}


RELATED_URI = "http://zotero.org/users/123456/items/ABCD1234"
RELATED_KEY = "ABCD1234"


# ---------------------------------------------------------------------------
# format_item_metadata (markdown)
# ---------------------------------------------------------------------------

class TestFormatItemMetadataRelations:
    def test_single_relation_appears_in_output(self):
        item = _make_item(relations={"dc:relation": [RELATED_URI]})
        output = format_item_metadata(item, include_abstract=False)
        assert "## Related Items" in output
        assert f"- {RELATED_KEY}" in output

    def test_multiple_relations(self):
        uris = [
            "http://zotero.org/users/123456/items/AAAA0001",
            "http://zotero.org/users/123456/items/BBBB0002",
        ]
        item = _make_item(relations={"dc:relation": uris})
        output = format_item_metadata(item, include_abstract=False)
        assert "- AAAA0001" in output
        assert "- BBBB0002" in output

    def test_string_relation_normalized_to_list(self):
        """Zotero API may return a single URI as a plain string instead of a list."""
        item = _make_item(relations={"dc:relation": RELATED_URI})
        output = format_item_metadata(item, include_abstract=False)
        assert f"- {RELATED_KEY}" in output

    def test_no_relation_section_when_empty(self):
        item = _make_item(relations={})
        output = format_item_metadata(item, include_abstract=False)
        assert "## Related Items" not in output

    def test_no_relation_section_when_missing(self):
        item = _make_item(relations=None)
        output = format_item_metadata(item, include_abstract=False)
        assert "## Related Items" not in output


# ---------------------------------------------------------------------------
# generate_bibtex — relations must NOT appear (non-standard field)
# ---------------------------------------------------------------------------

class TestGenerateBibtexRelations:
    def test_related_field_absent_when_relations_present(self):
        """BibTeX output must not include a `related` field even if relations exist."""
        item = _make_item(relations={"dc:relation": [RELATED_URI]})
        output = generate_bibtex(item)
        assert "related" not in output

    def test_related_field_absent_when_empty(self):
        item = _make_item(relations={})
        output = generate_bibtex(item)
        assert "related" not in output
