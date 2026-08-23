"""Tests for base-field title resolution in format_item_metadata.

Several Zotero item types store the title under a type-specific key instead
of "title" itself: a statute's is "nameOfAct", a case's is "caseName", an
email's is "subject" (see zotero_mcp.schema). The write path
(zotero_update_item) already routes through schema.resolve_field for this
(commit 94e3c1f, #402); format_item_metadata did a raw data.get("title", ...)
lookup, so every one of these item types rendered as "# Untitled" even
though the real title was present under its type-specific key.
"""

import importlib.util
import pathlib
import sys
from unittest.mock import MagicMock

_SRC = pathlib.Path(__file__).parent.parent / "src" / "zotero_mcp"

# Stub out heavy optional dependencies so client.py can be imported in
# isolation; schema is loaded for real below, not stubbed.
for _mod_name in (
    "pyzotero",
    "pyzotero.zotero",
    "dotenv",
    "fastmcp",
    "mcp",
    "mcp.server",
    "zotero_mcp",
    "zotero_mcp.utils",
    "zotero_mcp._app",
):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

if "zotero_mcp.schema" not in sys.modules:
    _schema_spec = importlib.util.spec_from_file_location("zotero_mcp.schema", _SRC / "schema.py")
    _schema_mod = importlib.util.module_from_spec(_schema_spec)
    sys.modules["zotero_mcp.schema"] = _schema_mod
    _schema_spec.loader.exec_module(_schema_mod)

_client_spec = importlib.util.spec_from_file_location("zotero_mcp.client", _SRC / "client.py")
_client_mod = importlib.util.module_from_spec(_client_spec)
sys.modules["zotero_mcp.client"] = _client_mod
_client_spec.loader.exec_module(_client_mod)
format_item_metadata = _client_mod.format_item_metadata


def _make_item(item_type, title_field, title):
    data = {
        "key": "TESTKEY1",
        "itemType": item_type,
        title_field: title,
        "creators": [{"creatorType": "author", "lastName": "Smith", "firstName": "J."}],
        "date": "2024",
    }
    return {"data": data}


class TestBaseFieldTitle:
    def test_statute_title_stored_under_name_of_act(self):
        item = _make_item("statute", "nameOfAct", "The Clean Air Act")
        output = format_item_metadata(item, include_abstract=False)
        assert "# The Clean Air Act" in output
        assert "# Untitled" not in output

    def test_case_title_stored_under_case_name(self):
        item = _make_item("case", "caseName", "Marbury v. Madison")
        output = format_item_metadata(item, include_abstract=False)
        assert "# Marbury v. Madison" in output
        assert "# Untitled" not in output

    def test_email_title_stored_under_subject(self):
        item = _make_item("email", "subject", "Re: manuscript review")
        output = format_item_metadata(item, include_abstract=False)
        assert "# Re: manuscript review" in output
        assert "# Untitled" not in output

    def test_ordinary_type_still_reads_title_directly(self):
        """journalArticle's title field is its own base field; unaffected types
        must keep working exactly as before."""
        item = _make_item("journalArticle", "title", "An Ordinary Article")
        output = format_item_metadata(item, include_abstract=False)
        assert "# An Ordinary Article" in output

    def test_missing_title_field_falls_back_to_untitled(self):
        item = {"data": {"key": "TESTKEY1", "itemType": "statute"}}
        output = format_item_metadata(item, include_abstract=False)
        assert "# Untitled" in output
