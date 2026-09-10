"""The filters example in zotero_semantic_search's description is copied
verbatim by models. It must (a) parse as a dict, (b) use only keys the index
stores, and (c) have exactly one key: chromadb rejects a multi-key ``where``
without an explicit ``$and`` ("Expected where to have exactly one operator").
"""

import ast
import asyncio
import re

import pytest

import zotero_mcp.server  # noqa: F401
from zotero_mcp._app import mcp

# Keys written per document in semantic_search.py (_build_metadata and the
# chunk path). 'itemType' is accepted as an alias and renamed to 'item_type'.
STORED_METADATA_KEYS = {
    "item_key",
    "item_type",
    "itemType",
    "title",
    "date",
    "date_added",
    "date_modified",
    "creators",
    "publication",
    "url",
    "doi",
    "tags",
    "citation_key",
    "group_id",
    "has_fulltext",
    "fulltext_source",
    "attachment_keys",
    "attachment_priority",
    "parent_item_key",
    "chunk_index",
    "n_chunks",
    "char_start",
    "char_end",
    "page",
}

EXAMPLE_RE = re.compile(r"e\.g\. (\{[^}]*\})")


@pytest.fixture(scope="module")
def semantic_search_description():
    tools = asyncio.run(mcp.list_tools())
    by_name = {t.name: t for t in tools}
    return by_name["zotero_semantic_search"].description


def test_filters_example_is_a_single_stored_key(semantic_search_description):
    m = EXAMPLE_RE.search(semantic_search_description)
    assert m, "no '{...}' filters example found in the description"
    example = ast.literal_eval(m.group(1))
    assert isinstance(example, dict)
    assert len(example) == 1, "multi-key filters need $and; the example must show one key"
    (key,) = example
    assert key in STORED_METADATA_KEYS, f"{key!r} is not a key the index stores"
