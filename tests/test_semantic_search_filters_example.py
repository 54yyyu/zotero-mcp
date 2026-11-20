"""The filters example in zotero_semantic_search's description is copied
verbatim by models. It must (a) parse as a dict, (b) use a key search knows,
and (c) have exactly one key, as the description asks.

Keys are split by ``semantic_filters.split_filters`` (#826): stored ones go to
ChromaDB, bibliographic ones are evaluated against live Zotero data.
"""

import ast
import asyncio
import re

import pytest

import zotero_mcp.server  # noqa: F401
from zotero_mcp._app import mcp
from zotero_mcp.semantic_filters import CHROMA_FIELDS, LIVE_FIELDS

KNOWN_FILTER_KEYS = CHROMA_FIELDS | LIVE_FIELDS

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
    assert key in KNOWN_FILTER_KEYS, f"{key!r} is not a key semantic search filters on"
