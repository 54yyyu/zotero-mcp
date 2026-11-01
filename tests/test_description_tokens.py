"""Token-usage tests for @mcp.tool descriptions.

Every tool description is sent to the client on every request, so the real
guard is the total-surface ceiling in ``test_context_cost_claim.py``, which
counts descriptions, parameter schemas and output schemas together. This file
keeps two per-description checks that the total cannot express:

* a hard cap, so one tool cannot quietly absorb the savings of the others;
* a floor, so a covered tool never regresses to a purpose-only one-liner,
  the most common smell in Hasan et al., *"Model Context Protocol (MCP) Tool
  Descriptions Are Smelly!"* (arXiv:2602.14878).

There are deliberately no per-tool minimum budgets: they existed to stop
descriptions shrinking, and shrinking without losing guidance is the goal.

Skipped when `tiktoken` isn't installed.
"""

import pytest

tiktoken = pytest.importorskip("tiktoken")


#: Tools whose descriptions must stay above the one-liner floor.
COVERED_TOOLS = frozenset({
    # tools/annotations.py
    "zotero_get_annotations",
    "zotero_get_notes",
    "zotero_manage_note",
    "zotero_create_annotation",
    # tools/local_auth.py
    "zotero_authorize_local_writes",
    "zotero_write_capabilities",
    # tools/retrieval.py
    "zotero_get_tags",
    "zotero_get_item_children",
    # tools/write.py
    "zotero_add_item",
    "zotero_update_item",
    "zotero_batch_update",
    "zotero_set_item_collections",
    "zotero_update_collection",
    "zotero_find_duplicates",
    "zotero_merge_duplicates",
    "zotero_attach_file",
    # tools/search.py
    "zotero_search_items",
    "zotero_search_by_tag",
    "zotero_search_by_citation_key",
    "zotero_advanced_search",
    "zotero_semantic_search",
    "zotero_update_search_database",
    "zotero_get_search_database_status",
})

#: Below this a description is a purpose-only one-liner.
FLOOR = 30

#: No single description should need more than this.
PER_TOOL_HARD_MAX = 450


def _collect_tool_descriptions():
    """Return {tool_name: description_string} by parsing @mcp.tool blocks
    from the source files directly.

    We don't introspect the FastMCP runtime because its internal layout
    has churned across versions (no stable `tools` or `_tool_manager.tools`
    attribute). Parsing the source gives a stable, FastMCP-version-
    independent check.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "zotero_mcp" / "tools"
    files = sorted(root.glob("*.py"))

    block_re = re.compile(r"@mcp\.tool\(\s*(.*?)\n\s*\)(?:\s*\n@[\w.]+(?:\([^)]*\))?)*\s*\n(?:async\s+)?def ", re.DOTALL)
    name_re = re.compile(r'name="([^"]+)"')

    descriptions: dict[str, str] = {}
    for f in files:
        # encoding is explicit: the tool descriptions contain non-ASCII, and
        # the default on Windows is cp1252, which cannot decode them.
        content = f.read_text(encoding="utf-8")
        for m in block_re.finditer(content):
            block = m.group(1)
            name_m = name_re.search(block)
            if not name_m:
                continue
            name = name_m.group(1)
            desc_idx = block.find("description=")
            if desc_idx == -1:
                continue
            desc_text = block[desc_idx + len("description="):].strip()
            if desc_text.endswith(","):
                desc_text = desc_text[:-1].strip()
            try:
                desc_val = eval(desc_text, {"__builtins__": {}}, {})
            except Exception:
                desc_val = desc_text
            descriptions[name] = desc_val
    return descriptions


@pytest.fixture(scope="module")
def enc():
    # cl100k_base covers both GPT-4 and Claude-family tokenizers closely
    # enough for a budget check.
    return tiktoken.get_encoding("cl100k_base")


@pytest.fixture(scope="module")
def descriptions():
    descs = _collect_tool_descriptions()
    if not descs:
        pytest.skip("could not enumerate MCP tools — FastMCP internals changed?")
    return descs


class TestCoverageStaysInSync:
    """COVERED_TOOLS must not accumulate entries for tools that no longer exist.

    A renamed or merged tool would otherwise silently drop out of the floor
    check while the suite reports green.
    """

    def test_no_entries_for_missing_tools(self, descriptions):
        stale = sorted(COVERED_TOOLS - set(descriptions))
        assert not stale, (
            f"COVERED_TOOLS references tools that are no longer registered: {stale}. "
            f"Remove the entry if the tool is gone, or rename it if the tool was renamed."
        )


class TestGlobalCeiling:
    """No single tool description should blow past the hard cap."""

    def test_no_tool_exceeds_hard_max(self, enc, descriptions):
        over = [
            (name, len(enc.encode(desc)))
            for name, desc in descriptions.items()
            if len(enc.encode(desc)) > PER_TOOL_HARD_MAX
        ]
        assert not over, (
            f"Tools over {PER_TOOL_HARD_MAX}-token hard cap: {over}. "
            f"Compact rubric-compliant descriptions rarely need more."
        )


class TestRubricFloor:
    def test_no_trivial_descriptions_for_covered_tools(self, enc, descriptions):
        trivial = [
            (name, n)
            for name in sorted(COVERED_TOOLS)
            if (desc := descriptions.get(name)) is not None
            and (n := len(enc.encode(desc))) < FLOOR
        ]
        assert not trivial, (
            f"Regression: covered tools now < {FLOOR} tokens (likely one-liners): "
            f"{trivial}."
        )
