"""Token-usage tests for @mcp.tool descriptions.

Locks in the token footprint of each tool's top-level description so that
future edits don't silently regress to smelly single-line descriptions OR
balloon far past the rubric-compliant range.

Inspired by Hasan et al., *"Model Context Protocol (MCP) Tool Descriptions
Are Smelly!"* (arXiv:2602.14878), which finds a compact variant of all six
rubric components (purpose, guidelines, limitations, parameter explanation,
examples, length/completeness) tends to produce the best cost/accuracy
trade-off.

Budgets are set per tool based on the post-rubric-rewrite values, with a
generous ±50% band so small wording changes don't break the test — only
regressions to one-liners or runaway growth do.

Skipped when `tiktoken` isn't installed.
"""

import pytest

tiktoken = pytest.importorskip("tiktoken")


# Per-tool token budgets (min, max). Measured on the post-rubric-rewrite
# descriptions; min ≈ 0.67×, max ≈ 1.5× of the current value.
# If you legitimately need to exceed a max, update the budget and mention
# why in the PR — that's an active choice, not an accident.
TOOL_BUDGETS = {
    # tools/annotations.py
    "zotero_get_annotations":          (110, 245),
    "zotero_get_notes":                (100, 230),
    "zotero_search_notes":             ( 90, 205),
    "zotero_create_note":              (100, 225),
    "zotero_update_note":              (100, 225),
    "zotero_delete_note":              ( 90, 205),
    "zotero_create_annotation":        (130, 295),
    "zotero_create_area_annotation":   (175, 390),
    # tools/retrieval.py
    "zotero_get_tags":                 ( 85, 195),
    # tools/write.py
    "zotero_batch_update_tags":        (155, 350),
    # tools/search.py
    "zotero_search_items":             (175, 400),
    "zotero_search_by_tag":            (115, 265),
    "zotero_search_by_citation_key":   (125, 280),
    "zotero_advanced_search":          (175, 400),
    "zotero_semantic_search":          (130, 295),
    "zotero_update_search_database":   (130, 295),
    "zotero_get_search_database_status": ( 75, 170),
}

# Global ceiling: even rubric-rich descriptions shouldn't exceed this.
# The paper's RQ-2 data shows diminishing returns (and AS inflation) past
# this range for compact variants.
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

    block_re = re.compile(r"@mcp\.tool\(\s*(.*?)\n\)\s*\n(?:async\s+)?def ", re.DOTALL)
    name_re = re.compile(r'name="([^"]+)"')

    descriptions: dict[str, str] = {}
    for f in files:
        content = f.read_text()
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


class TestDescriptionTokenBudgets:
    """Each tool's description must fit within its rubric-compliant band."""

    @pytest.mark.parametrize("tool_name,bounds", list(TOOL_BUDGETS.items()))
    def test_tool_within_budget(self, enc, descriptions, tool_name, bounds):
        lo, hi = bounds
        desc = descriptions.get(tool_name)
        if desc is None:
            pytest.skip(f"{tool_name} not registered (may be gated by extras)")
        n = len(enc.encode(desc))
        assert lo <= n <= hi, (
            f"{tool_name}: {n} tokens outside budget [{lo}, {hi}]. "
            f"Too few tokens usually means a smelly one-liner; "
            f"too many usually means bloat. Update the budget here if the "
            f"change is intentional."
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
    """No tool we've committed to keeping rubric-compliant should regress
    to a trivial one-liner. 'Purpose-only' descriptions are the most
    common smell from Hasan et al. and cost the most in practice.

    Only tools in TOOL_BUDGETS are enforced here — other tools in the
    codebase are known-smelly work not yet addressed; extending coverage
    means adding the tool to TOOL_BUDGETS in a follow-up PR.
    """

    def test_no_trivial_descriptions_for_covered_tools(self, enc, descriptions):
        trivial = []
        for name in TOOL_BUDGETS:
            desc = descriptions.get(name)
            if desc is None:
                continue
            n = len(enc.encode(desc))
            if n < 30:
                trivial.append((name, n))
        assert not trivial, (
            f"Regression — covered tools now < 30 tokens (likely one-liners): "
            f"{trivial}."
        )

    def test_known_smelly_tools_are_tracked(self, enc, descriptions):
        """Catalog which *other* tools are still smelly so contributors see
        the backlog. This test is INFORMATIONAL — it prints but does not
        fail. Promote a tool to TOOL_BUDGETS once you've rewritten it."""
        remaining = []
        for name, desc in descriptions.items():
            if name in TOOL_BUDGETS:
                continue
            n = len(enc.encode(desc))
            if n < 30:
                remaining.append((name, n))
        if remaining:
            print(
                f"\n[info] {len(remaining)} tool(s) still below 30-token floor "
                f"(not yet in TOOL_BUDGETS): {remaining}"
            )
