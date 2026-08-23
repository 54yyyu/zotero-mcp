#!/usr/bin/env python3
"""Measure the fixed context cost of each way to reach a Zotero library.

Two routes exist, and they charge for context differently:

* **MCP server.** Every registered tool's name, description and JSON parameter
  schema is sent to the model on *every* request, before the user has typed
  anything. The cost is paid whether or not a single tool is called.

* **CLI + agent skill.** Only the skill's frontmatter (name + description)
  sits in context until the model decides the skill is relevant; the body is
  read on demand, and `reference.md` only if the body sends it there. Command
  output costs what it costs either way, and is not what this measures.

This measures the *fixed* cost -- the tax before any work happens. It does
not measure task success, output size, or number of round trips: those need a
live task benchmark against a real library, which this script deliberately is
not. Reporting the fixed cost as though it settled the question would be the
easy mistake here, so the output says what it covers.

    python scripts/measure_context_cost.py            # table
    python scripts/measure_context_cost.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

SKILL_DIR = REPO / "src" / "zotero_mcp" / "skills" / "zotero-cli"

#: cl100k_base is not any Claude model's exact tokenizer, but it tracks them
#: closely enough for a ratio between two bodies of English + JSON, which is
#: all this reports. The same encoder is used for both sides, so any bias
#: applies equally and cancels in the comparison.
ENCODING = "cl100k_base"


def _encoder():
    try:
        import tiktoken
    except ImportError:
        print("tiktoken is required: pip install tiktoken", file=sys.stderr)
        raise SystemExit(2)
    return tiktoken.get_encoding(ENCODING)


def _serialize_tool(tool) -> str:
    """The tool as the client actually receives it.

    Counting only the description would flatter MCP considerably: parameter
    schemas are most of a tool's wire size. This mirrors an MCP `tools/list`
    entry -- name, description, inputSchema -- so the number is what a session
    really pays.
    """
    payload = {
        "name": tool.name,
        "description": tool.description or "",
        "inputSchema": tool.parameters or {},
    }
    if getattr(tool, "output_schema", None):
        payload["outputSchema"] = tool.output_schema
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def measure_mcp_surface(toolsets: str | None) -> dict:
    """Token cost of the MCP tool surface under one ZOTERO_MCP_TOOLSETS value.

    Imported fresh per profile: the registry is built at import time from the
    environment, so measuring several profiles in one process would report the
    first one three times.
    """
    env_backup = dict(os.environ)
    # Re-importing under a different profile means evicting the package from
    # sys.modules -- but other code in this process may hold references to the
    # old module objects, and leaving the eviction in place would make its
    # monkeypatches apply to modules nobody uses any more. Snapshot and put
    # everything back on the way out, so calling this is invisible from
    # outside.
    module_backup = {
        name: module for name, module in sys.modules.items()
        if name.startswith("zotero_mcp")
    }
    for name in module_backup:
        del sys.modules[name]
    try:
        os.environ["ZOTERO_LOCAL"] = "true"
        if toolsets is None:
            os.environ.pop("ZOTERO_MCP_TOOLSETS", None)
        else:
            os.environ["ZOTERO_MCP_TOOLSETS"] = toolsets

        from zotero_mcp import server  # noqa: F401  (registers the tools)
        from zotero_mcp._app import mcp

        # The public list_tools() is the one that honours the toolset
        # profile. The internal _list_tools() returns every *registered* tool,
        # disabled ones included, which would report an identical surface for
        # every profile and quietly make this whole measurement meaningless.
        tools = asyncio.run(mcp.list_tools())
        enc = _encoder()
        per_tool = {t.name: len(enc.encode(_serialize_tool(t))) for t in tools}
        return {
            "profile": toolsets if toolsets is not None else "(unset - default)",
            "tools": len(tools),
            "tokens": sum(per_tool.values()),
            "per_tool": dict(sorted(per_tool.items(), key=lambda kv: -kv[1])),
        }
    finally:
        os.environ.clear()
        os.environ.update(env_backup)
        for name in [n for n in sys.modules if n.startswith("zotero_mcp")]:
            del sys.modules[name]
        sys.modules.update(module_backup)


def measure_skill() -> dict:
    """Token cost of the CLI route, split by when each part is paid for."""
    enc = _encoder()
    skill_md = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    reference = (SKILL_DIR / "reference.md").read_text(encoding="utf-8")

    # Everything before the closing --- is what sits in context unconditionally.
    parts = skill_md.split("---", 2)
    frontmatter = parts[1] if len(parts) > 2 else ""
    body = parts[2] if len(parts) > 2 else skill_md

    return {
        "frontmatter": len(enc.encode(frontmatter)),
        "body": len(enc.encode(body)),
        "skill_total": len(enc.encode(skill_md)),
        "reference": len(enc.encode(reference)),
    }


def build_report() -> dict:
    profiles = [None, "none", "all"]
    mcp_results = [measure_mcp_surface(p) for p in profiles]
    skill = measure_skill()

    default = next(r for r in mcp_results if r["profile"].startswith("(unset"))
    return {
        "encoding": ENCODING,
        "mcp": mcp_results,
        "skill": skill,
        "comparison": {
            "mcp_default_always_loaded": default["tokens"],
            "cli_always_loaded": skill["frontmatter"],
            "cli_after_skill_fires": skill["skill_total"],
            "cli_worst_case_with_reference": skill["skill_total"] + skill["reference"],
        },
    }


def print_table(report: dict) -> None:
    c = report["comparison"]
    print("Fixed context cost: MCP tool surface vs CLI + skill")
    print(f"(tokens, {report['encoding']}; measured, not estimated)\n")

    print("MCP server -- sent on every request, before any tool is called")
    print(f"  {'profile':<24} {'tools':>6} {'tokens':>8}")
    for row in report["mcp"]:
        print(f"  {row['profile']:<24} {row['tools']:>6} {row['tokens']:>8,}")

    s = report["skill"]
    print("\nCLI + skill -- paid in stages")
    print(f"  {'frontmatter (always in context)':<40} {s['frontmatter']:>8,}")
    print(f"  {'+ SKILL.md body (when it fires)':<40} {s['skill_total']:>8,}")
    print(f"  {'+ reference.md (only if needed)':<40} "
          f"{s['skill_total'] + s['reference']:>8,}")

    always = c["mcp_default_always_loaded"] / max(c["cli_always_loaded"], 1)
    fired = c["mcp_default_always_loaded"] / max(c["cli_after_skill_fires"], 1)
    print("\nRatio, MCP default profile vs CLI")
    print(f"  before either is used:        {always:>6.1f}x")
    print(f"  once the skill has fired:     {fired:>6.1f}x")

    print("\nWhat this does and does not show")
    print("  Measured: the fixed tax each route puts in context before work starts.")
    print("  Not measured: task success, output size, or round trips. A cheaper")
    print("  surface that gets the answer wrong is not cheaper. Those need a live")
    print("  benchmark against a real library.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    parser.add_argument("--per-tool", action="store_true",
                        help="Also list each tool's cost in the default profile")
    args = parser.parse_args()

    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print_table(report)
    if args.per_tool:
        default = next(r for r in report["mcp"] if r["profile"].startswith("(unset"))
        print("\nPer-tool cost, default profile (most expensive first)")
        for name, tokens in default["per_tool"].items():
            print(f"  {tokens:>6,}  {name}")


if __name__ == "__main__":
    main()
