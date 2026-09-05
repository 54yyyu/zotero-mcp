#!/usr/bin/env python3
"""Regenerate the zotero-cli skill's command reference from the real parser.

Hand-written CLI documentation drifts the moment a flag is added, and a skill
that names a flag which no longer exists is worse than one that omits it: the
agent will confidently run the wrong command. Deriving the reference from
`build_parser()` means it cannot say anything the CLI does not accept.

Run after changing the CLI surface:

    python scripts/gen_skill_reference.py

`tests/test_skill_reference_current.py` fails if the checked-in file is stale.
"""

import argparse
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "src" / "zotero_mcp" / "skills" / "zotero-cli" / "reference.md"

HEADER = """# zotero-cli command reference

Generated from the CLI's own argument parser by
`scripts/gen_skill_reference.py` -- do not edit by hand.

Every command also accepts `--json` (machine-readable envelope on stdout) and
`-v` (diagnostics on stderr), before or after the command name. Run
`zotero-cli --json-schema` for the output contract.

"""


def _format_action(action) -> str | None:
    if action.dest in ("help", "json_out", "verbose"):
        return None
    if isinstance(action, argparse._SubParsersAction):
        return None
    if action.option_strings:
        names = ", ".join(action.option_strings)
    else:
        names = f"<{action.dest}>"
    bits = [f"`{names}`"]
    if action.choices:
        bits.append("one of " + ", ".join(f"`{c}`" for c in action.choices))
    if action.required and action.option_strings:
        bits.append("**required**")
    if action.default not in (None, False, argparse.SUPPRESS) and action.option_strings:
        bits.append(f"default `{action.default}`")
    help_text = (action.help or "").strip()
    if help_text and help_text != argparse.SUPPRESS:
        bits.append(help_text)
    return " - " + " -- ".join(bits)


def _render(name: str, parser, out: io.StringIO, depth: int = 2,
            child_prefix: str | None = None) -> None:
    """Render one parser. *child_prefix* is what subcommands are named under --
    the canonical command, so a subheading reads `get metadata` rather than
    repeating the alias list from the parent heading."""
    heading = "#" * depth
    desc = (parser.description or "").strip()
    out.write(f"\n{heading} `{name}`\n")
    if desc:
        out.write(f"\n{desc}\n")

    lines = [ln for ln in (_format_action(a) for a in parser._actions) if ln]
    if lines:
        out.write("\n")
        out.write("\n".join(lines))
        out.write("\n")

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub_name, sub in action.choices.items():
                # argparse registers aliases as separate entries pointing at
                # the same parser object; render each parser once.
                if getattr(sub, "_rendered", False):
                    continue
                sub._rendered = True
                _render(f"{child_prefix or name} {sub_name}", sub, out, depth + 1)


def build() -> str:
    from zotero_mcp.cli_standalone import build_parser

    parser = build_parser()
    out = io.StringIO()
    out.write(HEADER)

    subparsers = [
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    ][0]

    # Aliases share a parser object; list them with their canonical command.
    aliases: dict[int, list[str]] = {}
    for name, sub in subparsers.choices.items():
        aliases.setdefault(id(sub), []).append(name)

    seen: set[int] = set()
    for name, sub in subparsers.choices.items():
        if id(sub) in seen:
            continue
        seen.add(id(sub))
        names = aliases[id(sub)]
        title = names[0]
        if len(names) > 1:
            title = f"{names[0]} (alias: {', '.join(names[1:])})"
        _render(title, sub, out, child_prefix=names[0])

    return out.getvalue()


if __name__ == "__main__":
    # An explicit destination lets a caller (the test suite, most of all)
    # generate without writing over the checked-in copy.
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    text = build()
    out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" is load-bearing on Windows: write_text's default
    # translates every "\n" to "\r\n", so a run of this script rewrote the
    # tracked file in CRLF. The repo is LF throughout, and read_text()
    # normalizes on the way back in, so nothing downstream reported the drift.
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    shown = out.relative_to(REPO) if out.is_relative_to(REPO) else out
    print(f"wrote {shown} ({len(text)} chars)")
