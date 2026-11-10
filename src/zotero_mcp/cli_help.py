"""The ``zotero-mcp help`` reference, generated from the argument parser.

argparse's own top-level help lists the commands but none of their options, so
the options were only discoverable one command at a time. This renders a
single grouped reference straight from the parser, which means it cannot drift
from the real command line: add an option and it shows up here.

Imports nothing heavy; ``cli`` must stay cheap to import (#445).
"""

from __future__ import annotations

import argparse
import shutil
import textwrap

#: How commands are grouped. A command missing from every group is still shown,
#: under "Other"; a test fails so it gets classified deliberately.
HELP_GROUPS: list[tuple[str, list[str]]] = [
    ("Run the server", ["serve"]),
    ("Set up and maintain", [
        "setup", "setup-info", "authorize-local", "install-skill",
        "schema-refresh", "update", "version",
    ]),
    ("Semantic search index", ["update-db", "db-status", "db-inspect", "repair-db", "migrate-index"]),
    ("Batch API indexing", ["batch-status", "batch-import"]),
    ("Deprecated (still work)", ["openai-batch-status", "openai-batch-import"]),
]

FOOTER = """\
Batch API indexing (OpenAI or Gemini, ~50% of realtime cost):
  zotero-mcp update-db --batch              Submit embeddings asynchronously
  zotero-mcp update-db --batch --auto-loop  Submit, then poll and import
  zotero-mcp batch-status                   Check submitted batch status
  zotero-mcp batch-import                   Import completed embeddings

More:
  zotero-mcp help <command>                 Show one command on its own
  zotero-mcp                                Same as 'zotero-mcp serve'
"""

_LABEL_COLUMN = 34


def _option_label(action: argparse.Action) -> str:
    names = ", ".join(action.option_strings)
    if action.nargs == 0:  # store_true / store_false / store_const
        return names
    if action.metavar:
        metavar = action.metavar
    elif action.choices:
        metavar = "{" + ",".join(str(c) for c in action.choices) + "}"
    else:
        metavar = action.dest.upper()
    return f"{names} {metavar}"


def command_options(sub: argparse.ArgumentParser) -> list[argparse.Action]:
    """The options of one command, everything but -h."""
    return [
        a for a in sub._actions
        if a.option_strings and a.dest != "help" and a.help is not argparse.SUPPRESS
    ]


def _entry(label: str, text: str, indent: int, width: int) -> list[str]:
    """``label`` and wrapped ``text``; a label too long for its column gets a line to itself."""
    body = textwrap.wrap(text, max(width - _LABEL_COLUMN, 30)) or [""]
    rest = [" " * _LABEL_COLUMN + line for line in body[1:]]
    if indent + len(label) + 2 > _LABEL_COLUMN:
        return [" " * indent + label, " " * _LABEL_COLUMN + body[0], *rest]
    return [" " * indent + label.ljust(_LABEL_COLUMN - indent) + body[0], *rest]


def render_full_help(parser: argparse.ArgumentParser, commands, width: int | None = None) -> str:
    """Every command with every option, grouped by what it is for.

    ``commands`` is the object ``parser.add_subparsers()`` returned.
    """
    width = width or min(shutil.get_terminal_size((100, 24)).columns, 110)
    summaries = {a.dest: a.help or "" for a in commands._choices_actions}

    grouped = {name for _, names in HELP_GROUPS for name in names}
    groups = list(HELP_GROUPS)
    leftover = [name for name in commands.choices if name not in grouped]
    if leftover:
        groups.append(("Other", leftover))

    out = [
        parser.description or "zotero-mcp",
        "",
        "Usage: zotero-mcp <command> [options]",
        "       zotero-mcp help <command>",
        "",
    ]
    for title, names in groups:
        out.append(title)
        out.append("-" * len(title))
        for name in names:
            sub = commands.choices.get(name)
            if sub is None:
                continue
            out.extend(_entry(name, summaries.get(name, ""), 2, width))
            for opt in command_options(sub):
                out.extend(_entry(_option_label(opt), opt.help or "", 6, width))
            out.append("")
    out.append(FOOTER.rstrip("\n"))
    return "\n".join(out) + "\n"


class FullHelpAction(argparse.Action):
    """``-h`` / ``--help`` at the top level prints the full reference.

    Pass the ``add_subparsers()`` result as ``commands=``.
    """

    def __init__(self, option_strings, commands, dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings, dest=dest, default=default, nargs=0, help=help)
        self.commands = commands

    def __call__(self, parser, namespace, values, option_string=None):
        print(render_full_help(parser, self.commands), end="")
        parser.exit()
