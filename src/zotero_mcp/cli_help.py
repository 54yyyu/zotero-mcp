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
    ("Semantic search index", ["update-db", "db-status", "db-inspect", "repair-db"]),
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


def _subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise ValueError("parser has no subcommands")


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


def _help_text(action: argparse.Action) -> str:
    text = action.help or ""
    try:
        return text % {**vars(action), "default": action.default}
    except (KeyError, ValueError, TypeError):
        return text


def command_options(sub: argparse.ArgumentParser) -> list[argparse.Action]:
    """The user-facing options of one command (everything but -h)."""
    return [
        a for a in sub._actions
        if a.option_strings
        and not isinstance(a, argparse._HelpAction)
        and a.help is not argparse.SUPPRESS
    ]


def _entry(label: str, text: str, indent: int, width: int) -> list[str]:
    pad = " " * indent
    column = max(_LABEL_COLUMN, indent + len(label) + 2)
    if indent + len(label) + 2 > _LABEL_COLUMN:  # long label: text on the next line
        lines = [pad + label]
        column = _LABEL_COLUMN
        first_prefix = " " * column
    else:
        lines = []
        first_prefix = pad + label + " " * (column - indent - len(label))
    body = textwrap.wrap(text, max(width - column, 30)) or [""]
    lines.append(first_prefix + body[0])
    lines.extend(" " * column + line for line in body[1:])
    return lines


def render_full_help(parser: argparse.ArgumentParser, width: int | None = None) -> str:
    """Every command with every option, grouped by what it is for."""
    width = width or min(shutil.get_terminal_size((100, 24)).columns, 110)
    action = _subparsers_action(parser)
    commands = action.choices
    summaries = {a.dest: a.help or "" for a in action._choices_actions}

    grouped = {name for _, names in HELP_GROUPS for name in names}
    groups = list(HELP_GROUPS)
    leftover = [name for name in commands if name not in grouped]
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
            sub = commands.get(name)
            if sub is None:
                continue
            out.extend(_entry(name, summaries.get(name, ""), 2, width))
            for opt in command_options(sub):
                out.extend(_entry(_option_label(opt), _help_text(opt), 6, width))
            out.append("")
    out.append(FOOTER.rstrip("\n"))
    return "\n".join(out) + "\n"


class FullHelpAction(argparse.Action):
    """``-h`` / ``--help`` at the top level prints the full reference."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings, dest=dest, default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        print(render_full_help(parser), end="")
        parser.exit()
