"""Optional tool groups ("toolsets") and the logic that decides which ship.

Every registered ``@mcp.tool`` is sent to the client on *every* request as
part of the tool list, so the tool surface is a fixed tax on each session's
context window. The full surface costs roughly 23k tokens; a reading or
literature-review session typically touches a small fraction of it.

This module carves the optional groups out of that surface. Anything not
named here is **core** and always available, so merging or renaming a core
tool never requires touching this file. Only the opt-in groups are
enumerated, and :func:`validate_toolsets` (exercised by the test suite)
fails loudly if a name here ever drifts from the real tool registry —
FastMCP silently ignores unknown names, so drift is otherwise invisible.

Selection happens through the ``ZOTERO_MCP_TOOLSETS`` environment variable:

===========================  ==================================================
Value                        Effect
===========================  ==================================================
unset                        Core plus :data:`DEFAULT_ON` (the default profile)
``all``                      Every tool, matching pre-0.9 behaviour
``none``                     Core only
``scite,feeds``              Core plus the named groups
``all,-scite``               Every group except the negated ones
===========================  ==================================================

Names are case-insensitive and may be separated by commas or whitespace.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastmcp import FastMCP

#: Environment variable that selects which optional toolsets are enabled.
TOOLSETS_ENV_VAR = "ZOTERO_MCP_TOOLSETS"

#: Optional tool groups, keyed by toolset name.
#:
#: Tools absent from every group are core and always enabled. Keep each group
#: coherent: a user turning one on should get a whole capability, not a
#: fragment of one.
TOOLSETS: dict[str, frozenset[str]] = {
    # Scite.ai citation-tally and retraction data. No account needed, but it
    # calls out to scite.ai and wants the optional `[scite]` extra installed.
    "scite": frozenset(
        {
            "scite_check_retractions",
            "scite_enrich_item",
            "scite_enrich_search",
        }
    ),
    # Library hygiene. Valuable, but a maintenance task rather than something
    # a research session reaches for.
    "duplicates": frozenset(
        {
            "zotero_find_duplicates",
            "zotero_merge_duplicates",
        }
    ),
    # Corpus-level exploration built on the semantic index.
    "discovery": frozenset(
        {
            "zotero_find_related_papers",
            "zotero_library_coverage",
        }
    ),
    # Zotero RSS feed subscriptions.
    "feeds": frozenset(
        {
            "zotero_list_feeds",
            "zotero_get_feed_items",
        }
    ),
    # Explicit item-to-item relations ("related items" in the Zotero UI).
    "relations": frozenset(
        {
            "zotero_add_item_relation",
            "zotero_remove_item_relation",
            "zotero_get_item_related",
        }
    ),
    # Group/personal library enumeration and switching. Only useful to users
    # who actually belong to group libraries.
    "libraries": frozenset(
        {
            "zotero_list_libraries",
            "zotero_switch_library",
        }
    ),
    # Semantic-index administration. The same operations are available from
    # the CLI (``zotero-mcp update-db``), so agent access is a convenience.
    "search-admin": frozenset(
        {
            "zotero_update_search_database",
            "zotero_get_search_database_status",
        }
    ),
    # PDF page geometry. Pairs with area annotations, which need a coordinate
    # space, and with outline-driven navigation of long documents.
    "pdf-geometry": frozenset(
        {
            "zotero_get_page_layout",
            "zotero_get_pdf_outline",
        }
    ),
    # The ChatGPT deep-research connector contract, which requires tools named
    # exactly ``search`` and ``fetch``. Meaningless over stdio, so this group
    # is transport-scoped rather than listed in DEFAULT_ON; see
    # :func:`resolve_enabled`.
    "chatgpt-connector": frozenset({"search", "fetch"}),
}

#: Toolsets enabled when ``ZOTERO_MCP_TOOLSETS`` is unset.
#:
#: The default profile keeps groups that pair directly with core workflows
#: (area annotations need page geometry; semantic search needs a way to
#: report on its index) and drops groups that need external services, serve
#: maintenance rather than research, or apply only to some users.
DEFAULT_ON: frozenset[str] = frozenset(
    {
        "libraries",
        "search-admin",
        "pdf-geometry",
    }
)

#: Transports over which the ChatGPT connector contract is reachable. ChatGPT
#: connects to a served endpoint, never to a stdio subprocess.
_HTTP_TRANSPORTS = frozenset({"streamable-http", "sse", "http"})

_CONNECTOR_TOOLSET = "chatgpt-connector"


class UnknownToolsetError(ValueError):
    """Raised when ``ZOTERO_MCP_TOOLSETS`` names a group that does not exist."""


def _split(raw: str) -> list[str]:
    """Split a comma/whitespace separated toolset spec into lowercase tokens."""
    return [token.lower() for token in raw.replace(",", " ").split() if token]


def resolve_enabled(
    raw: str | None = None,
    *,
    transport: str | None = None,
) -> set[str]:
    """Return the set of optional toolsets that should be enabled.

    Args:
        raw: Raw ``ZOTERO_MCP_TOOLSETS`` value. ``None`` reads the environment;
            an empty or whitespace-only value is treated as unset so that
            ``ZOTERO_MCP_TOOLSETS=`` does not silently mean "core only".
        transport: Transport the server is about to run under. HTTP-family
            transports add the ChatGPT connector group unless the spec already
            decided it explicitly.

    Raises:
        UnknownToolsetError: If the spec names an unknown toolset. Failing here
            beats silently serving a surface the operator did not ask for.
    """
    if raw is None:
        raw = os.environ.get(TOOLSETS_ENV_VAR)

    spec = _split(raw or "")
    # Groups the spec named literally. `all` deliberately does not count: it
    # means "every group that makes sense here", which leaves the
    # transport-scoped connector group to the transport rule below.
    named: set[str] = set()

    if not spec:
        enabled = set(DEFAULT_ON)
    else:
        enabled = set()
        for token in spec:
            negated = token.startswith("-")
            name = token[1:] if negated else token

            if name == "all":
                candidates = set(TOOLSETS)
            elif name == "none":
                candidates = set()
            elif name in TOOLSETS:
                candidates = {name}
                named.add(name)
            else:
                valid = ", ".join(sorted(TOOLSETS) + ["all", "none"])
                raise UnknownToolsetError(
                    f"Unknown toolset {name!r} in {TOOLSETS_ENV_VAR}. Valid values: {valid}"
                )

            if negated:
                enabled -= candidates
            else:
                enabled |= candidates

    # ChatGPT's connector contract is reachable only over HTTP transports, and
    # its generic `search`/`fetch` names collide badly with the Zotero tools in
    # a normal session. Turn it on by transport unless the spec spoke for it.
    if _CONNECTOR_TOOLSET not in named:
        if transport is not None and transport.lower() in _HTTP_TRANSPORTS:
            enabled.add(_CONNECTOR_TOOLSET)
        else:
            enabled.discard(_CONNECTOR_TOOLSET)

    return enabled


def apply_toolsets(
    mcp: FastMCP,
    *,
    raw: str | None = None,
    transport: str | None = None,
) -> set[str]:
    """Enable/disable optional tool groups on ``mcp``; return what was enabled.

    Safe to call more than once — the enabled and disabled sets are both passed
    explicitly every time, and FastMCP resolves visibility with last-write-wins,
    so a later call fully supersedes an earlier one. That matters because the
    server applies a transport-agnostic default at import time and the CLI
    re-applies once the real transport is known.
    """
    enabled = resolve_enabled(raw, transport=transport)

    on: set[str] = set()
    off: set[str] = set()
    for name, tools in TOOLSETS.items():
        (on if name in enabled else off).update(tools)

    # Disable first so a tool appearing in two groups stays on if any of its
    # groups is enabled.
    if off:
        mcp.disable(names=off)
    if on:
        mcp.enable(names=on)

    return enabled


def optional_tool_names() -> set[str]:
    """Every tool name belonging to some optional toolset."""
    return {name for tools in TOOLSETS.values() for name in tools}


def validate_toolsets(registered: Iterable[str]) -> list[str]:
    """Return toolset entries that no longer match a registered tool.

    FastMCP ignores unknown names in ``enable``/``disable``, so a tool that is
    renamed or removed would leave a dead entry here and silently ship in the
    default profile. The test suite calls this against the live registry to
    turn that silent drift into a failure.
    """
    known = set(registered)
    return sorted(name for name in optional_tool_names() if name not in known)
