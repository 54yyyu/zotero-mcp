"""Tests for optional toolset selection (``ZOTERO_MCP_TOOLSETS``).

Two things are being protected here:

1. The spec parser, which decides what an operator's configuration means.
2. The registry itself. FastMCP silently ignores unknown names in
   ``enable``/``disable``, so a tool renamed in ``tools/*.py`` without a
   matching update to ``toolsets.py`` would leave a dead entry and quietly
   ship in the default profile. :func:`validate_toolsets` turns that into a
   test failure.
"""

from __future__ import annotations

import asyncio

import pytest

from zotero_mcp.toolsets import (
    DEFAULT_ON,
    TOOLSETS,
    TOOLSETS_ENV_VAR,
    UnknownToolsetError,
    apply_toolsets,
    optional_tool_names,
    resolve_enabled,
    validate_toolsets,
)

CONNECTOR = "chatgpt-connector"


class TestResolveEnabled:
    def test_unset_uses_default_profile(self, monkeypatch):
        monkeypatch.delenv(TOOLSETS_ENV_VAR, raising=False)
        assert resolve_enabled() == set(DEFAULT_ON)

    def test_blank_value_is_treated_as_unset(self):
        # ZOTERO_MCP_TOOLSETS= (empty) must not silently mean "core only";
        # an operator clearing the variable expects defaults back.
        assert resolve_enabled("   ") == set(DEFAULT_ON)

    def test_all_enables_every_group(self):
        # stdio, so the transport-scoped connector group stays off.
        assert resolve_enabled("all") == set(TOOLSETS) - {CONNECTOR}

    def test_none_is_core_only(self):
        assert resolve_enabled("none") == set()

    def test_explicit_groups(self):
        assert resolve_enabled("scite,feeds") == {"scite", "feeds"}

    def test_explicit_selection_replaces_defaults(self):
        # Naming a group opts out of the default profile rather than adding
        # to it, so the result is predictable from the spec alone.
        assert resolve_enabled("scite") == {"scite"}

    def test_negation_after_all(self):
        expected = set(TOOLSETS) - {CONNECTOR, "scite"}
        assert resolve_enabled("all,-scite") == expected

    def test_whitespace_and_case_insensitive(self):
        assert resolve_enabled("  SCITE   Feeds ") == {"scite", "feeds"}

    def test_unknown_toolset_raises_with_valid_values(self):
        with pytest.raises(UnknownToolsetError) as exc:
            resolve_enabled("nope")
        message = str(exc.value)
        assert "nope" in message
        assert "scite" in message  # lists the valid options

    def test_env_var_is_read_when_raw_is_none(self, monkeypatch):
        monkeypatch.setenv(TOOLSETS_ENV_VAR, "feeds")
        assert resolve_enabled() == {"feeds"}


class TestConnectorTransportScoping:
    def test_off_for_stdio(self):
        assert CONNECTOR not in resolve_enabled("all", transport="stdio")

    @pytest.mark.parametrize("transport", ["streamable-http", "sse", "http"])
    def test_on_for_http_transports(self, transport):
        assert CONNECTOR in resolve_enabled("", transport=transport)

    def test_explicit_request_wins_over_stdio_default(self):
        assert CONNECTOR in resolve_enabled("chatgpt-connector", transport="stdio")

    def test_explicit_negation_wins_over_http_default(self):
        enabled = resolve_enabled("all,-chatgpt-connector", transport="streamable-http")
        assert CONNECTOR not in enabled


class TestToolsetRegistry:
    def test_groups_are_disjoint(self):
        seen: dict[str, str] = {}
        for group, tools in TOOLSETS.items():
            for tool in tools:
                assert tool not in seen, f"{tool} in both {seen.get(tool)} and {group}"
                seen[tool] = group

    def test_default_on_names_are_real_groups(self):
        assert DEFAULT_ON <= set(TOOLSETS)

    def test_registry_matches_live_tools(self):
        """Every name in TOOLSETS must still be a registered tool."""
        from zotero_mcp.server import mcp

        # list_tools reflects the applied profile, so re-enable everything
        # first; otherwise disabled groups would look like drift.
        mcp.enable(names=optional_tool_names())
        registered = {t.name for t in asyncio.run(mcp.list_tools())}
        stale = validate_toolsets(registered)
        assert not stale, (
            f"toolsets.py references tools that no longer exist: {stale}. "
            "Update TOOLSETS after renaming or removing a tool."
        )


class TestApplyToolsets:
    def test_apply_hides_disabled_groups_and_is_reversible(self):
        from zotero_mcp.server import mcp

        def listed() -> set[str]:
            return {t.name for t in asyncio.run(mcp.list_tools())}

        try:
            apply_toolsets(mcp, raw="none", transport="stdio")
            core_only = listed()
            assert not (core_only & optional_tool_names())

            apply_toolsets(mcp, raw="all", transport="streamable-http")
            everything = listed()
            assert optional_tool_names() <= everything
            assert len(everything) > len(core_only)

            # Idempotent: re-applying the same spec is stable, and a later
            # call fully supersedes an earlier one.
            apply_toolsets(mcp, raw="none", transport="stdio")
            assert listed() == core_only
        finally:
            apply_toolsets(mcp, raw="all", transport="streamable-http")

    def test_default_profile_is_smaller_than_full_surface(self):
        from zotero_mcp.server import mcp

        try:
            apply_toolsets(mcp, raw="all", transport="streamable-http")
            full = len(asyncio.run(mcp.list_tools()))
            apply_toolsets(mcp, raw=None, transport="stdio")
            default = len(asyncio.run(mcp.list_tools()))
            assert default < full
        finally:
            apply_toolsets(mcp, raw="all", transport="streamable-http")
