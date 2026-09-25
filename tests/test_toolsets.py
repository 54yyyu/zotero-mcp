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
    PROFILE_ENV_VAR,
    PROFILES,
    TOOLSETS,
    TOOLSETS_ENV_VAR,
    UnknownProfileError,
    UnknownToolsetError,
    apply_profile,
    apply_toolsets,
    optional_tool_names,
    profile_tool_names,
    resolve_enabled,
    resolve_profile,
    validate_profiles,
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


class TestResolveProfile:
    def test_unset_is_none(self, monkeypatch):
        monkeypatch.delenv(PROFILE_ENV_VAR, raising=False)
        assert resolve_profile() is None

    def test_blank_value_is_treated_as_unset(self):
        assert resolve_profile("   ") is None

    def test_known_profile_is_normalized(self):
        assert resolve_profile("  RESEARCH ") == "research"

    def test_unknown_profile_raises_with_valid_values(self):
        with pytest.raises(UnknownProfileError) as exc:
            resolve_profile("nope")
        message = str(exc.value)
        assert "nope" in message
        assert "research" in message  # lists the valid options

    def test_env_var_is_read_when_raw_is_none(self, monkeypatch):
        monkeypatch.setenv(PROFILE_ENV_VAR, "research")
        assert resolve_profile() == "research"


class TestProfileRegistry:
    def test_registry_matches_live_tools(self):
        """Every name in PROFILES must still be a registered tool."""
        from zotero_mcp.server import mcp

        # list_tools reflects whatever is currently hidden, so re-enable
        # everything named by a profile first; otherwise a disabled tool
        # would look like drift rather than an intentional exclusion.
        mcp.enable(names=profile_tool_names())
        registered = {t.name for t in asyncio.run(mcp.list_tools())}
        stale = validate_profiles(registered)
        assert not stale, (
            f"toolsets.py PROFILES references tools that no longer exist: {stale}. "
            "Update PROFILES after renaming or removing a tool."
        )

    def test_research_profile_excludes_irreversible_deletes(self):
        # The two operations with no undo path must never be in a profile
        # meant for an unsupervised local assistant with full write access.
        assert "zotero_delete_collection" not in PROFILES["research"]
        assert "zotero_delete_annotation" not in PROFILES["research"]

    def test_research_profile_keeps_recoverable_delete(self):
        # zotero_delete_item moves to Trash, so it stays available.
        assert "zotero_delete_item" in PROFILES["research"]


class TestApplyProfile:
    """``apply_profile`` uses FastMCP's ``only=True`` allowlist mode, which
    (unlike everything ``apply_toolsets`` does) disables *every* tool via a
    match-all transform before re-enabling the named set. There is no public
    API to undo a match-all transform other than removing it, so each test
    here snapshots ``mcp.transforms`` and truncates back to it afterwards —
    otherwise a profile applied in one test would leak into every test that
    runs after it in the same process.
    """

    def test_no_profile_leaves_toolset_surface_untouched(self):
        from zotero_mcp.server import mcp

        def listed() -> set[str]:
            return {t.name for t in asyncio.run(mcp.list_tools())}

        snapshot = len(mcp.transforms)
        try:
            apply_toolsets(mcp, raw="all", transport="streamable-http")
            before = listed()
            assert apply_profile(mcp, raw=None) is None
            assert listed() == before
        finally:
            del mcp._transforms[snapshot:]
            apply_toolsets(mcp, raw="all", transport="streamable-http")

    def test_research_profile_is_a_hard_allowlist(self):
        from zotero_mcp.server import mcp

        def listed() -> set[str]:
            return {t.name for t in asyncio.run(mcp.list_tools())}

        snapshot = len(mcp.transforms)
        try:
            # Start from the full surface so this proves the profile *removes*
            # core tools, not merely that it fails to add optional ones back.
            apply_toolsets(mcp, raw="all", transport="streamable-http")
            assert apply_profile(mcp, raw="research") == "research"
            assert listed() == set(PROFILES["research"])
            assert "zotero_delete_collection" not in listed()
        finally:
            del mcp._transforms[snapshot:]
            apply_toolsets(mcp, raw="all", transport="streamable-http")

    def test_unknown_profile_raises(self):
        from zotero_mcp.server import mcp

        with pytest.raises(UnknownProfileError):
            apply_profile(mcp, raw="nope")
