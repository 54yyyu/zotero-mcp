"""Regression tests: `zotero-mcp update` must never offer a downgrade.

The check was ``current_version != latest_version``, so any install ahead of
the last PyPI release — every git checkout and dev build — was told an update
was available, and running it replaced the newer code with the older release.
Ordering is what matters, not inequality.
"""

import builtins

import pytest

from zotero_mcp import updater


class TestIsNewerVersion:
    @pytest.mark.parametrize(
        "current,latest,expected",
        [
            ("0.6.3", "0.6.2", False),   # the bug: ahead of PyPI
            ("0.6.2", "0.6.3", True),
            ("0.6.3", "0.6.3", False),
            ("0.6.3", "v0.6.4", True),   # GitHub tags carry a leading v
            ("v0.6.3", "0.6.3", False),
            ("0.6.3", "0.10.0", True),   # numeric, not lexical
            ("0.10.0", "0.9.0", False),
            ("0.7.0.dev1", "0.7.0", True),
            ("0.7.0", "0.7.0rc1", False),
            (" 0.6.3 ", "0.6.3", False),
        ],
    )
    def test_ordering(self, current, latest, expected):
        assert updater.is_newer_version(current, latest) is expected

    def test_fallback_without_packaging(self, monkeypatch):
        """`packaging` is not a declared dependency — the fallback must hold."""
        real_import = builtins.__import__

        def no_packaging(name, *args, **kwargs):
            if name.startswith("packaging"):
                raise ImportError("simulated: packaging not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_packaging)

        assert updater.is_newer_version("0.6.3", "0.6.2") is False
        assert updater.is_newer_version("0.6.2", "0.6.3") is True
        assert updater.is_newer_version("0.6.3", "0.6.3") is False
        assert updater.is_newer_version("0.10.0", "0.9.0") is False


class TestUpdateCheck:
    def test_ahead_of_pypi_reports_up_to_date(self, monkeypatch):
        """The user-visible regression."""
        monkeypatch.setattr(updater, "get_current_version", lambda: "0.7.0")
        monkeypatch.setattr(updater, "get_latest_version", lambda: "0.6.3")

        result = updater.update_zotero_mcp(check_only=True)

        assert result["needs_update"] is False
        assert result["success"] is True
        assert "ahead" in result["message"]

    def test_behind_pypi_still_offers_update(self, monkeypatch):
        monkeypatch.setattr(updater, "get_current_version", lambda: "0.6.2")
        monkeypatch.setattr(updater, "get_latest_version", lambda: "0.6.3")

        result = updater.update_zotero_mcp(check_only=True)

        assert result["needs_update"] is True
        assert "0.6.2 → 0.6.3" in result["message"]

    def test_equal_versions_are_up_to_date(self, monkeypatch):
        monkeypatch.setattr(updater, "get_current_version", lambda: "0.6.3")
        monkeypatch.setattr(updater, "get_latest_version", lambda: "0.6.3")

        result = updater.update_zotero_mcp(check_only=True)

        assert result["needs_update"] is False
        assert "ahead" not in result["message"]
