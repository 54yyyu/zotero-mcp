"""Tests for the security-hardening fixes from issue #326.

Covers the credential-file permission helper (#3) and the pdfannots2json
subprocess timeout handling (#5). The SSRF guard (#1) is covered in
test_pdf_cascade.py. The plaintext-key gating (#2), --api-key getpass
fallback (#4), and Dockerfile USER (#6) are exercised manually.
"""

import os
import stat
import subprocess
import tempfile

import pytest


def test_restrict_file_permissions_sets_owner_only():
    """_restrict_file_permissions tightens a world-readable file to 0o600."""
    if os.name != "posix":
        pytest.skip("POSIX file permissions only")
    from zotero_mcp.setup_helper import _restrict_file_permissions

    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        os.chmod(path, 0o644)
        _restrict_file_permissions(path)
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    finally:
        os.unlink(path)


def test_restrict_file_permissions_swallows_errors():
    """Missing file must not raise (best-effort hardening)."""
    from zotero_mcp.setup_helper import _restrict_file_permissions

    # Should not raise even though the path does not exist.
    _restrict_file_permissions("/nonexistent/path/to/config.json")


def test_new_config_directory_is_owner_only(tmp_path):
    """The directory holding the index and credentials is created 0700 (#401)."""
    if os.name != "posix":
        pytest.skip("POSIX directory permissions only")
    from zotero_mcp.utils import ensure_private_dir

    target = tmp_path / "home" / ".config" / "zotero-mcp"
    old_umask = os.umask(0o022)
    try:
        ensure_private_dir(target)
    finally:
        os.umask(old_umask)

    assert stat.S_IMODE(target.stat().st_mode) == 0o700


def test_existing_open_directory_is_left_alone_but_reported(tmp_path, caplog):
    """An existing directory's mode may be deliberate: warn once, never chmod."""
    if os.name != "posix":
        pytest.skip("POSIX directory permissions only")
    from zotero_mcp import utils

    target = tmp_path / "zotero-mcp"
    target.mkdir()
    os.chmod(target, 0o755)

    with caplog.at_level("WARNING", logger="zotero_mcp.utils"):
        utils.ensure_private_dir(target)
        utils.ensure_private_dir(target)

    assert stat.S_IMODE(target.stat().st_mode) == 0o755
    warnings = [r for r in caplog.records if "chmod 700" in r.getMessage()]
    assert len(warnings) == 1


def test_local_write_config_creates_a_private_directory(tmp_path, monkeypatch):
    """client._write_config is often the first thing to create the directory."""
    if os.name != "posix":
        pytest.skip("POSIX directory permissions only")
    from zotero_mcp import client

    config_path = tmp_path / "fresh" / "zotero-mcp" / "config.json"
    monkeypatch.setattr(client, "ZOTERO_MCP_CONFIG_PATH", config_path)
    client._write_config({"local_api": {}})

    assert stat.S_IMODE(config_path.parent.stat().st_mode) == 0o700


def test_pdfannots_timeout_returns_empty(monkeypatch):
    """A pdfannots2json timeout is handled gracefully (returns [])."""
    import zotero_mcp.pdfannots_helper as ph

    monkeypatch.setattr(ph, "ensure_pdfannots_installed", lambda: True)
    monkeypatch.setattr(ph, "get_pdfannots_executable", lambda: "/bin/true")

    def _raise_timeout(*args, **kwargs):
        # Confirm the call is bounded by a timeout.
        assert kwargs.get("timeout"), "subprocess.run must pass a timeout"
        raise subprocess.TimeoutExpired(cmd="pdfannots2json", timeout=kwargs["timeout"])

    monkeypatch.setattr(ph.subprocess, "run", _raise_timeout)

    result = ph.extract_annotations_from_pdf("/nonexistent.pdf", output_dir=".")
    assert result == []
