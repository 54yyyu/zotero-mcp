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
